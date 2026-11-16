"""
sphere_oxygen_connect_MPO.py

Mn-O cluster connectivity analysis / Mn-O 团簇连通性分析，
用于研究 0-TM channel。

Pipeline per supercell / 单个超胞处理流程：
    1. modify_structure          : strip Li/Na, Ni→Mn, classify O as O or Cl
                                   去 Li/Na，Ni→Mn，按 Mn 配位数将 O 分类
    2. extract_sphere            : carve an R-radius sphere around a random Mn
                                   以随机 Mn 为中心，取半径 R 的球形区域
    3. build_cl_graph            : connect Cl atoms sharing an Mn neighbor
                                   在 Cl 原子间建图（共享 Mn 即视为相邻）
    4. largest_component_size    : size of the biggest connected component
                                   求最大连通分量大小
    5. replace_component_with_f  : boundary-touching Cl clusters → F
                                   将与球边界相交的连通分量中的 Cl 替换为 F

Reported ratios / 输出比值（均针对 Replaced 结构）：
    replace_cl_f_ratio = (Cl+F) / (Cl+F+O)   —— 球内非桥接位点占原本 O 的比例
    replace_f_ratio    = F     / (Cl+F)      —— 原 Cl 中被替换为 F 的比例

    (注：Empty 侧的 Cl/(O+Cl) 与 (Cl+F)/(Cl+F+O) 完全等价——替换只把 Cl 变为 F，
    O 的数目在替换前后不变，故不再单独输出 Empty 侧比值。)
"""

from __future__ import annotations

import concurrent.futures
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import networkx as nx
import numpy as np
from ase import Atoms
from ase.build import sort
from ase.io import read, write
from ase.neighborlist import NeighborList


# ---------------------------------------------------------------------------
# Constants / 常量
# ---------------------------------------------------------------------------

# Neighbor cutoffs used in modify_structure (Å) / modify_structure 的邻居半径（Å）
RADIUS_MN_STRUCTURE = 1.4
RADIUS_P = 1.2
RADIUS_O = 1.4

# Neighbor cutoffs used in build_cl_graph (Å) / build_cl_graph 的邻居半径（Å）
RADIUS_CL_GRAPH = 1.7
RADIUS_MN_GRAPH = 1.2

# Padding added to the sphere's bounding cube (Å) / 球形外接正方体的额外留边（Å）
SPHERE_PADDING = 5.0


# ---------------------------------------------------------------------------
# Internal helpers / 内部工具
# ---------------------------------------------------------------------------

def _element_cutoffs(structure: Atoms, radii: dict[str, float]) -> list[float]:
    """Per-atom cutoff array for ``NeighborList``.
    按原子元素返回 ``NeighborList`` 所需 cutoff 数组。
    """
    return [radii.get(atom.symbol, 0.0) for atom in structure]


def _count_symbols(structure: Atoms, symbols: Iterable[str]) -> int:
    """Count atoms whose symbol belongs to ``symbols``.
    统计符号属于 ``symbols`` 集合的原子数。
    """
    wanted = set(symbols)
    return sum(1 for a in structure if a.symbol in wanted)


def _chunk_ranges(n: int, p: int) -> list[tuple[int, int]]:
    """Split ``[0, n)`` into ``p`` roughly equal ``(start, end)`` chunks.
    将 ``[0, n)`` 大致均分为 ``p`` 个 ``(start, end)`` 区间。

    Using ceil-div instead of ``n // p`` guarantees coverage of the full
    range even when ``n`` is not a multiple of ``p``.
    使用向上取整而非 ``n // p``，避免 ``n`` 不能被 ``p`` 整除时遗漏尾部样本。
    """
    chunk = (n + p - 1) // p
    ranges: list[tuple[int, int]] = []
    for i in range(p):
        start = i * chunk
        if start >= n:
            break
        ranges.append((start, min(n, start + chunk)))
    return ranges


# ---------------------------------------------------------------------------
# Step 1 — structure pre-processing / 结构预处理
# ---------------------------------------------------------------------------

def modify_structure(structure: Atoms) -> Atoms:
    """Convert the structure into the form required for cluster analysis.
    将结构转换为团簇分析所需形态（就地修改并返回）。

    - Delete all Li / Na atoms.                删除所有 Li / Na。
    - Replace Ni with Mn.                      Ni → Mn。
    - For each O atom:                         对每个 O：
        * if adjacent to P              → keep O    与 P 相邻           → 保持 O
        * else if Mn-CN ∈ {0, 1, 2}     → become Cl 否则 Mn 配位数 ≤ 2   → 改为 Cl
        * else (Mn-CN ≥ 3)              → keep O    否则 Mn 配位数 ≥ 3   → 保持 O
    """
    # Remove Li / Na / 删除 Li、Na
    del structure[[atom.index for atom in structure
                   if atom.symbol in ('Li', 'Na')]]

    # Ni → Mn / Ni 替换为 Mn
    for atom in structure:
        if atom.symbol == 'Ni':
            atom.symbol = 'Mn'

    # Pre-collect indices / 预收集各类原子索引
    p_indices = {i for i, a in enumerate(structure) if a.symbol == 'P'}
    mn_indices = {i for i, a in enumerate(structure) if a.symbol == 'Mn'}
    o_indices = [i for i, a in enumerate(structure) if a.symbol == 'O']

    # Build neighbor list over Mn/P/O / 构建 Mn-P-O 邻居表
    cutoffs = _element_cutoffs(structure, {
        'Mn': RADIUS_MN_STRUCTURE,
        'P':  RADIUS_P,
        'O':  RADIUS_O,
    })
    nl = NeighborList(cutoffs, self_interaction=False, bothways=True, skin=0)
    nl.update(structure)

    # Decide per-O symbol / 逐个 O 判定最终符号
    for o_idx in o_indices:
        neighbors, _ = nl.get_neighbors(o_idx)
        num_mn = sum(1 for n in neighbors if n in mn_indices)
        num_p = sum(1 for n in neighbors if n in p_indices)

        if num_p > 0:            # P-adjacent O never changes / 与 P 相邻的 O 永不改动
            continue
        if num_mn <= 2:          # only Mn-CN ≤ 2 becomes Cl / 仅 Mn-CN ≤ 2 转为 Cl
            structure[o_idx].symbol = 'Cl'

    return structure


# ---------------------------------------------------------------------------
# Step 2 — spherical truncation / 球形截断
# ---------------------------------------------------------------------------

def _select_atoms_in_sphere(
    structure: Atoms, center_idx: int, radius: float,
) -> tuple[list[np.ndarray], list[str], np.ndarray]:
    """Return in-sphere positions, symbols, and the original center.
    返回球内原子位置、符号，以及原始中心坐标。
    """
    center = structure[center_idx].position
    cell = structure.cell

    # Only the center has a non-zero cutoff → single-target search.
    # 仅中心原子有非零 cutoff → 只搜索该原子的邻域。
    cutoffs = [radius if i == center_idx else 0.0
               for i in range(len(structure))]
    nl = NeighborList(cutoffs, self_interaction=True, bothways=True, skin=0)
    nl.update(structure)

    indices, offsets = nl.get_neighbors(center_idx)

    positions: list[np.ndarray] = []
    symbols: list[str] = []
    for idx, offset in zip(indices, offsets):
        # Apply periodic image / 应用周期性镜像
        pos = structure[idx].position + offset @ cell
        if np.linalg.norm(pos - center) <= radius:
            positions.append(pos)
            symbols.append(structure[idx].symbol)

    return positions, symbols, center


def extract_sphere(structure: Atoms, radius: float) -> Atoms:
    """Carve an R-sphere around a random Mn and wrap it into a new cube.
    以随机 Mn 为中心提取半径 R 的球形区域，装入新立方晶胞。
    """
    mn_indices = [i for i, a in enumerate(structure) if a.symbol == 'Mn']
    if not mn_indices:
        raise ValueError("No Mn atoms to center the sphere. / 找不到可用于取球的 Mn。")

    center_idx = random.choice(mn_indices)
    positions, symbols, center_pos = _select_atoms_in_sphere(
        structure, center_idx, radius,
    )

    # Bounding cube / 球形外接正方体
    box_length = 2 * radius + SPHERE_PADDING
    new_cell = np.eye(3) * box_length
    box_center = np.full(3, box_length / 2.0)

    # Translate center to box center, wrap to [0, L) / 把球心平移到盒心并按周期折回
    shift = box_center - center_pos
    shifted = [(p + shift) % box_length for p in positions]

    return Atoms(positions=shifted, symbols=symbols,
                 cell=new_cell, pbc=True)


# ---------------------------------------------------------------------------
# Step 3 — Cl connectivity graph / Cl 连通图
# ---------------------------------------------------------------------------

def build_cl_graph(structure: Atoms) -> nx.Graph:
    """Build the Cl graph: two Cl atoms sharing ≥1 Mn neighbor are adjacent,
    **provided** they are also spatial neighbours (Cl–Cl < 2×RADIUS_CL_GRAPH = 3.4 Å).

    构建 Cl 图：两个 Cl 需同时满足
      (1) 空间相邻：Cl–Cl 距离 < 2 * RADIUS_CL_GRAPH = 3.4 Å
      (2) 共享至少一个 Mn 邻居（Mn–Cl < RADIUS_CL_GRAPH + RADIUS_MN_GRAPH = 2.9 Å）
    才连边。
    —— 与原始 analyze_oxygen_network_in_sphere.py 中 modify_graph 的判断一致。
    """
    cl_indices = [i for i, a in enumerate(structure) if a.symbol == 'Cl']

    graph = nx.Graph()
    graph.add_nodes_from(cl_indices, element='Cl')
    if not cl_indices:
        return graph

    cutoffs = _element_cutoffs(structure, {
        'Cl': RADIUS_CL_GRAPH,       # 1.7 Å
        'Mn': RADIUS_MN_GRAPH,       # 1.2 Å
    })
    nl = NeighborList(cutoffs, self_interaction=False, bothways=True, skin=0)
    nl.update(structure)

    # 对每个 Cl 同时收集：Mn 邻居集合、Cl 空间邻居集合
    cl_to_mn: dict[int, set[int]] = {}
    cl_to_cl: dict[int, set[int]] = {}
    for cl in cl_indices:
        neighbors, _ = nl.get_neighbors(cl)
        mn_set, cl_set = set(), set()
        for n in neighbors:
            sym = structure[n].symbol
            if sym == 'Mn':
                mn_set.add(n)
            elif sym == 'Cl':
                cl_set.add(n)
        cl_to_mn[cl] = mn_set
        cl_to_cl[cl] = cl_set

    # 双重判据：空间相邻 AND 共享 Mn
    for cl, spatial_neighbors in cl_to_cl.items():
        for other in spatial_neighbors:
            if other <= cl:                      # 每条边只处理一次
                continue
            if cl_to_mn[cl] & cl_to_mn[other]:   # 共享 Mn 才连边
                graph.add_edge(cl, other)

    return graph

def largest_component_size(graph: nx.Graph) -> int:
    """Size of the largest connected component (0 if empty).
    返回最大连通分量的节点数（空图返回 0）。
    """
    return max((len(c) for c in nx.connected_components(graph)), default=0)


# ---------------------------------------------------------------------------
# Step 4/5 — boundary detection & F replacement / 边界识别与 F 替换
# ---------------------------------------------------------------------------

def find_cl_near_boundary(structure: Atoms, radius: float) -> list[int]:
    """Return Cl indices with distance-to-center ∈ (R-1, R].
    返回到晶胞中心距离在 (R-1, R] 内的 Cl 原子索引。
    """
    center = structure.get_cell().diagonal() / 2.0
    return [
        i for i, atom in enumerate(structure)
        if atom.symbol == 'Cl'
        and radius - 1 < np.linalg.norm(atom.position - center) <= radius
    ]


def replace_component_with_f(
    structure: Atoms,
    seed_indices: Iterable[int],
    graph: nx.Graph,
) -> tuple[Atoms, list[int]]:
    """Replace every Cl atom in the same component(s) as ``seed_indices`` by F.
    将与 ``seed_indices`` 同属一个连通分量的所有 Cl 替换为 F。

    Returns the new structure and the sorted list of affected indices.
    返回新结构与受影响的节点索引（已排序）。
    """
    seed_set = set(seed_indices)
    if not seed_set:
        return structure.copy(), []

    # Collect every component that touches any seed node.
    # 收集所有与 seed 有交集的连通分量。
    affected: set[int] = set()
    for component in nx.connected_components(graph):
        if component & seed_set:
            affected |= component

    new_structure = structure.copy()
    for idx in affected:
        if new_structure[idx].symbol == 'Cl':
            new_structure[idx].symbol = 'F'

    return new_structure, sorted(affected)


# ---------------------------------------------------------------------------
# Per-structure processing / 单结构处理
# ---------------------------------------------------------------------------

@dataclass
class StructureResult:
    """Statistics from one supercell (Replace side only).
    单个超胞的统计结果（仅 Replace 侧；默认值用于失败样本占位）。

    说明：Empty 侧 Cl/(O+Cl) 与 Replace 侧 (Cl+F)/(Cl+F+O) 数值完全相等，
    因此只保留 Replace 侧两个比值。
    """
    # Replaced-sphere counts / Replaced 结构计数
    o_count:    int = 0    # number of O    / 球内 O 数（替换前后不变）
    cl_f_count: int = 0    # number of Cl+F / 球内 Cl+F 数（= 替换前的 Cl 数）
    f_count:    int = 0    # number of F    / 球内 F 数

    # Cluster info / 团簇信息
    size_1:     int = 0    # largest Cl-component / Cl 图最大连通分量大小

    # Ratios / 两个比值
    replace_cl_f_ratio: float = 0.0   # (Cl+F) / (Cl+F+O)
    replace_f_ratio:    float = 0.0   # F     / (Cl+F)


def process_structure(
    idx: int,
    radius: float,
    input_dir: Path,
    output_dir: Path,
) -> StructureResult:
    """Run the full pipeline on a single supercell.
    对单个超胞执行完整流程。
    """
    # Read input supercell / 读取输入超胞
    structure = read(input_dir / f'supercell_3_{idx}.vasp', format='vasp')

    # Step 1-2: pre-process then carve sphere / 预处理 + 球形截断
    modified = modify_structure(structure)
    sphere = extract_sphere(modified, radius)
    write(output_dir / f'Empty_R{radius:g}_{idx}.vasp',
          images=sort(sphere), direct=True)

    # Step 3: build Cl graph and size of largest component / 构建 Cl 图并求最大分量
    graph  = build_cl_graph(sphere)
    size_1 = largest_component_size(graph)

    # Step 4-5: find boundary Cl, replace their whole components with F
    #           找到边界 Cl，整分量替换为 F
    near_boundary = find_cl_near_boundary(sphere, radius)
    replaced, _ = replace_component_with_f(sphere, near_boundary, graph)
    write(output_dir / f'Replace_R{radius:g}_{idx}.vasp',
          images=sort(replaced), direct=True)

    # Replaced-sphere counts / Replaced 结构计数
    n_o_replace    = _count_symbols(replaced, ('O',))
    n_cl_f_replace = _count_symbols(replaced, ('Cl', 'F'))
    n_f_replace    = _count_symbols(replaced, ('F',))

    # Ratios (guard against zero denominators) / 计算比值，防零除
    denom_cl_f = n_cl_f_replace                     # Cl+F
    denom_all  = n_o_replace + n_cl_f_replace       # O+Cl+F

    return StructureResult(
        o_count=n_o_replace,
        cl_f_count=n_cl_f_replace,
        f_count=n_f_replace,
        size_1=size_1,
        replace_cl_f_ratio=(n_cl_f_replace / denom_all) if denom_all else 0.0,
        replace_f_ratio=(n_f_replace / denom_cl_f) if denom_cl_f else 0.0,
    )


def process_batch(
    start: int,
    end: int,
    radius: float,
    input_dir: Path,
    output_dir: Path,
) -> list[StructureResult]:
    """Process index range ``[start, end)``; one failure does not abort the batch.
    处理索引区间 ``[start, end)``，单结构失败不影响整批。
    """
    results: list[StructureResult] = []
    for i in range(start, end):
        try:
            r = process_structure(i, radius, input_dir, output_dir)
            print(f"{i}  O={r.o_count}  Cl+F={r.cl_f_count}  "
                  f"F={r.f_count}  size_1={r.size_1}  "
                  f"cl_f_ratio={r.replace_cl_f_ratio:.4f}  "
                  f"f_ratio={r.replace_f_ratio:.4f}")
            results.append(r)
        except Exception as exc:                    # noqa: BLE001
            print(f"[Error] structure {i}: {exc}")
            results.append(StructureResult())
    return results


# ---------------------------------------------------------------------------
# Orchestration / 编排
# ---------------------------------------------------------------------------

@dataclass
class Config:
    """Run configuration / 运行配置。"""
    n_structures: int = 100
    radius:       float = 30.0
    n_processes:  int = 50
    input_dir:    Path = Path('..')
    output_dir:   Path = Path('.')


def run(config: Config) -> list[StructureResult]:
    """Run all supercells in parallel and return results in original order.
    并行处理全部超胞，并按原始索引顺序返回结果。
    """
    n = config.n_structures
    p = config.n_processes
    ranges = _chunk_ranges(n, p)

    # Pre-fill placeholders so we can splice by index / 预填占位，便于按下标回填
    results: list[StructureResult] = [StructureResult() for _ in range(n)]

    with concurrent.futures.ProcessPoolExecutor(max_workers=p) as executor:
        futures = {
            executor.submit(
                process_batch, start, end,
                config.radius, config.input_dir, config.output_dir,
            ): (start, end)
            for start, end in ranges
        }

        for future in concurrent.futures.as_completed(futures):
            start, end = futures[future]
            results[start:end] = future.result()

    return results


def summarize(results: list[StructureResult]) -> None:
    """Print mean / variance / std for each metric.
    打印各指标的均值 / 方差 / 标准差。

    Note: ``numpy.var/std`` default to *population* statistics (ddof=0),
    matching the original script. Use ``ddof=1`` for sample statistics.
    注：``numpy.var/std`` 默认是总体统计（ddof=0），与原脚本一致；
    若需样本统计请改用 ``ddof=1``。
    """
    def collect(attr: str) -> np.ndarray:
        return np.array([getattr(r, attr) for r in results])

    def show(name: str, values: np.ndarray) -> None:
        if values.size == 0:
            print(f"  {name:<22} (no data)")
            return
        print(f"  {name:<22} {values.mean():>10.4f}  "
              f"{values.var():>10.4f}  {values.std():>10.4f}")

    print(f"result:                    mean      variance         std   "
          f"(N={len(results)})")
    print("-" * 62)
    show("o_count",             collect("o_count"))
    show("cl_f_count",          collect("cl_f_count"))
    show("f_count",             collect("f_count"))
    show("size_1",              collect("size_1"))
    print("-" * 62)
    show("replace_cl_f_ratio",  collect("replace_cl_f_ratio"))
    show("replace_f_ratio",     collect("replace_f_ratio"))


if __name__ == '__main__':
    cfg = Config()
    summarize(run(cfg))
