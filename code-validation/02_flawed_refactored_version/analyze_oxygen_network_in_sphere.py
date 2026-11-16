# -*- coding: utf-8 -*-
"""
分析晶体结构中氧原子配位环境及氯原子（Cl）连通性，并统计相关量的分布。
流程：
1. 读取一系列超胞结构（NiO666 变体），删除 Li/Na，将 Ni 替换为 Mn。
2. 根据 Mn 的配位数将 O 分类，并将非桥接 O（不与 P 相邻）标记为 Cl。
3. 以随机 Mn 为中心，截取半径 R 的球形区域，构建新晶胞。
4. 构建 Cl 原子图（共享 Mn 邻居 AND Cl-Cl 空间相邻），计算最大连通分量大小。
5. 找出位于球壳边界附近的 Cl，将其所在连通分量的所有 Cl 替换为 F。
6. 统计最大连通分量大小和替换 Cl 数量及其与原始 O 原子数的比例。

并行版：ProcessPoolExecutor + 分块处理，每个 idx 独立随机种子。
"""

import random
import concurrent.futures
import numpy as np
import networkx as nx
from ase.io import read, write
from ase.build import sort
from ase import Atoms
from ase.neighborlist import NeighborList

# ----------------------------------------------------------------------
# 常量定义（原子半径，单位 Å）
RADIUS_MN = 1.4       # Mn-O 邻居判断半径
RADIUS_P  = 1.2       # P-O 邻居判断半径（修复：此前 P 半径=0 导致 P-O 截断仅 1.4 Å）
RADIUS_O  = 1.4       # O 的邻居半径（用于 O-Mn 检测）
RADIUS_CL = 1.7       # Cl-Mn 邻居判断半径（Cl 稍大）
RADIUS_MN_FOR_CL = 1.2 # Mn 在 Cl-Mn 检测中的半径
# ----------------------------------------------------------------------


# ----------------------------------------------------------------------
# 步骤 1：结构预处理
# ----------------------------------------------------------------------
def modify_structure(lattice):
    """
    修改原始结构：
      - 删除所有 Li 和 Na 原子
      - 将 Ni 替换为 Mn
      - 将所有 O 改为 Cl，但保留与 P 相邻的 O（保护 P-O 键）。

    修复：P 的邻居半径由此前 0.0 改为 RADIUS_P=1.2，
          使 P-O 判据截断 = 1.4 + 1.2 = 2.6 Å（此前仅 1.4 Å，P-O 键 1.5+ Å 测不到）。
    """
    del_idx = [atom.index for atom in lattice if atom.symbol in ('Li', 'Na')]
    del lattice[del_idx]

    for atom in lattice:
        if atom.symbol == 'Ni':
            atom.symbol = 'Mn'

    o_indices = [i for i, atom in enumerate(lattice) if atom.symbol == 'O']
    p_indices = {i for i, atom in enumerate(lattice) if atom.symbol == 'P'}

    # 为每个原子分配邻居半径
    radii = []
    for atom in lattice:
        if atom.symbol == 'Mn':
            radii.append(RADIUS_MN)
        elif atom.symbol == 'P':
            radii.append(RADIUS_P)       # ← 修复：P 也参与邻居检测
        elif atom.symbol == 'O':
            radii.append(RADIUS_O)
        else:
            radii.append(0.0)

    nl = NeighborList(radii, self_interaction=False, bothways=True, skin=0)
    nl.update(lattice)

    # 找出与 P 相邻的 O
    p_adjacent_oxygens = set()
    for idx in o_indices:
        neighbors, _ = nl.get_neighbors(idx)
        if any(n in p_indices for n in neighbors):
            p_adjacent_oxygens.add(idx)

    # 所有 O → Cl
    for idx in o_indices:
        lattice[idx].symbol = 'Cl'
    # P 相邻的 O 改回 O
    for idx in p_adjacent_oxygens:
        lattice[idx].symbol = 'O'

    return lattice


# ----------------------------------------------------------------------
# 步骤 2：球形截取
# ----------------------------------------------------------------------
def select_atoms_in_sphere(atoms, center_index, radius):
    """选以 center_index 为心、半径 radius 内的所有原子（考虑 PBC）。"""
    center = atoms[center_index].position
    cell = atoms.cell

    radii = [radius if i == center_index else 0.0 for i in range(len(atoms))]
    nl = NeighborList(radii, self_interaction=True, bothways=True, skin=0)
    nl.update(atoms)

    indices, offsets = nl.get_neighbors(center_index)
    positions, symbols = [], []
    for idx, off in zip(indices, offsets):
        pos = atoms[idx].position + np.dot(off, cell)
        if np.linalg.norm(pos - center) <= radius:
            positions.append(pos)
            symbols.append(atoms[idx].symbol)

    return positions, symbols, center


def get_sphere_str(atoms, radius):
    """随机 Mn 为中心截取球体，装入 (2R+5) 立方晶胞。"""
    mn_indices = [i for i, atom in enumerate(atoms) if atom.symbol == 'Mn']
    if not mn_indices:
        raise ValueError("No Mn atoms to center the sphere.")
    center_idx = random.choice(mn_indices)

    positions, symbols, original_center = select_atoms_in_sphere(atoms, center_idx, radius)

    cell_length = 2 * radius + 5.0
    new_cell = np.eye(3) * cell_length
    new_center = np.array([cell_length / 2.0] * 3)

    shift = new_center - original_center
    shifted_pos = [(pos + shift) % cell_length for pos in positions]

    return Atoms(positions=shifted_pos, symbols=symbols, cell=new_cell, pbc=True)


# ----------------------------------------------------------------------
# 步骤 3：Cl 图构建（关键：同时要求 Cl-Cl 空间相邻 且 共享 Mn）
# ----------------------------------------------------------------------
def modify_graph(lattice):
    """
    构建 Cl 原子图。两个 Cl 原子需同时满足：
      1) Cl-Cl 空间距离 < RADIUS_CL * 2 = 3.4 Å
      2) 共享至少一个 Mn 邻居（Cl-Mn 距离 < 1.7 + 1.2 = 2.9 Å）
    才连边。

    —— 与原始 analyze_oxygen_network_in_sphere.py 的 modify_graph 判断一致；
       特别地，trans 位（Cl-Cl ≈ 4.2 Å > 3.4 Å）的 Cl 对不会连边。
    """
    cl_indices = [i for i, atom in enumerate(lattice) if atom.symbol == 'Cl']
    if not cl_indices:
        return nx.Graph()

    radii = []
    for atom in lattice:
        if atom.symbol == 'Cl':
            radii.append(RADIUS_CL)
        elif atom.symbol == 'Mn':
            radii.append(RADIUS_MN_FOR_CL)
        else:
            radii.append(0.0)

    nl = NeighborList(radii, self_interaction=False, bothways=True, skin=0)
    nl.update(lattice)

    cl_mn_neighbors, cl_cl_neighbors = {}, {}
    for cl_idx in cl_indices:
        neighbors, _ = nl.get_neighbors(cl_idx)
        cl_mn_neighbors[cl_idx] = {n for n in neighbors if lattice[n].symbol == 'Mn'}
        cl_cl_neighbors[cl_idx] = {n for n in neighbors if lattice[n].symbol == 'Cl'}

    graph = nx.Graph()
    graph.add_nodes_from(cl_indices, element='Cl')

    # 双重判据：Cl-Cl 空间相邻 且 共享 Mn
    for cl1 in cl_indices:
        for cl2 in cl_cl_neighbors[cl1]:
            if cl2 <= cl1:
                continue
            if cl_mn_neighbors[cl1] & cl_mn_neighbors[cl2]:
                graph.add_edge(cl1, cl2)

    return graph


def get_largest_component_size(graph):
    """最大连通分量节点数（空图 → 0）。"""
    if not graph:
        return 0
    components = list(nx.connected_components(graph))
    return max(len(c) for c in components) if components else 0


# ----------------------------------------------------------------------
# 步骤 4：边界 Cl 检测
# ----------------------------------------------------------------------
def find_cl_atoms_near_boundary(atoms, radius):
    """找出距晶胞中心 ∈ (R-1, R] 的 Cl 原子。"""
    cell_center = atoms.get_cell().diagonal() / 2.0
    boundary_cl = []
    for i, atom in enumerate(atoms):
        if atom.symbol == 'Cl':
            d = np.linalg.norm(atom.position - cell_center)
            if radius - 1.0 < d <= radius:
                boundary_cl.append(i)
    return boundary_cl


# ----------------------------------------------------------------------
# 步骤 5：整分量替换为 F
# ----------------------------------------------------------------------
def replace_cl_in_connected_components(lattice, cl_indices, graph):
    """将与 cl_indices 有交集的连通分量中所有 Cl 替换为 F。"""
    new_lattice = lattice.copy()
    if not graph or not cl_indices:
        return new_lattice, []

    replaced_indices = []
    for comp in nx.connected_components(graph):
        if any(idx in comp for idx in cl_indices):
            for cl_idx in comp:
                if new_lattice[cl_idx].symbol == 'Cl':
                    new_lattice[cl_idx].symbol = 'F'
                    replaced_indices.append(cl_idx)

    return new_lattice, replaced_indices


# ----------------------------------------------------------------------
# 单结构处理（顶层函数，供多进程 pickle）
# ----------------------------------------------------------------------
def process_structure(idx, radius, prefix):
    """
    处理单个结构，返回 (idx, largest_size, replaced_count, num_O)。
    为每个 idx 设置独立随机种子，避免 fork 后所有进程选出同一个中心 Mn。
    """
    seed = idx * 7919 + 12345
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))

    infile = f'{prefix}{idx}.vasp'
    structure = read(filename=infile, format='vasp')
    num_O = sum(1 for atom in structure if atom.symbol == 'O')

    # 步骤1：预处理
    empty_struct = modify_structure(structure)

    # 步骤2：球形截取
    sphere = get_sphere_str(empty_struct, radius)
    write(f'Empty_R{radius}_{idx}.vasp', sort(sphere), direct=True)

    # 步骤3：Cl 图 + 最大连通分量
    cl_graph = modify_graph(sphere)
    largest_size = get_largest_component_size(cl_graph)

    # 步骤4：边界 Cl → 整分量替换为 F
    boundary_cl = find_cl_atoms_near_boundary(sphere, radius)
    replaced_struct, replaced_cl = replace_cl_in_connected_components(
        sphere, boundary_cl, cl_graph
    )
    write(f'Replace_R{radius}_{idx}.vasp', sort(replaced_struct), direct=True)

    return (idx, largest_size, len(replaced_cl), num_O)


def process_batch(start, end, radius, prefix):
    """处理 [start, end) 区间，单样本失败不影响整批。"""
    results = []
    for idx in range(start, end):
        try:
            results.append(process_structure(idx, radius, prefix))
        except Exception as e:
            print(f"[Error] structure {idx}: {e}")
            results.append((idx, 0, 0, 1))    # 占位
    return results


def _split_ranges(n, p):
    """把 [0, n) 均分为 p 块（尽量均匀，覆盖全部 n）。"""
    base, rem = divmod(n, p)
    ranges, start = [], 0
    for i in range(p):
        size = base + (1 if i < rem else 0)
        if size == 0:
            continue
        ranges.append((start, start + size))
        start += size
    return ranges


# ----------------------------------------------------------------------
# 主程序
# ----------------------------------------------------------------------
def main():
    N = 100
    RADIUS = 30
    num_processes = 50
    prefix = '../supercell_3_'

    ranges = _split_ranges(N, num_processes)
    print(f"使用 {num_processes} 个进程，共 {len(ranges)} 个批次：{ranges[:3]} ... {ranges[-1:] if len(ranges) > 3 else ''}")

    # 预填占位，按 idx 回填
    results = [None] * N

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_processes) as ex:
        futures = {
            ex.submit(process_batch, s, e, RADIUS, prefix): (s, e)
            for s, e in ranges
        }
        for fut in concurrent.futures.as_completed(futures):
            s, e = futures[fut]
            try:
                batch = fut.result()
            except Exception as exc:
                print(f"[Batch error] {s}-{e}: {exc}")
                batch = [(i, 0, 0, 1) for i in range(s, e)]
            for k, r in enumerate(batch):
                results[s + k] = r
                idx, largest_size, replaced_count, _ = r
                print(f"{idx:3d}  {largest_size:5d}  {replaced_count:5d}")

    # ---- 统计 ----
    size_1_list = [r[1] for r in results]
    comp_len_list = [r[2] for r in results]
    size_1_ratio_list = [r[1] / r[3] if r[3] > 0 else 0 for r in results]
    comp_ratio_list = [r[2] / r[3] if r[3] > 0 else 0 for r in results]

    arr_size = np.array(size_1_list)
    arr_comp = np.array(comp_len_list)
    arr_size_ratio = np.array(size_1_ratio_list)
    arr_comp_ratio = np.array(comp_ratio_list)

    print("\\n---------- 统计结果（绝对值） ----------")
    print(f"最大连通分量大小：均值 {arr_size.mean():.2f}, 方差 {arr_size.var():.2f}, 标准差 {arr_size.std():.2f}")
    print(f"替换 Cl 数量    ：均值 {arr_comp.mean():.2f}, 方差 {arr_comp.var():.2f}, 标准差 {arr_comp.std():.2f}")

    print("\\n---------- 统计结果（相对于 O 原子数） ----------")
    print(f"最大连通分量大小：均值 {arr_size_ratio.mean():.3f}, 方差 {arr_size_ratio.var():.3f}, 标准差 {arr_size_ratio.std():.3f}")
    print(f"替换 Cl 数量    ：均值 {arr_comp_ratio.mean():.3f}, 方差 {arr_comp_ratio.var():.3f}, 标准差 {arr_comp_ratio.std():.3f}")


if __name__ == '__main__':
    main()
