# -*- coding: utf-8 -*-
"""
分析晶体结构中氧原子配位环境及氯原子（Cl）连通性，并统计相关量的分布。
流程：
1. 读取一系列超胞结构（NiO666 变体），删除 Li/Na，将 Ni 替换为 Mn。
2. 根据 Mn 的配位数将 O 分为不同类别，并将非桥接 O（不与 P 相邻）标记为 Cl。
3. 以随机 Mn 为中心，截取半径 R 的球形区域，构建新晶胞。
4. 构建 Cl 原子图（共享 Mn 邻居的 Cl 之间连边），计算最大连通分量大小。
5. 找出位于球壳边界附近的 Cl，将其所在连通分量的所有 Cl 替换为 F。
6. 统计最大连通分量大小和替换 Cl 数量及其与原始 O 原子数的比例。
"""

import random
import numpy as np
import networkx as nx
from ase.io import read, write
from ase.build import sort
from ase import Atoms
from ase.neighborlist import NeighborList

# ----------------------------------------------------------------------
# 常量定义（原子半径，单位 Å）
RADIUS_MN = 1.4       # Mn-O 邻居判断半径
RADIUS_P  = 1.2       # P-O 邻居判断半径
RADIUS_O  = 1.4       # O 的邻居半径（用于 O-Mn 检测）
RADIUS_CL = 1.7       # Cl-Mn 邻居判断半径（Cl 稍大）
RADIUS_MN_FOR_CL = 1.2 # Mn 在 Cl-Mn 检测中的半径
# ----------------------------------------------------------------------


def modify_structure(lattice):
    """
    修改原始结构：
      - 删除所有 Li 和 Na 原子
      - 将 Ni 替换为 Mn
      - 根据 Mn 的配位数将 O 分类：与 0/1/2 个 Mn 相连的 O 全部改为 Cl，
        但保留与 P 相邻的 O 为 O（即 P-O 键不被破坏）。
    返回修改后的 Atoms 对象。
    """
    # 删除 Li/Na
    del_idx = [atom.index for atom in lattice if atom.symbol in ('Li', 'Na')]
    del lattice[del_idx]

    # Ni -> Mn
    for atom in lattice:
        if atom.symbol == 'Ni':
            atom.symbol = 'Mn'

    # 获取各类原子索引
    mn_indices = [i for i, atom in enumerate(lattice) if atom.symbol == 'Mn']
    p_indices  = [i for i, atom in enumerate(lattice) if atom.symbol == 'P']
    o_indices  = [i for i, atom in enumerate(lattice) if atom.symbol == 'O']

    # 为每个原子设置邻居半径（仅 O 和 Mn 需要）
    radii = []
    for atom in lattice:
        if atom.symbol in ('Mn', 'O'):
            radii.append(RADIUS_MN)   # 这里统一用 Mn 的半径，用于 O-Mn 探测
        else:
            radii.append(0.0)

    nl = NeighborList(radii, self_interaction=False, bothways=True, skin=0)
    nl.update(lattice)

    # 分类 O 原子：与 Mn 相连的个数
    two_mn_oxygens = []   # 连 2 个 Mn
    one_mn_oxygens  = []   # 连 1 个 Mn
    zero_mn_oxygens = []   # 连 0 个 Mn
    p_adjacent_oxygens = [] # 与 P 相邻的 O

    for idx in o_indices:
        neighbors, _ = nl.get_neighbors(idx)
        mn_count = sum(1 for n in neighbors if lattice[n].symbol == 'Mn')
        p_count  = sum(1 for n in neighbors if lattice[n].symbol == 'P')

        if mn_count == 2:
            two_mn_oxygens.append(idx)
        elif mn_count == 1:
            one_mn_oxygens.append(idx)
        else:  # mn_count == 0
            zero_mn_oxygens.append(idx)

        if p_count > 0:
            p_adjacent_oxygens.append(idx)

    # 将所有 O 暂时改为 Cl
    for idx in o_indices:
        lattice[idx].symbol = 'Cl'

    # 将与 P 相邻的 O 改回 O（保护 P-O 键）
    for idx in p_adjacent_oxygens:
        lattice[idx].symbol = 'O'

    return lattice


def select_atoms_in_sphere(atoms, center_index, radius):
    """
    选择以给定原子为中心、半径 radius 的球内的所有原子（考虑周期性边界）。
    返回：
      - 球内原子坐标（绝对坐标，已考虑 PBC 偏移）
      - 原子符号列表
      - 中心原子的原始坐标
    """
    center = atoms[center_index].position
    cell = atoms.cell

    # 仅中心原子有非零半径，其他原子半径设为 0（用于邻居检测）
    radii = [radius if i == center_index else 0.0 for i in range(len(atoms))]
    nl = NeighborList(radii, self_interaction=True, bothways=True, skin=0)
    nl.update(atoms)

    indices, offsets = nl.get_neighbors(center_index)
    positions = []
    symbols = []
    for idx, off in zip(indices, offsets):
        pos = atoms[idx].position + np.dot(off, cell)
        dist = np.linalg.norm(pos - center)
        if dist <= radius:
            positions.append(pos)
            symbols.append(atoms[idx].symbol)

    return positions, symbols, center


def get_sphere_str(atoms, radius):
    """
    随机选取一个 Mn 原子作为中心，截取半径 radius 的球形区域，
    并以该球的外切正方体作为新晶胞，返回包含球内所有原子的 Atoms 对象。
    新晶胞中心与球心重合。
    """
    # 随机选择一个 Mn
    mn_indices = [i for i, atom in enumerate(atoms) if atom.symbol == 'Mn']
    center_idx = random.choice(mn_indices)

    positions, symbols, original_center = select_atoms_in_sphere(atoms, center_idx, radius)

    # 新晶胞边长 = 2*radius + 5.0（留出余量避免原子贴边）
    cell_length = 2 * radius + 5.0
    new_cell = np.eye(3) * cell_length
    new_center = np.array([cell_length / 2.0] * 3)

    # 将球内原子坐标平移到新晶胞中心
    shift = new_center - original_center
    shifted_pos = [pos + shift for pos in positions]
    # 应用周期性边界条件，确保坐标在 [0, cell_length) 内
    shifted_pos = [pos % cell_length for pos in shifted_pos]

    return Atoms(positions=shifted_pos, symbols=symbols, cell=new_cell, pbc=True)


def modify_graph(lattice):
    """
    构建 Cl 原子之间的图。两个 Cl 原子若共享至少一个 Mn 邻居，则在图中添加一条边。
    返回 networkx.Graph 对象。
    """
    cl_indices = [i for i, atom in enumerate(lattice) if atom.symbol == 'Cl']
    if not cl_indices:
        return nx.Graph()

    # 为 Cl 和 Mn 设置邻居半径
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

    # 获取每个 Cl 的 Mn 邻居集合
    cl_mn_neighbors = {}
    for cl_idx in cl_indices:
        neighbors, _ = nl.get_neighbors(cl_idx)
        mn_set = {n for n in neighbors if lattice[n].symbol == 'Mn'}
        cl_mn_neighbors[cl_idx] = mn_set

    # 构建图
    graph = nx.Graph()
    graph.add_nodes_from(cl_indices, element='Cl')

    # 两两比较 Cl，若共享 Mn 则连边
    for i in range(len(cl_indices)):
        for j in range(i + 1, len(cl_indices)):
            cl1, cl2 = cl_indices[i], cl_indices[j]
            if cl_mn_neighbors[cl1] & cl_mn_neighbors[cl2]:
                graph.add_edge(cl1, cl2)

    return graph


def get_largest_component_size(graph):
    """返回图 graph 中最大连通分量的节点数（若图为空则返回 0）。"""
    if not graph:
        return 0
    components = list(nx.connected_components(graph))
    return max(len(c) for c in components) if components else 0


def find_cl_atoms_near_boundary(atoms, radius):
    """
    找出 Cl 原子中到晶胞中心距离在 (radius-1, radius] 范围内的原子索引。
    此范围用于定位球壳边界附近的 Cl。
    """
    cell_center = atoms.get_cell().diagonal() / 2.0
    boundary_cl = []
    for i, atom in enumerate(atoms):
        if atom.symbol == 'Cl':
            dist = np.linalg.norm(atom.position - cell_center)
            if radius - 1.0 < dist <= radius:
                boundary_cl.append(i)
    return boundary_cl


def replace_cl_in_connected_components(lattice, cl_indices, graph):
    """
    查找包含 cl_indices 中任一 Cl 的连通分量，将该分量中所有 Cl 替换为 F。
    返回 (新 lattice, 被替换的 Cl 索引列表)。
    """
    new_lattice = lattice.copy()
    if not graph or not cl_indices:
        return new_lattice, []

    components = list(nx.connected_components(graph))
    replaced_indices = []

    for comp in components:
        # 若连通分量与给定的 cl_indices 有交集，则替换整个分量
        if any(idx in comp for idx in cl_indices):
            for cl_idx in comp:
                if new_lattice[cl_idx].symbol == 'Cl':
                    new_lattice[cl_idx].symbol = 'F'
                    replaced_indices.append(cl_idx)

    return new_lattice, replaced_indices


def main():
    """主程序：循环处理 100 个结构，统计并输出结果。"""
    N = 100                 # 处理文件数量
    RADIUS = 40             # 球半径（Å）
    prefix = '../supercell_3_'   # 输入文件前缀

    size_1_list = []          # 最大连通分量大小
    comp_len_list = []        # 被替换的 Cl 数量
    size_1_ratio_list = []    # 最大连通分量大小 / 原始 O 数
    comp_ratio_list = []      # 替换 Cl 数 / 原始 O 数

    for i in range(N):
        # 读取原始结构
        infile = f'{prefix}{i}.vasp'
        structure = read(filename=infile, format='vasp')

        # 记录原始 O 原子数（用于归一化）
        num_O = sum(1 for atom in structure if atom.symbol == 'O')

        # 步骤1：修改结构（删除 Li/Na，Ni->Mn，O->Cl 等）
        empty_struct = modify_structure(structure)

        # 步骤2：截取球体并写入 Empty 文件
        sphere = get_sphere_str(empty_struct, RADIUS)
        write(f'Empty_R{RADIUS}_{i}.vasp', sort(sphere), direct=True)

        # 步骤3：构建 Cl 图，计算最大连通分量大小
        cl_graph = modify_graph(sphere)
        largest_size = get_largest_component_size(cl_graph)
        size_1_list.append(largest_size)

        # 步骤4：找出边界 Cl，替换所在连通分量为 F
        boundary_cl = find_cl_atoms_near_boundary(sphere, RADIUS)
        replaced_struct, replaced_cl = replace_cl_in_connected_components(
            sphere, boundary_cl, cl_graph
        )
        comp_len_list.append(len(replaced_cl))

        # 步骤5：写入替换后的结构
        write(f'Replace_R{RADIUS}_{i}.vasp', sort(replaced_struct), direct=True)

        # 计算比例
        size_1_ratio_list.append(largest_size / num_O if num_O > 0 else 0)
        comp_ratio_list.append(len(replaced_cl) / num_O if num_O > 0 else 0)

        # 进度输出
        print(f"{i:3d}  {largest_size:5d}  {len(replaced_cl):5d}")

    # ---- 统计输出 ----
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