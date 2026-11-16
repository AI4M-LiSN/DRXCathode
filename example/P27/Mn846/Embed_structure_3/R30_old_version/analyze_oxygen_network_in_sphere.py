from ase.db import connect
from ase.io import read, write
from ase.build import bulk, sort, make_supercell
from ase.neighborlist import NeighborList
from ase import atom, atoms
from icet import ClusterSpace, StructureContainer, ClusterExpansion
from trainstation import CrossValidationEstimator
from mchammer.calculators import ClusterExpansionCalculator
from mchammer.ensembles import CanonicalAnnealing, CanonicalEnsemble
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import random
import math
import concurrent.futures
from itertools import combinations


#将结构转换为考虑的结构：O的连接数<=2,同时P-O之间的距离<3.6时认为O> 2bond
def modify_structure(lattice):
    # 删除所有的 Li 和 Na 原子
    del lattice[[atom.index for atom in lattice if atom.symbol in ['Li', 'Na']]]

    # 替换 Ni 为 Mn
    for atom in lattice:
        if atom.symbol == 'Ni':
            atom.symbol = 'Mn'

    index_Mn = []
    index_P = []
    index_O = []
    index_b2 = []
    index_b1 = []
    index_b0 = []
    index_linkP = []

    # 分类出 Mn, P, O 的原子索引
    for index in range(len(lattice)):
        if lattice[index].symbol == 'Mn':
            index_Mn.append(index)
        elif lattice[index].symbol == 'P':
            index_P.append(index)
        elif lattice[index].symbol == 'O':
            index_O.append(index)

    # 设置邻居关系半径
    tmp_rc = []
    radius_Mn = 1.4
    radius_P = 1.2
    radius_O = 1.4  # 用于 O 和 Mn 的邻接

    for atom in lattice:
        if atom.symbol == 'Mn':
            tmp_rc.append(radius_Mn)
        elif atom.symbol == 'P':
            tmp_rc.append(radius_P)
        elif atom.symbol == 'O':
            tmp_rc.append(radius_O)  # 选择最大邻居半径
        else:
            tmp_rc.append(0)

    # 使用 NeighborList 查找 O-Mn 和 O-P 关系
    n1 = NeighborList(tmp_rc, self_interaction=False, bothways=True, skin=0)
    n1.update(lattice)

    # 判断 O 周围 Mn+P 原子的个数
    for index in index_O:
        indices, offsets = n1.get_neighbors(index)
        num_Mn_neighbors = 0
        num_P_neighbors = 0

        # 计算 O 周围的 Mn 和 P 原子数
        for i, neighbor in enumerate(indices):
            if lattice[neighbor].symbol == 'Mn':
                num_Mn_neighbors += 1
            elif lattice[neighbor].symbol == 'P':
                num_P_neighbors += 1

        # 根据 Mn 邻居的数量分类 O
        if num_Mn_neighbors == 2:
            index_b2.append(index)
        elif num_Mn_neighbors == 1:
            index_b1.append(index)
        elif num_Mn_neighbors == 0:
            index_b0.append(index)

        # 如果 O 与 P 原子相邻，则加入 linkP 列表
        if num_P_neighbors > 0:
            index_linkP.append(index)
    #print(index_b2)
    # 修改 O 原子种类
    for index in index_b2:
        lattice[index].symbol = 'Cl'
    for index in index_b1:
        lattice[index].symbol = 'Cl'
    for index in index_b0:
        lattice[index].symbol = 'Cl'
    
    # 保持与 P 原子相邻的 O 不被修改为 Cl
    for index in index_linkP:
        lattice[index].symbol = 'O'

    return lattice


def select_atoms_in_sphere(atoms, center_atom_index, radius):
    """
    选择以给定原子为中心、半径为R的球内的所有原子，考虑周期性边界条件。
    
    参数：
    atoms: ASE Atoms 对象, 包含晶体结构
    center_atom_index: int, 作为中心的原子索引
    radius: float, 半径R
    
    返回：
    new_atoms: ASE Atoms 对象, 包含球内的原子
    """
    # 获取中心原子的坐标
    center = atoms[center_atom_index].position
    #print(center)
    cell = atoms.cell
    #print(cell)   
    # 使用 neighbor_list 获取半径内的原子，考虑周期性边界条件

    tmp_rc=[]   # the radius of each atoms
    for index in range(len(atoms.numbers)):   #just consider the 0-TM channel so we only stat the O-O bond
        if(index == center_atom_index):
            tmp_rc.append(radius)
        else:
            tmp_rc.append(0)

    n1=NeighborList(tmp_rc,self_interaction=True,bothways=True,skin=0) 
    n1.update(atoms)
    indices,offsets=n1.get_neighbors(center_atom_index)

    # 计算距离，找到在半径R范围内的原子
    new_atoms_positions = []
    new_atom_symbols = []
    for i, offset in zip(indices, offsets):
        # 考虑周期性边界条件的偏移
        position = atoms[i].position + np.dot(offset, cell)
        distance = np.linalg.norm(position - center)
        if distance <= radius:
            new_atoms_positions.append(position)
            new_atom_symbols.append(atoms[i].symbol)

    return new_atoms_positions, new_atom_symbols, center


def get_sphere_str(atoms,R):
    # 随机选择一个 Mn 原子作为中心
    mn_indices = [atom.index for atom in atoms if atom.symbol == 'Mn']
    center_index = random.choice(mn_indices)
    #center_index = 2

    # 选取半径R包裹的球中所有的原子，考虑周期性边界条件
    new_positions, new_symbols, original_center_position = select_atoms_in_sphere(atoms, center_index, R)

    #print(original_center_position)
    # 计算新的晶胞（该球的外切正方体）
    new_cell_length = 2 * R +5.0  # 外切正方体的边长
    new_cell = np.eye(3) * new_cell_length

    # 新晶胞的中心坐标
    new_cell_center = np.array([new_cell_length / 2] * 3)
    #print(new_cell_center)
    # 计算从原中心到新晶胞中心的位移
    shift_vector = new_cell_center - original_center_position

    # 将所有原子的坐标移动到新晶胞的中心
    shifted_positions = [pos + shift_vector for pos in new_positions]

    # 确保所有原子坐标都在新晶胞内 [0, new_cell_length)
    shifted_positions = [pos % new_cell_length for pos in shifted_positions]
    # 构建新的 ASE Atoms 对象
    from ase import Atoms
    new_atoms = Atoms(positions=shifted_positions, symbols=new_symbols, cell=new_cell, pbc=True)

    return new_atoms

def modify_graph(lattice):
    graph_Cl = nx.Graph()
    index_Mn = []
    index_P = []
    index_O = []
    index_Cl = []

    # 分类出 Mn, P, O, Cl 的原子索引
    for index in range(len(lattice)):
        if lattice[index].symbol == 'Mn':
            index_Mn.append(index)
        elif lattice[index].symbol == 'P':
            index_P.append(index)
        elif lattice[index].symbol == 'O':
            index_O.append(index)
        elif lattice[index].symbol == 'Cl':
            index_Cl.append(index)

    # 为 Cl 和 Mn 的相邻关系设定半径
    tmp_rc = []
    radius_Cl = 1.7
    radius_Mn = 1.2
    for atom in lattice:
        if atom.symbol == 'Cl':
            tmp_rc.append(radius_Cl)
        elif atom.symbol == 'Mn':
            tmp_rc.append(radius_Mn)
        else:
            tmp_rc.append(0)

    # 使用 NeighborList 查找 Cl-Mn 关系
    n1 = NeighborList(tmp_rc, self_interaction=False, bothways=True, skin=0)
    n1.update(lattice)

    # 将 Cl 原子添加为节点
    for i in range(len(index_Cl)):
        graph_Cl.add_node(index_Cl[i], element='Cl')

    # 找出每个 Cl 原子的邻居并构建 Cl 的环境
    Cl_env_lists = []
    for index in index_Cl:
        indices, offsets = n1.get_neighbors(index)
        Cl_env_tmp = [index]
        Cl_env_tmp.extend(indices)
        Cl_env_lists.append(Cl_env_tmp)

    # 检查两个 Cl 是否连接到同一个 Mn 原子
    for Cl_env in Cl_env_lists:
        center_Cl = Cl_env[0]
        neighbors = Cl_env[1:]  # 忽略中心 Cl 本身
        connected_Mn = []

        # 找出与 Cl 相连的 Mn 原子
        for neighbor in neighbors:
            if lattice[neighbor].symbol == 'Mn':
                connected_Mn.append(neighbor)

        # 如果有多个 Cl 连接到同一个 Mn，添加边
        for i in range(len(neighbors)):
            if lattice[neighbors[i]].symbol == 'Cl':
                # 遍历每个 Mn 邻居，看是否有共同的 Mn 原子
                for mn in connected_Mn:
                    indices, offsets = n1.get_neighbors(neighbors[i])
                    if mn in indices:  # 如果此 Cl 也与该 Mn 连接
                        graph_Cl.add_edge(center_Cl, neighbors[i])

    return graph_Cl



def get_largest_connected_component(graph_Cl):
    """
    获取 graph_Cl 中最大的连通分量并返回其子图，同时输出其连通分量的大小。
    
    参数：
    graph_Cl: networkx.Graph - 输入图
    
    返回：
    largest_subgraph: networkx.Graph - 最大连通分量的子图
    largest_component_size: int - 最大连通分量的大小
    """
    # 获取所有连通分量
    connected_components = list(nx.connected_components(graph_Cl))
    
    # 找到最大的连通分量
    largest_component = max(connected_components, key=len)
    
    # 构建最大连通分量的子图
    largest_subgraph = graph_Cl.subgraph(largest_component).copy()
    
    # 输出最大连通分量的大小
    largest_component_size = len(largest_component)
    #print(f"最大连通分量的大小: {largest_component_size}")
    
    return  largest_component_size

def find_cl_atoms_near_boundaries(atoms, R):
    """
    查找 Cl 原子到晶胞中心的距离，并输出距离大于 (R-1) Å 且小于 R Å 的 Cl 原子的序号。

    参数：
    atoms: ASE Atoms 对象，包含晶体结构
    R: float, 给定的最大半径 
    
    返回：
    indices_in_range: list, 满足条件的 Cl 原子序号
    """
    # 获取晶胞中心
    cell_center = atoms.get_cell().diagonal() / 2
    
    # 用于存储满足条件的 Cl 原子的序号
    cl_indices_in_range = []

    # 遍历所有原子，计算 Cl 原子到晶胞中心的距离
    for i, atom in enumerate(atoms):
        if atom.symbol == 'Cl':
            # 计算 Cl 原子到晶胞中心的距离
            distance = np.linalg.norm(atom.position - cell_center)
            
            # 判断距离是否在 (R-1) 到 R 之间
            if R - 1 < distance <= R:
                cl_indices_in_range.append(i)

    return cl_indices_in_range

def replace_cl_in_connected_components(lattice, Cl_indices, graph_Cl):
    """
    查找 graph_Cl 中的连通分量，如果某个连通分量中包含 Cl_indices 列表中的节点，
    将该连通分量中的 Cl 原子替换为 F 并输出新的 lattice 对象，同时返回该连通分量中所有的 Cl 序号。
    
    参数：
    lattice: ASE Atoms 对象，晶体结构
    Cl_indices: list - Cl 原子的索引列表，用于查找连通分量
    graph_Cl: networkx.Graph - 包含 Cl 原子连接关系的图结构
    
    返回：
    lattice_replace: ASE Atoms 对象，替换后的晶体结构
    cl_component_indices: list - 被替换为 F 的 Cl 原子序号列表
    """
    # 获取 graph_Cl 中的所有连通分量
    connected_components = list(nx.connected_components(graph_Cl))
    
    # 用于存储包含 Cl_indices 中原子序号的连通分量
    cl_component_indices = []

    # 遍历所有连通分量，查找包含 Cl_indices 的连通分量
    for component in connected_components:
        # 如果连通分量和 Cl_indices 有交集
        if any(idx in component for idx in Cl_indices):
            # 记录该连通分量中所有的 Cl 原子序号
            cl_component_indices.extend(component)

            # 将连通分量中的 Cl 原子替换为 F
            for cl_index in component:
                if lattice[cl_index].symbol == 'Cl':
                    lattice[cl_index].symbol = 'F'

    # 创建 lattice 的副本作为 lattice_replace
    lattice_replace = lattice.copy()

    # 输出提取的连通分量中所有的 Cl 原子序号
    #print(f"被替换为 F 的 Cl 原子序号: {cl_component_indices}")
    
    return lattice_replace, cl_component_indices

def process_structure(start, end, R):
    # 初始化结果列表，每个位置存储一个结构的结果
    size_1_list = [None] * (end - start)
    cl_component_len_list = [None] * (end - start)
    size_1_ratio_list = [None] * (end - start)
    cl_component_ratio_list = [None] * (end - start)

    try:
        for i, idx in enumerate(range(start, end)):
            pri_file = f'../supercell_3_{idx}.vasp'
            structure = read(filename=pri_file, format='vasp')
            lattice_file = f'Empty_R{R}_{idx}.vasp'
            lattice_re_file = f'Replace_R{R}_{idx}.vasp'

            num_O_atoms = sum(1 for atom in structure if atom.symbol == 'O')

            lattice_empty = modify_structure(structure)
            lattice = get_sphere_str(lattice_empty, R)
            write(filename=lattice_file, images=sort(lattice), direct=True)

            graph_cl = modify_graph(lattice)
            size_1 = get_largest_connected_component(graph_cl)
            cl_near_boundary = find_cl_atoms_near_boundaries(lattice, R)

            lattice_re, cl_component = replace_cl_in_connected_components(lattice, cl_near_boundary, graph_cl)
            write(filename=lattice_re_file, images=sort(lattice_re), direct=True)

            size_1_ratio = size_1 / num_O_atoms
            cl_component_ratio = len(cl_component) / num_O_atoms

            # 将结果存储在正确的位置上
            size_1_list[i] = size_1
            cl_component_len_list[i] = len(cl_component)
            size_1_ratio_list[i] = size_1_ratio
            cl_component_ratio_list[i] = cl_component_ratio
            # 打印每次计算的结果
            print(f"{idx}    {size_1}    {len(cl_component)}")
    except Exception as e:
        print(f"Error processing range {start} to {end}: {e}")

    return size_1_list, cl_component_len_list, size_1_ratio_list, cl_component_ratio_list

# 设置多进程处理
N = 100
R = 30
num_processes = 50
batch_size = N // num_processes

# 初始化总结果列表，保持顺序
size_1_results = [None] * N
cl_component_len_results = [None] * N
size_1_ratio_results = [None] * N
cl_component_ratio_results = [None] * N

with concurrent.futures.ProcessPoolExecutor(max_workers=num_processes) as executor:
    futures = {
        executor.submit(process_structure, i * batch_size, (i + 1) * batch_size, R): i
        for i in range(num_processes)
    }

    for future in concurrent.futures.as_completed(futures):
        # 获取起始和结束位置
        start_idx = futures[future] * batch_size
        size_1_list, cl_len_list, size_1_ratio_list, cl_ratio_list = future.result()

        # 把每个批次的结果放入总结果列表中的相应位置
        size_1_results[start_idx:start_idx + batch_size] = size_1_list
        cl_component_len_results[start_idx:start_idx + batch_size] = cl_len_list
        size_1_ratio_results[start_idx:start_idx + batch_size] = size_1_ratio_list
        cl_component_ratio_results[start_idx:start_idx + batch_size] = cl_ratio_list

# 计算统计结果
size_1_array = np.array(size_1_results)
cl_component_len_array = np.array(cl_component_len_results)
print("result: ave var std")
print(f"size_1 {np.mean(size_1_array)} {np.var(size_1_array)} {np.std(size_1_array)}")
print(f"cl_component_len {np.mean(cl_component_len_array)} {np.var(cl_component_len_array)} {np.std(cl_component_len_array)}")