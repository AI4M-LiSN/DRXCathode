import numpy as np
import random
import os
from ase.io import read, write
from ase.build import make_supercell,sort


def generate_supercell(file_list, output_file,N):
    """
    从文件列表中选取N^3个VASP文件，生成N*N*N的超晶胞并保存为VASP文件。
    
    参数：
    file_list: 可选的VASP文件列表
    output_file: 输出的VASP文件路径
    N: 超晶胞的维度，如2x2x2、3x3x3等
    """
    # 随机选择N^3个文件
    selected_files = random.sample(file_list, N**3)

    # 读取第一个文件，作为拼接的基础晶胞
    base_cell = read(filename=selected_files[0],format='vasp')

    # 拼接 N*N*N 的超晶胞矩阵
    supercell_matrix = np.eye(3) * N

    # 将第一个文件的结构进行 N*N*N  的超晶胞扩展
    new_structure = make_supercell(base_cell, supercell_matrix)

    # 创建空列表存储每个晶胞的原子结构
    atoms_list = []

    # 依次读取每个 VASP 文件，添加到列表中
    for file in selected_files:
        atoms = read(filename=file,format='vasp')
        atoms_list.append(atoms)

    # 检查原子数量是否一致
    atom_count = len(atoms_list[0])
    for atoms in atoms_list:
        if len(atoms) != atom_count:
            raise ValueError("所有晶胞的原子数量必须一致")

    # 超晶胞共 N*N*N 个单元格
    unit_cell_count = N ** 3

    # 确保新的超晶胞可以容纳所有原子
    assert len(new_structure) == atom_count * unit_cell_count, "超晶胞中的原子数量不正确"

    # 替换每个新单元格中的原子为相应的结构
    for i, atoms in enumerate(atoms_list):
        # 计算每个方向上的位置偏移
        x_offset = (i % N) * base_cell.cell[0]
        y_offset = ((i // N) % N) * base_cell.cell[1]
        z_offset = (i // (N * N)) * base_cell.cell[2]
        offset = x_offset + y_offset + z_offset

        # 更新第 i 个单元格中的原子坐标
        start_idx = i * atom_count
        end_idx = (i + 1) * atom_count
        new_structure.positions[start_idx:end_idx] = atoms.positions + offset

    # 输出新的超晶胞结构为 VASP 文件
    write(filename=output_file, images=sort(new_structure),direct=True)

    return 0


#拼接原始结构的文件名
file_list = [f'../T1000/MC_'+str(i)+'_666.vasp' for i in range(100)]  

N=3

# 生成100个新的拼接结构
for i in range(100):
    output_file = 'supercell_'+str(N)+'_'+str(i)+'.vasp'
    generate_supercell(file_list, output_file,N)

print("100个新的拼接结构已生成。")
