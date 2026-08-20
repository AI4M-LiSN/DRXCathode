#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
利用团簇展开（Cluster Expansion）和蒙特卡洛退火（MC）对随机结构进行优化，
生成能量较低的稳定构型。仅保留模拟核心，不包含后续连通性分析。
"""

from ase.io import read, write
from ase.build import sort
from mchammer.calculators import ClusterExpansionCalculator
from mchammer.ensembles import CanonicalAnnealing

# -------------------- 可调参数 --------------------
N_SAMPLES = 100                  # 需要处理的随机结构数量
INPUT_PREFIX = '../random_666/Random_'   # 输入文件前缀（如 Random_0_666.vasp）
OUTPUT_PREFIX = 'MC_'            # 输出文件前缀（如 MC_0_666.vasp）
T_START = 1000                   # 退火起始温度 (K)
T_STOP = 1000                    # 退火终止温度 (K)
N_STEPS = 1000000                # 蒙特卡洛步数
COOLING = 'linear'               # 降温方式（linear / exponential）
# --------------------------------------------------


def main():
    # 1. 读取预先训练好的团簇展开模型
    ce = ClusterExpansion.read('mixing_energy.ce')

    # 用于记录能量（可选）
    energy_list = []

    # 2. 循环处理每个随机初始结构
    for i in range(N_SAMPLES):
        # 读取随机结构
        infile = f'{INPUT_PREFIX}{i}_666.vasp'
        structure = read(infile, format='vasp')

        # 3. 设置 MC 计算器
        calculator = ClusterExpansionCalculator(structure, ce)

        # 4. 执行退火模拟（此处采用等温退火，T_start = T_stop）
        mc = CanonicalAnnealing(
            structure=structure,
            calculator=calculator,
            T_start=T_START,
            T_stop=T_STOP,
            n_steps=N_STEPS,
            cooling_function=COOLING
        )
        mc.run()

        # 5. 获取最终优化后的结构
        final_structure = mc.structure

        # 6. 计算最终结构的能量
        energy = ce.predict(final_structure)
        energy_list.append(energy)

        # 7. 输出结构文件（排序原子便于可视化）
        outfile = f'{OUTPUT_PREFIX}{i}_666.vasp'
        write(outfile, sort(final_structure), direct=True)

        # 8. 打印进度
        print(f"Sample {i:3d}  Energy: {energy:.6f} eV")

    '''
    # 9. 输出所有能量到文件
    with open('energies.txt', 'w') as f:
        for idx, e in enumerate(energy_list):
            f.write(f"{idx}  {e:.6f}\\n")
    '''

if __name__ == '__main__':
    main()