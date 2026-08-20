#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
生成随机构型：将 Li/Mn/P 原子在四面体位点和八面体位点内分别打乱，
输出 100 个随机结构文件。
"""

from ase.io import read, write
from ase.build import sort
import numpy as np

# -------------------- 可调参数 --------------------
N_SAMPLES = 100              # 生成随机结构的数量
INPUT_FILE = 'LMO666.vasp'   # 原始结构文件（包含 Li, Mn, P, Na 等）
OUTPUT_PREFIX = 'Random_'    # 输出文件前缀，如 Random_0_666.vasp
# --------------------------------------------------


def MC_sweep(lattice):
    """
    对输入的晶格进行随机化重排：
      - 前 432 个原子（四面体位点）包含 Li、Mn、P，按各自数量随机打乱位置。
      - 后 864 个原子（八面体位点）包含 Na、Mn、P，按各自数量随机打乱位置， Na代表空位。
    返回修改后的晶格对象。
    """
    # ---------- 四面体位点（索引 0 ~ 431） ----------
    index_tr = []
    num_Li = num_Mn = num_P = 0

    for idx in range(432):  # 前432个原子
        sym = lattice[idx].symbol
        if sym == 'Li':
            num_Li += 1
            index_tr.append(idx)
        elif sym == 'Mn':
            num_Mn += 1
            index_tr.append(idx)
        elif sym == 'P':
            num_P += 1
            index_tr.append(idx)

    # 随机打乱这些位点
    np.random.shuffle(index_tr)

    # 按数量依次填充 P, Li, Mn（顺序可调，但需保持数量一致）
    pos = 0
    for _ in range(num_P):
        lattice[index_tr[pos]].symbol = 'P'
        pos += 1
    for _ in range(num_Li):
        lattice[index_tr[pos]].symbol = 'Li'
        pos += 1
    for _ in range(num_Mn):
        lattice[index_tr[pos]].symbol = 'Mn'
        pos += 1

    # ---------- 八面体位点（索引 432 ~ 1295） ----------
    index_oc = []
    num_Na = num_Mn = num_P = 0

    for idx in range(432, 1296):  # 后864个原子
        sym = lattice[idx].symbol
        if sym == 'Na':
            num_Na += 1
            index_oc.append(idx)
        elif sym == 'Mn':
            num_Mn += 1
            index_oc.append(idx)
        elif sym == 'P':
            num_P += 1
            index_oc.append(idx)

    np.random.shuffle(index_oc)

    pos = 0
    for _ in range(num_P):
        lattice[index_oc[pos]].symbol = 'P'
        pos += 1
    for _ in range(num_Na):
        lattice[index_oc[pos]].symbol = 'Na'
        pos += 1
    for _ in range(num_Mn):
        lattice[index_oc[pos]].symbol = 'Mn'
        pos += 1

    return lattice


def main():
    # 读取原始结构（需包含 Li, Na, Mn, P 等）
    structure = read(INPUT_FILE, format='vasp')

    # 循环生成随机结构
    for i in range(N_SAMPLES):
        # 复制原始结构以避免修改原对象
        lattice = structure.copy()

        # 执行随机重排
        lattice = MC_sweep(lattice)

        # 输出排序后的结构（便于可视化）
        outfile = f'{OUTPUT_PREFIX}{i}_666.vasp'
        write(outfile, sort(lattice), direct=True)

        # 打印进度
        print(f"Generated random structure {i+1}/{N_SAMPLES} -> {outfile}")

    print("\\n所有随机结构生成完毕。")


if __name__ == '__main__':
    main()