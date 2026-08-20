import numpy as np

def calculate_selected_statistics(file_path):
    data = np.loadtxt(file_path,skiprows=1)  # 读取文件为numpy数组
    
    col1 = data[:, 0]  # 第一列
    col2 = data[:, 1]  # 第二列
    col_sum = col1 + col2  # 第一列与第二列之和
    
    # 计算并输出平均值和标准差
    print(f"Column_1: Mean = {np.mean(col1):.2f}, Standard Deviation = {np.std(col1):.2f}")
    print(f"Column_2: Mean = {np.mean(col2):.2f}, Standard Deviation = {np.std(col2):.2f}")
    print(f"Sum_Column_1+2: Mean = {np.mean(col_sum):.2f}, Standard Deviation = {np.std(col_sum):.2f}")


# 使用方法：将文件路径替换为你自己的文件路径
file_path = 'oxygen_result'
calculate_selected_statistics(file_path)