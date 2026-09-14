import os

file_path = "/home/yangx/code/new_deform/GeoTransformer/output/geotransformer.p2p_liver.rtor.full/test_a5_ranker.json"

try:
    with open(file_path, 'r', encoding='utf-8') as f:
        print(f"--- 正在读取 {file_path} 的前 300 行 ---\n")
        for i in range(300):
            line = f.readline()
            if not line:
                break # 文件不足 300 行时提前结束
            print(line, end='')
        print("\n--- 读取完毕 ---")
except FileNotFoundError:
    print(f"错误：找不到文件 {file_path}")
except Exception as e:
    print(f"发生错误：{e}")