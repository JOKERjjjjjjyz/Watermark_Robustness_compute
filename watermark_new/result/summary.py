import os

# 定义文件范围
i_range = range(2, 70)  # 修改为你的实际文件范围

# 打开 summary_result.txt 文件准备写入
with open("summary_result_new.txt", "w") as summary_file:
    for i in i_range:
        file_name = f"{i}_result_new.txt"
        if os.path.exists(file_name):
            with open(file_name, "r") as result_file:
                first_line = result_file.readline().strip()
                try:
                    # 确保第一行是浮点数
                    float_value = float(first_line)
                    # 写入 summary_result.txt
                    summary_file.write(f"{float_value}\n")
                except ValueError:
                    print(f"Warning: {file_name} 的第一行不是有效的浮点数，跳过该文件。")
        else:
            print(f"Warning: {file_name} 不存在，跳过该文件。")

print("所有浮点数已写入 summary_result.txt")
