from util_v2 import *


# 定义 i 的范围
i_range = range(2, 70)

# 并行执行任务
Parallel(n_jobs=68)(delayed(process_file)(i) for i in i_range)

# process_file(2)

# from util_v1_k_5 import *

# pp = p_computation()
# print("n=5,eta=10:",pp[0][20][20],pp[1][20][20])
# print("n=5,eta=20:",pp[0][40][40],pp[1][40][40])
# print("n=6,eta=0:",pp[0][0][0],pp[1][0][0],1-pp[0][0][0]-pp[1][0][0])
# print("n=6,eta=5:",pp[0][10][10],pp[1][10][10],1-pp[0][10][10]-pp[1][10][10])
# print("n=6,eta=10:",pp[0][20][20],pp[1][20][20],1-pp[0][20][20]-pp[1][20][20])
# print("n=6,eta=15:",pp[0][30][30],pp[1][30][30],1-pp[0][30][30]-pp[1][30][30])
# print("n=6,eta=20:",pp[0][40][40],pp[1][40][40],1-pp[0][40][40]-pp[1][40][40])