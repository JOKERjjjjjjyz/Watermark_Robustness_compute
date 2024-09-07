from util_v2 import *
# from joblib import Parallel, delayed
# import numpy as np


# i_range = range(31, 61)

# # 将31到60分成15个区间
# num_splits = 30
# splits = np.array_split(i_range, num_splits)
# print(splits)
# # 并行执行
# Parallel(n_jobs=num_splits)(delayed(parallel_f_computation)(i) for split in splits for i in split)

# np.save(visit_f_file, visit_f)
# np.save(F_file, F)

for i in range(61,81):
    print(i,80)
    for j in range(i+1):
        for m in range(61):
            for n in range(i+1):
                if (not visit_f[i,j,m,n]):
                    F[i,j,m,n] = f_computation(i,j,m,n)
                    visit_f[i,j,m,n] = 1
    np.save(visit_f_file, visit_f)
    np.save(F_file, F)