import numpy as np
from math import comb
import time
from itertools import permutations
import os
from joblib import Parallel, delayed
import numpy as np

#eta变量意为：binary search时候的最大eta值
eta = 40

kk = None
z_array = None
x_array = None
zk = None
x_const = None
f_index_max = 80

visit=np.zeros((500,500,13),dtype=np.int32)
BIN=np.zeros((500,500,13),dtype=np.float32)
def Bin(n,k,B):
    # 算二项分布pr
    if k>n:
        return 0
    if visit[n][k][B]:
        return BIN[n][k][B]
    p=1/B
    BIN[n][k][B]=comb(n,k)*(p**k)*((1-p)**(n-k))
    visit[n][k][B]=1
    return BIN[n][k][B]

#读取hyper↓

# visit_Hyper_file = 'visit_Hyper.npy'
HYPER_file = 'HYPER.npy'

if not os.path.exists(HYPER_file):
# if not os.path.exists(visit_Hyper_file) or not os.path.exists(HYPER_file):
    visit_Hyper=np.zeros((400+1,max(2*eta+1,f_index_max + 1),max(2*eta+1,f_index_max + 1),max(2*eta+1,f_index_max + 1)),dtype=np.int32)
    HYPER=np.zeros((400+1,max(2*eta+1,f_index_max + 1),max(2*eta+1,f_index_max + 1),max(2*eta+1,f_index_max + 1)),dtype=np.float32)
else:
    # visit_Hyper = np.load(visit_Hyper_file)
    HYPER = np.load(HYPER_file)

#读取f↓
# visit_f_file = 'visit_f.npy'
F_file = 'F.npy'

# if not os.path.exists(visit_f_file) or not os.path.exists(F_file):
if not os.path.exists(F_file):
    visit_f = np.zeros(
        (f_index_max + 1, f_index_max + 1, max(2 * eta + 1, f_index_max + 1), max(2 * eta + 1, f_index_max + 1)),
        dtype=np.int32
    )
    F = np.zeros(
        (f_index_max + 1, f_index_max + 1, max(2 * eta + 1, f_index_max + 1), max(2 * eta + 1, f_index_max + 1)),
        dtype=np.float32
    )
else:
    # visit_f = np.load(visit_f_file)
    F = np.load(F_file)

#p表计算↓
def p_computation():
    pp = np.zeros((kk+2,2, 2*eta+1, 2*eta+1))
    for k in range(2,kk+1):
        z = z_array[k]
        x = x_array[k]
        zk_sum = 0
        for j in range(1,k+1):
            zk_sum += z_array[j]
        sub_HYPER = HYPER[zk_sum,z]
        sub_F = F[z,x]
        #把sub表提前存好可能会节省解析index需要的时间
        if k == 2:
            #分类是因为k由k-1的表更新，但是k=1值是确定的，不需要循环去存表
            sub_F1 = F[z_array[k-1],x_array[k-1]]
            for a in range(2*eta+1):
                for b in range(2*eta+1):
                    sum1 = 0
                    sum2 = 0
                    for ak in range(a+1):
                        p_ak = Bin(a,ak,k)
                        for bk in range(b+1):
                            p_bk = sub_HYPER[b][bk] * p_ak
                            sum1 += p_bk * (1-sub_F1[a-ak][b-bk])*(1-sub_F[ak][bk])
                            sum2 += p_bk * ((1-2*sub_F1[a-ak][b-bk]) * sub_F[ak][bk] + sub_F1[a-ak][b-bk])
                    pp[k][0][a][b] = sum1
                    pp[k][1][a][b] = sum2
        else:
            for a in range(2*eta+1):
                for b in range(2*eta+1):
                    sum1 = 0
                    sum2 = 0
                    for ak in range(a+1):
                        p_ak = Bin(a,ak,k)
                        for bk in range(b+1):
                            p_bk = sub_HYPER[b][bk] * p_ak
                            sum1 += p_bk * pp[k-1][0][a-ak][b-bk]*(1-sub_F[ak][bk])
                            sum2 += p_bk * ((pp[k-1][0][a-ak][b-bk]-pp[k-1][1][a-ak][b-bk]) * sub_F[ak][bk] + pp[k-1][1][a-ak][b-bk])
            pp[k][0][a][b] = sum1
            pp[k][1][a][b] = sum2
    return pp

def eta_max_binary_search(line):
    #先计算p表，然后根据p表去binary search
    start_time = time.time()
    global z_array
    global x_array
    global kk
    global zk
    global x_const
    global f_index_max
    data = line.split(';')
    array_a = eval(data[0])
    array_b = eval(data[1])
    kk = len(array_a)
    z_array = np.array([0] + array_a[0:])
    x_array = np.array([0] + [int(x) for x in array_b[0:]])
    zk = np.max(z_array)
    x_const = np.max(x_array)
    f_index_max = max(80,zk)

    if np.any(z_array > 60):
        return 0,0

    eta_max = eta
    eta_min = 0
    left = eta_min
    right = eta_max
    max_eta_prime = eta_min  # 如果所有eta都使p_computation=0, 返回eta_min
    pp = p_computation()
    while left <= right:
        mid = (left + right) // 2
        if (1-pp[kk][0][2*mid][2*mid]-pp[kk][1][2*mid][2*mid]) < 0.001:
            max_eta_prime = mid  # 更新最大eta_prime
            left = mid + 1  # 继续搜索更大的值
        else:
            right = mid - 1  # 搜索更小的值
    return max_eta_prime, time.time()-start_time

def process_file(i):
    #用于并行执行给每个file里面的每行去算bound
    file_name = f"./tok_assign/{i}.txt"
    count = 0
    result = np.zeros((31,))
    time = np.zeros((31,))
    
    with open(file_name, 'r') as file:
        for line in file:
            if count >= 30:  # 修正了条件，避免超过数组边界
                break
            count += 1
            print(i,count)
            result[count], time[count] = eta_max_binary_search(line.strip())
            if time[count] == 0:
                count -= 1

    Average_eta_max = result.sum() / 30.
    
    with open(f"./result/{i}_result_new.txt", "a") as file:
        file.write(f"{Average_eta_max}\n")
        file.write(f"{result}\n")
        file.write(f"used time: {time.sum()}\n")

def parallel_f_computation(i):
    #用于f pre-comoute的，可以不用管
    global F
    global visit_f
    print(F[i,i,60,i],i,"begin")
    for j in range(i + 1):
        for m in range(61):
            for n in range(i+1):
                if not visit_f[i, j, m, n]:
                    F[i, j, m, n] = f_computation(i, j, m, n)
                    visit_f[i, j, m, n] = 1
    print(F[i,i,60,i],i,"end")