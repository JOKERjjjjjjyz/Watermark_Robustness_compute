import numpy as np
from math import comb
import time
from itertools import permutations

eta = 10
zk = 33
x_const = 27

visit=np.zeros((500,500,13),dtype=np.int32)
BIN=np.zeros((500,500,13),dtype=np.float32)
def Bin(n,k,B):
    if k>n:
        return 0
    if visit[n][k][B]:
        return BIN[n][k][B]
    p=1/B
    BIN[n][k][B]=comb(n,k)*(p**k)*((1-p)**(n-k))
    visit[n][k][B]=1
    return BIN[n][k][B]

def Multi_PDF_table_generation(n, b):
    '''
    generate PDF and CDF table for x
    :return: save PDF file
    '''
    e = n
    p_x = np.zeros((e+1, e+1, e+1, e+1, e+1))
    A=time.time()
    T = n
    CNT=0
    for x in range(0, e + 1):
        if x>T:
            break
        p_x1 = Bin(n, x, b)
        print(x,e,CNT)
        for x1 in range(0, e + 1):
            if (x + x1)>T:
                break
            p_x2 = Bin(n - x, x1, b - 1)
            for x2 in range(0, e + 1):
                if (x+x1+x2) > T:
                    break
                p_x3 = Bin(n - x - x1, x2, b - 2)
                for x3 in range(0, e + 1):
                    if (x+x1+x2+x3)>T:
                        break
                    p_x4 = Bin(n - x - x1 - x2, x3, b - 3)
                    for x4 in range(0, e + 1):
                        if (x+x1+x2+x3+x4)>T:
                            break
                        if (T - x - x1 - x2 - x3 - x4) <= e :  # Ensure the index is within bounds
                            # Generate all permutations of indices
                            p_x5 = Bin(n - x - x1 - x2 - x3, x4, b - 4)
                            #indices = [x, x1, x2, x3, x4, T-x-x1-x2-x3-x4]
                            value = p_x1 * p_x2 * p_x3 * p_x4 * p_x5
                            CNT+=1
                            p_x[x,x1,x2,x3,x4] = value
    print(time.time()-A)
    np.save(f"{n}_PDF_BIN",p_x)
    return p_x

visit_Hyper=np.zeros((200+1,max(2*eta+1,zk+1),max(2*eta+1,zk+1),max(2*eta+1,zk+1)),dtype=np.int32)
HYPER=np.zeros((200+1,max(2*eta+1,zk+1),max(2*eta+1,zk+1),max(2*eta+1,zk+1)),dtype=np.float32)
def Hyper(N,K,n,k):
    if k>n:
        return 0
    if n>N:
        return 0
    if k>K:
        return 0
    if visit_Hyper[N][K][n][k]:
        return HYPER[N][K][n][k]
    HYPER[N][K][n][k]=comb(K,k)*comb(N-K,n-k)/comb(N,n)
    visit_Hyper[N][K][n][k]=1
    return HYPER[N][K][n][k]

def Hyper_PDF_table_generation(N,K):
    # N = T actually
    # K = z1...z6
    # ball number: 2*eta
    e = 2 * eta
    p_x = np.zeros((e+1, e+1, e+1, e+1, e+1))
    A=time.time()
    CNT=0
    for x in range(0, e + 1):
        if x>e:
            break
        p_x1 = Hyper(N, K, 2*eta, x)
        print(x,e,CNT)
        for x1 in range(0, e + 1):
            if (x + x1)>e:
                break
            p_x2 = Hyper(N - K, K, 2*eta - x, x1)
            for x2 in range(0, e + 1):
                if (x+x1+x2) > e:
                    break
                p_x3 = Hyper(N - 2*K, K, 2*eta - x - x1, x2)
                for x3 in range(0, e + 1):
                    if (x+x1+x2+x3)>e:
                        break
                    p_x4 = Hyper(N - 3*K, K, 2*eta - x - x1 - x2, x3)
                    for x4 in range(0, e + 1):
                        if (x+x1+x2+x3+x4)>e:
                            break
                        if x4 <= (2*eta - x - x1 - x2 - x3) :  # Ensure the index is within bounds
                            # Generate all permutations of indices
                            p_x5 = Hyper(N - 4*K, K, 2*eta - x - x1 - x2 - x3, x4)
                            #indices = [x, x1, x2, x3, x4, T-x-x1-x2-x3-x4]
                            value = p_x1 * p_x2 * p_x3 * p_x4 * p_x5
                            CNT+=1
                            p_x[x,x1,x2,x3,x4] = value
    print(time.time()-A)
    np.save(f"{N}_PDF_HYPER",p_x)
    return p_x

def f_computation(z,x,a,b):
    m = 3
    sum = 0
    for x_prime in range(a+x+1):
        #max: B(a,1/2) = a, H(z,x,b,k) = 0, x' = x+a
        #min: B(a,1/2) = 0, H(z,x,b,k) = x, x' = 0
        for c in range(a+1):
            p_c = Bin(a,c,2)
            d = x+c-x_prime
            if d<0:
                continue
            if d>b:
                continue
            print(z,x,b,d)
            p_d = Hyper(z,x,b,d)

            if (x_prime < (z + a - b)/2):
                sum += p_c * p_d
            else:
                p_aid = 0
                for i in range(x_prime,z+a-b+1):
                    p_aid += Bin(z+a-b,i,2)
                sum += p_c * p_d * (1-((1-2*p_aid)**(pow(2,m-1)-1)))
            # p_aid = 0
            # for i in range(x_prime,z+a-b+1):
            #     p_aid += Bin(z+a-b,i,2)
            # sum += p_c * p_d * (1-((1-p_aid)**(pow(2,m)-1)))
    if sum < 0:
        input("check")
    return sum

visit_f=np.zeros((max(2*eta+1,zk+1),max(2*eta+1,zk+1)),dtype=np.int32)
F=np.zeros((max(2*eta+1,zk+1),max(2*eta+1,zk+1)),dtype=np.float32)

def f(z,x,a,b):
    # f as table
    if (z!=zk) or (x!=x_const):
        input(f"error, not f({zk},{x_const},a,b)")
        return 0
    if visit_f[a][b]:
        return F[a][b]
    F[a][b]=f_computation(z,x,a,b)
    visit_f[a][b]=1
    return F[a][b]

def p_computation_table(k,pp_previous):
    pp = np.zeros((2, 2*eta+1, 2*eta+1))
    for a in range(2*eta+1):
        print(f"table{k} {a} {2*eta}")
        for b in range(2*eta+1):
            sum1 = 0
            sum2 = 0
            for ak in range(a+1):
                p_ak = Bin(a,ak,k)
                for bk in range(b+1):
                    p_bk = Hyper(zk*k,zk,b,bk)
                    sum1 += p_ak * p_bk * pp_previous[0][a-ak][b-bk]*(1-f(zk,x_const,ak,bk))
                    sum2 += p_ak * p_bk * (pp_previous[0][a-ak][b-bk]*f(zk,x_const,ak,bk)+pp_previous[1][a-ak][b-bk]*(1-f(zk,x_const,ak,bk)))
            pp[0][a][b] = sum1
            pp[1][a][b] = sum2
    return pp
    

def p_computation():
    pp_previous = np.zeros((2, 2*eta+1, 2*eta+1))
    for k in range(2,7):
        if k == 2:
            pp = np.zeros((2, 2*eta+1, 2*eta+1))
            for a in range(2*eta+1):
                print(f"table{k} {a} {2*eta}")
                for b in range(2*eta+1):
                    sum1 = 0
                    sum2 = 0
                    for ak in range(a+1):
                        p_ak = Bin(a,ak,k)
                        for bk in range(b+1):
                            p_bk = Hyper(zk*k,zk,b,bk)
                            sum1 += p_ak * p_bk * (1-f(zk,x_const,a,b))*(1-f(zk,x_const,ak,bk))
                            sum2 += p_ak * p_bk * ((1-f(zk,x_const,a,b))*f(zk,x_const,ak,bk)+f(zk,x_const,a,b)*(1-f(zk,x_const,ak,bk)))
                    pp[0][a][b] = sum1
                    pp[1][a][b] = sum2
            np.save(f"p_{k}_a_b_table",pp)
            pp_previous = pp
        else:
            pp = p_computation_table(k,pp_previous)
            np.save(f"p_{k}_a_b_table",pp)
            pp_previous = pp
    return 1