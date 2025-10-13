from util_v2 import *

for i in range(401):
    print(i,400)
    for j in range(min(81,i+1)):
        for m in range(min(81,i+1)):
            for n in range(min(81,min(j,m)+1)):
                if (not visit_Hyper[i,j,m,n]):
                    HYPER[i,j,m,n] = comb(j,n)*comb(i-j,m-n)/comb(i,m)
                    visit_Hyper[i,j,m,n] = 1
    np.save(visit_Hyper_file, visit_Hyper)
    np.save(HYPER_file, HYPER)