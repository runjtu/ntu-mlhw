import numpy as np#for array compute
from numpy import *
import random

def pla():
    W=np.ones(4)#initial all weight with 1
    count=0
    dataset=[[1,0.10723,0.64385, 0.29556    ,1],
            [1 ,0.2418, 0.83075, 0.42741,   1],
            [1 ,0.23321 ,0.81004 ,0.98691,  1],
            [1 ,0.36163, 0.14351 ,0.3153,   -1],
            [1, 0.46984, 0.32142, 0.00042772,   -1],
            [1, 0.25969, 0.87208 ,0.075063, -1],
            ]

    while True:
        count+=1
        iscompleted=True
        for i in range(0,len(dataset)):
            X=dataset[i][:-1]
            Y=np.dot(W,X)#matrix multiply
            if sign(Y)==sign(dataset[i][-1]):
                continue
            else:
                iscompleted=False
                W=W+(dataset[i][-1])*np.array(X)
        if iscompleted:
            break
    print("final W is :",W)
    print("count is :",count)
    return W

def main():
    pla()

if __name__ == '__main__':
    main()
    