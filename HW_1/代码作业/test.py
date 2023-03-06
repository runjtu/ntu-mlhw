import numpy as np
from numpy.core.numeric import ones
data = np.genfromtxt("exampledata.txt")
X = data[:, :-1]
y = data[:, -1]
ret = np.linalg.norm(X,ord=None,axis=1) # 对象是逐行，求每行元素平方和开放
r,d=data.shape
d,c=data.shape
#print(r)
#print(c)

#print(ret)
X1=ones([r,c-1])

for i in range(r):
    X1[i]=np.array(X[i]/ret[i])
print(X1)
#print(y)

