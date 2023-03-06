import numpy as np
import myPLA as my
from numpy.core.numeric import ones

#將網頁上的資料下載到桌面，存儲到同一個目錄下，打開命名
data = np.genfromtxt("hw1_train.dat.txt")
#獲取幾行幾列
n, d = data.shape
#分离出X
X = data[:, :-1]
#偏置增廣列向量1
X = np.c_[np.ones(n), X]
#從資料剝離y
y = data[:, -1]

ret = np.linalg.norm(X,ord=None,axis=1) # 對象是逐列，求每行元素平方和開方
r,c=data.shape
#定義一個X1存儲標準化後的資料
X1=ones([r,c])
for i in range(r):
    X1[i]=np.array(X[i]/ret[i])

#problem 15，迭代1000次
print(my.PLA(X1, y))
my.f(my.PLA, X1, y, 1000, 1)
