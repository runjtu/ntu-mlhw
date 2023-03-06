import numpy as np
import myPLA as my

#將網頁上的資料下載到桌面，存儲到同一個目錄下，打開命名
data = np.genfromtxt("hw1_train.dat.txt")
#獲取幾行幾列
n, d = data.shape
#分离出X
X = data[:, :-1]
#偏置增廣列向量0
X = np.c_[np.ones(n)*0, X]
#從資料剝離y
y = data[:, -1]

#problem 16，迭代1000次  
print(my.PLA(X, y))
my.f(my.PLA, X, y, 1000, 1)
