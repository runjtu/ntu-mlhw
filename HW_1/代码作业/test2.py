import numpy as np

x = np.array([[1000, 10, 0.5],

[ 765, 5, 0.35],

[ 800, 7, 0.09]])

x_normed = x / x.max(axis=0)

print(x_normed)