import numpy as np

data = np.loadtxt("../data/lab2/phoneme.csv", delimiter=',')

print(data.shape)

print(data)
print(data[:, :-1], data[:, -1])