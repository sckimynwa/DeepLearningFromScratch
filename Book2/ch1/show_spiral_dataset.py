import sys
sys.path.append('..')
import matplotlib.pyplot as plt
from dataset import spiral

x, t = spiral.load_data()

print(len(x))
print('x', x.shape)
print('t', t.shape)

N = 100
CLASS_NUM = 3
markers = ['o', 'x', '+']

for i in range(CLASS_NUM):
    plt.scatter(x[i*N:(i+1)*N, 0], x[i*N:(i+1)*N, 1], s=40, marker=markers[i])
plt.show()
