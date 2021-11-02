import sys
sys.path.append('..')
import numpy as np
from forward_net import SGD, TwoLayerNet
from dataset import spiral
import matplotlib.pyplot as plt

# hyperparameter
max_epoch = 300
batch_size = 30
hidden_size = 10
learning_rate = 1.0

# read data & generate model
x, t = spiral.load_data()
model = TwoLayerNet(input_size=2, hidden_size=hidden_size, output_size=3)
optimizer = SGD(lr=learning_rate)

# learning variable
data_size = len(x)
max_iters = data_size // batch_size
total_loss = 0
loss_count = 0
loss_list = []

for epoch in range(max_epoch):
  idx = np.random.permutation(data_size)
  x = x[idx]
  t = t[idx]

  for iters in range(max_iters):
    batch_x = x[iters*batch_size:(iters+1)*batch_size]
    batch_t = t[iters*batch_size:(iters+1)*batch_size]

    loss = model.forward(batch_x, batch_t)
    model.backward()
    optimizer.update(model.params, model.grads)

    total_loss += loss
    loss_count += 1
  
    if (iters+1) % 10 == 0:
      avg_loss = total_loss / loss_count
      print("| epoch %d | iter %d / %d | loss %.2f" % (epoch+1, iters+1, max_iters, avg_loss))
      loss_list.append(avg_loss)
      total_loss, loss_count = 0, 0

# 학습 결과 플롯
# plt.plot(np.arange(len(loss_list)), loss_list, label='train')
# plt.xlabel('반복 (x10)')
# plt.ylabel('손실')
# plt.show()

# 데이터점 플롯
x, t = spiral.load_data()
N = 100
CLS_NUM = 3
markers = ['o', 'x', '^']
for i in range(CLS_NUM):
    plt.scatter(x[i*N:(i+1)*N, 0], x[i*N:(i+1)*N, 1], s=40, marker=markers[i])
plt.show()