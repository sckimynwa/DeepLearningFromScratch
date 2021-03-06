import numpy as np

def softmax(x):
  if x.ndim == 2:
    x = x - x.max(axis=1, keepdims=True)
    x = np.exp(x)
    x /= x.sum(axis=1, keepdims=True)
  elif x.ndim == 1:
    x = x - np.max(x)
    x = np.exp(x) / np.sum(np.exp(x))

  return x

def cross_entropy_error(y, t):
  if y.ndim == 1:
    t = t.reshape(1, t.size)
    y = y.reshape(1, y.size)
    
  # 정답 데이터가 원핫 벡터일 경우 정답 레이블 인덱스로 변환
  if t.size == y.size:
    t = t.argmax(axis=1)
            
  batch_size = y.shape[0]

  return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size


class Sigmoid:
  def __init__(self):
    self.params, self.grads = [], []
    self.out = None
  
  def forward(self, x):
    out = 1 / (1 + np.exp(-x))
    self.out = out
    return out

  def backward(self, dout):
    dx = dout * (1.0 - self.out) * self.out
    return dx
  
class Affine:
  def __init__(self, W, b):
    self.params = [W, b]
    self.grads = [np.zeros_like(W), np.zeros_like(b)]
    self.x = None

  def forward(self, x):
    W, b = self.params
    out = np.dot(x, W) + b
    self.x = x
    return out

  def backward(self, dout):
    W, b = self.params
    dx = np.dot(dout, W.T)
    dW = np.dot(self.x.T, dout)
    db = np.sum(dout, axis=0)

    self.grads[0][...] = dW
    self.grads[1][...] = db
    return dx

class MatMul:
  def __init__(self, W):
    self.params = [W]
    self.grads = [np.zeros_like(W)]
    self.x = None
  
  def forward(self, x):
    W, = self.params
    out = np.matmul(x, W)
    self.x = x
    return out
  
  def backward(self, dout):
    W, = self.params
    dx = np.matmul(dout, W.T)
    dW = np.matmul(self.x.T, dout)
    self.grads[0][...] = dW
    return dx

class SoftmaxWithLoss:
  def __init__(self):
    self.params, self.grads = [], []
    self.loss = None
    self.y = None
    self.t = None

  def forward(self, x, t):
    self.t = t
    self.y = softmax(x)
    self.loss = cross_entropy_error(self.y, self.t)
    
    return self.loss
  
  def backward(self, dout):
    batch_size = self.t.shape[0]
    dx = (self.y - self.t) / batch_size

    return dx

class SGD:
  def __init__(self, lr=0.01):
    self.lr = lr
  
  def update(self, params, grads):
    for i in range(len(params)):
      params[i] -= self.lr * grads[i]

class TwoLayerNet:
  def __init__(self, input_size, hidden_size, output_size):
    I, H, O = input_size, hidden_size, output_size

    # initialize
    W1 = 0.01 * np.random.randn(I, H)
    b1 = np.zeros(H)
    W2 = 0.01 * np.random.randn(H, O)
    b2 = np.zeros(O)

    # layers
    self.layers = [
      Affine(W1, b1),
      Sigmoid(),
      Affine(W2, b2)
    ]
    self.loss_layer = SoftmaxWithLoss()

    # list
    self.grads, self.params = [], []
    for layer in self.layers:
      self.params += layer.params
      self.grads += layer.grads

  def predict(self, x):
    for layer in self.layers:
      x = layer.forward(x)
    return x
  
  def forward(self, x, t):
    score = self.predict(x)
    loss = self.loss_layer.forward(score, t)
    return loss

  def backward(self, dout=1):
    dout = self.loss_layer.backward(dout)
    for layer in reversed(self.layers):
      dout = layer.backward(dout)
    return dout
    