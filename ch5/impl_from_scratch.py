from collections import OrderedDict
import numpy as np


class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size):
        self.params = {}
        self.params['W1'] = np.random.rand(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = np.random.rand(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

        # Layers
        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['Sigmoid'] = Sigmoid()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])
        
        self.lastLayer = SoftmaxWithLoss()

    def predict(self, x):
      for layer in self.layers.values():
        x = layer.forward(x)
      
      return x

    def loss(self, x, t):
      y = self.predict(x)
      return self.lastLayer.forward(y, t)
    
    def accuracy(self, x, t):
      y = self.predict(x)
      y = np.argmax(y, axis=1)
      if t.ndim != 1: t = np.argmax(t, axis=1)

      accuracy = np.sum(y == t) / float(x.shape[0])
      return accuracy
    
    def gradient(self, x, t):
      self.loss(x, t)

      dout = 1
      dout = self.lastLayer.backward(dout)

      layers = list(self.layers.values())
      layers.reverse()

      for layer in layers:
        dout = layer.backward(dout)
      
      grads = {}
      grads['W1'] = self.layers['Affine1'].dW
      grads['b1'] = self.layers['Affine1'].db
      grads['W2'] = self.layers['Affine2'].dW
      grads['b2'] = self.layers['Affine2'].db

      return grads



# Layer Class
class MulLayer:
    def __init__(self):
        self.x = None
        self.y = None

    def forward(self, x, y):
        self.x = x
        self.y = y
        out = x * y

        return out

    def backward(self, dout):
        dx = dout * self.y
        dy = dout * self.x

        return dx, dy


class AddLayer:
    def __init__(self):
        pass

    def forward(self, x, y):
        return x + y

    def backward(self, dout):
        dx = dout * 1
        dy = dout * 1

        return dx, dy


class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        self.out = out

        return out

    def backward(self, dout):
        dx = dout * (self.out * (1 - self.out))

        return dx

class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None

    def forward(self, x, t):
      self.t = t
      self.y = softmax(x)
      self.loss = cross_entropy_error(self.y, self.t)
      
      return self.loss

    def backward(self, dout = 1):
      batch_size = self.t.shape[0]
      dx = (self.y - self.t) / batch_size

      return dx



class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.dW = None
        self.db = None

    def forward(self, x):
        self.x = x
        out = np.dot(x, self.W) + self.b

        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)

        return dx

## Functions
def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a-c)  # prevent overflow
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a

    return y

def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
      
    batch_size = y.shape[0]
    return -np.sum(t * np.log(y + 1e-7)) / batch_size