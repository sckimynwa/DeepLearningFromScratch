import numpy as np
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

  def backward(dout):
    dx = dout * 1
    dy = dout * 1
    return dx, dy

class Relu:
  def __init__(self):
    self.mask = None
  
  def forward(self, x):
    self.mask = (x <= 0)
    out = x.copy()
    out[self.mask] = 0

    return out
  
  def backward(self, dout):
    dout[self.mask] = 0
    dx = dout

    return dx

class Sigmoid:
  def __init__(self):
    self.out = None
  
  def forward(self, x):
    out = 1 / (1 + np.exp(-x))
    self.out = out

    return out

  def backward(self, dout):
    y = self.out
    dx = dout * y * (1.0 - y)

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
    self.dw = np.dot(self.x.T, dout)
    self.db = np.sum(dout, axis = 0)

    return dx

# Examples
apple = 100
apple_num = 2
orange = 150
orange_num = 3
tax = 1.1

# Layers
mul_apple_layer = MulLayer()
mul_orange_layer = MulLayer()
add_apple_orange_layer = MulLayer()
mul_tax_layer = MulLayer()

# Forward Propagation
apple_price = mul_apple_layer.forward(apple, apple_num)
orange_price = mul_orange_layer.forward(orange, orange_num)
all_price = add_apple_orange_layer.forward(apple_price, orange_price)
price = mul_tax_layer.forward(all_price, tax)

print("##### Forward Propagation Result #####")
print(price)
print('\n')

# Backward Propagation
dprice = 1
dall_price, dtax = mul_tax_layer.backward(dprice)
dapple_price, dorange_price = add_apple_orange_layer.backward(dall_price)
dapple, dapple_num = mul_apple_layer.backward(dapple_price)
dorange, dorange_num = mul_orange_layer.backward(dorange_price)

print("##### Backward Propagation Result #####")
print(dprice, dall_price, dtax, dapple_price, dorange_price, dapple_price, dapple_num, dorange_price, dorange_num)
print('\n')

print("##### Relu Layer #####")
x = np.array([[1.0, -0.5], [-2.0, 3.0]])
print("original array")
print(x)
print('\n')
mask = (x <= 0)
print("mask array")
print(mask)
print('\n')