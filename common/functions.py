import numpy as np

def step_function(x):
    y = x > 0
    return y.astype(int)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def relu(x):
    return np.maximum(0, x)


def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a-c)  # prevent overflow
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a

    return y

# Sum of Squares Error (SSE)
def sum_squares_error(y, t):
    return 0.5 * np.sum((y-t) ** 2)


# Cross Entrophy Error (CEE)
def cross_entrophy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
      
    batch_size = y.shape[0]
    return -np.sum(t * np.log(y + 1e-7)) / batch_size


def sigmoid_grad(x):
    return (1.0 - sigmoid(x)) * sigmoid(x)