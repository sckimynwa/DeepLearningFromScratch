import numpy as np
import matplotlib.pylab as plt


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


def draw(x, y):
    plt.plot(x, y)
    plt.ylim(-0.1, 1.1)
    plt.show()
