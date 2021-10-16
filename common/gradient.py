import numpy as np

def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)

    for idx in range(x.size):
        tmp_val = x[idx]
        x[idx] = tmp_val + h
        f_x1 = f(x)

        x[idx] = tmp_val - h
        f_x2 = f(x)

        grad[idx] = (f_x1 - f_x2) / (2*h)
        x[idx] = tmp_val
        
    return grad