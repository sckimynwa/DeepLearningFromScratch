import numpy as np


# Sum of Squares Error (SSE)
def sum_squares_error(y, t):
    return 0.5 * np.sum((y-t) ** 2)


# Cross Entrophy Error (CEE)
def cross_entrophy_error(y, t):
    delta = 1e-7
    return -np.sum(t * np.log(y + delta))


t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]

print("######## ERROR Methods ########")
print("SSE: ", + sum_squares_error(np.array(y), np.array(t)))
print("CEE: ", + cross_entrophy_error(np.array(y), np.array(t)))
