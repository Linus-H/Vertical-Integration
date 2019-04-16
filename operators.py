import numpy as np


def laplace(data, delta_x):
    data_plus1 = np.roll(data, -1)
    data_minus1 = np.roll(data, 1)
    output = (data_minus1 - 2 * data + data_plus1) / (delta_x ** 2)
    return output
