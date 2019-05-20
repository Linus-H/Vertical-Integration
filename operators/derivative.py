import numpy as np


def diff_backward_n1_e1(data, delta_x):
    data_minus1 = np.roll(data, 1, -1)

    output = (data - data_minus1) / delta_x
    return output


def diff_forward_n1_e1(data, delta_x):
    data_plus1 = np.roll(data, -1, -1)

    output = (data_plus1 - data) / delta_x
    return output


def diff_n1_e2(data, delta_x):
    data_plus1 = np.roll(data, -1, -1)
    data_minus1 = np.roll(data, 1, -1)

    output = (data_plus1 - data_minus1) / (2 * delta_x)
    return output


def diff_n1_e4(data, delta_x):
    data_plus2 = np.roll(data, -2, -1)
    data_plus1 = np.roll(data, -1, -1)
    data_minus1 = np.roll(data, 1, -1)
    data_minus2 = np.roll(data, 2, -1)

    output = (-data_plus2 + 8 * data_plus1 - 8 * data_minus1 + data_minus2) / (12 * delta_x)
    return output
