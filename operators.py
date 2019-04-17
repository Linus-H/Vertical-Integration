import numpy as np


def derivative_O1(data, delta_x):
    data_plus1 = np.roll(data, -1)

    output = (data_plus1 - data) / delta_x
    return output


def derivative_O2(data, delta_x):
    data_plus1 = np.roll(data, -1)
    data_minus1 = np.roll(data, 1)

    output = (data_plus1 - data_minus1) / (2 * delta_x)
    return output


def derivative_O4(data, delta_x):
    data_plus2 = np.roll(data, -2)
    data_plus1 = np.roll(data, -1)
    data_minus1 = np.roll(data, 1)
    data_minus2 = np.roll(data, 2)

    output = (-data_plus2 + 8 * data_plus1 - 8 * data_minus1 + data_minus2) / (12 * delta_x)
    return output


def laplace_O2(data, delta_x):
    """
    :param data:
    :param delta_x:
    :return: calculates the laplacian of the data, assuming that "data[0]==data[len(data)]"
    """
    data_plus1 = np.roll(data, -1)
    data_minus1 = np.roll(data, 1)
    output = (data_minus1 - 2 * data + data_plus1) / (delta_x ** 2)
    return output


def laplace_O4(data, delta_x):
    data_plus2 = np.roll(data, -2)
    data_plus1 = np.roll(data, -1)
    data_minus1 = np.roll(data, 1)
    data_minus2 = np.roll(data, 2)
    output = ((16. * data_plus1 + 16. * data_minus1) - (data_plus2 + data_minus2 + 30. * data)) / (
            12.0 * (delta_x ** 2))
    return output
