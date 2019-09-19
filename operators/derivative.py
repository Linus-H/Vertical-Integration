import numpy as np


def diff_backward_n1_e1(data, delta_x):
    data_minus1 = np.roll(data, 1, -1)

    output = (data - data_minus1) / delta_x
    return output


def diff_forward_n1_e1(data, delta_x):
    data_plus1 = np.roll(data, -1, -1)

    output = (data_plus1 - data) / delta_x
    return output


def diff_offset_n1_e2(aligned_data, delta_x):
    return diff_forward_n1_e1(aligned_data, delta_x)


def diff_align_n1_e2(offset_data, delta_x):
    return diff_backward_n1_e1(offset_data, delta_x)


def diff_s_offset_n1_e2(aligned_data, delta_x):
    return diff_backward_n1_e1(aligned_data, delta_x)


def diff_s_align_n1_e2(offset_data, delta_x):
    return diff_forward_n1_e1(offset_data, delta_x)


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


def diff_fft(data, delta_x):  # this has not been properly unit-tested, but seems to work for the wave-equation
    transformed = np.fft.fft(data) / len(data)
    shifted = np.fft.fftshift(transformed)
    N = len(data)
    L = N * delta_x
    k = np.arange(0, N) * 1.0
    if N % 2 == 0:
        k -= N / 2
    else:
        k -= (N - 1) / 2
    k = k * 2j * np.pi / delta_x
    shifted = shifted * k
    transformed = np.fft.ifftshift(shifted)
    return np.fft.ifft(transformed)
