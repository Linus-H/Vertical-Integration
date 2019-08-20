import numpy as np


def avg_forward_e1(data):
    rolled_backwards = np.roll(data, -1)
    return (rolled_backwards + data) * 0.5


def avg_backward_e1(data):
    rolled_forwards = np.roll(data, +1)
    return (rolled_forwards + data) * 0.5

def avg_s_align(offset_data):
    aligned_data = avg_forward_e1(offset_data)
    aligned_data[-1] = offset_data[-1]
    return aligned_data

def avg_s_offset(aligned_data):
    offset_data = avg_backward_e1(aligned_data)
    offset_data[0] = aligned_data[0]
    return offset_data


def avg_e2(data):
    rolled_backwards = np.roll(data, -1)
    rolled_forwards = np.roll(data, +1)
    return (rolled_backwards + 2 * data + rolled_forwards) * 0.25
