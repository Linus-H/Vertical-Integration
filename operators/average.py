import numpy as np


def avg_forward_1(data):
    rolled_backwards = np.roll(data, -1)
    return (rolled_backwards + data) * 0.5


def avg_backward_1(data):
    rolled_forwards = np.roll(data, +1)
    return (rolled_forwards + data) * 0.5

def avg_2(data):
    rolled_backwards = np.roll(data, -1)
    rolled_forwards = np.roll(data, +1)
    return (rolled_backwards + 2 * data + rolled_forwards) * 0.25