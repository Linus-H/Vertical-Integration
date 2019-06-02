import numpy as np


def flat(a):
    return 0 * a


class GaussianBump:
    def __init__(self, middle, squish_factor):
        self.middle = middle
        self.squish_factor = squish_factor

    def start_cond(self, a):
        return np.exp(- self.squish_factor * (a - self.middle) * (a - self.middle))

    def derivative(self, a):
        return - self.squish_factor * 2 * (a - self.middle) * self.start_cond(a)


class Sinc:
    def __init__(self, middle, squish_factor):
        self.middle = middle
        self.squish_factor = squish_factor

    def start_cond(self, a):
        return np.sinc(self.squish_factor * (a - self.middle))

    def derivative(self, a):
        return None
