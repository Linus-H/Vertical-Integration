import numpy as np


def flat(a):
    return 0 * a


class GaussianBump:
    def __init__(self, middle, squish_factor):
        self.middle = middle
        self.squish_factor = squish_factor

    def get_start_condition(self, a):
        return np.exp(- self.squish_factor * (a - self.middle) * (a - self.middle))

    def get_derivative(self, a):
        return - self.squish_factor * 2 * (a - self.middle) * self.get_start_condition(a)


class Sinc:
    def __init__(self, middle, squish_factor):
        self.middle = middle
        self.squish_factor = squish_factor

    def get_start_condition(self, a):
        return np.sinc(self.squish_factor * (a - self.middle))

    def get_derivative(self, a):
        return None
