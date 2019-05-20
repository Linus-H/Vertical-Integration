import numpy as np


class GaussianBump:
    def __init__(self, squish_factor):
        self.squish_factor = squish_factor

    def start_cond(self, a):
        return np.exp(- self.squish_factor * (a - 0.5) * (a - 0.5))

    def derivative(self, a):
        return - self.squish_factor * (2 * a - 1) * self.start_cond(a)


class Sinc:
    def __init__(self, squish_factor):
        self.squish_factor = squish_factor

    def start_cond(self, a):
        return np.sinc(self.squish_factor * (a - 0.5))

    def derivative(self, a):
        return None