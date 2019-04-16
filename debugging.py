import numpy as np


class Timer:
    def __init__(self, dt):
        self.dt = dt

    def __iter__(self):
        self.n = 0
        return self

    def __next__(self):
        self.n += 1
        return self.n * self.dt


class TrackError:
    def __init__(self):
        self.timestamps = []
        self.abs_error = []

    def add_entry(self, time, real_value, calc_value):
        self.timestamps.append(time)
        self.abs_error.append(np.sum((np.abs(real_value - calc_value))[0]))

    def print(self):
        for t, err in zip(self.timestamps, self.abs_error):
            print(str(t) + " " + str(err))
