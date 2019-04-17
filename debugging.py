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
    def __init__(self, num_points, mode="l_inf"):
        self.timestamps = []
        self.abs_error = []
        self.num_points = num_points
        self.mode = mode

    def add_entry(self, time, real_value, calc_value):
        self.timestamps.append(time)
        if self.mode == "l_1":
            self.abs_error.append(np.sum((np.abs(real_value - calc_value))[0]))
        elif self.mode == "l_1norm":
            self.abs_error.append(np.sum((np.abs(real_value - calc_value))[0]) / self.num_points)
        elif self.mode == "l_inf":
            self.abs_error.append(np.max((np.abs(real_value - calc_value))))

    def print(self):
        for t, err in zip(self.timestamps, self.abs_error):
            print(str(t) + " " + str(err))
