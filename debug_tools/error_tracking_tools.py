import numpy as np


class TimeIterator:
    def __init__(self, dt):
        self.dt = dt

    def __iter__(self):
        self.n = 0
        return self

    def __next__(self):
        self.n += 1
        return self.n * self.dt


class ErrorTracker:
    def __init__(self, num_points, x_label, mode="l_inf"):
        self.labels = []
        self.label_name = x_label
        self.error_name = "Error ({})".format(mode)
        self.abs_error = []
        self.num_points = num_points
        self.mode = mode

    def add_entry(self, label, real_value, calc_value):
        self.labels.append(label)
        if self.mode == "l_1":
            self.abs_error.append(np.sum((np.abs(real_value - calc_value))))
        elif self.mode == "l_1norm":
            self.abs_error.append(np.sum((np.abs(real_value - calc_value))) / self.num_points)
        elif self.mode == "l_inf":
            self.abs_error.append(np.max(np.abs(real_value - calc_value)))

    def print(self, with_labels=False):
        if with_labels:
            for label, err in zip(self.labels, self.abs_error):
                print(str(label) + " " + str(err))
        else:
            for label, err in zip(self.labels, self.abs_error):
                print(str(err), end="\t")
            print("", end="\n")
