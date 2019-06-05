import numpy as np


class TimeIterator:
    def __init__(self, t0, dt):
        self.dt = dt
        self.t0 = t0

    def __iter__(self):
        self.n = 0
        return self

    def __next__(self):
        self.n += 1
        return self.t0 + self.n * self.dt

    def get_time(self):
        return self.t0 + self.n * self.dt


class ErrorTracker:  # TODO: add method to convert abs_error to numpy array
    def __init__(self, num_points, x_label, norm="l_inf"):
        """
        :param num_points: number of points in an axis.
        :param x_label: name of the labels applied to the errors tracked by this object.
        :param norm: which norm to use. "l_1" sum of absolute error, "l_1norm" sum of absolute error divided by the
number of points, "l_inf" maximum absolute error.
        """
        self.labels = []
        self.label_name = x_label
        self.error_name = "Error ({})".format(norm)
        self.abs_error = []
        self.num_points = num_points
        self.mode = norm

    def add_entry(self, label, real_value, calc_value):
        self.labels.append(label)
        if self.mode == "l_1":
            new_value = (np.sum((np.abs(real_value - calc_value))))
        elif self.mode == "l_1norm":
            new_value = (np.sum((np.abs(real_value - calc_value))) / self.num_points)
        elif self.mode == "l_inf":
            new_value = (np.max(np.abs(real_value - calc_value)))
        else:
            new_value = None
        self.abs_error.append(new_value)

    def print(self, with_labels=False):
        if with_labels:
            for label, err in zip(self.labels, self.abs_error):
                print(str(label) + " " + str(err))
        else:
            for label, err in zip(self.labels, self.abs_error):
                print(str(err), end="\t")
            print("", end="\n")


class ErrorIntegrator:  # TODO: add method to convert abs_error to numpy array
    def __init__(self, num_points, norm="l_inf"):
        """
        :param num_points: number of points in an axis.
        :param norm: which norm to use. "l_1" sum of absolute error, "l_1norm" sum of absolute error divided by the
number of points, "l_inf" maximum absolute error.
        """
        self.error_name = "Error ({})".format(norm)
        self.tot_error = 0
        self.num_points = num_points
        self.mode = norm

    def add_entry(self, label, real_value, calc_value):
        if self.mode == "l_1":
            error = (np.sum((np.abs(real_value - calc_value))))
        elif self.mode == "l_1norm":
            error = (np.sum((np.abs(real_value - calc_value))) / self.num_points)
        elif self.mode == "l_inf":
            error = (np.max(np.abs(real_value - calc_value)))
        else:
            error = None
        self.tot_error += error
