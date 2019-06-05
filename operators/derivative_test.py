from unittest import TestCase

import numpy as np

from matplotlib import pyplot as plt

from debug_tools import error_tracking_tools
import operators.derivative
from debug_tools.visualization import WindowManager


def run_test(test_object, op, expected_order):
    tracker = error_tracking_tools.ErrorTracker(num_points=None, x_label="#Samples", norm="l_1norm")

    max_res_exp = 7
    # take measurements
    for resolution_exp in range(max_res_exp):
        num_points = 100 * (2 ** resolution_exp)

        # shifting axis by irrational number e to avoid 'random' perfect results at low resolutions
        axis = np.linspace(0, 1.0, num_points + 1)[:-1] - np.e
        dx = 1.0 / num_points

        factor = 2 * np.pi * 10

        signal = np.sin(axis * factor)
        accurate_derivative = factor * np.cos(axis * factor)

        calc_derivative = op(signal, dx)
        tracker.num_points = num_points
        tracker.add_entry(num_points, accurate_derivative, calc_derivative)

    # evaluate measurements by comparing every measurement with every other measurement
    for span in range(1, max_res_exp):
        expected_improvement = 2 ** (span * expected_order)
        for i in range(0, max_res_exp - span):
            j = i + span
            actual_improvement = tracker.abs_error[i] / tracker.abs_error[j]
            test_object.assertTrue(expected_improvement * 0.95 < actual_improvement < expected_improvement * 1.05,
                                   msg="Mistake found at resolutions {} x {}. Expected an improvement of {} but got {}".format(
                                       tracker.labels[i], tracker.labels[j], expected_improvement, actual_improvement))


class TestDiff_backward_n1_e1(TestCase):
    def test_diff_backward_n1_e1(self):
        run_test(self, operators.derivative.diff_backward_n1_e1, 1)


class TestDiff_forward_n1_e1(TestCase):
    def test_diff_forward_n1_e1(self):
        run_test(self, operators.derivative.diff_forward_n1_e1, 1)


class TestDiff_n1_e2(TestCase):
    def test_diff_n1_e2(self):
        run_test(self, operators.derivative.diff_n1_e2, 2)


class TestDiff_n1_e4(TestCase):
    def test_diff_n1_e4(self):
        run_test(self, operators.derivative.diff_n1_e4, 4)


def visual_test():
    window_manager = WindowManager(1, 1)

    for op, name in zip([operators.derivative.diff_backward_n1_e1, operators.derivative.diff_n1_e2,
                         operators.derivative.diff_n1_e4],
                        ["Derivative Order: 1", "Derivative Order: 2", "Derivative Order: 4"]):
        print(name)
        tracker = error_tracking_tools.ErrorTracker(num_points=None, x_label="Number of Samples", norm="l_inf")
        for resolution_exp in range(20):
            num_points = 5 * (2 ** resolution_exp)
            L = 1

            # shifting axis by irrational number e to avoid 'random' perfect results at low resolutions
            axis = np.linspace(0, L, num_points + 1)[:-1] - np.e
            dx = L * 1.0 / num_points

            factor = np.pi * 2 * 10  # 2 pi to have a periodic signal, other factors to increase the frequency
            signal = np.sin(axis * factor)

            calculated_result = op(signal, dx)
            accurate_result = factor * np.cos(axis * factor)

            tracker.add_entry(num_points, accurate_result, calculated_result)
            print(str(num_points) + "\t" + str(tracker.abs_error[-1]) + "\t" + str(1 / dx ** 2))

        window_manager.display_error(1, tracker, double_log=True, line_name=name, clear_axis=False)

    window_manager.draw_loglog_oder_line(1, 10, 1e-6, 1, 1)
    window_manager.draw_loglog_oder_line(1, 10, 1e-6, 1, 2)
    window_manager.draw_loglog_oder_line(1, 10, 1e-6, 1, 4)
    plt.grid()
    plt.show()
