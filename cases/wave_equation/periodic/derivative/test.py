from unittest import TestCase

import math

from cases.run_utils import gen_test_data
from cases.wave_equation.periodic.derivative.derivative import PeriodicWaveTimeDerivative
from cases.wave_equation.periodic.derivative.solution import CaseSolution
from integrators import Heun
from starting_conditions import GaussianBump


class Test(TestCase):
    def test_with_solution(self):
        c = 1.0
        num_grid_points = 100000

        err_lists = [[], []]
        dt_list = []

        a = 0.5
        expected_order = 2

        t = 20 / (4.0 * num_grid_points * c)

        for i in range(10):
            dt = (a ** i) * (2 ** 10) * 64.0 / (4.0 * num_grid_points * c)

            params = {
                'num_grid_points': num_grid_points,
                'domain_size': 1.0,
                'dt': dt,
                'end_time': t
            }

            time_derivative_input = [c]

            case_sol_input = [c, GaussianBump(params['domain_size'] * 0.5, 2).get_start_condition]

            error_tracker_list = gen_test_data(params, Heun.Explicit,
                                               PeriodicWaveTimeDerivative, time_derivative_input,
                                               CaseSolution, case_sol_input)
            err_lists[0].append(error_tracker_list[0].tot_error)
            err_lists[1].append(error_tracker_list[1].tot_error)

            dt_list.append(dt)

        for i in range(len(err_lists[0]) - 1):
            actual_order = math.log(err_lists[0][i + 1] / err_lists[0][i], a)
            self.assertTrue(expected_order * 0.95 < actual_order < expected_order * 1.05,
                            msg="Mistake found at time-resolutions {} x {} for u. Expected order of {} but got {}".format(
                                dt_list[i], dt_list[i + 1], expected_order, actual_order))

            actual_order = math.log(err_lists[1][i + 1] / err_lists[1][i], a)
            self.assertTrue(expected_order * 0.95 < actual_order < expected_order * 1.05,
                            msg="Mistake found at time-resolutions {} x {} for v. Expected order of {} but got {}".format(
                                dt_list[i], dt_list[i + 1], expected_order, actual_order))
