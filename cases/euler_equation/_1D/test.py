from unittest import TestCase

import math

from cases.euler_equation._1D.derivative import LogTimeDerivative
from cases.run_utils import gen_test_data
from cases.numerical_ref_solution import CaseSolution, ReferenceSolutionCalculator
from cases.euler_equation._1D.solution import StationarySolution
from integrators import Euler
from starting_conditions import GaussianBump


class Test(TestCase):
    def test_with_stationary_solution(self):
        expected_order = 1
        num_grid_points = 100000

        err_lists = [[], []]
        dt_list = []

        a = 0.5

        max_exp = 5

        baseline_dt = 1.0 / ((2 ** 16) * num_grid_points)  # largest dt that will be used

        simul_t = 100 * baseline_dt

        for i in range(4):
            dt = (a ** i) * baseline_dt

            params = {
                'num_grid_points': num_grid_points,
                'domain_size': 10.0,
                'dt': dt,
                'end_time': simul_t
            }

            g = -10

            time_derivative_input = [True, True, g]

            case_sol_input = [g]

            error_tracker_list = gen_test_data(params, Euler.Explicit,
                                               LogTimeDerivative, time_derivative_input,
                                               StationarySolution, case_sol_input)
            err_lists[0].append(error_tracker_list[0].tot_error)
            err_lists[1].append(error_tracker_list[1].tot_error)

            dt_list.append(dt)

        for i in range(len(err_lists[0])):
            # actual_order = math.log(err_lists[0][i + 1] / err_lists[0][i], a)
            self.assertTrue(err_lists[0][i] < 1e-16,
                            msg="Mistake found at time-resolution {} for u. Expected error of {} but got {}".format(
                                dt_list[i], 1e-16, err_lists[0][i]))

            # actual_order = math.log(err_lists[1][i + 1] / err_lists[1][i], a)
            self.assertTrue(err_lists[1][i] < 1e-16,
                            msg="Mistake found at time-resolution {} for v. Expected error of {} but got {}".format(
                                dt_list[i], 1e-16, err_lists[1][i]))

    def test_with_numerical_solution(self):
        expected_order = 1
        c = 1.0
        num_grid_points = 100000

        err_lists = [[], []]
        dt_list = []

        a = 0.5

        max_exp = 5

        overresolution = 16

        baseline_dt = 1.0 / (40.0 * num_grid_points * c)  # largest dt that will be used

        ref_dt = (a ** max_exp) * baseline_dt / overresolution

        ref_t = 33 * baseline_dt
        simul_t = 32 * baseline_dt

        for i in range(3, 8):
            dt = (a ** i) * baseline_dt

            params = {
                'num_grid_points': num_grid_points,
                'domain_size': 1.0,
                'dt': dt,
                'end_time': simul_t / 8.0
            }

            time_derivative_input = [False]

            ref_solution_generator = ReferenceSolutionCalculator(num_grid_points, 2, LogTimeDerivative,
                                                                 time_derivative_input, [0.5, 0], params['domain_size'],
                                                                 down_sampling_rate=overresolution)

            start_cond = GaussianBump(params['domain_size'] * 0.5, 100)
            case_sol_input = [2, ref_solution_generator,
                              [lambda x: start_cond.start_cond(x) * 0.1 + 1, lambda x: 0 * x], ref_dt, ref_t]

            error_tracker_list = gen_test_data(params, Euler.Explicit,
                                               LogTimeDerivative, time_derivative_input,
                                               CaseSolution, case_sol_input)
            err_lists[0].append(error_tracker_list[0].tot_error)
            err_lists[1].append(error_tracker_list[1].tot_error)

            dt_list.append(dt)

        for i in range(len(err_lists[0]) - 1):
            actual_order = math.log(err_lists[0][i + 1] / err_lists[0][i], a)
            self.assertTrue(expected_order * 0.95 < actual_order < expected_order * 1.05,
                            msg="Mistake found at time-resolutions {} x {} for u. Expected order of {} but got {}".format(
                                dt_list[i], dt_list[i + 1], expected_order, actual_order))

            # actual_order = math.log(err_lists[1][i + 1] / err_lists[1][i], a)
            # self.assertTrue(expected_order * 0.95 < actual_order < expected_order * 1.05,
            #                msg="Mistake found at time-resolutions {} x {} for v. Expected order of {} but got {}".format(
            #                    dt_list[i], dt_list[i + 1], expected_order, actual_order))
