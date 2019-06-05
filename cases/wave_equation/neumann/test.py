from unittest import TestCase

import math

from cases.run_utils import gen_test_data
from cases.wave_equation.neumann.derivative import TimeDerivativeLaplace
from cases.numerical_ref_solution import CaseSolution, ReferenceSolutionCalculator
from integrators import Euler, Heun
from starting_conditions import GaussianBump


class Test(TestCase):
    def test_with_solution(self):
        expected_order = 2
        c = 1.0
        num_grid_points = 100000

        err_lists = [[], []]
        dt_list = []

        a = 0.5

        max_exp = 5

        overresolution = 16

        baseline_dt = 1.0 / (4.0 * num_grid_points * c)  # largest dt that will be used

        ref_dt = (a ** max_exp) * baseline_dt / overresolution

        ref_t = 33 * baseline_dt
        simul_t = 32 * baseline_dt

        for i in range(5):
            dt = (a ** i) * baseline_dt

            params = {
                'num_grid_points': num_grid_points,
                'domain_size': 1.0,
                'dt': dt,
                'end_time': simul_t
            }

            time_derivative_input = [c]

            ref_solution_generator = ReferenceSolutionCalculator(num_grid_points, 2,
                                                                 TimeDerivativeLaplace, time_derivative_input,
                                                                 [0, 0], params['domain_size'],
                                                                 down_sampling_rate=overresolution)

            start_cond = GaussianBump(params['domain_size'] * 0.5, 100)
            case_sol_input = [2, ref_solution_generator, [start_cond.start_cond, start_cond.derivative], ref_dt, ref_t]

            error_tracker_list = gen_test_data(params, Heun.Explicit,
                                               TimeDerivativeLaplace, time_derivative_input,
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
