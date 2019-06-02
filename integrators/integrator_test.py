from unittest import TestCase

from debug_tools import error_tracking_tools
import integrators.Euler
import integrators.Heun
import integrators.RungeKutta
from cases.debug_case.derivative import TimeDerivative
from cases.debug_case.solution import CaseSolution


def run_test(test_obj, integrator_class, expected_order_power, test_case, initial_time_factor, start_time=0,
             end_time=10, start_value=1):
    num_time_res = 6

    # simulate until time reaches 10s and store error at every full second at num_time_res different time-resolutions
    error_tracking_list = []
    for j in range(num_time_res):
        time_factor = 2 ** j
        dt = 1.0 / (initial_time_factor * time_factor)
        error_tracker = error_tracking_tools.ErrorTracker(1, "time")
        error_tracking_list.append(error_tracker)
        timer = error_tracking_tools.TimeIterator(start_time, dt)

        solution = CaseSolution(dt, start_time, start_value, test_case)
        state = solution.get_initial_state()
        derivative = TimeDerivative(test_case)

        # choose integrator
        integrator = integrator_class(state, derivative, start_time, dt)

        for (time, state, state_sol) in zip(timer, integrator, solution):
            if time >= end_time:
                error_tracker.add_entry(time, state_sol.get_state_vars(), state.get_state_vars())
                break
    last_error_list = [error_tracker.abs_error[-1] for error_tracker in error_tracking_list]
    for i in range(num_time_res - 1):
        actual_order = last_error_list[i] / last_error_list[i + 1]
        dt = 1.0 / (initial_time_factor * (2 ** i))
        test_obj.assertTrue((2 ** expected_order_power) * 0.95 < actual_order < (2 ** expected_order_power) * 1.05,
                            msg="Mistake found in test case {} resolutions {} x {}. Expected an improvement of {} but got {}".format(
                                test_case, dt, dt * 0.5, (2 ** expected_order_power), actual_order))


class TestExplicitEuler(TestCase):
    def test_integrator(self):
        run_test(self, integrators.Euler.Explicit, 1, 0, 40)
        run_test(self, integrators.Euler.Explicit, 1, 1, 40, end_time=2)
        run_test(self, integrators.Euler.Explicit, 1, 2, 40, start_time=1)
        run_test(self, integrators.Euler.Explicit, 1, 3, 40)


class TestExplicitHeun(TestCase):
    def test_integrator(self):
        run_test(self, integrators.Heun.Explicit, 2, 0, 40)
        run_test(self, integrators.Heun.Explicit, 2, 1, 40, end_time=2)
        run_test(self, integrators.Heun.Explicit, 2, 2, 40, start_time=1)
        run_test(self, integrators.Heun.Explicit, 2, 3, 40)


class TestExplicitRungeKutta(TestCase):
    def test_integrator(self):
        run_test(self, integrators.RungeKutta.Explicit, 4, 0, 16)
        run_test(self, integrators.RungeKutta.Explicit, 4, 1, 32, end_time=2)
        run_test(self, integrators.RungeKutta.Explicit, 4, 2, 4, start_time=1)
        run_test(self, integrators.RungeKutta.Explicit, 4, 3, 32)
