import debug_tools.visualization
from debug_tools import error_tracking_tools
import integrators.Euler
import integrators.Heun
import integrators.RungeKutta
# from cases.wave_equation.wrap_around.derivative.derivative import TimeDerivative
# from cases.wave_equation.wrap_around.staggered_grid_derivative.solution import CaseSolution
from cases.debug_case.derivative import TimeDerivative
from cases.debug_case.solution import CaseSolution
from matplotlib import pyplot as plt

import numpy as np


def initial_f(a):
    return np.exp(-100 * (a - 0.5) * (a - 0.5))


def derivative_f(a):
    return -(200 * a - 100) * initial_f(a)


integrator_list = [integrators.Euler.Explicit,
                   integrators.Heun.Explicit,
                   integrators.RungeKutta.Explicit]
integrator_names = ["Euler.Explicit",
                    "Heun.Explicit",
                    "RungeKutta.Explicit"]


def unit_test():
    expected_orders = [1, 2, 4]
    num_time_res = 6
    initial_time_factor = 4
    t0 = 0
    y0 = 4

    # simulate until time reaches 10s and store error at every full second at num_time_res different time-resolutions
    for integrator_class, expected_order, name in zip(integrator_list, expected_orders, integrator_names):
        print(name)
        error_tracking_list = []
        for j in range(num_time_res):
            time_factor = 2 ** j
            dt = 1.0 / (initial_time_factor * time_factor)
            error_tracker = error_tracking_tools.ErrorTracker(1, "time")
            error_tracking_list.append(error_tracker)
            timer = error_tracking_tools.TimeIterator(t0, dt)

            solution = CaseSolution(dt, t0, y0)
            state = solution.get_initial_state()
            derivative = TimeDerivative()

            # choose integrator
            integrator = integrator_class(state, derivative, t0, dt)
            time_list = []
            arr = []
            acc_arr = []
            for i, (time, state, state_sol) in enumerate(zip(timer, integrator, solution), 1):
                if j == num_time_res - 1 and expected_order == 4:
                    acc_arr.append(state_sol.get_state_vars()[0, 0])
                arr.append(state.get_state_vars()[0, 0])
                time_list.append(time)
                if i % (initial_time_factor * time_factor) == 0:
                    error_tracker.add_entry(time, state_sol.get_state_vars(), state.get_state_vars())
                if time >= 10:
                    break
            plt.plot(time_list, arr, label=name)

            if j == num_time_res - 1 and expected_order == 4:
                plt.plot(time_list, acc_arr, 'k--', label="accurate")
            # print(error_tracker.abs_error)
        for error_tracker in error_tracking_list:
            print(error_tracker.abs_error[-1] / error_tracking_list[-1].abs_error[-1], end="\t")
            print(error_tracker.abs_error[-1])

    plt.legend()
    plt.grid()
    plt.show()


unit_test()


def visual_test():
    time_factors = [2, 4, 8]  # , 16]
    table = {}
    time_list = []

    # set up display window
    window_manager = debug_tools.visualization.WindowManager(1, 1)

    for abbr, integrator_class in zip(integrator_names, integrator_list):
        for time_factor in time_factors:
            # choose constants
            num_grid_points = time_factor * 1000
            start = 0.0
            L = 1.0
            f = 2.0
            dx = L / num_grid_points
            c = 2.0

            dt = 1 / (4.0 * num_grid_points * c)

            # setup
            solution = CaseSolution(c, num_grid_points, dt, initial_f)
            state = solution.get_initial_state()

            # choose border condition
            derivative = TimeDerivative(dx, c)

            # choose integrator
            integrator = integrator_class(state, derivative, dt)

            # debugging
            error_tracker = error_tracking_tools.ErrorTracker(num_grid_points, "time")
            timer = error_tracking_tools.TimeIterator(dt)
            time_list = []

            # simulation loop
            for i, (time, state, state_sol) in enumerate(zip(timer, integrator, solution), 1):
                if i % (100 * time_factor) == 0:
                    # print(i)
                    time_list.append(time)
                    error_tracker.add_entry(time, state_sol.get_state_vars(), state.get_state_vars())
                if i == 10000 * time_factor:
                    break
            name = abbr + " " + str(dt)
            print(name, end="\t")
            error_tracker.print(with_labels=False)
            window_manager.display_error(1, error_tracker, True, name, clear_axis=False)
    from matplotlib import pyplot as plt

    plt.show()
