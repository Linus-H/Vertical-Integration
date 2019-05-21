import numpy as np

import debug_tools.visualization
from debug_tools import error_tracking_tools
from utils import State


def run_with_solution(Integrator, TimeDerivative, params, CaseSolution, case_sol_inputs):
    """
    :param Integrator: Class of the integrator to be used.
    :param TimeDerivative: Class of the time derivative to be used.
    :param params: dictionary with values for 'num_grid_points' and wave speed 'c'
    :param CaseSolution: Class for the solution of the case.
    :param case_sol_inputs: iterable list of special inputs for the class of the solution.
    """
    # choose constants
    num_grid_points = params['num_grid_points']  # 1000
    c = params['c']  # 2.0

    time_factor = 4
    start = 0.0
    L = 1.0

    dx = L / num_grid_points
    dt = 1 / (time_factor * 4.0 * num_grid_points * c)

    # setup starting condition
    standing_wave_sol = CaseSolution(c, num_grid_points, dt, *case_sol_inputs)  # , starting_cond.derivative)
    state = standing_wave_sol.get_initial_state()

    # choose border condition
    derivative = TimeDerivative(dx, c)

    # choose integrator
    integrator = Integrator(state, derivative, 0, dt)

    # debugging
    error_tracker_u = error_tracking_tools.ErrorTracker(num_grid_points, "Time", mode="l_1norm")
    error_tracker_v = error_tracking_tools.ErrorTracker(num_grid_points, "Time", mode="l_1norm")
    timer = error_tracking_tools.TimeIterator(0,dt)
    time_list = []

    # set up display window
    window_manager = debug_tools.visualization.WindowManager(4, 2)

    # simulation loop
    for i, (time, state, state_sol) in enumerate(zip(timer, integrator, standing_wave_sol), 1):
        if i % (100 * time_factor) == 0:
            # print(i)
            time_list.append(time)
            error_tracker_u.add_entry(time, state_sol.get_state_vars()[0], state.get_state_vars()[0])
            error_tracker_v.add_entry(time, state_sol.get_state_vars()[1], state.get_state_vars()[1])

            print(time, end="\t")
            print(error_tracker_u.abs_error[-1], end="\t")
            print(error_tracker_v.abs_error[-1])

            difference = state - state_sol

            window_manager.display_state(1, difference, 0)
            window_manager.display_state(3, state, 0, -2, 2)
            window_manager.display_state(5, state_sol, 0, -2, 2)
            window_manager.display_error(7, error_tracker_u, double_log=False, line_name="Error", clear_axis=True)

            window_manager.display_state(2, difference, 1)
            window_manager.display_state(4, state, 1, -2, 2)
            window_manager.display_state(6, state_sol, 1, -2, 2)
            window_manager.display_error(8, error_tracker_v, double_log=False, line_name="Error", clear_axis=True)

        # vis.display(2, state, 1, -10, 10)
        # if i == 10000 * time_factor:
        #    break

    # for e in error_tracker.abs_error:
    #    print(e, end="\t")
    #    print("", end="\n")


def run_without_solution(Integrator, TimeDerivative, params, state):
    """
    :param Integrator: Class of the integrator to be used.
    :param TimeDerivative: Class of the time derivative to be used.
    :param params: dictionary with values for 'num_grid_points', 'squish_factor', and wave speed 'c'
    :param StartCond: Class of the starting condition.
    """
    # choose constants
    num_grid_points = params['num_grid_points']  # 1000
    squish_factor = params['squish_factor']  # 2.0
    c = params['c']  # 2.0

    time_factor = 4
    start = 0.0
    L = 1.0

    dx = L / num_grid_points
    dt = 1 / (time_factor * 4.0 * num_grid_points * c)

    # choose border condition
    derivative = TimeDerivative(dx, c)

    # choose integrator
    integrator = Integrator(state, derivative, 0, dt)

    # set up display window
    window_manager = debug_tools.visualization.WindowManager(2, 1)

    # simulation loop
    for i, state in enumerate(integrator, 1):
        if i % (100 * time_factor) == 0:
            window_manager.display_state(1, state, 0, -2, 2)
            window_manager.display_state(2, state, 1, -2, 2)
