import utils
import debug_tools.visualization
from debug_tools import error_tracking_tools


def gen_test_data(params, integrator_class,
                  time_derivative_class, time_derivative_inputs,
                  case_solution_class, case_sol_inputs):
    """
    Outputs a list of error trackers that contain data about the difference between the given solution and the
integrators output over time.
    :param params: dictionary with values for 'num_grid_points', time-step-size 'dt', time at which to evaluate the difference 'end_time' btw solution and simulation, size of the domain 'domain_size',
    :param integrator_class: Class of the integrator to be used for the simulation.
    :param time_derivative_class: Class of the time derivative to be used.
    :param time_derivative_inputs: Iterable list of special inputs for the class of the time derivative.
    :param case_solution_class: Class for the solution of the case.
    :param case_sol_inputs: Iterable list of special inputs for the class of the solution.
    """
    # choose constants
    num_grid_points = params['num_grid_points']
    domain_size = params['domain_size']
    dt = params['dt']
    end_time = params['end_time']

    dx = domain_size / num_grid_points

    # setup starting condition
    solution = case_solution_class(num_grid_points, dt, domain_size, *case_sol_inputs)
    state = solution.get_initial_state()

    # choose border condition
    derivative = time_derivative_class(dx, *time_derivative_inputs)

    # choose integrator
    integrator = integrator_class(state, derivative, 0, dt)

    # debugging
    num_vars = len(state.get_names())
    error_trackers = []
    for _ in range(num_vars):
        error_trackers.append(error_tracking_tools.ErrorIntegrator(num_grid_points, norm="l_1norm"))

    timer = error_tracking_tools.TimeIterator(0, dt)

    # simulation loop
    for i, (time, state, state_sol) in enumerate(zip(timer, integrator, solution), 1):
        if time >= end_time:
            for j in range(num_vars):
                error_trackers[j].add_entry(time, state_sol.get_state_vars()[j], state.get_state_vars()[j])
            break
    return error_trackers


def run_visual_with_solution(params, integrator_class,
                             time_derivative_class, time_derivative_inputs,
                             case_solution_class, case_sol_inputs):
    """
    :param params: dictionary with values for 'num_grid_points', time-step-size 'dt', size of the domain 'domain_size',
every how many steps the state shall be displayed 'sampling_rate'.
    :param integrator_class: Class of the integrator to be used.
    :param time_derivative_class: Class of the time derivative to be used.
    :param time_derivative_inputs: Iterable list of special inputs for the class of the time derivative.
    :param case_solution_class: Class for the solution of the case.
    :param case_sol_inputs: iterable list of special inputs for the class of the solution.
    """
    # choose constants
    num_grid_points = params['num_grid_points']
    domain_size = params['domain_size']
    dt = params['dt']
    sampling_rate = params['sampling_rate']

    dx = domain_size / num_grid_points

    # setup starting condition
    solution = case_solution_class(num_grid_points, dt, domain_size, *case_sol_inputs)
    state = solution.get_initial_state()

    # choose border condition
    derivative = time_derivative_class(dx, *time_derivative_inputs)

    # choose integrator
    integrator = integrator_class(state, derivative, 0, dt)

    # debugging
    num_vars = len(state.get_names())
    error_trackers = []
    for _ in range(num_vars):
        error_trackers.append(error_tracking_tools.ErrorTracker(num_grid_points, "Time", norm="l_inf"))

    timer = error_tracking_tools.TimeIterator(0, dt)

    # set up display window
    window_manager = debug_tools.visualization.WindowManager(4, num_vars)

    # simulation loop
    for i, (time, state, state_sol) in enumerate(zip(timer, integrator, solution), 1):
        if i % sampling_rate == 0:
            # print(time, end="\t")
            difference = state - state_sol
            for j in range(num_vars):
                error_trackers[j].add_entry(time, state_sol.get_state_vars()[j], state.get_state_vars()[j])
                # print(error_trackers[j].abs_error[-1], end="\t")
                window_manager.display_state(1 + j + 0 * num_vars, difference, j)
                window_manager.display_state(1 + j + 1 * num_vars, state, j, -2, 2)
                window_manager.display_state(1 + j + 2 * num_vars, state_sol, j, -2, 2)
                window_manager.display_error(1 + j + 3 * num_vars, error_trackers[j], double_log=False,
                                             line_name="Error", clear_axis=True)


def run_visual_without_solution(params, integrator_class,
                                time_derivative_class, time_derivative_inputs,
                                state, display_diagram_exponentially=None):
    """
    :param params: dictionary with values for 'num_grid_points', time-step-size 'dt', size of the domain 'domain_size',
every how many steps the state shall be displayed 'sampling_rate'.
    :param integrator_class: Class of the integrator to be used.
    :param time_derivative_class: Class of the time derivative to be used.
    :param time_derivative_inputs: Iterable list of special inputs for the class of the time derivative.
    :param state: An object of class utils.State, which contains the initial state of the system.
    :param display_diagram_exponentially: A list of booleans, where it can be specified whether a system variable should
be exponentiated before being displayed (True), or not (False).
    """
    # choose constants
    num_grid_points = params['num_grid_points']
    domain_size = params['domain_size']
    dt = params['dt']
    sampling_rate = params['sampling_rate']

    dx = domain_size / num_grid_points

    # choose border condition
    derivative = time_derivative_class(dx, *time_derivative_inputs)

    # choose integrator
    integrator = integrator_class(state, derivative, 0, dt)

    # set up display window
    num_vars = len(state.get_names())
    window_manager = debug_tools.visualization.WindowManager(num_vars, 1)

    if display_diagram_exponentially is None:
        display_diagram_exponentially = [False] * num_vars

    # simulation loop
    for i, state in enumerate(integrator, 1):
        if i % sampling_rate == 0:
            for j in range(num_vars):
                window_manager.display_state(1 + j, state, j, exp=display_diagram_exponentially[j])
