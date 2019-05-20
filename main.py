from cases.wave_equation.wrap_around.derivative.derivative import TimeDerivative
import debug_tools.visualization
from debug_tools import error_tracking_tools
from cases.wave_equation.wrap_around.derivative.solution import CaseSolution
import starting_conditions

import integrators.Heun
import integrators.Euler
import integrators.RungeKutta
import numpy as np

# choose constants
num_grid_points = 1000
start = 0.0
L = 1.0
squish_factor = 2.0
dx = L / num_grid_points
c = 2.0

time_factor = 4
dt = 1 / (time_factor * 4.0 * num_grid_points * c)

# set up display window
window_manager = debug_tools.visualization.WindowManager(4, 2)


def initial_f(a):
    return np.exp(- squish_factor * 100 * (a - 0.5) * (a - 0.5))
    # return np.sinc(np.pi * 100 * (a - 0.5))


def derivative_f(a):
    return - squish_factor * (200 * a - 100) * initial_f(a)


# setup starting condition
starting_cond = starting_conditions.GaussianBump(200.0)
standing_wave_sol = CaseSolution(c, num_grid_points, dt, starting_cond.start_cond)#, starting_cond.derivative)
# known_problems.StandingWaveFixedEnd(c, num_grid_points, dt, [(1, 1.0), (2, 1.0)])
state = standing_wave_sol.get_initial_state()
print(np.max(state.get_state_vars()[1]))

# choose border condition
derivative = TimeDerivative(dx, c)

# choose integrator
integrator = integrators.RungeKutta.Explicit(state, derivative, dt)

# debugging
error_tracker_u = error_tracking_tools.ErrorTracker(num_grid_points, "Time", mode="l_1norm")
error_tracker_v = error_tracking_tools.ErrorTracker(num_grid_points, "Time", mode="l_1norm")
timer = error_tracking_tools.TimeIterator(dt)
time_list = []

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
