from cases.wave_equation.wrap_around.staggered_grid_derivative.derivative import TimeDerivative
import debug_tools.visualization
from debug_tools import error_tracking_tools
import integrators.Euler
import integrators.Heun
from cases.wave_equation.wrap_around.staggered_grid_derivative.solution import CaseSolution
import numpy as np


def initial_f(a):
    return np.exp(-100 * (a - 0.5) * (a - 0.5))


def derivative_f(a):
    return -(200 * a - 100) * initial_f(a)


integrator_list = [integrators.Euler.Explicit, integrators.Euler.ModifiedExplicit, integrators.Heun.Explicit,
               integrators.Heun.ModifiedExplicit]
abbrs = ["EE", "MEE", "EH", "MEH"]
time_factors = [2, 4, 8, 16]
table = {}
time_list = []

for abbr, integrator_class in zip(abbrs, integrator_list):
    for time_factor in time_factors:
        # choose constants
        num_grid_points = 1000
        start = 0.0
        L = 1.0
        f = 2.0
        dx = L / num_grid_points
        c = 2.0

        dt = 1 / (time_factor * 4.0 * num_grid_points * c)

        # set up display window
        vis = debug_tools.visualization.WindowManager(2, 1)

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
