from cases.euler_equation._1D.derivative import LogTimeDerivative
import debug_tools.visualization
import utils

import integrators.RungeKutta
import integrators.Heun

import starting_conditions
import numpy as np


def exp_curve(a):
    T = 273.15
    R = 8.314
    return np.exp(10 * a / (R * T))


def sinc_start_cond(a):
    return np.sinc(np.pi * 10 * (a - 0.5)) + 2


def const_start_cond(a):
    return a * 0 + 1


# set up display window
window_manager = debug_tools.visualization.WindowManager(2, 1)

# choose constants
num_grid_points = 400
domain_start = 0.0
domain_size = 1.0
dx = domain_size / num_grid_points

time_factor = 1000.
dt = 1.0 / (time_factor * 4.0 * (num_grid_points))

# setup
axes = np.tile(np.linspace(0, 1, num_grid_points + 1)[:-1], (2, 1))
axes[0] = axes[0] + 0.5 * dx
state = utils.State(2, num_grid_points, axes, [("x", "rho"), ("x", "u_x")])
print(np.max(state.get_state_vars()[1]))

# choose starting condition
data = state.get_state_vars()
# data[0] =starting_conditions.GaussianBump(200).start_cond(axes[0])*0.1 + 1
# data[0] = sinc_start_cond(axes[0])
# data[0] = const_start_cond(axes[0])
data[0] = exp_curve(axes[0])

# choose derivative-function
derivative = LogTimeDerivative(dx)
data[0] = np.log(data[0])

# choose integrator
integrator = integrators.RungeKutta.Explicit(state, derivative, 0, dt)

# simulation loop
for i, state in enumerate(integrator, 1):
    if i % (1 * time_factor / 2) == 0:
        print(np.sum(np.exp(state.get_state_vars()[0])))
        window_manager.display_state(1, state, 0, exp=True)
        window_manager.display_state(2, state, 1)
