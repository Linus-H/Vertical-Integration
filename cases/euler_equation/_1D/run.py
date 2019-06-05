from cases import run_utils
from cases.euler_equation._1D.derivative import LogTimeDerivative
import utils

import integrators.RungeKutta
import integrators.Heun

import starting_conditions
import numpy as np


def exp_curve(a):
    T = 273.15
    R = 8.314
    return np.exp(-10 * a / (R * T))


def sinc_start_cond(a):
    return np.sinc(np.pi * 10 * (a - 0.5)) + 2


def const_start_cond(a):
    return a * 0 + 1


# choose simulation-parameters
num_grid_points = 10000
domain_start = 0.0
domain_size = 1.0
dx = domain_size / num_grid_points

time_factor = 1.0
dt = 16.0 / (1 * num_grid_points)

params = {
    'num_grid_points': num_grid_points,
    'domain_size': 1000.0,
    'dt': dt,
    'sampling_rate': time_factor * 2
}

# setup the state
axes = np.tile(np.linspace(0, params['domain_size'], num_grid_points + 1)[:-1], (2, 1))  # setup the axes
axes[0] = axes[0] + 0.5 * dx  # offset the lnrho-axis
state = utils.State(2, num_grid_points, axes, [("x", "rho"), ("x", "u_x")])  # create the state

# choose starting condition
data = state.get_state_vars()  # get the underlying numpy-array

# specify rho-values
data[0] = starting_conditions.GaussianBump(0.5 * params['domain_size'], 0.001).start_cond(axes[0]) * 0.1 + 1
# data[0] = sinc_start_cond(axes[0])
# data[0] = const_start_cond(axes[0])
# data[0] = exp_curve(axes[0])

data[0] = np.log(data[0])  # apply logarithm in order to have ln(rho)-axis

# choose inputs for the time_derivative
time_derivative_input = [True,  # non-linear
                         True,  # viscosity
                         -10]  # gravity

run_utils.run_visual_without_solution(params, integrators.RungeKutta.Explicit,
                                      LogTimeDerivative, time_derivative_input,
                                      state, [True, False])
