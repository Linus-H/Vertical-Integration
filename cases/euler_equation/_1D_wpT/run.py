from cases import run_utils
from cases.euler_equation._1D_wpT.derivative import LogTimeDerivativeCP
from cases.euler_equation._1D_wpT.derivative import LogTimeDerivativeLorenz
import utils

import cases.euler_equation.consts as const
import integrators.RungeKutta
import integrators.Heun

import starting_conditions
import numpy as np


def exp_curve(a):
    return np.exp(-const.g * a / (const.R * const.T))


def sinc_start_cond(a):
    return np.sinc(np.pi * 10 * (a - 0.5)) + 2


def const_start_cond(a):
    return a * 0 + 1


# "L" for Lorenz grid, "CP" for charney phillip grid
mode = "CP"  # "CP"

# choose simulation-parameters
num_grid_points = 1000
domain_size = 10000.0
dx = domain_size / num_grid_points

time_factor = 1.0
dt = 1e-3  # 1.0 / (1000 * num_grid_points)

params = {
    'num_grid_points': num_grid_points,
    'domain_size': domain_size,
    'dt': dt,
    'sampling_rate': time_factor * 800
}

# setup the state
axes = np.tile(np.linspace(0, params['domain_size'], num_grid_points + 1)[:-1], (3, 1))  # setup the axes
axes[0] = axes[0] + 0.5 * dx  # offset the lnp-axis
if mode is "L":
    axes[1] = axes[1] + 0.5 * dx  # offset the Tp-axis
state = utils.State(3, num_grid_points, axes, [("z", "p"), ("z", "T"), ("z", "w")])  # create the state

# choose starting condition
data = state.get_state_vars()  # get the underlying numpy-array

# specify p-values
data[0] = starting_conditions.GaussianBump(0.5 * params['domain_size'], 0.000001).get_start_condition(axes[0]) * 0.1 + 1
# data[0] = sinc_start_cond(axes[0])
# data[0] = const_start_cond(axes[0])
# data[0] = exp_curve(axes[0])

data[0] = np.log(data[0])  # apply logarithm in order to have ln(rho)-axis

# specify T-values
data[
    1] = const.T  # + starting_conditions.GaussianBump(0.5*params['domain_size'],0.000001).get_start_condition(axes[0])*50

# choose inputs for the time_derivative
time_derivative_input = [0]  # gravity

run_utils.run_visual_without_solution(params, integrators.RungeKutta.Explicit,
                                      LogTimeDerivativeLorenz if mode is "L" else LogTimeDerivativeCP,
                                      time_derivative_input,
                                      state, [(lambda x: x, np.exp), None, None])
