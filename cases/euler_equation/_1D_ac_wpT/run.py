from cases import run_utils
# from cases.euler_equation._1D_ac_wpT.derivative import LogTimeDerivativeCP
from cases.euler_equation._1D_ac_wpT.derivative import LogTimeDerivativeLorenz
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


# "L" for Lorenz grid, "CP" for charney phillip grid
mode = "L"  # "CP"

# choose simulation-parameters
num_grid_points = 1000
domain_size = 1.0
dx = domain_size / num_grid_points

time_factor = 1.0
dt = 2.5e-3  # 1.0 / (1000 * num_grid_points)
g = 9.81

params = {
    'num_grid_points': num_grid_points,
    'domain_size': domain_size,
    'dt': dt,
    'sampling_rate': time_factor * 1000
}

# setup the state
axes = np.tile(np.linspace(0, params['domain_size'], num_grid_points + 1)[:-1], (3, 1))  # setup the axes
axes[0] = axes[0] + 0.5 * dx  # offset the lnp-axis
if mode is "L":
    axes[1] = axes[1] + 0.5 * dx  # offset the T-axis
state = utils.State(3, num_grid_points, axes, [("s", "p"), ("s", "T"), ("s", "w")])  # create the state

# choose starting condition
data = state.get_state_vars()  # get the underlying numpy-array

# specify p-values
f = lambda s: 11.39e4 * s + (s - 1) * s
df_ds = lambda s: 11.39e4 + (2 * s - 1)

data[0] = f(axes[1]) + 0.00011
T = 273
z = np.flip(286.99 / g * np.cumsum(np.flip((T * df_ds(axes[0])/data[0]))) / num_grid_points)

data[0] += starting_conditions.GaussianBump(0.5 * params['domain_size'], 0.0000001).get_start_condition(z)
# data[0] = sinc_start_cond(axes[0])
# data[0] = const_start_cond(axes[0])
# data[0] = exp_curve(axes[0])


# data[0] = data[0][0]
data[0] = np.log(data[0])  # apply logarithm in order to have ln(rho)-axis

# specify T-values
data[1] = T  # + starting_conditions.GaussianBump(0.5*params['domain_size'],0.000001).get_start_condition(axes[0])*50

# choose inputs for the time_derivative

time_derivative_input = [axes[2],  # s-axis
                         df_ds,  # dpi/ds
                         g]  # gravity

def converter(ax):
    R = 286.99
    lnp = data[0]
    T = data[1]
    z = (R / g * np.cumsum(np.flip((T * np.exp(-lnp) * df_ds(axes[0])))) / num_grid_points)
    z = np.flip(z)
    # z = np.concatenate([z,])
    # print(z)
    return z

run_utils.run_visual_without_solution(params, integrators.RungeKutta.Explicit,
                                      LogTimeDerivativeLorenz if mode is "L" else None,  # LogTimeDerivativeCP
                                      time_derivative_input,
                                      state, [(converter, lambda x: np.exp(x) - f(axes[1]) - 0.00011),
                                              (converter, lambda x: x), (converter, lambda x: x)])
