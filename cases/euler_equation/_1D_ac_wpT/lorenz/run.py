import time
from functools import partial
from cases import run_utils
from cases.euler_equation._1D_ac_wpT.lorenz.derivative import LogTimeDerivativeLorenz
import utils
import cases.euler_equation.consts as const
from cases.euler_equation._1D_ac_wpT.lorenz.functions import *

import integrators.RungeKutta
import starting_conditions
import numpy as np

# choose simulation-parameters
num_grid_points = 1000
domain_size = 1.0
dx = domain_size / num_grid_points

time_factor = 1.0
dt = 2.5e-3  # 1.0 / (1000 * num_grid_points)

params = {
    'num_grid_points': num_grid_points,
    'domain_size': domain_size,
    'dt': dt,
    'sampling_rate': time_factor * 1000
}

# setup the state
axes = np.tile(np.linspace(0, params['domain_size'], num_grid_points + 1)[:-1], (3, 1))  # setup the axes
axes[0] = axes[0] + 0.5 * dx  # offset the lnp-axis
axes[1] = axes[1] + 0.5 * dx  # offset the T-axis

state = utils.State(3, num_grid_points, axes, [("z", "p"), ("z", "T"), ("z", "w")])  # create the state

# choose starting condition
data = state.get_state_vars()  # get the underlying numpy-array

# specify T-values
data[1] = const.T

# specify p-values
pi = lambda s: const.sea_level_pressure * s  # + (s - 1) * s
dpi_ds = lambda s: const.sea_level_pressure  # + (2 * s - 1) #+ 0.02
data[0] = pi(axes[0])

z = s_offset_to_z(axes[0], np.log(data[0]), data[1], dpi_ds, num_grid_points)

#data[0] += starting_conditions.GaussianBump(0.5 * params['domain_size'], 0.0000001).get_start_condition(z)
data[0] += starting_conditions.GaussianBump(0.5 * np.max(z), 0.0000001).get_start_condition(z) * 0.1
# data[0] = data[0][0]
data[0] = np.log(data[0])  # apply logarithm in order to have ln(rho)-axis
data[0][0] = -np.inf

# choose inputs for the time_derivative
time_derivative_input = [axes[2],  # s-axis without offset
                         dpi_ds]  # dpi/ds

baseline_energy = 0


def state_handler(state, t):
    global baseline_energy
    print(t)
    state_vars = state.get_state_vars()
    lnp = state_vars[0]
    T = state_vars[1]
    w = state_vars[2]
    z = s_offset_to_z(axes[0], lnp, T, dpi_ds, num_grid_points)
    next_energy = calc_energy(w, T, z, dpi_ds, num_grid_points, axes[0])
    print((next_energy - baseline_energy))
    if baseline_energy == 0:
        baseline_energy = next_energy
    print(calc_mass())


run_utils.run_visual_without_solution(
    params,
    integrators.RungeKutta.Explicit,
    LogTimeDerivativeLorenz, time_derivative_input,
    state,
    [(partial(s_offset_to_z, lnp=data[0], T=data[1], dpi_ds=dpi_ds, num_grid_points=num_grid_points),
      lambda x: np.exp(x)),# - pi(axes[0])),  # lnp
     (partial(s_offset_to_z, lnp=data[0], T=data[1], dpi_ds=dpi_ds, num_grid_points=num_grid_points), lambda x: x),  # T
     (partial(s_to_z, lnp=data[0], T=data[1], dpi_ds=dpi_ds, num_grid_points=num_grid_points), lambda x: x)],  # w
    state_handler)
