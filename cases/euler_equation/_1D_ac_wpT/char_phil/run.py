import time
from functools import partial

from cases import run_utils
from cases.euler_equation._1D_ac_wpT.char_phil.derivative import LogTimeDerivativeCP
from cases.euler_equation._1D_ac_wpT.lorenz.derivative import LogTimeDerivativeLorenz
import utils
import cases.euler_equation.consts as const
from cases.euler_equation._1D_ac_wpT.char_phil.functions import *

import integrators.RungeKutta

import starting_conditions
import numpy as np

# choose simulation-parameters
num_grid_points = 1000
domain_size = 1.0
ds = domain_size / num_grid_points

time_factor = 1.0
dt = 2.5e-3 / time_factor  # 1.0 / (1000 * num_grid_points)

params = {
    'num_grid_points': num_grid_points,
    'domain_size': domain_size,
    'dt': dt,
    'sampling_rate': int(time_factor * 1000)
}

# setup the state
axes = np.tile(np.linspace(0, params['domain_size'], num_grid_points + 1)[:-1], (3, 1))  # setup the axes
axes[0] = axes[0] + 0.5 * ds  # offset the lnp-axis

state = utils.State(3, num_grid_points, axes, [("z in $m$", "p in $\\frac{N}{m^2}$"), ("z in $m$", "T in $K$"), ("z in $m$", "w in $\\frac{m}{s}$")])  # create the state

# choose starting condition
data = state.get_state_vars()  # get the underlying numpy-array

# specify T-values
data[1] = const.T

# specify p-values
pi = lambda s: const.sea_level_pressure * s  # + (s - 1) * s
dpi_ds = lambda s: const.sea_level_pressure  # + (2 * s - 1) + 0.02
data[0] = pi(axes[0])

z = s_offset_to_z(axes[0], np.log(data[0]), data[1], dpi_ds, num_grid_points)

data[0] += starting_conditions.GaussianBump(0*0.5 * params['domain_size'], 0.0000001).get_start_condition(z)
# data[0] += starting_conditions.GaussianBump(0.5 * np.max(z), 0.0000001).get_start_condition(z) * 0.01
#data[0] += starting_conditions.GaussianBump(35000, 0.0000001).get_start_condition(z) * 0.1
# data[0] = data[0][0]
data[0] = np.log(data[0])  # apply logarithm in order to have ln(rho)-axis
# data[0][0] = -np.inf

# choose inputs for the time_derivative
time_derivative_input = [axes[2],  # s-axis without offset
                         dpi_ds]  # dpi/ds

A, B, C, D, iteration_cnt = 0, 0, 0, 0, 0
energy_arr = np.zeros((5, 97))

def state_handler(state, t):
    global A, B, C, D, iteration_cnt
    state_vars = state.get_state_vars()
    lnp = state_vars[0]
    T = state_vars[1]
    w = state_vars[2]
    z = s_offset_to_z(axes[0], lnp, T, dpi_ds, num_grid_points)
    A_next, B_next, C_next, D_next = calc_energy(w, T, z, dpi_ds, num_grid_points, axes[0])
    a, b, c, d = (A_next - A, B_next - B, C_next - C, D_next - D)
    #print("{}\t{}\t{}\t{}\t{}".format(t, a, b, c, d))
    if A == 0:
        A, B, C, D = A_next, B_next, C_next, D_next

    max_ind = np.argmax(np.exp(lnp) - pi(axes[0]))
    print("{}\t{}".format(t, z[max_ind]))

    energy_arr[0][iteration_cnt] = t
    energy_arr[1][iteration_cnt] = a
    energy_arr[2][iteration_cnt] = b
    energy_arr[3][iteration_cnt] = c
    energy_arr[4][iteration_cnt] = d
    iteration_cnt+=1
    if t >=240:
        np.save(utils.data_path+"cp_energy.npy", energy_arr)
        np.save(utils.data_path+"cp_state.npy", state_vars)
        np.save(utils.data_path+"cp_axes.npy", axes)
        #time.sleep(2000)


run_utils.run_visual_without_solution(
    params,
    integrators.RungeKutta.Explicit,
    LogTimeDerivativeCP, time_derivative_input,
    state,
    [(partial(s_offset_to_z, lnp=data[0], T=data[1], dpi_ds=dpi_ds, num_grid_points=num_grid_points),
      lambda x: np.exp(x) - pi(axes[0])),  # lnp
     (partial(s_to_z, lnp=data[0], T=data[1], dpi_ds=dpi_ds, num_grid_points=num_grid_points), lambda x: x),  # T
     (partial(s_to_z, lnp=data[0], T=data[1], dpi_ds=dpi_ds, num_grid_points=num_grid_points), lambda x: x)],  # w
    state_handler)
#from matplotlib import pyplot as plt
#plt.savefig(utils.data_path + "cp_fig.pdf")