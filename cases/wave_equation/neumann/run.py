import numpy as np

from cases import run_utils
from integrators.RungeKutta import Explicit
from cases.wave_equation.periodic.derivative.derivative import TimeDerivative
from starting_conditions import GaussianBump
from utils import State

c = 2.0
num_grid_points = 1000
dt = 1 / (16 * num_grid_points * c)

params = {
    'num_grid_points': num_grid_points,
    'domain_size': 1.0,
    'dt': dt,
    'sampling_rate': 1000
}

time_derivative_input = [c]

axes = np.tile(np.linspace(0, params['domain_size'], num_grid_points + 1)[:-1], (2, 1))  # setup axes
state = State(2, num_grid_points, axes, [("u", "x"), ("v", "x")])  # create state

state_vars = state.get_state_vars()  # get underlying numpy-array

starting_cond = GaussianBump(params['domain_size'] * 0.5, 200.0)  # choose a starting-condition

state_vars[0] = starting_cond.start_cond(axes[0])  # apply the starting condition variables
# state_vars[1] = starting_cond.derivative(axes[1])

run_utils.run_visual_without_solution(params, Explicit,
                                      TimeDerivative, time_derivative_input,
                                      state)
