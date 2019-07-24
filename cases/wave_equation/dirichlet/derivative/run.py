import numpy as np

import starting_conditions
from cases import run_utils
from integrators.RungeKutta import Explicit
from cases.wave_equation.dirichlet.derivative.derivative import WaveEquationDerivative
from cases.wave_equation.dirichlet.derivative.solution import StandingWaveFixedEnd
from utils import State

c = 2.0
num_grid_points = 1000
dt = 1 / (16 * num_grid_points * c)

params = {
    'num_grid_points': num_grid_points,
    'domain_size': 1.0,
    'dt': dt,
    'sampling_rate': 100
}

time_derivative_input = [c]

# case_sol_input = [c, [(1, 1.0), (2, 2.0)]]
axes = np.tile(np.linspace(0, params['domain_size'], num_grid_points + 1)[:-1], (2, 1))  # setup the axes
state = State(2, num_grid_points, axes, [("x", "u"), ("x", "v")])

state_vars = state.get_state_vars()
starting_cond = starting_conditions.GaussianBump(params['domain_size'] * 0.5, 50)

#state_vars[0] = starting_cond.get_start_condition(axes[0])
state_vars[0] = np.sin(axes[0] * 2 * np.pi / params['domain_size'])

run_utils.run_visual_without_solution(params, Explicit,
                                      WaveEquationDerivative, time_derivative_input,
                                      state)