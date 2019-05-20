import numpy as np

from cases.wave_equation import run
from integrators.RungeKutta import Explicit
from cases.wave_equation.wrap_around.derivative.derivative import TimeDerivative
from starting_conditions import GaussianBump

params = {}
params['num_grid_points'] = 1000
params['c'] = 2

starting_cond = GaussianBump(200.0)
axes = np.tile(np.linspace(0, 1, num_grid_points + 1)[:-1], (2, 1))
state = State(2, num_grid_points, axes, [("u", "x"), ("v", "x")])
state_vars = state.get_state_vars()
state_vars[0] = starting_cond.start_cond(axes[0])
state_vars[1] = starting_cond.derivative(axes[1])

run.run_without_solution(Explicit, TimeDerivative, params, state)
