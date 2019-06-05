import numpy as np

import starting_conditions
from cases import run_utils
from cases.wave_equation.wrap_around.derivative.derivative import TimeDerivativeMatrix
from cases.wave_equation.wrap_around.derivative.solution import CaseSolution
from integrators.Euler import Explicit
from integrators.Exponential import Exponential
from utils import State


c = 10.0
num_grid_points = 1000
dt = 1 / (1)

params = {
    'num_grid_points': num_grid_points,
    'domain_size': 10.0,
    'dt': dt,
    'sampling_rate': 1
}

time_derivative_input = [c]

case_sol_input = [c, starting_conditions.GaussianBump(params['domain_size'] * 0.5, 5).start_cond]

run_utils.run_visual_with_solution(params, Exponential,
                                   TimeDerivativeMatrix, time_derivative_input,
                                   CaseSolution, case_sol_input)

"""domain_size = 1.0
num_grid_points = 1000
dx = domain_size / num_grid_points
dt = 0.1
c = 1.0


derivative = TimeDerivativeMatrix(dx, c)

starting_cond = starting_conditions.GaussianBump(domain_size * 0.5, 100)
solution = CaseSolution(num_grid_points, dt, domain_size, c, starting_cond.start_cond)
state = solution.get_initial_state()


integrator = Exponential(state, derivative, 0, 0.1)"""


