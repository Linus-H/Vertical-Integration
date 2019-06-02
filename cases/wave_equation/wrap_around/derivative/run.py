from cases import run_utils
from integrators.RungeKutta import Explicit
from cases.wave_equation.wrap_around.derivative.derivative import TimeDerivative
from cases.wave_equation.wrap_around.derivative.solution import CaseSolution
from starting_conditions import GaussianBump, Sinc

c = 4.0
num_grid_points = 100
dt = 1 / (16 * num_grid_points * c)

params = {
    'num_grid_points': num_grid_points,
    'domain_size': 1.0,
    'dt': dt,
    'sampling_rate': 50
}

time_derivative_input = [c]

case_sol_input = [c, GaussianBump(params['domain_size'] * 0.5, 100).start_cond]

run_utils.run_visual_with_solution(params, Explicit,
                                   TimeDerivative, time_derivative_input,
                                   CaseSolution, case_sol_input)
