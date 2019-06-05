from cases import run_utils
from integrators.RungeKutta import Explicit
from cases.wave_equation.periodic.staggered_grid_derivative.derivative import TimeDerivative
from cases.wave_equation.periodic.staggered_grid_derivative.solution import CaseSolution
from starting_conditions import GaussianBump

c = 2.0
num_grid_points = 1000
dt = 1 / (16 * num_grid_points * c)

params = {
    'num_grid_points': num_grid_points,
    'domain_size': 3.0,
    'dt': dt,
    'sampling_rate': 1000
}

time_derivative_input = [c]

case_sol_input = [c, GaussianBump(params['domain_size'] * 0.5, 200).get_start_condition]

run_utils.run_visual_with_solution(params, Explicit,
                                   TimeDerivative, time_derivative_input,
                                   CaseSolution, case_sol_input)
