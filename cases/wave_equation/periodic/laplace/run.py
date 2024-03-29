from cases import run_utils
from integrators.RungeKutta import Explicit
from cases.wave_equation.periodic.laplace.derivative import PeriodicWaveLaplace
from cases.wave_equation.periodic.laplace.solution import CaseSolution
from starting_conditions import GaussianBump

c = 4.0
num_grid_points = 1000
dt = 1 / (16 * num_grid_points * c)

params = {
    'num_grid_points': num_grid_points,
    'domain_size': 3.0,
    'dt': dt,
    'sampling_rate': 4000
}

time_derivative_input = [c]

start_cond = GaussianBump(params['domain_size'] * 0.5, 200)

case_sol_input = [c,
                  start_cond.get_start_condition,
                  start_cond.get_derivative]

run_utils.run_visual_with_solution(params, Explicit,
                                   PeriodicWaveLaplace, time_derivative_input,
                                   CaseSolution, case_sol_input)
