from cases.wave_equation import run
from integrators.RungeKutta import Explicit
from cases.wave_equation.wrap_around.laplace.derivative import TimeDerivative
from cases.wave_equation.wrap_around.laplace.solution import CaseSolution
from starting_conditions import GaussianBump

params = {}
params['num_grid_points'] = 1000
params['c'] = 2

special_input = []
special_input.append(GaussianBump(200).start_cond)
special_input.append(GaussianBump(200).derivative)

run.run_with_solution(Explicit, TimeDerivative, params, CaseSolution, special_input)
