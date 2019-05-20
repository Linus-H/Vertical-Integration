from cases.wave_equation import run
from integrators.RungeKutta import Explicit
from cases.wave_equation.fixed_end.derivative import TimeDerivativeLaplace
from cases.wave_equation.fixed_end.solution import StandingWaveFixedEnd
from starting_conditions import GaussianBump

params = {}
params['num_grid_points'] = 1000
params['c'] = 2

special_input = []
special_input.append([(1, 1.0), (2, 2.0)])

run.run_with_solution(Explicit, TimeDerivativeLaplace, params, StandingWaveFixedEnd, special_input)
