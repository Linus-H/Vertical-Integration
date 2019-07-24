from cases import run_utils
from integrators.RungeKutta import Explicit
from cases.wave_equation.dirichlet.laplace.derivative import WaveEquationLaplace
from cases.wave_equation.dirichlet.laplace.solution import StandingWaveFixedEnd

c = 2.0
num_grid_points = 1000
dt = 1 / (16 * num_grid_points * c)

params = {
    'num_grid_points': num_grid_points,
    'domain_size': 1.0,  # TODO: make solution adaptive to domain size, right now it has to stay at 1.0
    'dt': dt,
    'sampling_rate': 1000
}

time_derivative_input = [c]

case_sol_input = [c, [(1, 1.0), (2, 2.0)]]

run_utils.run_visual_with_solution(params, Explicit,
                                   WaveEquationLaplace, time_derivative_input,
                                   StandingWaveFixedEnd, case_sol_input)
