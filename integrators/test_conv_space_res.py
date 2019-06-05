import starting_conditions
from cases import run_utils
from cases.wave_equation.periodic.derivative.derivative import TimeDerivativeMatrix
from cases.wave_equation.periodic.derivative.solution import CaseSolution
import starting_conditions
from debug_tools.error_tracking_tools import ErrorTracker
from debug_tools.visualization import WindowManager
from integrators.Exponential import Exponential
from cases import run_utils
from matplotlib import pyplot as plt

# Total simulation time
sim_time = 0.1
# time step size
dt = 0.1
# wave speed
c = 10.0

errors_u = ErrorTracker(1, "resolution")
errors_v = ErrorTracker(1, "resolution")
resolutions = [16, 32, 64, 128, 256]

for resolution in resolutions:
    # Initialize and compute time integration

    num_grid_points = resolution

    params = {
        'num_grid_points': num_grid_points,
        'domain_size': 10.0,
        'dt': dt,
        'end_time': sim_time
    }

    time_derivative_input = [c]

    case_sol_input = [c, starting_conditions.GaussianBump(params['domain_size'] * 0.5, 5).get_start_condition]

    error_trackers = run_utils.gen_test_data(params, Exponential,
                                             TimeDerivativeMatrix, time_derivative_input,
                                             CaseSolution, case_sol_input)

    # Get error
    error_u = error_trackers[0].tot_error
    error_v = error_trackers[1].tot_error

    # cache error
    errors_u.add_entry(resolution, 0, error_u)
    errors_v.add_entry(resolution, 0, error_v)

vis = WindowManager(1, 1)
vis.display_error(1, errors_u, double_log=True, line_name="u_error")
vis.display_error(1, errors_u, double_log=True, line_name="v_error")
vis.draw_loglog_oder_line(1, 1e2, 1e-2, 0.5, 2)
vis.draw_loglog_oder_line(1, 1e2, 1e-2, 0.5, 3)
vis.draw_loglog_oder_line(1, 1e2, 1e-2, 0.5, 4)

print("u")
for i in range(len(resolutions) - 1):
    # comparing the previous error with current one
    print(errors_u.abs_error[i] / errors_u.abs_error[i + 1])
print("v")
for i in range(len(resolutions) - 1):
    # comparing the previous error with current one
    print(errors_v.abs_error[i] / errors_v.abs_error[i + 1])
