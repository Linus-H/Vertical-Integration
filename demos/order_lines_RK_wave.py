import time

import numpy as np

import utils
from cases.wave_equation.periodic.derivative.derivative import PeriodicWaveTimeDerivative
from cases.wave_equation.periodic.derivative.solution import CaseSolution
from debug_tools.error_tracking_tools import ErrorTracker
from debug_tools.visualization import WindowManager
from integrators.Euler import Explicit as RK1
from integrators.Heun import Explicit as RK2
from integrators.RungeKutta import Explicit as RK4
from cases import run_utils
from matplotlib import pyplot as plt

# Total simulation time
from starting_conditions import GaussianBump

sim_time = 0.5
# time step size
# dt = 0.1
# wave speed
c = 1.0

vis = WindowManager(1, 1)

for integrator, name, line_type in zip([RK1, RK2, RK4],
                                       ["RK1", "RK2", "RK4"],
                                       ["-", "--", ":"]):
    print(name)
    tracker = ErrorTracker(num_points=None, x_label="step_size", norm="l_inf")

    dt0 = sim_time

    for res_exp in range(20):
        print(res_exp)
        dt = dt0 / (1.5 ** res_exp)
        # Initialize and compute time integration

        num_grid_points = 100

        params = {
            'num_grid_points': num_grid_points,
            'domain_size': 10.0,
            'dt': dt,
            'end_time': sim_time
        }

        time_derivative_input = [c]  # wave speed

        case_sol_input = [c,  # wave speed
                          GaussianBump(params['domain_size'] * 0.5, 2).get_start_condition]  # start condition

        error_trackers = run_utils.gen_test_data(params, integrator,
                                                 PeriodicWaveTimeDerivative, time_derivative_input,
                                                 CaseSolution, case_sol_input)

        # Get error
        error = error_trackers[0].tot_error + error_trackers[1].tot_error

        # cache error
        tracker.add_entry(dt, 0, error)

    vis.display_error(1, tracker, double_log=True, line_name=name, linestyle=line_type)
vis.draw_loglog_oder_line(1, 1e-3, 1e-3, 1.0, -1)
vis.draw_loglog_oder_line(1, 1e-3, np.sqrt(10)*1e-8, 1.0, -2)
vis.draw_loglog_oder_line(1, 1e-2, 1e-10, 1.0, -4)
plt.grid()
plt.pause(15)
plt.savefig(utils.data_path + "wave_order_line.pdf")
plt.show()
"""print("u")
for i in range(len(resolutions) - 1):
    # comparing the previous error with current one
    print(errors_u.abs_error[i] / errors_u.abs_error[i + 1])
print("v")
for i in range(len(resolutions) - 1):
    # comparing the previous error with current one
    print(errors_v.abs_error[i] / errors_v.abs_error[i + 1])
plt.show()"""
