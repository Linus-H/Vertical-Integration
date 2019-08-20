import numpy as np

from cases.debug_case.derivative import DebugTimeDerivative
from cases.debug_case.solution import CaseSolution
from debug_tools.error_tracking_tools import ErrorTracker
from debug_tools.visualization import WindowManager
from integrators.Euler import Explicit as RK1
from integrators.Heun import Explicit as RK2
from integrators.RungeKutta import Explicit as RK4
from cases import run_utils
from matplotlib import pyplot as plt

# Total simulation time
sim_time = 0.1
# time step size
# dt = 0.1
# wave speed
c = 10.0

vis = WindowManager(1, 1)

for integrator, name, line_type in zip([RK1, RK2, RK4],
                            ["RK1", "RK2", "RK4"],
                            ["-","--",":"]):
    print(name)
    tracker = ErrorTracker(num_points=None, x_label="step_size", norm="l_inf")

    dt0 = 1

    for res_exp in range(30):
        dt = dt0 / (1.5**res_exp)
        # Initialize and compute time integration

        num_grid_points = 1

        params = {
            'num_grid_points': num_grid_points,
            'domain_size': 10.0,
            'dt': dt,
            'end_time': sim_time
        }

        case_num = 1
        time_derivative_input = [case_num]  # case num

        case_sol_input = [0,  # t0
                          1,  # start_value
                          case_num]  # case num

        error_trackers = run_utils.gen_test_data(params, integrator,
                                                 DebugTimeDerivative, time_derivative_input,
                                                 CaseSolution, case_sol_input)

        # Get error
        error = error_trackers[0].tot_error

        # cache error
        tracker.add_entry(dt, 0, error)

    vis.display_error(1, tracker, double_log=True, line_name=name,linestyle=line_type)
vis.draw_loglog_oder_line(1, 1e-4, 1e-3, 1.0, -1)
vis.draw_loglog_oder_line(1, 1e-4, 1e-7, 1.0, -2)
vis.draw_loglog_oder_line(1, 1e-2, 1e-11, 1.0, -4)
plt.grid()
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
