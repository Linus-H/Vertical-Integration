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
max_time_exp, max_res_factor = 11, 40

for integrator, name, line_type in zip([RK4],
                                       ["RK4"],
                                       ["-"]):
    print(name)
    arr = np.zeros((max_time_exp, max_res_factor))
    # tracker = ErrorTracker(num_points=None, x_label="step_size", norm="l_inf")

    dt0 = sim_time / 10

    for res_exp in range(max_time_exp):
        print(res_exp)
        # dt = dt0 / (1.5 ** res_exp)
        dt = 1e-3 + res_exp * (3e-2 - 1e-3) / 10
        for res_factor in range(max_res_factor):
            print(res_factor)
            # Initialize and compute time integration
            num_grid_points = 100 * (1 + res_factor)

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
            arr[res_exp][res_factor] = np.log(error)
    # arr[8][0] = 10000
    np.save(utils.data_path + "heatmap.npy", arr)
    plt.imshow(arr, origin="lower", extent=[100, 100 * (max_res_factor), 1e-3, 3e-2], aspect="auto", cmap="seismic",
               interpolation="gaussian", vmin=-20, vmax=20)

    # vis.draw_loglog_oder_line(1, 1e-2, 1e-2, 0.5, -1)
    # vis.draw_loglog_oder_line(1, 2.5 * 1e-2, 1e-3, 0.5, -2)
    # vis.draw_loglog_oder_line(1, np.sqrt(10) * 1e-1, 1e-3, 0.25, -4)
    # plt.grid()
    plt.xlabel("number of gridpoints")
    plt.ylabel("time step size")
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
