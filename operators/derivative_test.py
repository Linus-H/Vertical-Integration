import numpy as np

from matplotlib import pyplot as plt

from debug_tools import error_tracking_tools
import operators.derivative
from debug_tools.visualization import WindowManager

window_manager = WindowManager(1, 1)

for op, name in zip([operators.derivative.diff_backward_n1_e1, operators.derivative.diff_n1_e2, operators.derivative.diff_n1_e4],
                    ["Derivative Order: 1", "Derivative Order: 2", "Derivative Order: 4"]):
    print(name)
    tracker = error_tracking_tools.ErrorTracker(num_points=None, x_label="Number of Samples", mode="l_inf")
    for resolution_exp in range(20):
        num_points = 5 * (2 ** resolution_exp)
        L = 1

        # shifting axis by irrational number e to avoid 'random' perfect results at low resolutions
        axis = np.linspace(0, L, num_points + 1)[:-1] - np.e
        dx = L * 1.0 / num_points

        factor = np.pi * 2 * 10  # 2 pi to have a periodic signal, other factors to increase the frequency
        signal = np.sin(axis * factor)

        calculated_result = op(signal, dx)
        accurate_result = factor * np.cos(axis * factor)

        tracker.add_entry(num_points, accurate_result, calculated_result)
        print(str(num_points) + "\t" + str(tracker.abs_error[-1]) + "\t" + str(1 / dx ** 2))

    window_manager.display_error(1, tracker, double_log=True, line_name=name, clear_axis=False)

window_manager.draw_loglog_oder_line(1, 10, 1e-6, 1, 1)
window_manager.draw_loglog_oder_line(1, 10, 1e-6, 1, 2)
window_manager.draw_loglog_oder_line(1, 10, 1e-6, 1, 4)
plt.show()
