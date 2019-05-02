import numpy as np

from matplotlib import pyplot as plt

from debug_tools import error_tracking_tools
import operators.laplace
from debug_tools.visualization import WindowManager

window_manager = WindowManager(1, 1)
for op, name in zip([operators.laplace.diff_n2_e2, operators.laplace.diff_n2_e4], ["Laplace Order: 2", "Laplace Order: 4"]):
    print(name)
    tracker = error_tracking_tools.ErrorTracker(num_points=None, x_label="Number of Points", mode="l_inf")
    for resolution_exp in range(20):
        tracker.num_points = num_points = 2 * (2 ** resolution_exp)
        L = 1
        # shifting axis by irrational number e to avoid 'random' perfect results at low resolutions
        axis = np.linspace(0, L, num_points + 1)[:-1] - np.e
        dx = L * 1.0 / num_points

        factor = np.pi * 2 * 10  # 2 pi to have a periodic signal, other factors to increase
        signal = np.sin(axis * factor)

        calculated_result = op(signal, dx)
        accurate_result = - (factor ** 2) * np.sin(axis * factor)

        tracker.add_entry(num_points, accurate_result, calculated_result)
        print(str(num_points) + "\t" + str(tracker.abs_error[-1]) + "\t" + str(1 / dx ** 2))

    window_manager.display_error(1, tracker, double_log=True, line_name=name, clear_axis=False)

window_manager.draw_loglog_oder_line(1, 10, 1e-2, 1, 1)
window_manager.draw_loglog_oder_line(1, 10, 1e-2, 1, 2)
window_manager.draw_loglog_oder_line(1, 10, 1e-2, 1, 4)
plt.show()
