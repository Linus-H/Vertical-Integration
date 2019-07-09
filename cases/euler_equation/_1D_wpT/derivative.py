import numpy as np
from operators import derivative, average, laplace


class LogTimeDerivativeCP:
    def __init__(self, delta_z, g_z=0.0):
        self.delta_z = delta_z
        self.g_z = g_z

    def __call__(self, z, t):
        lnp = z[0]
        T = z[1]
        w = z[2]

        # TODO: check constants
        R = 286.99  # 8.314  # perfect gas constant
        C_p = 1000  # 28.97  # specific heat at constant pressure

        #alpha = (0.024 / C_p) * np.exp(-average.avg_backward_1(lnp))

        dlnp_dt = - (1 / (1 - R / C_p)) * derivative.diff_forward_n1_e1(w, self.delta_z) \
                  - average.avg_forward_1(w) * derivative.diff_n1_e2(lnp, self.delta_z) \
         #         + (alpha * R / C_p) * np.exp(- average.avg_backward_1(lnp)) * laplace.diff_n2_e2(
          #  average.avg_forward_1(T), self.delta_z)

        dT_dt = (R / C_p) * T * average.avg_backward_1(dlnp_dt) \
                - w * derivative.diff_n1_e2(T, self.delta_z) \
           #     + (alpha * R / C_p) * laplace.diff_n2_e2(T, self.delta_z)

        dw_dt = - R * T * derivative.diff_backward_n1_e1(lnp, self.delta_z) + self.g_z \
                - w * derivative.diff_n1_e2(w, self.delta_z) \
                + 1.5e-5 * R * T * np.exp(- average.avg_backward_1(lnp)) * laplace.diff_n2_e2(w, self.delta_z)

        dlnp_dt[-1] = dlnp_dt[-2]
        dw_dt[0] = dw_dt[-1] = 0  # wind at top and bottom stays constant at zero

        # generate output
        dz = np.stack((dlnp_dt, dT_dt, dw_dt), axis=-1)
        dz = dz.transpose()
        return dz

    def __str__(self):
        return "log_euler_wpT_CP"


class LogTimeDerivativeLorenz:
    def __init__(self, delta_z, g_z=0.0):
        self.delta_z = delta_z
        self.g_z = g_z

    def __call__(self, z, t):
        lnp = z[0]
        T = z[1]
        w = z[2]

        # TODO: check constants
        R = 286.99  # perfect gas constant
        C_p = 1000  # specific heat at constant pressure

        dlnp_dt = - (1 / (1 - R / C_p)) * derivative.diff_forward_n1_e1(w, self.delta_z) \
                  - average.avg_forward_1(w) * derivative.diff_n1_e2(lnp, self.delta_z)

        dT_dt = (R / C_p) * T * dlnp_dt - average.avg_forward_1(w) * derivative.diff_n1_e2(T, self.delta_z)

        dw_dt = - R * average.avg_backward_1(T) * derivative.diff_backward_n1_e1(lnp, self.delta_z) + self.g_z \
                - w * derivative.diff_n1_e2(w, self.delta_z)

        dlnp_dt[-1] = dlnp_dt[-2]
        dT_dt[-1] = dT_dt[-2]
        dw_dt[0] = dw_dt[-1] = 0  # wind at top and bottom stays constant at zero

        # generate output
        dz = np.stack((dlnp_dt, dT_dt, dw_dt), axis=-1)
        dz = dz.transpose()
        return dz

    def __str__(self):
        return "log_euler_wpT_lorenz"
