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
        R = 8.314  # perfect gas constant
        C_p = 1  # specific heat at constant pressure

        dlnp_dt = - (1 / (1 - R / C_p)) * derivative.diff_forward_n1_e1(w, self.delta_z)

        dT_dt = (R / C_p) * T * average.avg_backward_1(dlnp_dt)

        dw_dt = - R * T * derivative.diff_backward_n1_e1(lnp, self.delta_z) + self.g_z

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
        R = 8.314  # perfect gas constant
        C_p = 1  # specific heat at constant pressure

        dlnp_dt = - (1 / (1 - R / C_p)) * derivative.diff_forward_n1_e1(w, self.delta_z)

        dT_dt = (R / C_p) * T * dlnp_dt

        dw_dt = - R * average.avg_backward_1(T) * derivative.diff_backward_n1_e1(lnp, self.delta_z) + self.g_z

        dlnp_dt[-1] = dlnp_dt[-2]
        dT_dt[-1] = dT_dt[-2]
        dw_dt[0] = dw_dt[-1] = 0  # wind at top and bottom stays constant at zero

        # generate output
        dz = np.stack((dlnp_dt, dT_dt, dw_dt), axis=-1)
        dz = dz.transpose()
        return dz

    def __str__(self):
        return "log_euler_wpT_lorenz"
