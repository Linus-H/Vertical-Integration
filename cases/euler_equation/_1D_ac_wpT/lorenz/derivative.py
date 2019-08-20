import numpy as np
from operators import derivative, average, laplace
from utils import TimeDerivative
import cases.euler_equation.consts as const


class LogTimeDerivativeLorenz(TimeDerivative):
    def __init__(self, delta_s, s, dpi_ds):
        self.delta_s = delta_s
        self.dpi_ds = dpi_ds
        self.s = s

    def __call__(self, z, t):
        lnp = z[0]
        T = z[1]
        w = z[2]

        dlnp_dt = (const.g * np.exp(lnp) / ((1 - const.R / const.C_p) * (const.R * T))) \
                  * derivative.diff_s_offset_n1_e2(w, self.delta_s) \
                  / self.dpi_ds(self.s + self.delta_s / 2)

        dT_dt = (const.R / const.C_p) * T * dlnp_dt

        dw_dt = - const.g * (1 - derivative.diff_s_align_n1_e2(np.exp(lnp), self.delta_s) / self.dpi_ds(self.s))

        dlnp_dt[0] = 0  # fix pressure to 0 (ln0=-inf) above atmosphere
        dT_dt[0] = dT_dt[1]  # fix Temperature above atmosphere

        # wind at top and bottom stays constant at zero
        dw_dt[0] = 0  # top
        dw_dt[-1] = 0  # bottom

        # generate output
        dz = np.stack((dlnp_dt, dT_dt, dw_dt), axis=-1)
        dz = dz.transpose()
        return dz

    def __str__(self):
        return "log_euler_ac_wpT_lorenz_time_derivative"
