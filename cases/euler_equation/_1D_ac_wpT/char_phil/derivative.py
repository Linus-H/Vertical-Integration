import numpy as np
from operators import derivative, average, laplace
from utils import TimeDerivative
import cases.euler_equation.consts as const


class LogTimeDerivativeCP:
    def __init__(self, delta_s, s, dpi_ds):
        self.delta_s = delta_s
        self.dpi_ds = dpi_ds
        self.s = s

    def __call__(self, z, t):
        lnp_offset = z[0]
        T_align = z[1]
        w_align = z[2]

        T_offset = average.avg_backward_e1(T_align)

        dlnp_dt_offset = (const.g * np.exp(lnp_offset) / ((1 - const.R / const.C_p) * (const.R * T_offset))) \
                         * derivative.diff_backward_n1_e1(w_align, self.delta_s) \
                         / self.dpi_ds(self.s + self.delta_s / 2)
        dlnp_dt_offset[0] = 0  # fix pressure at the top
        dlnp_dt_align = average.avg_forward_e1(dlnp_dt_offset)
        dlnp_dt_align[-1] = dlnp_dt_offset[-1]

        dT_dt_align = (const.R / const.C_p) * T_align * dlnp_dt_align
        #dT_dt_align[0] = 0  # fix temperature at the top
        dT_dt_align[0] = dT_dt_align[1] # set temperature above top to temperature below top
        # dT_dt[-1] = 0
        # dT_dt[-1]=dT_dt[-2]

        dw_dt_align = - const.g * (1 - derivative.diff_forward_n1_e1(np.exp(lnp_offset), self.delta_s) / self.dpi_ds(self.s))
        # wind at top and bottom stays constant at zero
        dw_dt_align[0] = 0  # fix wind at top
        dw_dt_align[-1] = 0  # fix wind at bottom

        # generate output
        dz = np.stack((dlnp_dt_offset, dT_dt_align, dw_dt_align), axis=-1)
        dz = dz.transpose()
        return dz

    def __str__(self):
        return "log_euler_wpT_CP"
