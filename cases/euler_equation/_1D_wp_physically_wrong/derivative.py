import numpy as np
from operators import derivative, average, laplace
from utils import TimeDerivative

import cases.euler_equation.consts as const


class EulerTimeDerivative(TimeDerivative):
    def __init__(self, delta_z, non_linear=True):
        self.delta_z = delta_z
        self.non_linear = non_linear

    def __call__(self, z, t):
        rho = z[0]
        rho_top = np.ones_like(rho) * rho[0]
        rho_bottom = np.ones_like(rho) * rho[-1]
        rho = np.concatenate((rho_top, rho, rho_bottom))

        w = z[1]
        w_flip = np.zeros_like(w)
        w = np.concatenate((w_flip, w, w_flip))

        drho_dt = - derivative.diff_n1_e4(rho * average.avg_forward_e1(w), self.delta_z)

        if self.non_linear:
            pass

        rho_on_u_grid = average.avg_backward_e1(rho)

        p = rho_on_u_grid * const.R * const.T

        dw_dt = - w * derivative.diff_n1_e4(w, self.delta_z) \
                - derivative.diff_n1_e4(p, self.delta_z) / rho_on_u_grid  # + 3.0

        if self.non_linear:
            pass  # dw_dt +=

        drho_dt = np.split(drho_dt, 3)[1]
        drho_dt[-1] = drho_dt[-2]

        dw_dt = np.split(dw_dt, 3)[1]
        dw_dt[0] = dw_dt[-1] = 0  # wind at top and bottom stays constant at zero

        # generate output
        dz = np.stack((drho_dt, dw_dt), axis=-1)
        dz = dz.transpose()
        return dz

    def __str__(self):
        return "euler_wp_time_derivative"


class LogTimeDerivative(TimeDerivative):
    def __init__(self, delta_z, non_linear=True, viscosity=False):
        self.delta_z = delta_z
        self.non_linear = non_linear
        self.viscosity = viscosity

    def __call__(self, z, t):
        lnrho = z[0]
        w = z[1]

        dlnrho_dt = - derivative.diff_forward_n1_e1(w, self.delta_z)

        if self.non_linear:
            dlnrho_dt += - average.avg_forward_e1(w) * derivative.diff_n1_e2(lnrho, self.delta_z)

        dw_dt = - const.T * const.R * derivative.diff_backward_n1_e1(lnrho, self.delta_z) + const.g

        if self.non_linear:
            dw_dt += - w * derivative.diff_n1_e2(w, self.delta_z)
            if self.viscosity:
                dw_dt += 0.05 * np.exp(- average.avg_backward_e1(lnrho)) * laplace.diff_n2_e2(w, self.delta_z)

        dlnrho_dt[-1] = dlnrho_dt[-2]
        dw_dt[0] = dw_dt[-1] = 0  # wind at top and bottom stays constant at zero

        # generate output
        dz = np.stack((dlnrho_dt, dw_dt), axis=-1)
        dz = dz.transpose()
        return dz

    def __str__(self):
        return "log_euler_wp_time_derivative"


class MatrixLogTimeDerivative(TimeDerivative):  # only linear parts
    def __init__(self, delta_z, g_z=0.0):
        self.delta_z = delta_z
        self.g_z = g_z

    def __call__(self, z, t):
        num_vars, length, *_ = z.shape

        A = -np.eye(length)  # this imitates the u and v state vector as a matrix to extract the operation
        B = np.zeros((length, length))  # this is a blank padding matrix to get to a square matrix at the end

        dlnrho_dt = - derivative.diff_forward_n1_e1(A, self.delta_z)

        du_dt = - const.T * const.R * derivative.diff_backward_n1_e1(A, self.delta_z)

        dlnrho_dt[-1] = dlnrho_dt[-2]
        du_dt[0] = du_dt[-1] = 0  # wind at top and bottom stays constant at zero

        # generate output
        dlnrho_dt = np.concatenate((B, dlnrho_dt), axis=1)
        du_dt = np.concatenate((du_dt, B), axis=1)

        dz = np.stack((dlnrho_dt, du_dt), axis=0)

        return dz

    def __str__(self):
        return "log_euler_wp_time_derivative_matrix"
