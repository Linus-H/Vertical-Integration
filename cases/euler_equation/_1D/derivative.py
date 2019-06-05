import numpy as np
from operators import derivative, average, laplace


class TimeDerivative:
    def __init__(self, delta_z, non_linear=True):
        self.delta_z = delta_z
        self.non_linear = non_linear

    def __call__(self, z, t):
        rho = z[0]
        rho_top = np.ones_like(rho) * rho[0]
        rho_bottom = np.ones_like(rho) * rho[-1]
        rho = np.concatenate((rho_top, rho, rho_bottom))

        u_z = z[1]
        u_z_flip = np.zeros_like(u_z)
        u_z = np.concatenate((u_z_flip, u_z, u_z_flip))

        drho_dt = - derivative.diff_n1_e4(rho * average.avg_forward_1(u_z), self.delta_z)

        if self.non_linear:
            pass

        T = 273.15
        R = 8.314

        rho_on_u_grid = average.avg_backward_1(rho)

        p = rho_on_u_grid * R * T

        du_dt = - u_z * derivative.diff_n1_e4(u_z, self.delta_z) \
                - derivative.diff_n1_e4(p, self.delta_z) / rho_on_u_grid  # + 3.0

        if self.non_linear:
            pass  # du_dt +=

        drho_dt = np.split(drho_dt, 3)[1]
        drho_dt[-1] = drho_dt[-2]

        du_dt = np.split(du_dt, 3)[1]
        du_dt[0] = du_dt[-1] = 0  # wind at top and bottom stays constant at zero

        # generate output
        dz = np.stack((drho_dt, du_dt), axis=-1)
        dz = dz.transpose()
        return dz


class LogTimeDerivative:
    def __init__(self, delta_z, non_linear=True, viscosity=False, g_z=0.0):
        self.delta_z = delta_z
        self.non_linear = non_linear
        self.viscosity = viscosity
        self.g_z = g_z

    def __call__(self, z, t):
        lnrho = z[0]
        u_z = z[1]

        dlnrho_dt = - derivative.diff_forward_n1_e1(u_z, self.delta_z)

        if self.non_linear:
            dlnrho_dt += - average.avg_forward_1(u_z) * derivative.diff_n1_e2(lnrho, self.delta_z)

        T = 273.15
        R = 8.314

        du_dt = - T * R * derivative.diff_backward_n1_e1(lnrho, self.delta_z) + self.g_z

        if self.non_linear:
            du_dt += - u_z * derivative.diff_n1_e2(u_z, self.delta_z)
            if self.viscosity:
                du_dt += 0.05 * np.exp(- average.avg_backward_1(lnrho)) * laplace.diff_n2_e2(u_z, self.delta_z)

        dlnrho_dt[-1] = dlnrho_dt[-2]
        du_dt[0] = du_dt[-1] = 0  # wind at top and bottom stays constant at zero

        # generate output
        dz = np.stack((dlnrho_dt, du_dt), axis=-1)
        dz = dz.transpose()
        return dz

    def __str__(self):
        return "log_euler"


class MatrixLogTimeDerivative:  # only linear parts
    def __init__(self, delta_z, g_z=0.0):
        self.delta_z = delta_z
        self.g_z = g_z

    def __call__(self, z, t):
        num_vars, length, *_ = z.shape

        A = -np.eye(length)  # this imitates the u and v state vector as a matrix to extract the operation
        B = np.zeros((length, length))  # this is a blank padding matrix to get to a square matrix at the end

        dlnrho_dt = - derivative.diff_forward_n1_e1(A, self.delta_z)

        T = 273.15
        R = 8.314

        du_dt = - T * R * derivative.diff_backward_n1_e1(A, self.delta_z)

        dlnrho_dt[-1] = dlnrho_dt[-2]
        du_dt[0] = du_dt[-1] = 0  # wind at top and bottom stays constant at zero

        # generate output
        dlnrho_dt = np.concatenate((B, dlnrho_dt), axis=1)
        du_dt = np.concatenate((du_dt, B), axis=1)

        dz = np.stack((dlnrho_dt, du_dt), axis=0)

        return dz

    def __str__(self):
        return "matrix_log_euler"
