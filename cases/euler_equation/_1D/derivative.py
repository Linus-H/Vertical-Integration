import numpy as np
from operators import derivative, average


class TimeDerivative:
    def __init__(self, delta_z):
        self.delta_z = delta_z

    def __call__(self, z, t):
        rho = z[0]
        rho_top = np.ones_like(rho) * rho[0]
        rho_bottom = np.ones_like(rho) * rho[-1]
        rho = np.concatenate((rho_top, rho, rho_bottom))

        u_z = z[1]
        u_z_flip = np.zeros_like(u_z)
        u_z = np.concatenate((u_z_flip, u_z, u_z_flip))

        drho_dt = - derivative.diff_n1_e4(rho * average.avg_forward_1(u_z), self.delta_z)

        T = 1.0
        R = 8.314

        rho_on_u_grid = average.avg_backward_1(rho)

        p = rho_on_u_grid * R * T

        du_dt = - u_z * derivative.diff_n1_e4(u_z, self.delta_z) \
                - derivative.diff_n1_e4(p, self.delta_z) / rho_on_u_grid  # + 3.0

        drho_dt = np.split(drho_dt, 3)[1]
        drho_dt[-1] = drho_dt[-2]

        du_dt = np.split(du_dt, 3)[1]
        du_dt[0] = du_dt[-1] = 0  # wind at top and bottom stays constant at zero

        # generate output
        dz = np.stack((drho_dt, du_dt), axis=-1)
        dz = dz.transpose()
        return dz


class LogTimeDerivative:
    def __init__(self, delta_z, non_linear=True):
        self.delta_z = delta_z
        self.non_linear = non_linear

    def __call__(self, z, t):
        lnrho = z[0]
        lnrho_top = np.flip(lnrho)  # np.ones_like(lnrho) * lnrho[0]
        lnrho_bottom = np.flip(lnrho)  # np.ones_like(lnrho) * lnrho[-1]

        rho_grid_ind = np.ones_like(lnrho)
        rho_grid_ind[0] = rho_grid_ind[-2:] = 0
        rho_grid_ind = np.concatenate((np.zeros_like(lnrho), rho_grid_ind, np.zeros_like(lnrho))) > 0

        lnrho = np.concatenate((lnrho_top, lnrho, lnrho_bottom))

        u_z = z[1]
        u_z_flip = np.zeros_like(u_z)

        u_grid_ind = np.ones_like(u_z)
        u_grid_ind[0] = u_grid_ind[-1:] = 0
        u_grid_ind = np.concatenate((np.zeros_like(u_z), u_grid_ind, np.zeros_like(u_z))) > 0

        u_z = np.concatenate((u_z_flip, u_z, u_z_flip))

        u_z_on_rho_grid = average.avg_forward_1(u_z)
        # u_z_on_rho_grid[rho_grid_ind] = average.avg_2(u_z_on_rho_grid)[rho_grid_ind]

        dlnrho_dt = - derivative.diff_n1_e4(u_z_on_rho_grid, self.delta_z)

        if self.non_linear:
            dlnrho_dt -= u_z_on_rho_grid * derivative.diff_n1_e4(lnrho, self.delta_z)

        T = 273.15
        R = 8.314

        lnrho_on_u_grid = average.avg_backward_1(lnrho)
        # lnrho_on_u_grid[u_grid_ind] = average.avg_2((lnrho_on_u_grid))[u_grid_ind]

        du_dt = - T * R * derivative.diff_n1_e4(lnrho_on_u_grid, self.delta_z) + 10.0

        if self.non_linear:
            du_dt -= u_z * derivative.diff_n1_e4(u_z, self.delta_z)

        drho_dt = np.split(dlnrho_dt, 3)[1]
        drho_dt[-1] = drho_dt[-2]

        du_dt = np.split(du_dt, 3)[1]
        du_dt[0] = du_dt[-1] = 0  # wind at top and bottom stays constant at zero

        # generate output
        dz = np.stack((drho_dt, du_dt), axis=-1)
        dz = dz.transpose()
        return dz
