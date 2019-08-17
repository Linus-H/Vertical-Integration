import numpy as np
from operators import average

import cases.euler_equation.consts as const


def s_to_z(axis, lnp, T, dpi_ds, num_grid_points):
    # converting lnp on offset grid to p on non-offset grid
    bottom_val = lnp[-1]
    top_val = lnp[1]
    lnp = average.avg_forward_e1(lnp)
    p = np.exp(lnp)
    p[0] = np.exp(top_val)/2
    p[-1] = np.exp(bottom_val)

    # converting T on offset grid to T on non-offset grid
    top_val = T[1]
    bottom_val = T[-1]
    T = average.avg_forward_e1(T)
    T[1] = top_val
    T[-1] = bottom_val

    z = (const.R / const.g * np.cumsum(np.flip((T * dpi_ds(axis) / p))) / num_grid_points)
    return np.flip(z)


def s_offset_to_z(axis, lnp, T, dpi_ds, num_grid_points):
    z = (const.R / const.g * np.cumsum(np.flip((T * np.exp(-lnp) * dpi_ds(axis)))) / num_grid_points)
    return np.flip(z)


def calc_energy(w, T, z, dpi_ds, num_grid_points, axis):
    # transfer w from non-offset layer to offset layer
    w = average.avg_backward_e1(w)

    # cut out top-most offset layer, because it is outside the domain
    w = w[1:]
    T = T[1:]
    z = z[1:]
    axis = axis[1:]

    kinetic = 0.5 * w * w
    internal = (const.C_p - const.R) * T
    geopotential = const.g * z
    E = np.sum((kinetic + internal + geopotential) * dpi_ds(axis)) / (const.g * num_grid_points)
    return E


def calc_mass():
    pass