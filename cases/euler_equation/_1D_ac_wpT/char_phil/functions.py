import numpy as np

import cases.euler_equation.consts as const
from operators import average


def s_to_z(axis, lnp, T, dpi_ds, num_grid_points):
    # converting lnp on offset grid to p on non-offset grid
    bottom_val = lnp[-1]
    top_val = lnp[1]
    lnp = average.avg_forward_e1(lnp)
    p = np.exp(lnp)
    p[0] = np.exp(top_val) / 2
    p[-1] = np.exp(bottom_val)

    z = (const.R / const.g * np.cumsum(np.flip((T * dpi_ds(axis) / p))) / num_grid_points)
    # print("align: {}".format(np.max(z)))
    return np.flip(z)


def s_offset_to_z(axis, lnp, T, dpi_ds, num_grid_points):
    # converting lnp on offset grid to p on offset grid
    p = np.exp(lnp)

    # converting T on non-offset grid to T on offset grid
    top_val = T[0]
    T = average.avg_backward_e1(T)
    T[0] = top_val

    z = (const.R / const.g * np.cumsum(np.flip((T * dpi_ds(axis) / p))) / num_grid_points)
    # print("offset: {}".format(np.max(z)))
    return np.flip(z)


def calc_energy(w_align, T_align, z_offset, dpi_ds, num_grid_points, axis_offset):
    # transfer w & T from non-offset layer to offset layer
    w_offset = average.avg_backward_e1(w_align)
    T_offset = average.avg_backward_e1(T_align)

    T_offset[0] = T_align[0]

    # cut out top-most offset layer, because it is outside the domain
    w_offset = w_offset[1:]
    T_offset = T_offset[1:]
    z_offset = z_offset[1:]
    axis_offset = axis_offset[1:]

    kinetic = 0.5 * w_offset * w_offset
    internal = (const.C_p - const.R) * T_offset
    geopotential = const.g * z_offset

    kinetic_en = kinetic * dpi_ds(axis_offset) / (const.g * num_grid_points)
    internal_en = internal * dpi_ds(axis_offset) / (const.g * num_grid_points)
    geopotential_en = geopotential * dpi_ds(axis_offset) / (const.g * num_grid_points)

    E = np.sum((kinetic + internal + geopotential) * dpi_ds(axis_offset)) / (const.g * num_grid_points)
    # E0 = (((kinetic + internal + geopotential) * dpi_ds(axis)) / (const.g * num_grid_points))[-1]
    return (E, np.sum(kinetic_en), np.sum(internal_en), np.sum(geopotential_en))


def calc_mass(dpi_ds, axis, num_grid_points):
    return np.sum(dpi_ds(axis) / (const.g * num_grid_points))
