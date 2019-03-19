import helpers as h
import derivative_function as d
import integrators as i
import numpy as np

# set up constants
num_grid_points = 1000
start = 0.0
stop = 10.0
f = 0.5
dx = (stop - start) / num_grid_points
dt = 0.01
c = 1.0

# set up display windows
vis = h.StateVisualizer(2, 1)

# set up bordering condition
axis = np.linspace(start, stop, num_grid_points)
state = h.State(2, num_grid_points, axis, ["u", "v"])
data = state.get_vars()

# data[0] = np.sinc((axis - 5) * 2 * np.math.pi * f)
data[0] = (1 - np.cos(axis * 2 * np.math.pi * f)) * 0.5  # - 1
data[0] *= data[0]
data[0] *= data[0]

for k, x in enumerate(axis):
    if x > 2:
        data[0][k] = 0.0
        print(x)

derivative = d.WaveFunctionFixedEnd(dx, c)

data[1] = derivative(data)[1]

integrator = i.ExplicitEuler(state, derivative, dt)

for state in integrator:
    vis.display(1, state, 0, -2, 2)
    vis.display(2, state, 1, -2, 2)
