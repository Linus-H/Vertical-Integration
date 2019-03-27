import helpers as h
import derivative_function as d
import integrators as i
import numpy as np

# set up constants
num_grid_points = 250
start = 0.0
L = 1.0
f = 1.0
dx = L / num_grid_points
dt = 1/(4.0*num_grid_points)
c = 2.0

# set up display windows
vis = h.StateVisualizer(2, 1)

# setup
axis = np.linspace(start, start + L, num_grid_points)
state = h.State(2, num_grid_points, axis, [("x","u"), ("x", "v")])
data = state.get_vars()

# setup starting condition
data[0] = np.sinc((axis-0.5) * 2 * np.math.pi)

# choose border condition
derivative = d.WaveFunctionFixedEnd(dx, c)

# set up initial derivatives
#data[1] = derivative(data)[1]
# alternative: comment out to set data[1] = 0

# choose integrator
integrator = i.ExplicitHeun(state, derivative, dt)

# simulation loop
for state in integrator:
    vis.display(1, state, 0, -2, 2)
    vis.display(2, state, 1, -10, 10)
