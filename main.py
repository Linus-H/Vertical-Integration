import helpers as h
import derivative_function as d
import integrators as i
import numpy as np

# choose constants
num_grid_points = 1000
start = 0.0
L = 1.0
f = 2.0
dx = L / num_grid_points
c = 2.0
dt = 2 / (4.0 * num_grid_points * c)

# set up display window
vis = h.StateVisualizer(2, 1)

# setup
axes = np.tile(np.linspace(start, start + L, num_grid_points), (2, 1))
state = h.State(2, num_grid_points, axes, [("x", "u"), ("x", "v")])
data = state.get_state_vars()

# choose starting condition
data[0] = np.exp(-(axes[0] - 0.5) * (axes[0] - 0.5) * 100)  # np.sin((axis-0.5) * 2 * np.math.pi * f)

# choose border condition
derivative = d.WaveFunctionWraparound(dx, c)

# choose integrator
integrator = i.ModifiedExplicitHeun(state, derivative, dt)

# simulation loop
for i, state in enumerate(integrator):
    if i % 100 == 0:
        vis.display(1, state, 0, -2, 2)
        vis.display(2, state, 1, -10, 10)
