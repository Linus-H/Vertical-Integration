import debugging
import helpers as h
import derivative_function as d
import integrators as i
import numpy as np
import known_problems

# choose constants
num_grid_points = 1000
start = 0.0
L = 1.0
f = 2.0
dx = L / num_grid_points
c = 2.0

time_factor = 4
dt = 1 / (time_factor * 4.0 * num_grid_points * c)

# set up display window
vis = h.StateVisualizer(2, 1)


def initial_f(a):
    return np.exp(-100 * (a - 0.5) * (a - 0.5))


def derivative_f(a):
    return -(200 * a - 100) * initial_f(a)


# setup
standing_wave_sol = known_problems.WraparoundSolution(c, num_grid_points, dt, initial_f, derivative_f)
# known_problems.StandingWaveFixedEnd(c, num_grid_points, dt, [(1, 1.0), (2, 1.0)])
state = standing_wave_sol.get_initial_state()
print(np.max(state.get_state_vars()[1]))

# choose starting condition
# data[0] = np.exp(-(axes[0] - 0.5) * (axes[0] - 0.5) * 100)  # np.sin((axis-0.5) * 2 * np.math.pi * f)

# choose border condition
derivative = d.WaveFunctionWraparound(dx, c)

# choose integrator
integrator = i.ModifiedExplicitEuler(state, derivative, dt)

# debugging
error_tracker = debugging.TrackError(num_grid_points)
timer = debugging.Timer(dt)
time_list = []

# simulation loop
for i, (time, state, state_sol) in enumerate(zip(timer, integrator, standing_wave_sol), 1):
    if i % (10 * time_factor) == 0:
        # print(i)
        time_list.append(time)
        # error_tracker.add_entry(time, state_sol.get_state_vars(), state.get_state_vars())
        vis.display(1, state, 0, -2, 2)
        vis.display(2, state_sol, 0, -2, 2)
        #vis.display(2, state, 1, -10, 10)
    # if i == 10000 * time_factor:
    #    break

# for e in error_tracker.abs_error:
#    print(e, end="\t")
#    print("", end="\n")
