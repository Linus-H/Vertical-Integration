import debugging
import helpers as h
import derivative_function as d
import integrators as i
import numpy as np
import known_problems

integrators = [i.ExplicitEuler, i.ModifiedExplicitEuler, i.ExplicitHeun, i.ModifiedExplicitHeun]
abbrs = ["EE", "MEE", "EH", "MEH"]
time_factors = [2, 4, 8, 16]
table = {}
time_list = []

for abbr, integrator_class in zip(abbrs, integrators):
    for time_factor in time_factors:
        # choose constants
        num_grid_points = 1000
        start = 0.0
        L = 1.0
        f = 2.0
        dx = L / num_grid_points
        c = 2.0

        dt = 1 / (time_factor * 4.0 * num_grid_points * c)

        # set up display window
        vis = h.StateVisualizer(2, 1)

        # setup
        standing_wave_sol = known_problems.StandingWaveFixedEnd(c, [(1, 1.0), (2, 1.0)], num_grid_points, dt)
        state = standing_wave_sol.get_initial_state()

        # choose starting condition
        # data[0] = np.exp(-(axes[0] - 0.5) * (axes[0] - 0.5) * 100)  # np.sin((axis-0.5) * 2 * np.math.pi * f)

        # choose border condition
        derivative = d.WaveFunctionFixedEnd(dx, c)

        # choose integrator
        integrator = integrator_class(state, derivative, dt)

        # debugging
        error_tracker = debugging.TrackError()
        timer = debugging.Timer(dt)
        time_list = []

        # simulation loop
        for i, (time, state, state_sol) in enumerate(zip(timer, integrator, standing_wave_sol), 1):
            if i % (100 * time_factor) == 0:
                # print(i)
                time_list.append(time)
                error_tracker.add_entry(time, state_sol.get_state_vars(), state.get_state_vars())
                # vis.display(1, state, 0, -2, 2)
                # vis.display(2, state_sol, 0, -2, 2)
                # vis.display(2, state, 1, -10, 10)
            if i == 10000 * time_factor:
                break
        name = abbr + " " + str(dt)
        print(name, end="\t")
        for e in error_tracker.abs_error:
            print(e, end="\t")
        print("",end="\n")