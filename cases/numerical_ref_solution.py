import numpy as np

import utils
from debug_tools import error_tracking_tools
from utils import Solution
import integrators.RungeKutta

path = """D:/Workspace/Vertical-Integration/data/"""


class ReferenceSolutionCalculator:
    def __init__(self, num_grid_points, num_vars, time_derivative_class, time_derivative_params, axes_offsets,
                 domain_length=1.0, down_sampling_rate=1):
        self.num_grid_points = num_grid_points
        self.num_vars = num_vars
        self.time_derivative_class = time_derivative_class
        self.axes = np.tile(np.linspace(0, domain_length, self.num_grid_points + 1)[:-1], (num_vars, 1))
        self.dx = domain_length / num_grid_points
        self.domain_length = domain_length
        self.time_derivative = time_derivative_class(self.dx, *time_derivative_params)

        self.down_sampling_rate = down_sampling_rate

        for i in range(num_vars):
            self.axes[i] += axes_offsets[i] * self.dx

    def generate(self, dt, end_time, initial_cond):

        state = utils.State(self.num_vars, self.num_grid_points, self.axes)

        integrator = integrators.RungeKutta.Explicit(state, self.time_derivative, 0, dt)

        file_name = "{}_#{}_dt{}_t{}_w{}".format(str(self.time_derivative),
                                                 self.num_grid_points,
                                                 int(1 / dt),
                                                 end_time,
                                                 self.domain_length)

        import os
        if not os.path.isfile(path + file_name + ".npy"):
            init_data = state.get_state_vars()
            num_iterations = int(end_time / dt)
            # first index: go through time, second index: variables, third index: go through space
            reference_solution = np.zeros((int(num_iterations / self.down_sampling_rate) + 1,
                                           self.num_vars,
                                           self.num_grid_points))
            times = np.zeros(((int(num_iterations / self.down_sampling_rate) + 1)))

            for i in range(self.num_vars):
                reference_solution[0, i] = init_data[i] = initial_cond[i](self.axes[i])

            timer = error_tracking_tools.TimeIterator(0, dt)

            for i, (time, state) in enumerate(zip(timer, integrator), 1):
                if i % self.down_sampling_rate == 0:
                    print(i / num_iterations)
                    times[int(i / self.down_sampling_rate)] = time
                    reference_solution[int(i / self.down_sampling_rate)] = state.get_state_vars()
                if i == num_iterations:
                    break
            np.save(path + file_name, reference_solution)
            np.save(path + file_name + "_axes", self.axes)
            np.save(path + file_name + "_time_vector", times)
        return file_name


def interpolate(location, axis, data):
    index = np.searchsorted(axis, location)  # assume that location is always within axis
    if index + 1 == len(axis) and location == axis[index]:
        return data[index]

    #print("max is {}, but got{}".format(axis[-1], location), flush=True)
    a, b = axis[index], axis[index + 1]
    b_factor = location - a
    b_factor = b_factor / (b - a)
    a_factor = 1 - b_factor
    return data[index] * a_factor + data[index + 1] * b_factor


class CaseSolution(Solution):
    def __init__(self, num_grid_points, dt, domain_size, num_vars, ref_solution_generator: ReferenceSolutionCalculator,
                 initial_cond, ref_dt=None, ref_t=10):
        super().__init__(self.solution, 0, dt)
        self.num_grid_points = num_grid_points
        self.num_vars = num_vars
        self.axes = np.tile(np.linspace(0, domain_size, self.num_grid_points + 1)[:-1], (num_vars, 1))
        self.state = utils.State(num_vars=num_vars, dim_vars=self.num_grid_points, axes=self.axes)

        if ref_dt is None:
            ref_dt = dt / 1000

        file_name = ref_solution_generator.generate(ref_dt, ref_t, initial_cond)
        self.sol_data = np.load(path + file_name + ".npy")
        self.sol_axes = np.load(path + file_name + "_axes.npy")
        self.sol_time = np.load(path + file_name + "_time_vector.npy")

    def solution(self, t, new_object=False):
        # get object to return
        if new_object:
            state = utils.State(num_vars=self.num_vars, dim_vars=self.num_grid_points, axes=self.axes)
            state_vars = state.get_state_vars()
        else:
            state = self.state
            state_vars = state.get_state_vars()
            state_vars.fill(0.0)

        # TODO: interpolate to create proper state
        high_definition_sol = interpolate(t, self.sol_time, self.sol_data)

        # low_def_sol = np.zeros((self.num_vars, self.num_grid_points))
        for i in range(self.num_vars):
            state_vars[i] = np.interp(self.axes[i], self.sol_axes[i], high_definition_sol[i])

        # np.copyto(state_vars, low_def_sol)

        return state
