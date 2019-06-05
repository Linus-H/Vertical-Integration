import numpy as np
import scipy.linalg

import utils


class Exponential(utils.Integrator):
    def __init__(self, state, time_derivative, t0, delta_t):
        super().__init__(state, self.stepper, t0, delta_t)
        self.time_derivative = time_derivative
        self.dt = delta_t
        self.state = state

        # make the state into a vector
        self.init_state_vector = np.copy(np.reshape(state.get_state_vars(), (-1, 1)))

        # get the system matrix A
        self.num_vars, self.dim_vars = self.state.get_state_vars().shape
        self.A = np.zeros((self.dim_vars * self.num_vars, self.dim_vars * self.num_vars))
        self.A = np.reshape(self.A, (self.num_vars, self.dim_vars, -1))
        self.A = self.time_derivative(self.A, t0)
        self.A = np.reshape(self.A, (self.dim_vars * self.num_vars, -1))

    def stepper(self, state_vars, t):
        np.copyto(state_vars,
                  np.reshape(scipy.linalg.expm(self.A * (t + self.dt - self.t0)) @ self.init_state_vector,
                             (self.num_vars, self.dim_vars)))
