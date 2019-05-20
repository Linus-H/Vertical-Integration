import utils


class Explicit(utils.Integrator):
    def __init__(self, state, time_derivative, t0, delta_t):
        super().__init__(state, self.stepper, t0, delta_t)
        self.time_derivative = time_derivative
        self.dt = delta_t

    def stepper(self, state_vars_0, t):
        k_1 = self.time_derivative(state_vars_0, t)
        k_2 = self.time_derivative(state_vars_0 + k_1 * self.dt * 0.5, t + self.dt * 0.5)
        k_3 = self.time_derivative(state_vars_0 + k_2 * self.dt * 0.5, t + self.dt * 0.5)
        k_4 = self.time_derivative(state_vars_0 + k_3 * self.dt * 1.0, t + self.dt)

        state_vars_0 += self.dt * (k_1 + 2 * k_2 + 2*k_3 + k_4) / 6.0
