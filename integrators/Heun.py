import utils


class Explicit(utils.Integrator):
    def __init__(self, state, time_derivative, delta_t):
        super().__init__(state, self.stepper)
        self.time_derivative = time_derivative
        self.dt = delta_t

    def stepper(self, state_vars_0):
        d_state_vars_0_dt = self.time_derivative(state_vars_0)

        state_vars_1 = state_vars_0 + self.dt * d_state_vars_0_dt
        d_state_vars_1_dt = self.time_derivative(state_vars_1)

        state_vars_0 += 0.5 * self.dt * (d_state_vars_0_dt + d_state_vars_1_dt)


class ModifiedExplicit(utils.Integrator):
    def __init__(self, state, time_derivative, delta_t):
        super().__init__(state, self.stepper)
        self.time_derivative = time_derivative
        self.dt = delta_t

    def stepper(self, state_vars_0):
        for i in range(len(state_vars_0)):
            d_state_vars_0_dt = self.time_derivative(state_vars_0)

            state_vars_1 = state_vars_0 + self.dt * d_state_vars_0_dt
            d_state_vars_1_dt = self.time_derivative(state_vars_1)

            state_vars_0[i] += 0.5 * self.dt * (d_state_vars_0_dt + d_state_vars_1_dt)[i]
