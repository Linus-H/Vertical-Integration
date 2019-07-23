import utils


class Explicit(utils.Integrator):
    def __init__(self, state, time_derivative, t0, delta_t):
        super().__init__(state, time_derivative, t0, delta_t)
        self.dt = delta_t

    def stepper(self, state_vars, t):
        d_state_vars_dt = self.time_derivative(state_vars, t)
        state_vars += self.dt * d_state_vars_dt


class ModifiedExplicit(utils.Integrator):
    def __init__(self, state, time_derivative, t0, delta_t):
        super().__init__(state, time_derivative, t0, delta_t)
        self.dt = delta_t

    def stepper(self, state_vars, t):
        for i in range(len(state_vars)):
            d_state_vars_dt = self.time_derivative(state_vars, t)
            state_vars[i] += self.dt * d_state_vars_dt[i]
