import helpers as h


class ExplicitEuler(h.Integrator):
    def __init__(self, state, func, delta_t):
        super().__init__(state, self.stepper)
        self.f = func
        self.dt = delta_t

    def stepper(self, state_vars):
        d_state_vars_dt = self.f(state_vars)
        state_vars += self.dt * d_state_vars_dt


class ModifiedExplicitEuler(h.Integrator):
    def __init__(self, state, func, delta_t):
        super().__init__(state, self.stepper)
        self.f = func
        self.dt = delta_t

    def stepper(self, state_vars):
        for i in range(len(state_vars)):
            d_state_vars_dt = self.f(state_vars)
            state_vars[i] += self.dt * d_state_vars_dt[i]


class ExplicitHeun(h.Integrator):
    def __init__(self, state, func, delta_t):
        super().__init__(state, self.stepper)
        self.f = func
        self.dt = delta_t

    def stepper(self, state_vars_0):
        d_state_vars_0_dt = self.f(state_vars_0)

        state_vars_1 = state_vars_0 + self.dt * d_state_vars_0_dt
        d_state_vars_1_dt = self.f(state_vars_1)

        state_vars_0 += 0.5 * self.dt * (d_state_vars_0_dt + d_state_vars_1_dt)


class ModifiedExplicitHeun(h.Integrator):
    def __init__(self, state, func, delta_t):
        super().__init__(state, self.stepper)
        self.f = func
        self.dt = delta_t

    def stepper(self, state_vars_0):
        for i in range(len(state_vars_0)):
            d_state_vars_0_dt = self.f(state_vars_0)

            state_vars_1 = state_vars_0 + self.dt * d_state_vars_0_dt
            d_state_vars_1_dt = self.f(state_vars_1)

            state_vars_0[i] += 0.5 * self.dt * (d_state_vars_0_dt + d_state_vars_1_dt)[i]
