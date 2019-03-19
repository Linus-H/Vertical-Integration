import helpers as h


class ExplicitEuler(h.Integrator):
    def __init__(self, state, func, delta_t):
        super().__init__(state, self.stepper)
        self.f = func
        self.delta_t = delta_t

    def stepper(self, state):
        data = state.get_vars()

        derivative = self.f(data)
        data = data + self.delta_t * derivative

        return data
