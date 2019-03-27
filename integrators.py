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

class ModifiedExplicitEuler(h.Integrator):
    def __init__(self, state, func, delta_t):
        super().__init__(state, self.stepper)
        self.f = func
        self.delta_t = delta_t

    def stepper(self, state):
        data = state.get_vars()

        derivative = self.f(data)
        data[0] = (data + self.delta_t * derivative)[0]

        data[1] = (data + self.delta_t * derivative)[1]

        return data


class ExplicitHeun(h.Integrator):
    def __init__(self, state, func, delta_t):
        super().__init__(state, self.stepper)
        self.f = func
        self.delta_t = delta_t

    def helper(self, data):
        derivative_initial = self.f(data)
        pred_first = data + self.delta_t * derivative_initial

        derivative_additional = self.f(pred_first)

        pred_out = data + 0.5 * self.delta_t * (derivative_initial + derivative_additional)

        return pred_out

    def stepper(self, state):
        data = state.get_vars()

        data = self.helper(data)

        return data

class ModifiedExplicitHeun(h.Integrator):
    def __init__(self, state, func, delta_t):
        super().__init__(state, self.stepper)
        self.f = func
        self.delta_t = delta_t

    def helper(self, data):
        derivative_initial = self.f(data)
        pred_first = data + self.delta_t * derivative_initial

        derivative_additional = self.f(pred_first)

        pred_out = data + 0.5 * self.delta_t * (derivative_initial + derivative_additional)

        return pred_out

    def stepper(self, state):
        data = state.get_vars()

        data[0] = self.helper(data)[0]
        data[1] = self.helper(data)[1]

        return data
