# `integrators`

## file descriptions
* Runge Kutta methods
    * `Euler.py` RK1
    * `Heun.py` RK2
    * `RungeKutta.py` RK4
* `Exponential.py` exponential integrator prototype (not yet unit-tested). 
* `integrator_test.py` unit-tests for explicit versions of RK1, RK2, RK4 using the test-cases in `cases.debug_case`.

## comments
`Euler.py` and `Heun.py` also contain modified versions `ModifiedExplicit`, which work differently.
Instead of doing one step and then updating all variables at once, they do one step per variable and after every such step, they just update one variable.