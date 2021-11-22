"""
I implement several fixed time-step integrator to understand the continuous
time simulation.
"""


def rk4_constant_control(dynamics_fun, controller, x0, dt,
                         constant_control_steps):
    """
    Given a system ẋ=f(x, u) and a controller u=π(x), we simulate the system
    for duration = constant_control_steps * dt. During this duration the
    control action is fixed to u=controller(x0). dt is the step length in
    Runge-Kutta4 method. This function emulates using a piecewise constant
    control on the real robot.
    """
    u = controller(x0)
    x = x0
    for i in range(constant_control_steps):
        k1 = dynamics_fun(x, u)
        k2 = dynamics_fun(x + dt / 2 * k1, u)
        k3 = dynamics_fun(x + dt / 2 * k2, u)
        k4 = dynamics_fun(x + dt * k3, u)
        x = x + 1. / 6 * dt * (k1 + 2 * k2 + 2 * k3 + k4)
    return x, u
