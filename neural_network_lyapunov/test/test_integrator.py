import neural_network_lyapunov.integrator as integrator
import numpy as np
import unittest
import scipy.integrate


class TestRk4ConstantControl(unittest.TestCase):
    def test1(self):
        # Test a simple system xdot = 1
        def dynamics_fun(x, u):
            return u

        def controller_fun(x):
            return 1.

        x0 = 2.
        dt = 0.01
        constant_control_steps = 10
        x, u = integrator.rk4_constant_control(dynamics_fun, controller_fun,
                                               x0, dt, constant_control_steps)
        self.assertEqual(u, 1)
        self.assertAlmostEqual(x, x0 + dt * constant_control_steps)

    def test2(self):
        # Test a simple system xdot = x + 1
        def dynamics_fun(x, u):
            return x + u

        def controller_fun(x):
            return 2 * x

        x0 = np.array([0.5])
        dt = 0.001
        constant_control_steps = 5
        x, u = integrator.rk4_constant_control(dynamics_fun, controller_fun,
                                               x0, dt, constant_control_steps)
        result = scipy.integrate.solve_ivp(lambda t, x: x + 1,
                                           [0, dt * constant_control_steps],
                                           x0)
        np.testing.assert_allclose(u, np.array([1.]))
        np.testing.assert_allclose(x, result.y[:, -1])


if __name__ == "__main__":
    unittest.main()
