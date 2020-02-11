import robust_value_approx.simple_pwl_lyapunov as simple_pwl_lyapunov

import unittest

import numpy as np


class TestSimplePWLLyapunov(unittest.TestCase):
    def test1(self):
        """
        For a simple 1D piecewise linear system
        ẋ = -x if x >= 0
        ẋ = -2x if x <= 0
        there exists piecewise linear Lyapunov function.
        """
        rho = 0.1
        epsilon = 0.01
        dut = simple_pwl_lyapunov.SimplePWLLyapunov(
            1, 2, rho, epsilon, np.array([0.]), 0)
        dut.add_lyapunov_derivative_in_mode(
            0, np.array([[0.], [1.]]), np.array([[-1.]]), np.array([0.]))
        dut.add_lyapunov_derivative_in_mode(
            1, np.array([[0.], [-1.]]), np.array([[-2.]]), np.array([0.]))
        dut.add_lyapunov_positivity_in_mode(
            0, np.array([[0.], [1.]]), np.array([1]))
        dut.add_lyapunov_positivity_in_mode(
            1, np.array([[0.], [-1.]]), np.array([-1]))
        dut.add_continuity_constraint(0, 1, np.array([[0.]]))

        c_val, d_val, s1_val, s2_val = dut.solve(1.)
        # Now check constraint
        self.assertAlmostEqual(s1_val, 0, places=5)
        self.assertAlmostEqual(s2_val, 0, places=5)
        # Check continuity
        np.testing.assert_allclose(d_val, np.array([0., 0.]), atol=1e-6)
        # Check Vdot <= -epsilon * V
        self.assertLessEqual(
            c_val[0, 0] * -1, -epsilon * c_val[0, 0] + 1e-6)
        self.assertLessEqual(
            c_val[0, 1] * -2 * -1, -epsilon * c_val[0, 1] * -1 + 1e-6)
        # Check V >= rho * 1_norm(x - x*)
        self.assertGreaterEqual(c_val[0, 0] * 1, rho * 1)
        self.assertGreaterEqual(c_val[0, 1] * -1, rho * 1)

    def test2(self):
        """
        This is the system in Example 2 of Computation of piecewise quadratic
        Lyapunov functions for hybrid systems By M. Johansson and A. Rantzer.
        """
        rho = 0.1
        epsilon = 0.01
        dut = simple_pwl_lyapunov.SimplePWLLyapunov(
            2, 3, rho, epsilon, np.array([0., 0.]), 1)
        mode_vertices = [None] * 3
        A = [None] * 3
        g = [None] * 3
        mode_vertices[0] = np.array(
            [[-2., -1.], [-2., 1.], [-1., -1.], [-1., 1.]])
        A[0] = np.array([[-10., -10.5], [10.5, 9]])
        g[0] = np.array([-11., 7.5])
        mode_vertices[1] = np.array(
            [[-1., -1.], [-1., 1.], [1., -1.], [1., 1.]])
        A[1] = np.array([[-1., -2.5], [1., -1.]])
        g[1] = np.array([0., 0.])
        mode_vertices[2] = np.array([[1., -1.], [1., 1.], [2., -1.], [2., 1.]])
        A[2] = np.array([[-10., -10.5], [10.5, -20.]])
        g[2] = np.array([11., 50.5])
        for i in range(3):
            dut.add_lyapunov_derivative_in_mode(
                i, mode_vertices[i], A[i], g[i])

        lyapunov_positivity_args = [None] * 8
        lyapunov_positivity_args[0] = (
            0, np.array([[-2., -1.], [-2., 0.], [-1., 0.], [-1., -1.]]),
            np.array([-1, -1]))
        lyapunov_positivity_args[1] = (
            0, np.array([[-2., 0.], [-2., 1.], [-1., 1.], [-1., 0.]]),
            np.array([-1, 1]))
        lyapunov_positivity_args[2] = (
            1, np.array([[-1., -1.], [-1., 0.], [0., 0.], [0., -1.]]),
            np.array([-1, -1]))
        lyapunov_positivity_args[3] = (
            1, np.array([[-1., 0.], [-1., 1.], [0., 1.], [0., 0.]]),
            np.array([-1, 1]))
        lyapunov_positivity_args[4] = (
            1, np.array([[0., 0.], [0., 1.], [1., 1.], [1., 0.]]),
            np.array([1, 1]))
        lyapunov_positivity_args[5] = (
            1, np.array([[0., -1.], [0., 0.], [1., 0.], [1., -1.]]),
            np.array([1, -1]))
        lyapunov_positivity_args[6] = (
            2, np.array([[1., 0.], [1., 1.], [2., 1.], [2., 0.]]),
            np.array([1, 1]))
        lyapunov_positivity_args[7] = (
            2, np.array([[1., -1.], [1., 0.], [2., 0.], [2., -1.]]),
            np.array([1, -1]))
        for args in lyapunov_positivity_args:
            dut.add_lyapunov_positivity_in_mode(args[0], args[1], args[2])

        continuity_args = [None] * 2
        continuity_args[0] = (0, 1, np.array([[-1., -1.], [-1., 1.]]))
        continuity_args[1] = (1, 2, np.array([[1., -1.], [1., 1.]]))
        for args in continuity_args:
            dut.add_continuity_constraint(args[0], args[1], args[2])

        c_val, d_val, s1_val, s2_val = dut.solve(2.)
        # This system does not have a simple piecewise linear lyapunov
        # function.
        self.assertTrue(np.abs(s1_val) > 1e-6 or np.abs(s2_val) > 1e-6)
        # Now check the continuity constraint
        for args in continuity_args:
            for k in range(args[2].shape[0]):
                self.assertAlmostEqual(
                    c_val[:, args[0]] @ args[2][k] + d_val[args[0]],
                    c_val[:, args[1]] @ args[2][k] + d_val[args[1]], places=5)
        # Check the Lyapunov derivative constraint.
        for i in range(3):
            for j in range(mode_vertices[i].shape[0]):
                self.assertGreaterEqual(
                    s1_val, c_val[:, i] @ (A[i]@mode_vertices[i][j] + g[i]) +
                    epsilon * (c_val[:, i] @ mode_vertices[i][j] + d_val[i])
                    - 1e-5)
        # Check the Lyapunov positivity constraint
        for args in lyapunov_positivity_args:
            for j in range(args[1].shape[0]):
                self.assertGreaterEqual(
                    c_val[:, args[0]] @ args[1][j] + d_val[args[0]] -
                    rho * np.linalg.norm(args[1][j], ord=1), -s2_val - 1e-5)


if __name__ == "__main__":
    unittest.main()
