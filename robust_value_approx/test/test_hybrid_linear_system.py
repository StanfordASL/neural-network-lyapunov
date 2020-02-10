import torch
import numpy as np
from scipy.integrate import solve_ivp
import scipy.linalg
import unittest
import robust_value_approx.hybrid_linear_system as hybrid_linear_system
import cvxpy as cp


def setup_trecate_discrete_time_system():
    """
    The piecewise affine system is from "Analysis of discrete-time
    piecewise affine and hybrid systems" by Giancarlo Ferrari-Trecate
    et.al.
    """
    dtype = torch.float64
    system = hybrid_linear_system.AutonomousHybridLinearSystem(
        2, dtype)
    system.add_mode(
        torch.tensor([[-0.999, 0], [-0.139, 0.341]], dtype=dtype),
        torch.zeros((2,), dtype=dtype),
        torch.tensor([[1, 0], [-1, 0], [0, 1], [0, -1]], dtype=dtype),
        torch.tensor([1, 0, 0, 1], dtype=dtype))
    system.add_mode(
        torch.tensor([[0.436, 0.323], [0.388, -0.049]], dtype=dtype),
        torch.zeros((2,), dtype=dtype),
        torch.tensor([[1, 0], [-1, 0], [0, 1], [0, -1]], dtype=dtype),
        torch.tensor([1, 0, 1, 0], dtype=dtype))
    system.add_mode(
        torch.tensor([[-0.457, 0.215], [0.491, 0.49]], dtype=dtype),
        torch.zeros((2,), dtype=dtype),
        torch.tensor([[1, 0], [-1, 0], [0, 1], [0, -1]], dtype=dtype),
        torch.tensor([0, 1, 0, 1], dtype=dtype))
    system.add_mode(
        torch.tensor([[-0.022, 0.344], [0.458, 0.271]], dtype=dtype),
        torch.zeros((2,), dtype=dtype),
        torch.tensor([[1, 0], [-1, 0], [0, 1], [0, -1]], dtype=dtype),
        torch.tensor([0, 1, 1, 0], dtype=dtype))
    return system


def setup_transformed_trecate_system(theta, x_equilibrium):
    """
    In paper "Analysis of discrete-time piecewise affine and hybrid systems"
    by Giancarlo Ferrari-Trecate et.al. the authors define a piecewise affine
    system in 2D.
    We transform this system, such that
    1. The equilibrium point is not the origin.
    2. Each hybrid mode is not in one and only one quadrant.
    If the system in Trecate's paper is
    x[n+1] = Ai * x[n] if Pi * x[n] <= qi
    The the transformed system is
    R(θ)ᵀ*(x[n+1] - x*) = Ai*R(θ)ᵀ * (x[n] - x*) if Pi*R(θ)ᵀ*(x[n] - x*) <= qi
    where R(θ) is a rotation matrix of angle θ in 2D, x* is the equilibrium
    state.
    @param theta The rotation angle for the transformation
    @param x_equilibrium The equilibrium state of the transformed system.
    """
    assert(isinstance(theta, float))
    assert(isinstance(x_equilibrium, torch.Tensor))
    assert(x_equilibrium.shape == (2,))
    dtype = x_equilibrium.dtype
    system = hybrid_linear_system.AutonomousHybridLinearSystem(
        2, dtype)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    R = torch.tensor(
        [[cos_theta, -sin_theta], [sin_theta, cos_theta]], dtype=dtype)

    def add_mode(A, P, q):
        system.add_mode(
            R @ A @ (R.T), x_equilibrium - R @ A @ (R.T) @ x_equilibrium,
            P @ (R.T), q + P @ (R.T) @ x_equilibrium)

    add_mode(
        torch.tensor([[-0.999, 0], [-0.139, 0.341]], dtype=dtype),
        torch.tensor([[1, 0], [-1, 0], [0, 1], [0, -1]], dtype=dtype),
        torch.tensor([1, 0, 0, 1], dtype=dtype))
    add_mode(
        torch.tensor([[0.436, 0.323], [0.388, -0.049]], dtype=dtype),
        torch.tensor([[1, 0], [-1, 0], [0, 1], [0, -1]], dtype=dtype),
        torch.tensor([1, 0, 1, 0], dtype=dtype))
    add_mode(
        torch.tensor([[-0.457, 0.215], [0.491, 0.49]], dtype=dtype),
        torch.tensor([[1, 0], [-1, 0], [0, 1], [0, -1]], dtype=dtype),
        torch.tensor([0, 1, 0, 1], dtype=dtype))
    add_mode(
        torch.tensor([[-0.022, 0.344], [0.458, 0.271]], dtype=dtype),
        torch.tensor([[1, 0], [-1, 0], [0, 1], [0, -1]], dtype=dtype),
        torch.tensor([0, 1, 1, 0], dtype=dtype))
    return system


def setup_johansson_continuous_time_system1(box_half_length=1.):
    """
    This is the simple example from section 3 of
    Computation of piecewise quadratic Lyapunov functions for hybrid systems
    by M. Johansson and A.Rantzer, 1997.
    This system doesn't have a common quadratic Lyapunov function.
    We definte the domain of the system as
    [-box_half_length, -box_half_length] x [box_half_length, box_half_length]
    """
    dtype = torch.float64
    system = hybrid_linear_system.AutonomousHybridLinearSystem(
        2, dtype)
    system.add_mode(
        torch.tensor([[-5, -4], [-1, -2]], dtype=dtype),
        torch.tensor([0, 0], dtype=dtype),
        torch.tensor([[1, 0], [-1, 0], [0, 1], [0, -1]], dtype=dtype),
        torch.tensor(
            [0, box_half_length, box_half_length, box_half_length],
            dtype=dtype))
    system.add_mode(
        torch.tensor([[-2, -4], [20, -2]], dtype=dtype),
        torch.tensor([0, 0], dtype=dtype),
        torch.tensor([[1, 0], [-1, 0], [0, 1], [0, -1]], dtype=dtype),
        torch.tensor(
            [box_half_length, 0, box_half_length, box_half_length],
            dtype=dtype))
    return system


def setup_johansson_continuous_time_system2():
    """
    This is the simple example from section 4 (equation 8, 9) of
    Computation of piecewise quadratic Lyapunov functions for hybrid systems
    by M. Johansson and A.Rantzer, 1997.
    This system doesn't have a common quadratic Lyapunov function.
    """
    dtype = torch.float64
    system = hybrid_linear_system.AutonomousHybridLinearSystem(2, dtype)
    alpha = 5.
    omega = 1.
    epsilon = 0.1
    A1 = torch.tensor(
        [[-epsilon, omega], [-alpha*omega, -epsilon]], dtype=dtype)
    A2 = torch.tensor(
        [[-epsilon, alpha*omega], [-omega, -epsilon]], dtype=dtype)
    system.add_mode(
        A1, torch.tensor([0., 0.], dtype=dtype),
        torch.tensor([[1, -1], [-1, -1], [0, 1]], dtype=dtype),
        torch.tensor([0, 0, 1], dtype=dtype))
    system.add_mode(
        A1, torch.tensor([0., 0.], dtype=dtype),
        torch.tensor([[-1, 1], [1, 1], [0, -1]], dtype=dtype),
        torch.tensor([0, 0, 1], dtype=dtype))
    system.add_mode(
        A2, torch.tensor([0., 0.], dtype=dtype),
        torch.tensor([[-1, 1], [-1, -1], [1, 0]], dtype=dtype),
        torch.tensor([0, 0, 1], dtype=dtype))
    system.add_mode(
        A2, torch.tensor([0., 0.], dtype=dtype),
        torch.tensor([[1, -1], [1, 1], [-1, 0]], dtype=dtype),
        torch.tensor([0, 0, 1], dtype=dtype))
    return system


def setup_johansson_continuous_time_system3(x_equilibrium):
    """
    This is the simple example from section 5 (equation 18~21) of
    Computation of piecewise quadratic Lyapunov functions for hybrid systems
    by M. Johansson and A.Rantzer, 1997.
    """
    dtype = torch.float64
    assert(isinstance(x_equilibrium, torch.Tensor))
    assert(x_equilibrium.shape == (2,))
    A1 = torch.tensor([[-10, -10.5], [10.5, 9]], dtype=dtype)
    A2 = torch.tensor([[-1, -2.5], [1, -1]], dtype=dtype)
    A3 = torch.tensor([[-10, -10.5], [10.5, -20]], dtype=dtype)
    g1 = -A1 @ x_equilibrium + torch.tensor([-11, 7.5], dtype=dtype)
    g2 = -A2 @ x_equilibrium
    g3 = -A3 @ x_equilibrium + torch.tensor([11, 50.5], dtype=dtype)
    P1 = torch.tensor([[-1, 0], [1, 0], [0, -1], [0, 1]], dtype=dtype)
    q1 = torch.tensor([2, -1, 1, 1], dtype=dtype) + P1 @ x_equilibrium
    P2 = torch.tensor([[-1, 0], [1, 0], [0, -1], [0, 1]], dtype=dtype)
    q2 = torch.tensor([1, 1, 1, 1], dtype=dtype) + P2 @ x_equilibrium
    P3 = torch.tensor([[-1, 0], [1, 0], [0, -1], [0, 1]], dtype=dtype)
    q3 = torch.tensor([-1, 2, 1, 1], dtype=dtype) + P3 @ x_equilibrium
    system = hybrid_linear_system.AutonomousHybridLinearSystem(2, dtype)
    system.add_mode(A1, g1, P1, q1)
    system.add_mode(A2, g2, P2, q2)
    system.add_mode(A3, g3, P3, q3)
    return system


class TestJohanssonSystem3(unittest.TestCase):
    def test(self):
        x_equilibrium = torch.tensor([1., 2.], dtype=torch.float64)
        system = setup_johansson_continuous_time_system3(x_equilibrium)
        np.testing.assert_allclose(
            (system.A[1] @ x_equilibrium + system.g[1]).detach().numpy(),
            np.zeros(2))
        for pt in ([-2, -1], [-2, 1], [-1, -1], [-1, 1],):
            np.testing.assert_array_less(
                (system.P[0] @ (torch.tensor(pt, dtype=torch.float64) +
                                x_equilibrium)).detach().numpy(),
                system.q[0].detach().numpy() + 1E-12)
        for pt in ([-1, -1], [-1, 1], [1, -1], [1, 1]):
            np.testing.assert_array_less(
                (system.P[1] @ (torch.tensor(pt, dtype=torch.float64) +
                                x_equilibrium)).detach().numpy(),
                system.q[1].detach().numpy() + 1E-12)
        for pt in ([1, -1], [1, 1], [2, -1], [2, 1]):
            np.testing.assert_array_less(
                (system.P[2] @ (torch.tensor(pt, dtype=torch.float64) +
                                x_equilibrium)).detach().numpy(),
                system.q[2].detach().numpy() + 1E-12)
        np.testing.assert_allclose(
            system.x_lo[0],
            np.array([-2., -1.]) + x_equilibrium.detach().numpy(), atol=1e-6)
        np.testing.assert_allclose(
            system.x_up[0],
            np.array([-1., 1.]) + x_equilibrium.detach().numpy(), atol=1e-6)
        np.testing.assert_allclose(
            system.x_lo[1],
            np.array([-1., -1.]) + x_equilibrium.detach().numpy(), atol=1e-6)
        np.testing.assert_allclose(
            system.x_up[1],
            np.array([1., 1.]) + x_equilibrium.detach().numpy(), atol=1e-6)
        np.testing.assert_allclose(
            system.x_lo[2],
            np.array([1., -1.]) + x_equilibrium.detach().numpy(), atol=2e-6)
        np.testing.assert_allclose(
            system.x_up[2],
            np.array([2., 1.]) + x_equilibrium.detach().numpy(), atol=2e-6)
        np.testing.assert_allclose(
            system.x_lo_all,
            np.array([-2., -1.]) + x_equilibrium.detach().numpy(), atol=2e-6)
        np.testing.assert_allclose(
            system.x_up_all,
            np.array([2., 1.]) + x_equilibrium.detach().numpy(), atol=2e-6)

        Aeq_s, Aeq_gamma, Ain_x, Ain_s, Ain_gamma, rhs_in =\
            system.mixed_integer_constraints()

        def test_fun(mode, state, satisfied):
            s_val = torch.zeros(6, dtype=torch.float64)
            gamma_val = torch.zeros(3, dtype=torch.float64)
            s_val[2 * mode: 2 * (mode+1)] = state
            gamma_val[mode] = 1
            if satisfied:
                np.testing.assert_array_less(
                    (Ain_x @ state + Ain_s @ s_val + Ain_gamma @ gamma_val).
                    detach().numpy(), rhs_in.detach().numpy() + 1e-7)
            else:
                self.assertFalse(
                    torch.all(Ain_x @ state + Ain_s @ s_val +
                              Ain_gamma @ gamma_val <= rhs_in + 1e-7))

        test_fun(
            2, torch.tensor([1.5, 0.3], dtype=torch.float64) + x_equilibrium,
            True)
        test_fun(
            1, torch.tensor([0.5, 0.3], dtype=torch.float64) + x_equilibrium,
            True)
        test_fun(
            2, torch.tensor([0.5, 0.3], dtype=torch.float64) + x_equilibrium,
            False)


class HybridLinearSystemTest(unittest.TestCase):
    def setUp(self):
        pass

    def test_constructor(self):
        dut = hybrid_linear_system.HybridLinearSystem(3, 2, torch.float64)
        self.assertEqual(dut.x_dim, 3)
        self.assertEqual(dut.u_dim, 2)
        self.assertEqual(dut.dtype, torch.float64)
        self.assertEqual(dut.num_modes, 0)

    def test_add_mode(self):
        dut = hybrid_linear_system.HybridLinearSystem(2, 1, torch.float64)
        A0 = torch.tensor([[1, 2], [2, 1]], dtype=dut.dtype)
        B0 = torch.tensor([[2], [3]], dtype=dut.dtype)
        c0 = torch.tensor([-1, 2], dtype=dut.dtype)
        P0 = torch.cat((torch.eye(3, dtype=dut.dtype),
                        -torch.eye(3, dtype=dut.dtype)), dim=0)
        q0 = torch.tensor([1, 2, 3, 1, 2, 3], dtype=dut.dtype)
        dut.add_mode(A0, B0, c0, P0, q0, True)
        self.assertEqual(dut.num_modes, 1)

    def test_mode(self):
        dut = hybrid_linear_system.HybridLinearSystem(2, 1, torch.float64)
        A0 = torch.tensor([[1, 2], [2, 1]], dtype=dut.dtype)
        B0 = torch.tensor([[2], [3]], dtype=dut.dtype)
        c0 = torch.tensor([-1, 2], dtype=dut.dtype)
        P0 = torch.cat((torch.eye(3, dtype=dut.dtype),
                        -torch.eye(3, dtype=dut.dtype)), dim=0)
        q0 = torch.tensor([1, 2, 3, 1, 2, 3], dtype=dut.dtype)
        dut.add_mode(A0, B0, c0, P0, q0, True)
        A1 = torch.tensor([[3, 2], [-2, 1]], dtype=dut.dtype)
        B1 = torch.tensor([[-2], [4]], dtype=dut.dtype)
        c1 = torch.tensor([3, -2], dtype=dut.dtype)
        P1 = torch.cat((3 * torch.eye(3, dtype=dut.dtype),
                        -2 * torch.eye(3, dtype=dut.dtype),
                        torch.tensor([[1, 2, 3]], dtype=dut.dtype)), dim=0)
        q1 = torch.tensor([12, 2, 4, -1, 1, 3, 7], dtype=dut.dtype)
        dut.add_mode(A1, B1, c1, P1, q1)
        A2 = torch.tensor([[3, -2], [6, 1]], dtype=dut.dtype)
        B2 = torch.tensor([[2], [7]], dtype=dut.dtype)
        c2 = torch.tensor([1, -4], dtype=dut.dtype)
        P2 = torch.cat((2 * torch.eye(3, dtype=dut.dtype),
                        -5 * torch.eye(3, dtype=dut.dtype),
                        torch.tensor([[4, 2, 1]], dtype=dut.dtype)), dim=0)
        q2 = torch.tensor([1, 3, 3, -1, 1, 3, 4], dtype=dut.dtype)
        dut.add_mode(A2, B2, c2, P2, q2)

        self.assertEqual(
            dut.mode(torch.tensor([0, 0], dtype=dut.dtype),
                     torch.tensor([0], dtype=dut.dtype)), 0)
        self.assertEqual(
            dut.mode(torch.tensor([1, 0], dtype=dut.dtype),
                     torch.tensor([2], dtype=dut.dtype)), 0)
        self.assertIsNone(
            dut.mode(torch.tensor([10, 20], dtype=dut.dtype),
                     torch.tensor([5], dtype=dut.dtype)))

    def test_mixed_integer_constraints(self):
        dut = hybrid_linear_system.HybridLinearSystem(2, 1, torch.float64)
        A0 = torch.tensor([[1, 2], [2, 1]], dtype=dut.dtype)
        B0 = torch.tensor([[2], [3]], dtype=dut.dtype)
        c0 = torch.tensor([-1, 2], dtype=dut.dtype)
        P0 = torch.cat((torch.eye(3, dtype=dut.dtype),
                        -torch.eye(3, dtype=dut.dtype)), dim=0)
        q0 = torch.tensor([1, 2, 3, 1, 2, 3], dtype=dut.dtype)
        dut.add_mode(A0, B0, c0, P0, q0, True)
        A1 = torch.tensor([[3, 2], [-2, 1]], dtype=dut.dtype)
        B1 = torch.tensor([[-2], [4]], dtype=dut.dtype)
        c1 = torch.tensor([3, -2], dtype=dut.dtype)
        P1 = torch.cat((3 * torch.eye(3, dtype=dut.dtype),
                        -2 * torch.eye(3, dtype=dut.dtype),
                        torch.tensor([[1, 2, 3]], dtype=dut.dtype)), dim=0)
        q1 = torch.tensor([12, 2, 4, -1, 1, 3, 7], dtype=dut.dtype)
        dut.add_mode(A1, B1, c1, P1, q1)
        A2 = torch.tensor([[3, -2], [6, 1]], dtype=dut.dtype)
        B2 = torch.tensor([[2], [7]], dtype=dut.dtype)
        c2 = torch.tensor([1, -4], dtype=dut.dtype)
        P2 = torch.cat((2 * torch.eye(3, dtype=dut.dtype),
                        -5 * torch.eye(3, dtype=dut.dtype),
                        torch.tensor([[4, 2, 1]], dtype=dut.dtype)), dim=0)
        q2 = torch.tensor([1, 3, 3, -1, 1, 3, 4], dtype=dut.dtype)
        dut.add_mode(A2, B2, c2, P2, q2)

        x_lo = torch.tensor([-3, -2], dtype=dut.dtype)
        x_up = torch.tensor([10, 8], dtype=dut.dtype)
        u_lo = torch.tensor([-5], dtype=dut.dtype)
        u_up = torch.tensor([10], dtype=dut.dtype)

        def generate_xu(mode, expect_in_mode):
            # @param expect_in_mode. Do you want to generate x/u in that mode?
            is_in_mode = not expect_in_mode
            while is_in_mode != expect_in_mode:
                x = torch.empty((2), dtype=dut.dtype)
                u = torch.empty((1), dtype=dut.dtype)
                for i in range(dut.x_dim):
                    x[i] = torch.DoubleTensor(1, 1).\
                        uniform_(x_lo[i], x_up[i])[0, 0]
                for i in range(dut.u_dim):
                    u[i] = torch.DoubleTensor(1, 1).\
                        uniform_(u_lo[i], u_up[i])[0, 0]
                if torch.all(dut.P[mode] @ torch.cat((x, u), dim=0) <=
                             dut.q[mode]):
                    is_in_mode = True
                else:
                    is_in_mode = False
            return (x, u)

        def test_mode(mode, x_lo, x_up, u_lo, u_up):
            (Aeq_slack, Aeq_alpha, Ain_x, Ain_u, Ain_slack, Ain_alpha, rhs_in)\
                = dut.mixed_integer_constraints(x_lo, x_up, u_lo, u_up)
            (x, u) = generate_xu(mode, True)
            # First find x and u in this mode.
            x_next = dut.A[mode] @ x + dut.B[mode] @ u + dut.c[mode]

            alpha = torch.zeros(dut.num_modes, dtype=dut.dtype)
            alpha[mode] = 1
            s = torch.zeros(dut.num_modes * dut.x_dim, dtype=dut.dtype)
            t = torch.zeros(dut.num_modes * dut.u_dim, dtype=dut.dtype)
            s[dut.x_dim * mode: dut.x_dim * (mode + 1)] = x
            t[dut.u_dim * mode: dut.u_dim * (mode + 1)] = u
            slack = torch.cat((s, t), dim=0)
            self.assertTrue(
                torch.all(torch.abs(x_next - (Aeq_slack @ slack
                                              + Aeq_alpha @ alpha)) < 1E-12))
            lhs_in = Ain_x @ x + Ain_u @ u + Ain_slack @ slack\
                + Ain_alpha @ alpha
            self.assertTrue(torch.all(lhs_in <= rhs_in + 1E-12))

        for mode in range(dut.num_modes):
            test_mode(mode, x_lo, x_up, u_lo, u_up)
            test_mode(mode, x_lo, x_up, u_lo, u_up)
            test_mode(mode, x_lo, x_up, u_lo, u_up)
            test_mode(mode, None, x_up, u_lo, u_up)
            test_mode(mode, x_lo, None, u_lo, u_up)
            test_mode(mode, x_lo, x_up, None, u_up)
            test_mode(mode, x_lo, x_up, u_lo, None)
            test_mode(mode, x_lo, x_up, None, None)
            test_mode(mode, None, x_up, u_lo, None)
            test_mode(mode, x_lo, None, u_lo, None)
            test_mode(mode, x_lo, None, None, u_up)
            test_mode(mode, None, None, u_lo, None)
            test_mode(mode, None, None, None, None)

        def test_ineq(mode, x_lo, x_up, u_lo, u_up):
            # Randomly sample x and u. If x and u are not in that mode, then
            # there should be no slack variables such that the inequality
            # constraints are satisfied.
            (Aeq_slack, Aeq_alpha, Ain_x, Ain_u, Ain_slack, Ain_alpha, rhs_in)\
                = dut.mixed_integer_constraints(x_lo, x_up, u_lo, u_up)
            (x, u) = generate_xu(mode, False)
            alpha = torch.zeros(dut.num_modes, 1, dtype=dut.dtype)
            alpha[mode] = 1
            s = torch.zeros(dut.num_modes * dut.x_dim, dtype=dut.dtype)
            t = torch.zeros(dut.num_modes * dut.u_dim, dtype=dut.dtype)
            s[dut.x_dim * mode: dut.x_dim * (mode + 1)] = x
            t[dut.u_dim * mode: dut.u_dim * (mode + 1)] = u
            slack = torch.cat((s, t), dim=0)
            lhs = Ain_x @ x + Ain_u @ u + Ain_slack @ slack + Ain_alpha @ alpha
            self.assertFalse(torch.all(lhs < rhs_in + 1E-12))

        for mode in range(dut.num_modes):
            test_ineq(mode, x_lo, x_up, u_lo, u_up)
            test_ineq(mode, x_lo, x_up, u_lo, u_up)
            test_ineq(mode, x_lo, x_up, u_lo, u_up)
            test_ineq(mode, None, x_up, u_lo, u_up)
            test_ineq(mode, x_lo, None, u_lo, u_up)
            test_ineq(mode, x_lo, x_up, None, u_up)
            test_ineq(mode, x_lo, x_up, u_lo, None)
            test_ineq(mode, x_lo, x_up, None, None)
            test_ineq(mode, None, x_up, u_lo, None)
            test_ineq(mode, x_lo, None, u_lo, None)
            test_ineq(mode, x_lo, None, None, u_up)
            test_ineq(mode, None, None, u_lo, None)
            test_ineq(mode, None, None, None, None)


class AutonomousHybridLinearSystemTest(unittest.TestCase):
    def test_constructor(self):
        dut = hybrid_linear_system.AutonomousHybridLinearSystem(
            3, torch.float64)
        self.assertEqual(dut.x_dim, 3)
        self.assertEqual(dut.dtype, torch.float64)
        self.assertEqual(dut.num_modes, 0)

    def test_add_mode(self):
        dut = hybrid_linear_system.AutonomousHybridLinearSystem(
            2, torch.float64)
        A0 = torch.tensor([[1, 2], [2, 1]], dtype=dut.dtype)
        g0 = torch.tensor([-1, 2], dtype=dut.dtype)
        P0 = torch.tensor(
            [[1, 1], [-1, -1], [1, -1], [-1, 1]], dtype=dut.dtype)
        q0 = torch.tensor([2, 2, 3, 3], dtype=dut.dtype)
        dut.add_mode(A0, g0, P0, q0, True)
        self.assertEqual(dut.num_modes, 1)
        self.assertEqual(len(dut.x_lo), 1)
        self.assertEqual(len(dut.x_up), 1)
        np.testing.assert_array_almost_equal(dut.x_lo[0],
                                             np.array([-2.5, -2.5]))
        np.testing.assert_array_almost_equal(dut.x_up[0], np.array([2.5, 2.5]))
        Ax_lower, Ax_upper = dut.mode_derivative_bounds(0)
        np.testing.assert_allclose(Ax_lower, np.array([-4.5, -4.5]))
        np.testing.assert_allclose(Ax_upper, np.array([4.5, 4.5]))

    def test_mixed_integer_constraints(self):
        dut = hybrid_linear_system.AutonomousHybridLinearSystem(
            2, torch.float64)
        A0 = torch.tensor([[1, 2], [2, 1]], dtype=dut.dtype)
        g0 = torch.tensor([-1, 2], dtype=dut.dtype)
        P0 = torch.tensor(
            [[1, 1], [-1, -1], [1, -1], [-1, 1]], dtype=dut.dtype)
        q0 = torch.tensor([1, 1, 1, 1], dtype=dut.dtype)
        dut.add_mode(A0, g0, P0, q0)

        A1 = torch.tensor([[2, 3], [4, 5]], dtype=dut.dtype)
        g1 = torch.tensor([0.1, 0.4], dtype=dut.dtype)
        P1 = P0.clone()
        q1 = torch.tensor([3, -1, 3, -1], dtype=dut.dtype)
        dut.add_mode(A1, g1, P1, q1)

        def test_mode(mode, x_lo, x_up):
            # We want to generate a random state in the admissible region of
            # the given mode.
            is_in_mode = False
            (Aeq_s, Aeq_gamma, Ain_x, Ain_s, Ain_gamma, rhs_in) =\
                dut.mixed_integer_constraints(
                    None, torch.tensor([4, 1], dtype=dut.dtype))
            while not is_in_mode:
                x_sample = torch.from_numpy(np.random.uniform(-4, 4, (2,)))
                if torch.all(dut.P[mode] @ x_sample <= dut.q[mode]):
                    is_in_mode = True
            # Now first check the expected x, s, gamma satisfy the constraint.
            xdot_expected = dut.A[mode] @ x_sample + dut.g[mode]
            s = torch.zeros(dut.x_dim * dut.num_modes, dtype=dut.dtype)
            s[mode * dut.x_dim: (mode+1) * dut.x_dim] = x_sample
            gamma = torch.zeros(dut.num_modes, dtype=dut.dtype)
            gamma[mode] = 1
            np.testing.assert_allclose(
                Aeq_s @ s + Aeq_gamma @ gamma, xdot_expected)
            np.testing.assert_array_less(
                (Ain_x @ x_sample + Ain_s @ s +
                 Ain_gamma @ gamma).detach().numpy(),
                (rhs_in + 1E-14).detach().numpy())
            # Now solve the problem with the given constraints, the only
            # solution should be gamma and s
            gamma_var = cp.Variable(dut.num_modes, boolean=True)
            s_var = cp.Variable(dut.num_modes * dut.x_dim)
            objective = cp.Maximize(0)
            prob = cp.Problem(
                objective,
                [(Ain_x @ x_sample).detach().numpy() +
                 Ain_s.detach().numpy() @ s_var +
                 Ain_gamma.detach().numpy() @ gamma_var <=
                 rhs_in.detach().numpy(), cp.sum(gamma_var) == 1])
            prob.solve()
            self.assertEqual(prob.status, 'optimal')
            np.testing.assert_allclose(gamma.detach().numpy(), gamma_var.value)
            np.testing.assert_allclose(s.detach().numpy(), s_var.value)

        for mode in [0, 1]:
            test_mode(mode, torch.tensor([-1, -1], dtype=dut.dtype),
                      torch.tensor([4, 1], dtype=dut.dtype))
            test_mode(mode, torch.tensor([-2, -2], dtype=dut.dtype),
                      torch.tensor([5, 2], dtype=dut.dtype))
            test_mode(mode, torch.tensor([-1, -1], dtype=dut.dtype), None)
            test_mode(mode, None, torch.tensor([-1, -1], dtype=dut.dtype))
            test_mode(mode, None, None)

    def test_mode1(self):
        dut = setup_trecate_discrete_time_system()
        self.assertEqual(
            dut.mode(torch.tensor([0.4, 0.5], dtype=dut.dtype)), 1)
        self.assertEqual(
            dut.mode(torch.tensor([-0.4, 0.5], dtype=dut.dtype)), 3)
        self.assertEqual(
            dut.mode(torch.tensor([-0.4, -0.5], dtype=dut.dtype)), 2)
        self.assertEqual(
            dut.mode(torch.tensor([0.4, -0.5], dtype=dut.dtype)), 0)

    def test_mode2(self):
        theta = np.pi / 5
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        R = torch.tensor(
            [[cos_theta, -sin_theta], [sin_theta, cos_theta]],
            dtype=torch.float64)
        x_equilibrium = torch.tensor([0.2, 1.5], dtype=torch.float64)
        dut = setup_transformed_trecate_system(theta, x_equilibrium)
        self.assertEqual(dut.mode(
            R @ torch.tensor([0.4, 0.5], dtype=dut.dtype) + x_equilibrium), 1)
        self.assertEqual(dut.mode(
            R @ torch.tensor([-0.4, 0.5], dtype=dut.dtype) + x_equilibrium), 3)
        self.assertEqual(dut.mode(
            R @ torch.tensor([-0.4, -0.5], dtype=dut.dtype) + x_equilibrium),
            2)
        self.assertEqual(dut.mode(
            R @ torch.tensor([0.4, -0.5], dtype=dut.dtype) + x_equilibrium), 0)

    def test_step_forward1(self):
        dut = setup_trecate_discrete_time_system()

        def test_fun(x):
            mode = dut.mode(x)
            x_next_expected = dut.step_forward(x, mode)
            x_next_expected2 = dut.step_forward(x)
            x_next = dut.A[mode] @ x + dut.g[mode]
            np.testing.assert_array_almost_equal(
                x_next.detach().numpy(), x_next_expected.detach().numpy())
            np.testing.assert_array_almost_equal(
                x_next.detach().numpy(), x_next_expected2.detach().numpy())

        test_fun(torch.tensor([0.4, 0.5], dtype=dut.dtype))
        test_fun(torch.tensor([0.4, -0.5], dtype=dut.dtype))
        test_fun(torch.tensor([-0.4, -0.5], dtype=dut.dtype))
        test_fun(torch.tensor([-0.4, 0.5], dtype=dut.dtype))

    def test_step_forward2(self):
        dut1 = setup_trecate_discrete_time_system()
        theta = np.pi / 5
        x_equilibrium = torch.tensor([0.2, 1.5], dtype=torch.float64)
        dut2 = setup_transformed_trecate_system(theta, x_equilibrium)
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        R = torch.tensor(
            [[cos_theta, -sin_theta], [sin_theta, cos_theta]],
            dtype=torch.float64)

        def test_fun(x):
            x_next = dut1.step_forward(x)
            x_transformed = R @ x + x_equilibrium
            x_next_transformed = dut2.step_forward(x_transformed)
            np.testing.assert_allclose(
                (R @ x_next + x_equilibrium).detach().numpy(),
                x_next_transformed.detach().numpy())

        test_fun(torch.tensor([0.4, 0.5], dtype=torch.float64))
        test_fun(torch.tensor([-0.4, 0.5], dtype=torch.float64))
        test_fun(torch.tensor([-0.4, -0.5], dtype=torch.float64))
        test_fun(torch.tensor([0.4, -0.5], dtype=torch.float64))

    def test_possible_dx(self):
        dut = setup_trecate_discrete_time_system()

        x = torch.tensor([0.5, 0.6], dtype=dut.dtype)
        next_states = dut.possible_dx(x)
        self.assertEqual(len(next_states), 1)
        np.testing.assert_allclose(
            next_states[0].detach().numpy(),
            (dut.A[1] @ x + dut.g[1]).detach().numpy())

        x = torch.tensor([0.5, 0], dtype=dut.dtype)
        next_states = dut.possible_dx(x)
        self.assertEqual(len(next_states), 2)
        np.testing.assert_allclose(
            next_states[0].detach().numpy(),
            (dut.A[0] @ x + dut.g[0]).detach().numpy())
        np.testing.assert_allclose(
            next_states[1].detach().numpy(),
            (dut.A[1] @ x + dut.g[1]).detach().numpy())

        x = torch.tensor([1.5, 0], dtype=dut.dtype)
        next_states = dut.possible_dx(x)
        self.assertEqual(len(next_states), 0)


class TestComputeDiscreteTimeSystemCostToGo(unittest.TestCase):
    def test_fun(self):
        system = setup_trecate_discrete_time_system()

        def instantaneous_cost_fun(x):
            return x @ x

        def test_fun(x):
            num_steps = 100
            total_cost, x_steps, costs = hybrid_linear_system.\
                compute_discrete_time_system_cost_to_go(
                    system, x, num_steps, instantaneous_cost_fun)
            np.testing.assert_allclose(
                x.detach().numpy(), x_steps[:, 0].detach().numpy())
            self.assertEqual(total_cost.item(), costs[0].item())
            total_cost_expected = instantaneous_cost_fun(x)
            x_i = x.clone()
            for i in range(num_steps):
                for j in range(system.num_modes):
                    if (torch.all(system.P[j] @ x_i <= system.q[j])):
                        x_i = system.A[j] @ x_i + system.g[j]
                        break
                np.testing.assert_allclose(
                    x_i.detach().numpy(), x_steps[:, i + 1].detach().numpy())
                self.assertAlmostEqual(
                    (total_cost_expected + costs[i+1]).item(),
                    total_cost.item())
                total_cost_expected += instantaneous_cost_fun(x_i)
            self.assertAlmostEqual(total_cost.item(),
                                   total_cost_expected.item())

        x_sample, y_sample = torch.meshgrid(
            torch.linspace(-1., 1., 11).type(system.dtype),
            torch.linspace(-1., 1., 11).type(system.dtype))
        for i in range(x_sample.shape[0]):
            for j in range(x_sample.shape[1]):
                test_fun(torch.tensor(
                    [x_sample[i, j], y_sample[i, j]], dtype=system.dtype))

        # With a goal state
        x = torch.tensor([0.5, 0.4], dtype=system.dtype)
        x_next = system.step_forward(x)
        total_cost, x_steps, costs = \
            hybrid_linear_system.compute_discrete_time_system_cost_to_go(
                system, x, 100, lambda x: torch.norm(x), x_goal=x_next)
        self.assertEqual(x_steps.shape, (2, 2))
        self.assertEqual(costs.shape, (2,))
        np.testing.assert_allclose(
            x_steps[:, 0].detach().numpy(), x.detach().numpy())
        np.testing.assert_allclose(
            x_steps[:, 1].detach().numpy(), x_next.detach().numpy())
        self.assertAlmostEqual(
            costs[0].item(), (torch.norm(x) + torch.norm(x_next)).item())
        self.assertAlmostEqual(
            costs[1].item(), torch.norm(x_next).item())
        self.assertAlmostEqual(total_cost.item(), costs[0].item())


class TestComputeContinuousTimeSystemCostToGo(unittest.TestCase):
    def test_fun1(self):
        system = setup_johansson_continuous_time_system1()
        x0 = torch.tensor([0.3, 0.8], dtype=torch.float64)
        (cost, x_traj, cost_to_go_traj, sol) = hybrid_linear_system.\
            compute_continuous_time_system_cost_to_go(
                system, x0, 15., lambda x: torch.norm(x, p=2))
        np.testing.assert_allclose(
            x_traj[:, -1].detach().numpy(), np.zeros(2), atol=1e-6)
        self.assertAlmostEqual(cost_to_go_traj[-1], 0)
        self.assertTrue(torch.all(
            cost_to_go_traj[:-1] - cost_to_go_traj[1:] > 0))
        self.assertAlmostEqual(cost.item(), cost_to_go_traj[0].item())
        np.testing.assert_allclose(
            cost_to_go_traj.detach().numpy() + sol.y[-1, :], cost)
        (cost_next, x_traj_next, cost_to_go_traj_next, sol_next) = \
            hybrid_linear_system.compute_continuous_time_system_cost_to_go(
                system, x_traj[:, 1], sol.t[-1] - sol.t[1],
                lambda x: torch.norm(x, p=2))
        self.assertAlmostEqual(
            cost_to_go_traj[1].item(), cost_next.item(), places=4)
        # The trajectory starts in mode 1, will first hit the mode boundary
        # x[0] = 0, and then stay in mode 0.
        A1_numpy = system.A[1].detach().numpy()
        def hit_boundary(t, x): return x[0]
        hit_boundary.terminal = True
        sol1 = solve_ivp(
            lambda t, x: A1_numpy @ x, (0, np.inf),
            x0.detach().numpy(), events=[hit_boundary])
        # t1 is the time to hit the mode boundary x[0] = 0
        t1 = sol1.t[-1]
        x1 = sol1.y[:, -1]
        # The trajectory of x(t) is exp(A1*t) for 0 <= t <= t1
        num_sample1 = 10000
        t1_sample = np.linspace(0, t1, num_sample1)
        cost1_sample = np.empty(num_sample1)
        for i in range(num_sample1):
            cost1_sample[i] = np.linalg.norm(
                scipy.linalg.expm(A1_numpy * t1_sample[i]) @
                x0.detach().numpy(), ord=2)
        A0_numpy = system.A[0].detach().numpy()
        def hit_origin(t, x): return np.linalg.norm(x, ord=2) - 1e-4
        hit_origin.terminal = True
        sol2 = solve_ivp(
            lambda t, x: A0_numpy @ x, (0, np.inf), x1, events=[hit_origin])
        num_sample2 = 10000
        t2_sample = np.linspace(0, sol2.t[-1], num_sample2)
        cost2_sample = np.empty(num_sample2)
        for i in range(num_sample2):
            cost2_sample[i] = np.linalg.norm(
                scipy.linalg.expm(A0_numpy * t2_sample[i]) @ x1, ord=2)
            cost_expected = np.sum(
                (cost1_sample[:-1] + cost1_sample[1:])/2 *
                (t1_sample[1:] - t1_sample[:-1])) +\
                np.sum(
                    (cost2_sample[:-1] + cost2_sample[1:])/2 *
                    (t2_sample[1:] - t2_sample[:-1]))
        # cost_expected is computed with low accuracy (because it uses
        # trapezoidal integration), so only compare up to 4'th decimal place.
        self.assertAlmostEqual(cost.item(), cost_expected, places=4)

        x_equilibrium = torch.tensor([0, 0], dtype=system.dtype)
        cost_inf, x_inf_traj, cost_to_go_inf_traj, sol_inf = \
            hybrid_linear_system.compute_continuous_time_system_cost_to_go(
                system, x0, np.inf,
                lambda x: torch.norm(x, p=2), x_goal=x_equilibrium)
        self.assertLessEqual(
            np.linalg.norm(sol_inf.y[:2, -1]), 2e-3)


class TestGenerateCostToGoSamples(unittest.TestCase):
    def setUp(self):
        dtype = torch.float64
        samples_x, samples_y = torch.meshgrid(
            torch.linspace(-1., 1., 5, dtype=dtype),
            torch.linspace(-1., 1., 5, dtype=dtype))
        self.x0_samples = [None] * (samples_x.shape[0] * samples_x.shape[1])
        for i in range(samples_x.shape[0]):
            for j in range(samples_x.shape[1]):
                self.x0_samples[i * samples_x.shape[1] + j] = torch.tensor(
                    [samples_x[i, j], samples_y[i, j]], dtype=dtype)

    def test_continuous_time_system(self):
        system = setup_johansson_continuous_time_system1(5)
        x_equilibrium = torch.tensor([0, 0], dtype=system.dtype)
        T = 15.

        def pruner(x):
            return torch.norm(x - x_equilibrium, p=2) < 0.01
        x0_cost_pairs = hybrid_linear_system.generate_cost_to_go_samples(
            system, self.x0_samples, T, lambda x: torch.norm(x, p=2), False,
            x_equilibrium, pruner)
        self.assertIsInstance(x0_cost_pairs, list)
        self.assertGreater(len(x0_cost_pairs), len(self.x0_samples))
        for x0_cost_pair in x0_cost_pairs:
            self.assertFalse(pruner(x0_cost_pair[0]))

    def test_discrete_time_system(self):
        system = setup_trecate_discrete_time_system()
        x_equilibrium = torch.tensor([0, 0], dtype=system.dtype)

        def pruner(x):
            return torch.norm(x - x_equilibrium, p=2) < 0.01
        N = 20
        x0_cost_pairs = hybrid_linear_system.generate_cost_to_go_samples(
            system, self.x0_samples, N, lambda x: torch.norm(x, p=2), True,
            x_equilibrium, pruner)
        self.assertIsInstance(x0_cost_pairs, list)
        self.assertGreater(len(x0_cost_pairs), len(self.x0_samples))
        for x0_cost_pair in x0_cost_pairs:
            self.assertFalse(pruner(x0_cost_pair[0]))


class TestPartitionStateInputSpace(unittest.TestCase):
    def test_one_per_mode(self):
        dtype = torch.float64
        x_lo = torch.Tensor([-1., -2.]).type(dtype)
        x_up = torch.Tensor([1., 3.]).type(dtype)
        u_lo = torch.Tensor([-10., -5]).type(dtype)
        u_up = torch.Tensor([10., 5]).type(dtype)
        num_breaks_x = torch.Tensor([3, 5]).type(torch.int)
        num_breaks_u = torch.Tensor([2, 1]).type(torch.int)
        x_delta = torch.Tensor([0., 0.]).type(dtype)
        u_delta = torch.Tensor([0., 0.]).type(dtype)
        ss = hybrid_linear_system.partition_state_input_space(x_lo, x_up,
                                                              u_lo, u_up,
                                                              num_breaks_x,
                                                              num_breaks_u,
                                                              x_delta,
                                                              u_delta)
        (states_x, states_u,
         states_x_lo, states_x_up,
         states_u_lo, states_u_up) = ss

        def num_of_modes(x, u):
            num_of_modes = 0
            for k in range(states_x_lo.shape[0]):
                if (torch.all(x >= states_x_lo[k, :]) and
                    torch.all(x <= states_x_up[k, :]) and
                    torch.all(u >= states_u_lo[k, :]) and
                        torch.all(u <= states_u_up[k, :])):
                    num_of_modes += 1
            return num_of_modes
        for i in range(states_x.shape[0]):
            self.assertEqual(num_of_modes(states_x[i, :], states_u[i, :]), 1)

    def test_overlap(self):
        dtype = torch.float64
        x_lo = torch.Tensor([-1., -2.]).type(dtype)
        x_up = torch.Tensor([1., 3.]).type(dtype)
        u_lo = torch.Tensor([-10., -15]).type(dtype)
        u_up = torch.Tensor([10., 15]).type(dtype)
        num_breaks_x = torch.Tensor([3, 5]).type(torch.int)
        num_breaks_u = torch.Tensor([2, 1]).type(torch.int)
        x_delta = torch.Tensor([.5, .5]).type(dtype)
        u_delta = torch.Tensor([-.5, -.5]).type(dtype)
        ss = hybrid_linear_system.partition_state_input_space(x_lo, x_up,
                                                              u_lo, u_up,
                                                              num_breaks_x,
                                                              num_breaks_u,
                                                              x_delta,
                                                              u_delta)
        (states_x, states_u,
         states_x_lo, states_x_up,
         states_u_lo, states_u_up) = ss

        def num_of_modes(x, u):
            num_of_modes = 0
            for k in range(states_x_lo.shape[0]):
                if (torch.all(x >= states_x_lo[k, :]) and
                    torch.all(x <= states_x_up[k, :]) and
                    torch.all(u >= states_u_lo[k, :]) and
                        torch.all(u <= states_u_up[k, :])):
                    num_of_modes += 1
            return num_of_modes
        for i in range(states_x.shape[0]):
            self.assertGreater(num_of_modes(states_x[i, :], states_u[i, :]), 1)


if __name__ == "__main__":
    unittest.main()
