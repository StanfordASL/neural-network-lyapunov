import neural_network_lyapunov.control_lyapunov as control_lyapunov

import unittest
import numpy as np
import cvxpy as cp
import torch
import torch.nn as nn


class TestControlLyapunovFixedActivationPattern(unittest.TestCase):
    def test_constructor(self):
        # Set the problem data to arbitrary value.
        g = np.array([1, 2]).reshape((2, 1))
        P = np.array([[1, 0], [0, 2]])
        q = np.array([2, 3]).reshape((2, 1))
        A = np.array([[1, 1], [0, 1]])
        B = np.array([2, 3]).reshape((2, 1))
        d = np.array([3, 4]).reshape(2, 1)
        u_vertices = np.array([-1, 1]).reshape((1, 2))
        dut = control_lyapunov.ControlLyapunovFixedActivationPattern(
            g, P, q, A, B, d, u_vertices)
        prob = dut.construct_program()
        prob.solve(solver="GUROBI")
        self.assertEqual(prob.status, cp.OPTIMAL)
        self.assertTrue(np.less_equal(P.dot(dut.x.value), q).all())
        self.assertAlmostEqual(
            prob.value,
            g.T.dot(dut.x.value) + np.min(g.T.dot(B.dot(u_vertices))) +
            g.T.dot(d), 5)


class TestControlLyapunovFreeActivationPattern(unittest.TestCase):
    def setUp(self):
        self.dtype = torch.float64
        self.linear1 = nn.Linear(2, 3)
        self.linear1.weight.data = torch.tensor([[1, 2], [3, 4], [5, 6]],
                                                dtype=self.dtype)
        self.linear1.bias.data = torch.tensor([-11, 13, 4], dtype=self.dtype)
        self.linear2 = nn.Linear(3, 4)
        self.linear2.weight.data = torch.tensor(
            [[-1, 0.5, 1.5], [2, 5, 6], [-2, -3, -4], [1, 4, 6]],
            dtype=self.dtype)
        self.linear2.bias.data = torch.tensor([4, -1, -2, 3], dtype=self.dtype)
        self.linear3 = nn.Linear(4, 1)
        self.linear3.weight.data = torch.tensor([[4, 5, 6, 7]],
                                                dtype=self.dtype)
        self.linear3.bias.data = torch.tensor([-10], dtype=self.dtype)
        self.model = nn.Sequential(self.linear1, nn.ReLU(), self.linear2,
                                   nn.ReLU(), self.linear3)
        self.dut = control_lyapunov.ControlLyapunovFreeActivationPattern(
            self.model, self.dtype)

    def test_generate_program_verify_continuous_affine_system(self):
        A_dyn = torch.tensor([[0, 1], [2, 3]], dtype=self.dtype)
        B_dyn = torch.tensor([[2, 0], [0, 4]], dtype=self.dtype)
        d_dyn = torch.tensor([[1], [-2]], dtype=self.dtype)
        u_vertices = torch.tensor([[1, 2, 3], [-1, 3, 2]], dtype=self.dtype)
        x_lo = torch.tensor([-2, -3], dtype=self.dtype)
        x_up = torch.tensor([3, 4], dtype=self.dtype)
        (c1, c2, Ain1, Ain2, Ain3, Ain4, Ain5, rhs) = \
            self.dut.generate_program_verify_continuous_affine_system(
            A_dyn, B_dyn, d_dyn, u_vertices, x_lo, x_up)

        def test_generate_program_util(x, beta):
            """
            With given value of x and beta, compute s, t and alpha that
            should satisfy the constraint. Check if the generated constraint
            parameterized by Ain1,..., 5 are satisfied, and the cost
            evaluates as expected.
            """
            assert (torch.all(x.squeeze() <= x_up))
            assert (torch.all(x.squeeze() >= x_lo))

            # First compute alpha
            alpha = torch.zeros((12, 1), dtype=self.dtype)
            assert (len(beta) == 2)
            for i0 in range(3):
                for i1 in range(4):
                    alpha_index = \
                        self.dut.relu_free_pattern.compute_alpha_index(
                            (i0, i1))
                    alpha[alpha_index][0] = 1 if beta[0][i0] == 1 and \
                        beta[1][i1] == 1 else 0
            # Now compute s, s(i, j) = alpha(i) * x(j)
            s = torch.empty((alpha.shape[0] * x.shape[0], 1), dtype=self.dtype)
            for i in range(alpha.shape[0]):
                for j in range(x.shape[0]):
                    s[i * x.shape[0] + j][0] = alpha[i][0] * x[j][0]
            # Now compute t = min_i αᵀMBuᵢ
            (M, _, _, _) = self.dut.relu_free_pattern.output_gradient()
            t = torch.tensor([[
                torch.min(alpha.T @ M @ B_dyn @ u_vertices).item()
            ]]).to(dtype=self.dtype)
            beta_vec = torch.tensor([
                beta_val for beta_layer in beta for beta_val in beta_layer
            ]).reshape((-1, 1)).to(dtype=self.dtype)

            def compute_lhs(s_val, t_val, alpha_val):
                return Ain1.to_dense() @ x \
                    + Ain2.to_dense() @ s_val \
                    + Ain3.to_dense() @ t_val \
                    + Ain4.to_dense() @ alpha_val \
                    + Ain5.to_dense() @ beta_vec.to(dtype=self.dtype)

            precision = 1E-10
            lhs = compute_lhs(s, t, alpha)
            self.assertTrue(torch.all(lhs < rhs + precision))

            # Perturb s a bit, the constraint shouldn't be satisfied
            lhs = compute_lhs(
                s + torch.empty(s.shape[0], 1, dtype=self.dtype).uniform_(
                    -0.1, 0.1), t, alpha)
            self.assertFalse(torch.all(lhs < rhs + precision))
            # Now increase t a bit, the constraint shouldn't be satisfied.
            lhs = compute_lhs(s, t + 0.1, alpha)
            self.assertFalse(torch.all(lhs < rhs + precision))
            # Perturb alpha a bit, the constraint shouldn't be satisfied.
            alpha_perturbed = alpha.clone()
            alpha_perturbed_index = np.random.randint(0, alpha.numel())
            alpha_perturbed[alpha_perturbed_index] = 1. - \
                alpha_perturbed[alpha_perturbed_index]
            lhs = compute_lhs(s, t, alpha_perturbed)
            self.assertFalse(torch.all(lhs < rhs + precision))

            # Now compute the expected cost
            cost_expected = alpha.T @ M @ A_dyn @ x + \
                torch.min(alpha.T @ M @ B_dyn @ u_vertices) + \
                alpha.T @ M @ d_dyn
            cost = c1.T @ s + t + c2.T @ alpha
            self.assertAlmostEqual(cost.item(), cost_expected.item(), 4)

        test_generate_program_util(
            torch.tensor([[0.], [0.]], dtype=self.dtype),
            [[1, 1, 0], [0, 0, 1, 1]])
        test_generate_program_util(
            torch.tensor([[1.], [2.]], dtype=self.dtype),
            [[1, 1, 0], [0, 0, 1, 1]])
        test_generate_program_util(
            torch.tensor([[-1.2], [-0.9]], dtype=self.dtype),
            [[1, 0, 1], [0, 1, 0, 0]])


if __name__ == '__main__':
    unittest.main()
