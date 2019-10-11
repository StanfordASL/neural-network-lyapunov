from context import ReLUToOptimization
import unittest
import numpy as np
import torch
import torch.nn as nn
import cvxpy as cp


class TestReLU(unittest.TestCase):
    def setUp(self):
        self.dtype = torch.float64
        self.linear1 = nn.Linear(2, 3)
        self.linear1.weight.data = torch.tensor(
            [[1, 2], [3, 4], [5, 6]], dtype=self.dtype)
        self.linear1.bias.data = torch.tensor(
            [-11, 13, 4], dtype=self.dtype)
        self.linear2 = nn.Linear(3, 4)
        self.linear2.weight.data = torch.tensor(
            [[-1, 0.5, 1.5], [2, 5, 6], [-2, -3, -4], [1, 4, 6]],
            dtype=self.dtype)
        self.linear2.bias.data = torch.tensor(
            [4, -1, -2, -20], dtype=self.dtype)
        self.linear3 = nn.Linear(4, 1)
        self.linear3.weight.data = torch.tensor(
            [[4, 5, 6, 7]], dtype=self.dtype)
        self.linear3.bias.data = torch.tensor([-10], dtype=self.dtype)
        self.model = nn.Sequential(self.linear1, nn.ReLU(), self.linear2,
                                   nn.ReLU(),
                                   self.linear3)

    def test_compute_relu_activation_pattern(self):
        x = torch.tensor([-6, 4], dtype=self.dtype)
        activation_pattern = ReLUToOptimization.ComputeReLUActivationPattern(
            self.model, x)
        self.assertEqual(len(activation_pattern), 2)
        self.assertEqual(len(activation_pattern[0]), 3)
        self.assertEqual(len(activation_pattern[1]), 4)
        x_linear1 = self.linear1.forward(x)
        x_relu1 = nn.ReLU().forward(x_linear1)
        for i in range(3):
            self.assertEqual(x_linear1[i] >= 0, activation_pattern[0][i])
        x_linear2 = self.linear2.forward(x_relu1)
        for i in range(4):
            self.assertEqual(x_linear2[i] >= 0, activation_pattern[1][i])

    def test_relu_given_activation_pattern(self):
        def test_relu_given_activation_pattern_util(self, x):
            activation_pattern = ReLUToOptimization.\
                ComputeReLUActivationPattern(self.model, x)
            (g, h, P, q) = ReLUToOptimization.ReLUGivenActivationPattern(
                self.model, 2, activation_pattern, self.dtype)
            output_expected = self.model.forward(x)
            output = g.T @ x.reshape((2, 1)) + h
            self.assertAlmostEqual(output, output_expected, 10)
            self.assertTrue(torch.all(torch.le(P @ (x.reshape((-1, 1))), q)))

        test_relu_given_activation_pattern_util(
            self, torch.tensor([-6, 4], dtype=self.dtype))
        test_relu_given_activation_pattern_util(
            self, torch.tensor([-10, 4], dtype=self.dtype))
        test_relu_given_activation_pattern_util(
            self, torch.tensor([3, -4], dtype=self.dtype))
        test_relu_given_activation_pattern_util(
            self, torch.tensor([-3, -4], dtype=self.dtype))

    def test_relu_free_pattern_constructor(self):
        relu_free_pattern = ReLUToOptimization.ReLUFreePattern(self.model,
                                                               self.dtype)
        self.assertEqual(len(relu_free_pattern.relu_unit_index), 2)
        self.assertListEqual(relu_free_pattern.relu_unit_index[0], [0, 1, 2])
        self.assertListEqual(
            relu_free_pattern.relu_unit_index[1], [3, 4, 5, 6])
        self.assertEqual(relu_free_pattern.num_relu_units, 7)

    def test_relu_free_pattern_output_constraint(self):
        relu_free_pattern = ReLUToOptimization.ReLUFreePattern(self.model,
                                                               self.dtype)
        x_lo = torch.tensor([-1, -2], dtype=self.dtype)
        x_up = torch.tensor([2, 3], dtype=self.dtype)
        (Ain1, Ain2, Ain3, rhs_in, Aeq1, Aeq2, Aeq3, rhs_eq, a_out, b_out,
            z_lo, z_up) = relu_free_pattern.output_constraint(
            self.model, x_lo, x_up)
        print("z_lo:{}\nz_up:{}".format(z_lo, z_up))
        num_z_lo_positive = np.sum([z_lo_i >= 0 for z_lo_i in z_lo])
        num_z_up_negative = np.sum([z_up_i <= 0 for z_up_i in z_up])
        num_ineq = (relu_free_pattern.num_relu_units -
                    num_z_lo_positive - num_z_up_negative) * 4
        num_eq = (num_z_lo_positive + num_z_up_negative) * 2
        self.assertListEqual(
            list(Ain1.shape), [num_ineq, 2])
        self.assertListEqual(list(Ain2.shape), [
                             num_ineq, relu_free_pattern.num_relu_units])
        self.assertListEqual(list(Ain3.shape), [
                             num_ineq, relu_free_pattern.num_relu_units])
        self.assertListEqual(
            list(rhs_in.shape), [num_ineq, 1])
        self.assertListEqual(
            list(Aeq1.shape), [num_eq, 2])
        self.assertListEqual(list(Aeq2.shape), [
                             num_eq, relu_free_pattern.num_relu_units])
        self.assertListEqual(list(Aeq3.shape), [
                             num_eq, relu_free_pattern.num_relu_units])
        self.assertListEqual(
            list(rhs_eq.shape), [num_eq, 1])

        def test_input_x(x):
            # For an arbitrary input x, compute its activation pattern and
            # output of each ReLU unit, check if they satisfy the constraint
            # Ain1*x+Ain2*z+Ain3*β <= rhs_in
            # Aeq1*x+Aeq2*z+Aeq3*β <= rhs_eq
            assert(torch.all(torch.ge(x, x_lo.squeeze())))
            assert(torch.all(torch.le(x, x_up.squeeze())))
            (z, beta, output) = \
                relu_free_pattern.compute_relu_unit_outputs_and_activation(
                self.model, x)
            for i in range(relu_free_pattern.num_relu_units):
                self.assertTrue(
                    torch.le(z[i][0],
                             torch.max(z_up[i],
                                       torch.tensor(0., dtype=self.dtype))))
                self.assertTrue(
                    torch.ge(z[i][0],
                             torch.max(z_lo[i],
                                       torch.tensor(0., dtype=self.dtype))))
            # Check the output
            self.assertAlmostEqual(output, (a_out.T @ z + b_out).item(), 3)
            x_vec = x.reshape((-1, 1))
            lhs_in = Ain1 @ x_vec + Ain2 @z + Ain3 @ beta
            lhs_eq = Aeq1 @ x_vec + Aeq2 @z + Aeq3 @ beta
            precision = 1E-10
            self.assertTrue(torch.all(
                torch.le(lhs_in.squeeze(),
                         rhs_in.squeeze() + torch.tensor(precision))))
            self.assertTrue(
                torch.all(torch.le(torch.abs(lhs_eq.squeeze()
                                             - rhs_eq.squeeze()), precision)))
            # Now perturb beta by changing some entry from 1 to 0, and vice
            # versa. Now it should not satisfy the constraint.
            perturbed_beta_entry = np.random.randint(0, beta.numel())
            beta_perturbed = beta.clone()
            beta_perturbed[perturbed_beta_entry] =\
                1 - beta[perturbed_beta_entry]
            lhs_in_perturbed = Ain1 @ x_vec + Ain2 @ z + Ain3 @ beta_perturbed
            lhs_eq_perturbed = Aeq1 @ x_vec + Aeq2 @ z + Aeq3 @ beta_perturbed
            self.assertFalse(torch.all(
                torch.le(lhs_in_perturbed.squeeze(),
                         rhs_in.squeeze() + torch.tensor(precision))) and
                torch.all(torch.le(torch.abs(lhs_eq_perturbed - rhs_eq),
                                   precision)))

            # Now formulate an optimization problem, with fixed input, search
            # for z and beta. There should be only one solution.
            z_var = cp.Variable(relu_free_pattern.num_relu_units)
            beta_var = cp.Variable(relu_free_pattern.num_relu_units,
                                   boolean=True)
            x_np = x.detach().numpy()
            con = [Ain1.detach().numpy() @ x_np +
                   Ain2.detach().numpy() @ z_var +
                   Ain3.detach().numpy() @ beta_var <=
                   rhs_in.squeeze().detach().numpy(),
                   Aeq1.detach().numpy() @ x_np +
                   Aeq2.detach().numpy() @ z_var +
                   Aeq3.detach().numpy() @ beta_var ==
                   rhs_eq.squeeze().detach().numpy()]
            objective = cp.Minimize(0.)
            prob = cp.Problem(objective, con)
            prob.solve(solver=cp.GUROBI)
            z_opt_var = z_var.value
            beta_opt_var = beta_var.value
            self.assertTrue(np.all(np.abs(
                z_opt_var - z.squeeze().detach().numpy()) < 1E-5))
            self.assertTrue(np.all(np.abs(
                beta_opt_var - beta.squeeze().detach().numpy()) < 1E-5))

        # Test with different input x.
        test_input_x(torch.tensor([0.7, 0.2], dtype=self.dtype))
        test_input_x(torch.tensor([-0.3, 0.2], dtype=self.dtype))
        test_input_x(torch.tensor([-0.15, -0.2], dtype=self.dtype))
        test_input_x(torch.tensor([1.1, -0.22], dtype=self.dtype))
        test_input_x(torch.tensor([1.5, -0.8], dtype=self.dtype))

    def test_compute_alpha_index(self):
        relu_free_pattern = ReLUToOptimization.\
            ReLUFreePattern(self.model, self.dtype)
        self.assertEqual(relu_free_pattern.compute_alpha_index((0, 0)), 0)
        self.assertEqual(relu_free_pattern.compute_alpha_index((0, 1)), 1)
        self.assertEqual(relu_free_pattern.compute_alpha_index((0, 2)), 2)
        self.assertEqual(relu_free_pattern.compute_alpha_index((0, 3)), 3)
        self.assertEqual(relu_free_pattern.compute_alpha_index((1, 0)), 4)
        self.assertEqual(relu_free_pattern.compute_alpha_index((1, 1)), 5)
        self.assertEqual(relu_free_pattern.compute_alpha_index((1, 2)), 6)
        self.assertEqual(relu_free_pattern.compute_alpha_index((1, 3)), 7)
        self.assertEqual(relu_free_pattern.compute_alpha_index((2, 0)), 8)
        self.assertEqual(relu_free_pattern.compute_alpha_index((2, 1)), 9)
        self.assertEqual(relu_free_pattern.compute_alpha_index((2, 2)), 10)
        self.assertEqual(relu_free_pattern.compute_alpha_index((2, 3)), 11)

    def test_output_gradient(self):
        relu_free_pattern = ReLUToOptimization.\
            ReLUFreePattern(self.model, self.dtype)
        (M, B1, B2, d) = relu_free_pattern.output_gradient(self.model)
        num_alpha = 12
        self.assertListEqual(list(M.shape), [num_alpha, 2])

        # Enumerate all the possible activation pattern, with only one ReLU
        # unit active at each layer. Compute the gradient of the ReLU
        # network network for each activation pattern through
        # ReLUGivenActivationPattern(), and compare the result aginst M.
        activation_pattern = [[False, False, False],
                              [False, False, False, False]]
        precision = 1E-10
        for i0 in range(3):
            activation_pattern[0] = [False, False, False]
            activation_pattern[0][i0] = True
            for i1 in range(4):
                activation_pattern[1] = [False, False, False, False]
                activation_pattern[1][i1] = True
                (g, _, _, _) = ReLUToOptimization.ReLUGivenActivationPattern(
                    self.model, 2, activation_pattern, self.dtype)
                alpha_index = relu_free_pattern.compute_alpha_index((i0, i1))
                self.assertTrue(
                    torch.all(torch.abs(M[alpha_index] - g.reshape((1, -1)))
                              < precision))
                alpha_value = torch.zeros((num_alpha, 1), dtype=self.dtype)
                alpha_value[alpha_index][0] = 1.
                beta_value = torch.zeros((relu_free_pattern.num_relu_units,
                                          1), dtype=self.dtype)
                beta_value[relu_free_pattern.relu_unit_index[0][i0]][0] = 1.
                beta_value[relu_free_pattern.relu_unit_index[1][i1]][0] = 1.
                self.assertTrue(
                    torch.all(B1 @ alpha_value + B2 @beta_value - d <
                              precision))
                # Now perturb alpha value a bit, by negating a value from 1
                # to 0 or vice versa, the perturbed alpha and beta should
                # violate the constraint.
                perturbed_alpha_entry = np.random.randint(
                    0, alpha_value.numel())
                alpha_value[perturbed_alpha_entry] = 1. - \
                    alpha_value[perturbed_alpha_entry]
                self.assertFalse(
                    torch.all(B1 @ alpha_value + B2 @beta_value - d <
                              precision))


if __name__ == "__main__":
    unittest.main()
