import ReLUToOptimization
import unittest
import numpy as np
import torch
import torch.nn as nn


class TestReLU(unittest.TestCase):
    def setUp(self):
        self.datatype = torch.float
        self.linear1 = nn.Linear(2, 3)
        self.linear1.weight.data = torch.tensor(
            [[1, 2], [3, 4], [5, 6]], dtype=self.datatype)
        self.linear1.bias.data = torch.tensor(
            [-11, 13, 4], dtype=self.datatype)
        self.linear2 = nn.Linear(3, 4)
        self.linear2.weight.data = torch.tensor(
            [[-1, 0.5, 1.5], [2, 5, 6], [-2, -3, -4], [1, 4, 6]],
            dtype=self.datatype)
        self.linear2.bias.data = torch.tensor(
            [4, -1, -2, 3], dtype=self.datatype)
        self.linear3 = nn.Linear(4, 1)
        self.linear3.weight.data = torch.tensor(
            [4, 5, 6, 7], dtype=self.datatype)
        self.linear3.bias.data = torch.tensor(-10, dtype=self.datatype)
        self.model = nn.Sequential(self.linear1, nn.ReLU(), self.linear2,
                                   nn.ReLU(),
                                   self.linear3)

    def test_compute_relu_activation_path(self):
        x = torch.tensor([-6, 4], dtype=self.datatype)
        activation_path = ReLUToOptimization.ComputeReLUActivationPath(
            self.model, x)
        self.assertEqual(len(activation_path), 2)
        self.assertEqual(len(activation_path[0]), 3)
        self.assertEqual(len(activation_path[1]), 4)
        x_linear1 = self.linear1.forward(x)
        x_relu1 = nn.ReLU().forward(x_linear1)
        for i in range(3):
            self.assertEqual(x_linear1[i] >= 0, activation_path[0][i])
        x_linear2 = self.linear2.forward(x_relu1)
        for i in range(4):
            self.assertEqual(x_linear2[i] >= 0, activation_path[1][i])

    def test_relu_given_activation_path(self):
        def test_relu_given_activation_path_util(self, x):
            activation_path = ReLUToOptimization.ComputeReLUActivationPath(
                self.model, x)
            (g, h, P, q) = ReLUToOptimization.ReLUGivenActivationPath(
                self.model, 2, activation_path)
            output_expected = self.model.forward(x)
            output = g.T @ x.reshape((2, 1)) + h
            self.assertAlmostEqual(output, output_expected, 10)
            self.assertTrue(torch.all(torch.le(P @ (x.reshape((-1, 1))), q)))

        test_relu_given_activation_path_util(
            self, torch.tensor([-6, 4], dtype=self.datatype))
        test_relu_given_activation_path_util(
            self, torch.tensor([-10, 4], dtype=self.datatype))
        test_relu_given_activation_path_util(
            self, torch.tensor([3, -4], dtype=self.datatype))
        test_relu_given_activation_path_util(
            self, torch.tensor([-3, -4], dtype=self.datatype))

    def test_relu_free_path_constructor(self):
        relu_free_path = ReLUToOptimization.ReLUFreePath(self.model)
        self.assertEqual(len(relu_free_path.relu_unit_index), 2)
        self.assertListEqual(relu_free_path.relu_unit_index[0], [0, 1, 2])
        self.assertListEqual(relu_free_path.relu_unit_index[1], [3, 4, 5, 6])
        self.assertEqual(relu_free_path.num_relu_units, 7)

    def test_relu_free_path_output_constraint(self):
        relu_free_path = ReLUToOptimization.ReLUFreePath(self.model)
        x_lo = torch.tensor([-1, -2], dtype=self.datatype)
        x_up = torch.tensor([2, 3], dtype=self.datatype)
        (Ain1, Ain2, Ain3, rhs_in, Aeq1, Aeq2, Aeq3, rhs_eq, a_out, b_out, z_lo, z_up) = relu_free_path.output_constraint(
            self.model, x_lo, x_up)
        print("z_lo:{}\nz_up:{}".format(z_lo, z_up))
        num_z_lo_positive = np.sum([z_lo_i >= 0 for z_lo_i in z_lo])
        num_z_up_negative = np.sum([z_up_i <= 0 for z_up_i in z_up])
        num_ineq = (relu_free_path.num_relu_units -
                    num_z_lo_positive - num_z_up_negative) * 4
        num_eq = (num_z_lo_positive + num_z_up_negative) * 2
        self.assertListEqual(
            list(Ain1.shape), [num_ineq, 2])
        self.assertListEqual(list(Ain2.shape), [
                             num_ineq, relu_free_path.num_relu_units])
        self.assertListEqual(list(Ain3.shape), [
                             num_ineq, relu_free_path.num_relu_units])
        self.assertListEqual(
            list(rhs_in.shape), [num_ineq, 1])
        self.assertListEqual(
            list(Aeq1.shape), [num_eq, 2])
        self.assertListEqual(list(Aeq2.shape), [
                             num_eq, relu_free_path.num_relu_units])
        self.assertListEqual(list(Aeq3.shape), [
                             num_eq, relu_free_path.num_relu_units])
        self.assertListEqual(
            list(rhs_eq.shape), [num_eq, 1])

        def test_input_x(x):
            # For an arbitrary input x, compute its activation path and output
            # of each ReLU unit, check if they satisfy the constraint
            # Ain1*x+Ain2*z+Ain3*β <= rhs_in
            # Aeq1*x+Aeq2*z+Aeq3*β <= rhs_eq
            assert(torch.all(torch.ge(x, x_lo.squeeze())))
            assert(torch.all(torch.le(x, x_up.squeeze())))
            (z, beta, output) = relu_free_path.ComputeReLUUnitOutputsAndActivation(
                self.model, x)
            for i in range(relu_free_path.num_relu_units):
                self.assertTrue(
                    torch.le(z[i][0], torch.max(z_up[i], torch.tensor(0.))))
                self.assertTrue(
                    torch.ge(z[i][0], torch.max(z_lo[i], torch.tensor(0.))))
            # Check the output
            self.assertAlmostEqual(output, (a_out.T @ z + b_out).item(), 3)
            x_vec = x.reshape((-1, 1))
            lhs_in = Ain1 @ x_vec + Ain2 @z + Ain3 @ beta
            lhs_eq = Aeq1 @ x_vec + Aeq2 @z + Aeq3 @ beta
            precision = 1E-5
            self.assertTrue(torch.all(
                torch.le(lhs_in.squeeze(),
                         rhs_in.squeeze() + torch.tensor(precision))))
            self.assertTrue(
                torch.all(torch.le(torch.abs(lhs_eq.squeeze() - rhs_eq.squeeze()), precision)))
            # Now perturb beta by changing some entry from 1 to 0, and vice versa.
            # Now it should not satisfy the constraint.
            perturbed_beta_entry = np.random.randint(0, beta.numel())
            beta[perturbed_beta_entry] = 1 - beta[perturbed_beta_entry]
            lhs_in_perturbed = Ain1 @ x_vec + Ain2 @ z + Ain3 @ beta
            lhs_eq_perturbed = Aeq1 @ x_vec + Aeq2 @ z + Aeq3 @ beta
            self.assertFalse(torch.all(
                torch.le(lhs_in_perturbed.squeeze(),
                         rhs_in.squeeze() + torch.tensor(precision))) and
                torch.all(torch.le(torch.abs(lhs_eq_perturbed - rhs_eq), precision)))

        # Test with different input x.
        test_input_x(torch.tensor([0.7, 0.2]))
        test_input_x(torch.tensor([-0.3, 0.2]))
        test_input_x(torch.tensor([-0.15, -0.2]))
        test_input_x(torch.tensor([1.1, -0.22]))
        test_input_x(torch.tensor([1.5, -0.8]))


if __name__ == "__main__":
    unittest.main()
