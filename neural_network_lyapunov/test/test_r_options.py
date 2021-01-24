import neural_network_lyapunov.r_options as r_options
import neural_network_lyapunov.utils as utils
import torch
import numpy as np
import unittest


class TestSearchRwithSPDOptions(unittest.TestCase):
    def test_constructor(self):
        dut = r_options.SearchRwithSPDOptions((3, 2), 0.01)
        self.assertEqual(dut._variables.shape, (5, ))
        self.assertTrue(dut._variables.requires_grad)
        self.assertEqual(dut.epsilon, 0.01)
        self.assertFalse(dut.fixed_R)

        # Test a square R.
        dut = r_options.SearchRwithSPDOptions((2, 2), 0.01)
        self.assertEqual(dut._variables.shape, (3, ))

    def test_R(self):
        dut = r_options.SearchRwithSPDOptions((3, 2), 0.01)
        dut._variables = torch.tensor([0.2, 0.3, 0.5, 1.2, 1.5],
                                      dtype=torch.float64,
                                      requires_grad=True)
        self.assertTrue(dut._variables.requires_grad)
        R = dut.R()
        L_expected = np.array([[0.2, 0], [0.3, 0.5]])
        R_expected = np.vstack(
            (L_expected @ L_expected.T + dut.epsilon * np.eye(2),
             np.array([[1.2, 1.5]])))
        np.testing.assert_allclose(R_expected, R.detach().numpy())

        # Now check the gradient
        (R.trace() + R[2:, :].sum()).backward()
        np.testing.assert_allclose(dut._variables.grad[:3].detach().numpy(),
                                   2 * dut._variables[:3].detach().numpy())
        np.testing.assert_allclose(dut._variables.grad[3:].detach().numpy(),
                                   np.ones((2, )))

    def test_variables(self):
        dut = r_options.SearchRwithSPDOptions((3, 2), 0.01)
        variables = dut.variables()
        self.assertIsInstance(variables, list)
        self.assertEqual(len(variables), 1)
        self.assertEqual(variables[0].shape, (5, ))

    def test_set_variable_value(self):
        dut = r_options.SearchRwithSPDOptions((4, 2), 0.01)
        L_val = np.array([[0.1, 0], [0.5, 1.2]])
        R_val = np.vstack((L_val @ L_val.T + dut.epsilon * np.eye(2),
                           np.array([[0.5, -0.3], [0.2, 1.4]])))
        dut.set_variable_value(R_val)
        R = dut.R()
        np.testing.assert_allclose(R.detach().numpy(), R_val)
        self.assertTrue(dut._variables.requires_grad)


class TestFixedROptions(unittest.TestCase):
    def test(self):
        R_val = torch.tensor([[1., 0.], [0., 1.], [1., 1.]],
                             dtype=torch.float64)
        dut = r_options.FixedROptions(R_val)
        self.assertTrue(dut.fixed_R)
        np.testing.assert_allclose(dut.R().detach().numpy(),
                                   R_val.detach().numpy())
        variables = dut.variables()
        self.assertIsInstance(variables, list)
        self.assertEqual(len(variables), 0)


class TestSearchRwithSVDOptions(unittest.TestCase):
    def test_constructor(self):
        dut = r_options.SearchRwithSVDOptions((3, 2), np.array([0.2, 0.4]))
        self.assertEqual(dut.R_size, (3, 2))
        self.assertFalse(dut.fixed_R)
        self.assertEqual(dut._variables.shape, (2, ))

    def test_set_variable_value(self):
        R_val = np.array([[0.5, 1.3], [2.4, 0.3], [2.1, -3.2]])
        U, Sigma, V = np.linalg.svd(R_val)
        dut = r_options.SearchRwithSVDOptions((3, 2), Sigma / 2)
        dut.set_variable_value(R_val)
        np.testing.assert_allclose(U, dut.U)
        np.testing.assert_allclose(V, dut.V)
        np.testing.assert_allclose(Sigma / 2, dut.a)
        np.testing.assert_allclose(R_val, dut.R().detach().numpy())

        # test gradient of variables.

        def evaluator(v):
            return np.sum(dut.U @ np.vstack(
                (np.diag(dut.a + v**2), np.zeros((1, 2)))) @ dut.V)

        dut.R().sum().backward()
        var_gradient = dut._variables.grad.clone()

        var_gradient_numerical = utils.compute_numerical_gradient(
            evaluator,
            dut._variables.detach().numpy())
        np.testing.assert_allclose(var_gradient, var_gradient_numerical)


if __name__ == "__main__":
    unittest.main()
