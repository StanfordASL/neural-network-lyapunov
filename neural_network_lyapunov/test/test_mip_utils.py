import neural_network_lyapunov.mip_utils as mip_utils

import unittest

import torch
import numpy as np
import neural_network_lyapunov.utils as utils


class TestComputeRangeByLP(unittest.TestCase):
    def test_x_bounds(self):
        # Test with only bounds on x.
        x_lb = np.array([1., 2.])
        x_ub = np.array([3., 4.])
        A = np.array([[0., 1.], [1., 1.], [2., -1.]])
        b = np.array([0., 1., 3.])
        y_lb, y_ub = mip_utils.compute_range_by_lp(A, b, x_lb, x_ub, None,
                                                   None)
        y_lb_expected = np.array([2., 4, 1.])
        y_ub_expected = np.array([4, 8., 7.])
        np.testing.assert_allclose(y_lb, y_lb_expected, atol=1E-7)
        np.testing.assert_allclose(y_ub, y_ub_expected, atol=1E-7)

        # Some of the bounds should be infinity.
        x_lb = np.array([1., -np.inf])
        y_lb, y_ub = mip_utils.compute_range_by_lp(A, b, x_lb, x_ub, None,
                                                   None)
        y_lb_expected = np.array([-np.inf, -np.inf, 1.])
        y_ub_expected = np.array([4, 8., np.inf])
        np.testing.assert_allclose(y_lb, y_lb_expected, atol=1E-7)
        np.testing.assert_allclose(y_ub, y_ub_expected, atol=1E-7)

        # Now the problem is infeasible as x_lb > x_ub
        x_lb = np.array([4., 2])
        y_lb, y_ub = mip_utils.compute_range_by_lp(A, b, x_lb, x_ub, None,
                                                   None)
        np.testing.assert_allclose(y_lb, np.full((3, ), np.inf))
        np.testing.assert_allclose(y_ub, np.full((3, ), -np.inf))

    def test_ineq_bounds(self):
        # x has both bounds x_lb <= x <= x_ub and inequality bounds C * x <= d
        x_lb = np.array([-1., -2.])
        x_ub = np.array([3., 4.])
        C = np.array([[1., 1.], [1., -1], [-1, 1], [-1, -1.]])
        d = np.array([2, 2, 2, 2])
        A = np.array([[0., 1.], [1., 1.], [2., -1.]])
        b = np.array([0., 1., 3.])
        y_lb, y_ub = mip_utils.compute_range_by_lp(A, b, x_lb, x_ub, C, d)
        y_lb_expected = np.array([-2, -1, 0.])
        y_ub_expected = np.array([2., 3, 7.])
        np.testing.assert_allclose(y_lb, y_lb_expected, atol=1E-7)
        np.testing.assert_allclose(y_ub, y_ub_expected, atol=1E-7)


class TestComputeRangeByIA(unittest.TestCase):
    def test(self):
        dtype = torch.float64
        A = torch.tensor([[1., 2., -3.], [2., -1., -4.]], dtype=dtype)
        b = torch.tensor([2., 3.], dtype=dtype)
        x_lb = torch.tensor([-2., 3., -4.], dtype=dtype)
        x_ub = torch.tensor([1., 5., 7.], dtype=dtype)
        output_lb, output_ub = mip_utils.compute_range_by_IA(A, b, x_lb, x_ub)
        np.testing.assert_allclose(output_lb.detach().numpy(),
                                   np.array([-15, -34]))
        np.testing.assert_allclose(output_ub.detach().numpy(),
                                   np.array([25, 18]))

    def test_gradient(self):
        def test_fun(A_np, b_np, x_lb_np, x_ub_np):
            output_lb, output_ub = mip_utils.compute_range_by_IA(
                torch.from_numpy(A_np.reshape((2, 3))), torch.from_numpy(b_np),
                torch.from_numpy(x_lb_np), torch.from_numpy(x_ub_np))
            return torch.sum(output_lb * output_ub).item()

        dtype = torch.float64
        A = torch.tensor([[1., 2., -3.], [2., -1., -4.]],
                         dtype=dtype,
                         requires_grad=True)
        b = torch.tensor([2., 3.], dtype=dtype, requires_grad=True)
        x_lb = torch.tensor([-2., 3., -4.], dtype=dtype, requires_grad=True)
        x_ub = torch.tensor([1., 5., 7.], dtype=dtype, requires_grad=True)
        output_lb, output_ub = mip_utils.compute_range_by_IA(A, b, x_lb, x_ub)
        torch.sum(output_lb * output_ub).backward()
        A_grad = A.grad.detach().numpy()
        b_grad = b.grad.detach().numpy()
        x_lb_grad = x_lb.grad.detach().numpy()
        x_ub_grad = x_ub.grad.detach().numpy()

        A_grad_numerical, b_grad_numerical, x_lb_grad_numerical,\
            x_ub_grad_numerical = utils.compute_numerical_gradient(
                test_fun, A.detach().numpy().reshape((-1)), b.detach().numpy(),
                x_lb.detach().numpy(), x_ub.detach().numpy())
        np.testing.assert_allclose(A_grad,
                                   A_grad_numerical.reshape((2, 3)),
                                   atol=1E-6)
        np.testing.assert_allclose(b_grad, b_grad_numerical, atol=1E-6)
        np.testing.assert_allclose(x_lb_grad, x_lb_grad_numerical, atol=1E-6)
        np.testing.assert_allclose(x_ub_grad, x_ub_grad_numerical, atol=1E-6)


class TestPropagateBounds(unittest.TestCase):
    def test_relu(self):
        layer = torch.nn.ReLU()
        input_lo = torch.tensor([-2., 2., -3.], dtype=torch.float64)
        input_up = torch.tensor([3., 6., -1.], dtype=torch.float64)
        output_lo, output_up = mip_utils.propagate_bounds(
            layer, input_lo, input_up, mip_utils.PropagateBoundsMethod.IA)
        np.testing.assert_allclose(output_lo, np.array([0, 2., 0.]))
        np.testing.assert_allclose(output_up, np.array([3., 6., 0.]))

    def test_leaky_relu1(self):
        # negative_slope > 0
        layer = torch.nn.LeakyReLU(0.1)
        input_lo = torch.tensor([-2., 2., -3.], dtype=torch.float64)
        input_up = torch.tensor([3., 6., -1.], dtype=torch.float64)
        output_lo, output_up = mip_utils.propagate_bounds(
            layer, input_lo, input_up, mip_utils.PropagateBoundsMethod.IA)
        np.testing.assert_allclose(output_lo, np.array([-0.2, 2., -0.3]))
        np.testing.assert_allclose(output_up, np.array([3., 6., -0.1]))

    def test_leaky_relu2(self):
        # negative_slope < 0
        layer = torch.nn.LeakyReLU(-0.1)
        input_lo = torch.tensor([-10, 2, -8, -1], dtype=torch.float64)
        input_up = torch.tensor([0.5, 8, -2, 4], dtype=torch.float64)
        output_lo, output_up = mip_utils.propagate_bounds(
            layer, input_lo, input_up, mip_utils.PropagateBoundsMethod.IA)
        np.testing.assert_allclose(output_lo, np.array([0, 2, 0.2, 0]))
        np.testing.assert_allclose(output_up, np.array([1., 8., 0.8, 4]))

    def test_linear_layer_no_bias_IA(self):
        layer = torch.nn.Linear(3, 2, bias=False)
        layer.weight.data = torch.tensor([[-1, 0, 2], [3, -2, -1]],
                                         dtype=torch.float64)
        input_lo = torch.tensor([-2, 2, -3], dtype=torch.float64)
        input_up = torch.tensor([3, 6, -1], dtype=torch.float64)
        output_lo, output_up = mip_utils.propagate_bounds(
            layer, input_lo, input_up, mip_utils.PropagateBoundsMethod.IA)
        np.testing.assert_allclose(output_lo.detach().numpy(),
                                   np.array([-9., -17]))
        np.testing.assert_allclose(output_up.detach().numpy(),
                                   np.array([0., 8.]))
        # Now check we can take the gradient.
        loss = output_lo.sum() + output_up.sum()
        loss.backward()

        for s in np.linspace(0, 1, 11):
            output = layer(s * input_lo + (1 - s) * input_up)
            np.testing.assert_array_less(output.detach().numpy(),
                                         output_up.detach().numpy() + 1E-10)
            np.testing.assert_array_less(output_lo.detach().numpy() - 1E-10,
                                         output.detach().numpy())

    def test_linear_layer_bias_IA(self):
        layer = torch.nn.Linear(3, 2, bias=True)
        layer.weight.data = torch.tensor([[-1, 0, 2], [3, -2, -1]],
                                         dtype=torch.float64)
        layer.bias.data = torch.tensor([2., -1.], dtype=torch.float64)
        input_lo = torch.tensor([-2, 2, -3], dtype=torch.float64)
        input_up = torch.tensor([3, 6, -1], dtype=torch.float64)
        output_lo, output_up = mip_utils.propagate_bounds(
            layer, input_lo, input_up, mip_utils.PropagateBoundsMethod.IA)
        np.testing.assert_allclose(output_lo.detach().numpy(),
                                   np.array([-7., -18.]))
        np.testing.assert_allclose(output_up.detach().numpy(),
                                   np.array([2., 7.]))
        # Now check we can take the gradient.
        loss = output_lo.sum() + output_up.sum()
        loss.backward()

        for s in np.linspace(0, 1, 11):
            output = layer(s * input_lo + (1 - s) * input_up)
            np.testing.assert_array_less(output.detach().numpy(),
                                         output_up.detach().numpy() + 1E-10)
            np.testing.assert_array_less(output_lo.detach().numpy() - 1E-10,
                                         output.detach().numpy())

    def test_linear_layer_no_bias_LP(self):
        layer = torch.nn.Linear(3, 2, bias=False)
        layer.weight.data = torch.tensor([[-1, 0, 2], [3, -2, -1]],
                                         dtype=torch.float64)
        input_lo = torch.tensor([-2, 2, -3], dtype=torch.float64)
        input_up = torch.tensor([3, 6, -1], dtype=torch.float64)
        output_lo, output_up = mip_utils.propagate_bounds(
            layer, input_lo, input_up, mip_utils.PropagateBoundsMethod.LP)
        output_lo_expected, output_up_expected = mip_utils.compute_range_by_lp(
            layer.weight.detach().numpy(), np.zeros((2, )),
            input_lo.detach().numpy(),
            input_up.detach().numpy(), None, None)
        np.testing.assert_allclose(output_lo.detach().numpy(),
                                   output_lo_expected)
        np.testing.assert_allclose(output_up.detach().numpy(),
                                   output_up_expected)

        for s in np.linspace(0, 1, 11):
            output = layer(s * input_lo + (1 - s) * input_up)
            np.testing.assert_array_less(output.detach().numpy(),
                                         output_up.detach().numpy() + 1E-10)
            np.testing.assert_array_less(output_lo.detach().numpy() - 1E-10,
                                         output.detach().numpy())

    def test_linear_layer_bias_LP(self):
        layer = torch.nn.Linear(3, 2, bias=True)
        layer.weight.data = torch.tensor([[-1, 0, 2], [3, -2, -1]],
                                         dtype=torch.float64)
        layer.bias.data = torch.tensor([2., -1.], dtype=torch.float64)
        input_lo = torch.tensor([-2, 2, -3], dtype=torch.float64)
        input_up = torch.tensor([3, 6, -1], dtype=torch.float64)
        output_lo, output_up = mip_utils.propagate_bounds(
            layer, input_lo, input_up, mip_utils.PropagateBoundsMethod.LP)
        output_lo, output_up = mip_utils.propagate_bounds(
            layer, input_lo, input_up, mip_utils.PropagateBoundsMethod.LP)
        output_lo_expected, output_up_expected = mip_utils.compute_range_by_lp(
            layer.weight.detach().numpy(),
            layer.bias.detach().numpy(),
            input_lo.detach().numpy(),
            input_up.detach().numpy(), None, None)
        np.testing.assert_allclose(output_lo.detach().numpy(),
                                   output_lo_expected)
        np.testing.assert_allclose(output_up.detach().numpy(),
                                   output_up_expected)

        for s in np.linspace(0, 1, 11):
            output = layer(s * input_lo + (1 - s) * input_up)
            np.testing.assert_array_less(output.detach().numpy(),
                                         output_up.detach().numpy() + 1E-10)
            np.testing.assert_array_less(output_lo.detach().numpy() - 1E-10,
                                         output.detach().numpy())


if __name__ == "__main__":
    unittest.main()
