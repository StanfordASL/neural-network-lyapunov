import neural_network_lyapunov.train_utils as train_utils

import unittest

import torch
import numpy as np


def setup_relu(relu_layer_width, params):
    assert (isinstance(relu_layer_width, tuple))
    dtype = torch.float64

    def set_param(linear, param_count):
        linear.weight.data = params[param_count:param_count +
                                    linear.in_features *
                                    linear.out_features].clone().reshape(
                                        (linear.out_features,
                                         linear.in_features))
        param_count += linear.in_features * linear.out_features
        linear.bias.data = params[param_count:param_count +
                                  linear.out_features].clone()
        param_count += linear.out_features
        return param_count

    linear_layers = [None] * len(relu_layer_width)
    param_count = 0
    for i in range(len(relu_layer_width)):
        next_layer_width = relu_layer_width[i+1] if \
            i < len(relu_layer_width)-1 else 1
        linear_layers[i] = torch.nn.Linear(relu_layer_width[i],
                                           next_layer_width).type(dtype)
        if params is None:
            pass
        else:
            param_count = set_param(linear_layers[i], param_count)
    layers = [None] * (len(relu_layer_width) * 2 - 1)
    for i in range(len(relu_layer_width) - 1):
        layers[2 * i] = linear_layers[i]
        layers[2 * i + 1] = torch.nn.LeakyReLU(0.2)
    layers[-1] = linear_layers[-1]
    relu = torch.nn.Sequential(*layers)
    return relu


def test_project_gradient(relu, loss1, loss2, mode):
    for p in relu.parameters():
        if p.grad is not None:
            p.grad.data.zero_()
    loss1.backward(retain_graph=True)
    n1 = torch.cat([p.grad.clone().reshape((-1, )) for p in relu.parameters()])
    for p in relu.parameters():
        if p.grad is not None:
            p.grad.data.zero_()
    loss2.backward(retain_graph=True)
    n2 = torch.cat([p.grad.clone().reshape((-1, )) for p in relu.parameters()])
    for p in relu.parameters():
        if p.grad is not None:
            p.grad.data.zero_()
    need_projection, n1, n2 = train_utils.project_gradient(relu,
                                                           loss1,
                                                           loss2,
                                                           mode,
                                                           retain_graph=True)
    grad = torch.cat(
        [p.grad.clone().reshape((-1, )) for p in relu.parameters()])
    if n1 @ n2 < 0:
        np.testing.assert_equal(need_projection, True)
        n1_perp = n1 - n1 @ n2 / (n2 @ n2) * n2
        n2_perp = n2 - n1 @ n2 / (n1 @ n1) * n1
        if mode == train_utils.ProjectGradientMode.LOSS1:
            np.testing.assert_almost_equal((grad @ n2).item(), 0)
            np.testing.assert_allclose((n1 - grad), n1 @ n2 / (n2 @ n2) * n2)
            np.testing.assert_allclose(grad, n1_perp)
        elif mode == train_utils.ProjectGradientMode.LOSS2:
            np.testing.assert_almost_equal((grad @ n1).item(), 0)
            np.testing.assert_allclose((n2 - grad), n1 @ n2 / (n1 @ n1) * n1)
            np.testing.assert_allclose(grad, n2_perp)
        elif mode == train_utils.ProjectGradientMode.BOTH:
            np.testing.assert_almost_equal(grad @ n1, n1_perp @ n1_perp)
            np.testing.assert_almost_equal(grad @ n2, n2_perp @ n2_perp)
            np.testing.assert_allclose(grad, n1_perp + n2_perp)
        elif mode == train_utils.ProjectGradientMode.EMPHASIZE_LOSS1:
            np.testing.assert_allclose(grad, n1 + n2_perp)
        elif mode == train_utils.ProjectGradientMode.EMPHASIZE_LOSS2:
            np.testing.assert_allclose(grad, n2 + n1_perp)
        else:
            raise Exception()
    else:
        np.testing.assert_equal(need_projection, False)
        np.testing.assert_allclose(grad, n1 + n2)


class TestProjectGradient(unittest.TestCase):
    def test1(self):
        dtype = torch.float64
        relu1 = setup_relu((2, 3),
                           torch.tensor([
                               0.1, 0.2, 0.3, -0.1, 2.1, 3.2, 0.5, -0.2, 4.5,
                               1.4, 0.5, 2.5, -2.3
                           ],
                                        dtype=dtype))
        relu2 = setup_relu((2, 4),
                           torch.tensor([
                               0.1, 0.2, 0.3, -0.1, 2.1, 3.2, 0.5, -0.2, 4.5,
                               1.4, 0.5, 2.5, -2.3, 4.2, 0.3, 1.5, -0.3
                           ],
                                        dtype=dtype))
        x = torch.tensor([2.0, 1.5], dtype=dtype)
        for relu in (relu1, relu2):
            y = relu(x)
            loss1 = y * y
            loss2 = y - y * y
            loss3 = y + y * y
            for mode in list(train_utils.ProjectGradientMode):
                # The gradient of loss 1 and loss 2 should have angle > 90
                # degrees.
                test_project_gradient(relu, loss1, loss2, mode)
                # The gradient of loss 1 and loss 3 should have angle < 90
                # degrees.
                test_project_gradient(relu, loss1, loss3, mode)
                test_project_gradient(relu, loss2, loss3, mode)

            for mode in (train_utils.ProjectGradientMode.BOTH,
                         train_utils.ProjectGradientMode.LOSS1,
                         train_utils.ProjectGradientMode.LOSS2):
                # Now project the gradient of loss1 and -loss1, they have
                # exact opposite gradient, so the projected gradient is 0.
                train_utils.project_gradient(relu,
                                             loss1,
                                             -loss1,
                                             mode,
                                             retain_graph=True)
                grad = torch.cat(
                    [p.grad.reshape((-1, )) for p in relu.parameters()])
                np.testing.assert_allclose(grad.detach().numpy(),
                                           np.zeros(grad.shape),
                                           atol=3e-13)


if __name__ == "__main__":
    unittest.main()
