import robust_value_approx.line_search_gd as line_search_gd

import torch
import numpy as np

import unittest


class TestLineSearchGD(unittest.TestCase):
    def test(self):
        """
        Test a simple linear network with quadratic loss.
        The loss can be written as aᵀXa + 2baᵀx+n*bᵀb, where X = ∑ᵢ xᵢxᵢᵀ,
        x = ∑ᵢ xᵢ. a, b are the weight/bias of the linear layer.
        """
        dtype = torch.float64
        network = torch.nn.Linear(2, 1).type(dtype)

        def reset_network():
            network.weight.data = torch.tensor([[1., 2.]], dtype=dtype)
            network.bias.data = torch.tensor([3.], dtype=dtype)
        reset_network()

        def param_flat():
            return torch.cat([network.weight.reshape((-1,)), network.bias])

        def grad_flat():
            return torch.cat(
                [network.weight.grad.reshape((-1,)), network.bias.grad])

        x = torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.float64)

        def loss_fun(a, b):
            y = x @ a.reshape((-1, 1)) + b
            return y.squeeze() @ y.squeeze()

        def grad_fun(a, b):
            with torch.no_grad():
                X = torch.zeros((2, 2), dtype=dtype)
                for i in range(x.shape[0]):
                    X += x[i].reshape((-1, 1)) @ x[i].reshape((1, -1))
                x_sum = torch.sum(x, axis=0)

                return torch.cat(
                    [(2 * a.reshape((1, -1)) @ X).squeeze() + 2 * b * x_sum,
                     2 * a @ x_sum + 2 * b * x.shape[0]])

        def closure():
            y = network(x)
            return y.squeeze() @ y.squeeze()

        loss0 = closure()
        network.zero_grad()
        loss0.backward()
        np.testing.assert_allclose(
            grad_flat().detach().numpy(),
            grad_fun(network.weight.data.squeeze(), network.bias).
            detach().numpy())
        # Take a single step
        lr = 1e-3
        loss_minimal_decrement = 1e-4
        dut1 = line_search_gd.LineSearchGD(
            network.parameters(), lr=lr, momentum=0, min_step_size=1e-8,
            loss_minimal_decrement=loss_minimal_decrement,
            step_size_reduction=0.2)
        a0 = network.weight.data.clone().squeeze()
        b0 = network.bias.clone()
        grad0 = grad_fun(a0, b0)
        a1 = a0 - grad0[:2] * lr
        b1 = b0 - grad0[2] * lr
        loss1 = loss_fun(a1, b1)
        assert(loss1 <= loss0 -
               loss_minimal_decrement * lr * grad0 @ grad0)
        dut1.step(closure, loss0.item())
        np.testing.assert_allclose(
            a1.detach().numpy(),
            network.weight.data.squeeze().detach().numpy())
        np.testing.assert_almost_equal(b1.item(), network.bias.data.item())

        # Now change the learning rate. With a large initial learning rate,
        # gradient descent will overshoot. It then uses line search to find
        # an appropriate step size.
        reset_network()
        lr = 0.5
        loss_minimal_decrement = 1e-4
        step_size_reduction = 0.2
        dut2 = line_search_gd.LineSearchGD(
            network.parameters(), lr=lr, momentum=0, min_step_size=1e-8,
            loss_minimal_decrement=loss_minimal_decrement,
            step_size_reduction=step_size_reduction)
        a2 = a0 - grad0[:2] * lr
        b2 = b0 - grad0[2] * lr
        loss2 = loss_fun(a2, b2)
        # Make sure using step size = 0.5 would overshoot.
        assert(loss2 > loss0)
        step_size = 0.5
        a, b = a2, b2
        while loss_fun(a, b) > loss0 + \
                step_size * loss_minimal_decrement * (-grad0 @ grad0):
            a = a0 - grad0[:2] * step_size
            b = b0 - grad0[2] * step_size
            step_size *= step_size_reduction
        loss2 = dut2.step(closure, loss0.item())

        np.testing.assert_allclose(
            network.weight.data.squeeze().detach().numpy(), a.detach().numpy())
        np.testing.assert_almost_equal(network.bias.data.item(), b.item())
        np.testing.assert_almost_equal(
            loss2.item(), loss_fun(a, b).item())

        # Change boththe learning rate and the minimal step size. Now the line
        # search stops even though the Armijo's condition is not satisfied.
        reset_network()
        lr = 0.5
        loss_minimal_decrement = 1e-4
        step_size_reduction = 0.2
        min_step_size = step_size / step_size_reduction * 1.1

        dut3 = line_search_gd.LineSearchGD(
            network.parameters(), lr=lr, momentum=0,
            min_step_size=min_step_size,
            loss_minimal_decrement=loss_minimal_decrement,
            step_size_reduction=step_size_reduction)
        loss3 = dut3.step(closure, loss0.item())
        step_size = lr
        while step_size > min_step_size:
            a = a0 - grad0[:2] * step_size
            b = b0 - grad0[2] * step_size
            step_size *= step_size_reduction
        np.testing.assert_allclose(
            network.weight.data.squeeze().detach().numpy(), a.detach().numpy())
        np.testing.assert_almost_equal(network.bias.data.item(), b.item())
        np.testing.assert_almost_equal(
            loss3.item(), loss_fun(a, b).item())
        self.assertGreater(
            loss3.item(), loss0.item() +
            (loss_minimal_decrement * step_size/step_size_reduction *
             -grad0 @ grad0).item())

        reset_network()
        lr = 0.5
        loss_minimal_decrement = 1e-4
        step_size_reduction = 0.2
        min_step_size = 1e-6

        dut4 = line_search_gd.LineSearchGD(
            network.parameters(), lr=lr, momentum=0,
            min_step_size=min_step_size,
            loss_minimal_decrement=loss_minimal_decrement,
            step_size_reduction=step_size_reduction)
        loss = closure()
        while loss > 1e-6:
            dut4.zero_grad()
            loss.backward()
            loss = dut4.step(closure, loss.item())

        np.testing.assert_allclose(
            network(x).squeeze().detach().numpy(), np.zeros(3), atol=1e-3)


if __name__ == "__main__":
    unittest.main()
