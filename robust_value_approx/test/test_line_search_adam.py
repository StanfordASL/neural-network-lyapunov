import robust_value_approx.line_search_adam as line_search_adam

import torch
import numpy as np
import unittest


class TestLineSearchAdam(unittest.TestCase):
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
        dut1 = line_search_adam.LineSearchAdam(
            network.parameters(), lr=lr, min_step_size_decrease=1e-5,
            loss_minimal_decrement=loss_minimal_decrement,
            step_size_reduction=0.2)
        a0 = network.weight.data.clone().squeeze()
        b0 = network.bias.clone()
        grad0 = grad_fun(a0, b0)
        _, step_size_all, dp0 = dut1.step_direction()
        dp0_flat = torch.cat([dp.reshape((-1,)) for dp in dp0])
        a1 = a0 + dp0_flat[:2] * step_size_all[0]
        b1 = b0 + dp0_flat[2] * step_size_all[1]
        loss1 = loss_fun(a1, b1)
        assert(loss1 <= loss0 +
               loss_minimal_decrement * lr * grad0 @ dp0_flat)
        dut1.step(closure, loss0.item())
        np.testing.assert_allclose(
            a1.detach().numpy(),
            network.weight.data.squeeze().detach().numpy())
        np.testing.assert_almost_equal(b1.item(), network.bias.data.item())

        reset_network()
        lr = 0.5
        loss_minimal_decrement = 1e-4
        step_size_reduction = 0.2
        min_step_size = 1e-6

        dut4 = line_search_adam.LineSearchAdam(
            network.parameters(), lr=lr,
            min_step_size_decrease=min_step_size/lr,
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
