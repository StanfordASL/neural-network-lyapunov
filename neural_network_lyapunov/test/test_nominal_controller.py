import neural_network_lyapunov.nominal_controller as mut
import neural_network_lyapunov.utils as utils
import unittest
import torch
import numpy as np


class TestNominalNNController(unittest.TestCase):
    def setUp(self):
        self.dtype = torch.float64
        self.phi = utils.setup_relu((2, 4, 3),
                                    params=None,
                                    negative_slope=0.1,
                                    bias=True,
                                    dtype=self.dtype)
        self.phi[0].weight.data = torch.tensor(
            [[2, 1], [3, 1], [1, -2], [0, -1]], dtype=self.dtype)
        self.phi[0].bias.data = torch.tensor([1, -2, 0, 2], dtype=self.dtype)
        self.phi[2].weight.data = torch.tensor(
            [[1, 0, -2, 3], [0, 1, 3, -1], [0, 1, 3, -2]], dtype=self.dtype)
        self.phi[2].bias.data = torch.tensor([0, 1, 2], dtype=self.dtype)

    def test1(self):
        # Test with x_star and u_star
        x_star = torch.tensor([0, 2], dtype=self.dtype)
        u_star = torch.tensor([1, 2, -3], dtype=self.dtype)
        u_lo = torch.tensor([0, 1, -4], dtype=self.dtype)
        u_up = torch.tensor([4, 5, 10], dtype=self.dtype)
        dut = mut.NominalNNController(self.phi, x_star, u_star, u_lo, u_up)

        def check(x):
            u = dut.output(x)
            u_expected = dut.network(x) - dut.network(x_star) + u_star
            if len(x.shape) == 1:
                u_expected = torch.maximum(torch.minimum(u_expected, u_up),
                                           u_lo)
            else:
                u_expected = torch.minimum(
                    torch.maximum(u_expected, u_lo.repeat(x.shape[0], 1)),
                    u_up.repeat(x.shape[0], 1))
            np.testing.assert_allclose(u.detach().numpy(),
                                       u_expected.detach().numpy())

        check(torch.tensor([0, 3], dtype=self.dtype))
        check(torch.tensor([[0, 3], [1, 2], [1, 3]], dtype=self.dtype))
        check(torch.tensor([[0, -3], [-1, 2], [1, 3]], dtype=self.dtype))
        check(torch.tensor([[0, 2], [-1, 2], [1, 3]], dtype=self.dtype))

    def test2(self):
        # Both x_star and u_star are None
        x_star = None
        u_star = None
        u_lo = torch.tensor([0, 1, -4], dtype=self.dtype)
        u_up = torch.tensor([4, 5, 10], dtype=self.dtype)
        dut = mut.NominalNNController(self.phi, x_star, u_star, u_lo, u_up)

        def check(x):
            u = dut.output(x)
            u_expected = dut.network(x)
            if len(x.shape) == 1:
                u_expected = torch.maximum(torch.minimum(u_expected, u_up),
                                           u_lo)
            else:
                u_expected = torch.minimum(
                    torch.maximum(u_expected, u_lo.repeat(x.shape[0], 1)),
                    u_up.repeat(x.shape[0], 1))
            np.testing.assert_allclose(u.detach().numpy(),
                                       u_expected.detach().numpy())

        check(torch.tensor([0, 3], dtype=self.dtype))
        check(torch.tensor([[0, 3], [1, 2], [1, 3]], dtype=self.dtype))
        check(torch.tensor([[0, -3], [-1, 2], [1, 3]], dtype=self.dtype))
        check(torch.tensor([[0, 2], [-1, 2], [1, 3]], dtype=self.dtype))


if __name__ == "__main__":
    unittest.main()
