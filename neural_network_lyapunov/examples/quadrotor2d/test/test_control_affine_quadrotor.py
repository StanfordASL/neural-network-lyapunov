import neural_network_lyapunov.examples.quadrotor2d.control_affine_quadrotor\
    as mut
import neural_network_lyapunov.utils as utils
import neural_network_lyapunov.mip_utils as mip_utils
import numpy as np
import torch
import unittest


class TestControlAffineQuadrotor2d(unittest.TestCase):
    def setUp(self):
        self.dtype = torch.float64
        self.phi_b = utils.setup_relu((1, 4, 6),
                                      params=None,
                                      negative_slope=0.1,
                                      bias=True,
                                      dtype=self.dtype)
        self.phi_b[0].weight.data = torch.tensor([[
            1,
        ], [-2], [3], [1]],
                                                 dtype=self.dtype)
        self.phi_b[0].bias.data = torch.tensor([1, 2, -2, -1],
                                               dtype=self.dtype)
        self.phi_b[2].weight.data = torch.tensor(
            [[1, 3, -2, 1], [0, 1, -2, -3], [3, 1, 2, -3], [3, 2, -2, 1],
             [0, 5, 1.5, 3], [-1, -2, 1, 3]],
            dtype=self.dtype)
        self.phi_b[2].bias.data = torch.tensor([1, 3, -2, 0, -2, -1],
                                               dtype=self.dtype)

    def test_dynamics(self):
        x_lo = torch.tensor([-3, -3, -2, -5, -5, -4], dtype=self.dtype)
        x_up = -x_lo
        u_lo = torch.tensor([0, 0], dtype=self.dtype)
        u_up = torch.tensor([5, 5], dtype=self.dtype)
        u_equilibrium = torch.tensor([3, 3], dtype=self.dtype)

        dut = mut.ControlAffineQuadrotor2d(x_lo, x_up, u_lo, u_up, self.phi_b,
                                           u_equilibrium,
                                           mip_utils.PropagateBoundsMethod.IA)

        x_samples = utils.uniform_sample_in_box(x_lo, x_up, 100)
        u_samples = utils.uniform_sample_in_box(u_lo, u_up, 100)
        xdot_batch = dut.dynamics(x_samples, u_samples)
        self.assertEqual(xdot_batch.shape, x_samples.shape)
        for i in range(x_samples.shape[0]):
            vdot = -self.phi_b(
                torch.tensor([dut.theta_equilibrium],
                             dtype=self.dtype)).reshape(
                                 (3, 2)) @ u_equilibrium + self.phi_b(
                                     x_samples[i, 2].unsqueeze(0)).reshape(
                                         (3, 2)) @ u_samples[i]
            xdot = torch.cat((x_samples[i, 3:], vdot))
            np.testing.assert_allclose(
                dut.dynamics(x_samples[i], u_samples[i]).detach().numpy(),
                xdot.detach().numpy())
            np.testing.assert_allclose(xdot_batch[i].detach().numpy(),
                                       xdot.detach().numpy())


if __name__ == "__main__":
    unittest.main()
