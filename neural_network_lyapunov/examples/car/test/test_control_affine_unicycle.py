import neural_network_lyapunov.examples.car.control_affine_unicycle as mut
import neural_network_lyapunov.utils as utils
import neural_network_lyapunov.mip_utils as mip_utils
import neural_network_lyapunov.gurobi_torch_mip as gurobi_torch_mip
import unittest
import torch
import numpy as np
import gurobipy


class TestControlAffineUnicycle(unittest.TestCase):
    def setUp(self):
        self.dtype = torch.float64
        self.phi = utils.setup_relu((1, 3, 2),
                                    params=None,
                                    negative_slope=0.1,
                                    bias=True,
                                    dtype=self.dtype)
        self.phi[0].weight.data = torch.tensor([[1], [3], [-1]],
                                               dtype=self.dtype)
        self.phi[0].bias.data = torch.tensor([0.5, 1, -1], dtype=self.dtype)
        self.phi[2].weight.data = torch.tensor([[1, 2, 2], [2, -1, -2]],
                                               dtype=self.dtype)
        self.phi[2].bias.data = torch.tensor([1, -2], dtype=self.dtype)

    def test_dynamics(self):
        x_lo = torch.tensor([-1, -2, -2], dtype=self.dtype)
        x_up = -x_lo
        u_lo = torch.tensor([-1, -1], dtype=self.dtype)
        u_up = -u_lo
        dut = mut.ControlAffineUnicycle(
            self.phi,
            x_lo,
            x_up,
            u_lo,
            u_up,
            method=mip_utils.PropagateBoundsMethod.IA)
        x_samples = utils.uniform_sample_in_box(x_lo, x_up, 100)
        u_samples = utils.uniform_sample_in_box(u_lo, u_up, 100)
        xdot = dut.dynamics(x_samples, u_samples)
        self.assertEqual(xdot.shape, x_samples.shape)
        for i in range(x_samples.shape[0]):
            xdot_i = dut.dynamics(x_samples[i], u_samples[i])
            np.testing.assert_allclose(xdot_i[:2].detach().numpy(),
                                       (dut.phi(x_samples[i, 2:]) *
                                        u_samples[i, 0]).detach().numpy())
            self.assertAlmostEqual(xdot_i[2].item(), u_samples[i, 1])
            np.testing.assert_allclose(xdot_i.detach().numpy(),
                                       xdot[i].detach().numpy())

    def mixed_integer_constraints_tester(self, dut):
        mip = gurobi_torch_mip.GurobiTorchMIP(dtype=self.dtype)
        x = mip.addVars(dut.x_dim, lb=-gurobipy.GRB.INFINITY)
        ret = dut.mixed_integer_constraints()
        f = mip.addVars(dut.x_dim, lb=-gurobipy.GRB.INFINITY)
        G_flat = mip.addVars(dut.x_dim * dut.u_dim, lb=-gurobipy.GRB.INFINITY)
        mip.add_mixed_integer_linear_constraints(ret.mip_cnstr_f, x, f, "", "",
                                                 "", "", "")
        mip.add_mixed_integer_linear_constraints(ret.mip_cnstr_G, x, G_flat,
                                                 "", "", "", "", "")
        torch.manual_seed(0)
        x_samples = utils.uniform_sample_in_box(dut.x_lo, dut.x_up, 100)
        mip.gurobi_model.setParam(gurobipy.GRB.Param.OutputFlag, False)
        for i in range(x_samples.shape[0]):
            for j in range(dut.x_dim):
                x[j].lb = x_samples[i, j].item()
                x[j].ub = x_samples[i, j].item()
            mip.gurobi_model.optimize()
            self.assertEqual(mip.gurobi_model.status,
                             gurobipy.GRB.Status.OPTIMAL)
            f_val = dut.f(x_samples[i]).detach().numpy()
            G_flat_val = dut.G(x_samples[i]).reshape((-1, )).detach().numpy()
            np.testing.assert_allclose([v.x for v in f], f_val)
            np.testing.assert_allclose([v.x for v in G_flat], G_flat_val)
            np.testing.assert_array_less(f_val,
                                         ret.f_up.detach().numpy() + 1E-6)
            np.testing.assert_array_less(ret.f_lo.detach().numpy() - 1E-6,
                                         f_val)
            np.testing.assert_array_less(G_flat_val,
                                         ret.G_flat_up.detach().numpy() + 1E-6)
            np.testing.assert_array_less(ret.G_flat_lo.detach().numpy() - 1E-6,
                                         G_flat_val)

    def test_mixed_integer_constraints(self):
        x_lo = torch.tensor([-1, -2, -2], dtype=self.dtype)
        x_up = -x_lo
        u_lo = torch.tensor([-1, -1], dtype=self.dtype)
        u_up = -u_lo
        for method in list(mip_utils.PropagateBoundsMethod):
            dut = mut.ControlAffineUnicycle(self.phi, x_lo, x_up, u_lo, u_up,
                                            method)
            self.mixed_integer_constraints_tester(dut)


if __name__ == "__main__":
    unittest.main()
