import neural_network_lyapunov.examples.quadrotor3d.control_affine_quadrotor \
    as mut
import neural_network_lyapunov.utils as utils
import neural_network_lyapunov.mip_utils as mip_utils
import neural_network_lyapunov.gurobi_torch_mip as gurobi_torch_mip

import unittest
import torch
import numpy as np
import gurobipy


class TestControlAffineQuadrotor(unittest.TestCase):
    def setUp(self):
        self.dtype = torch.float64
        self.phi_a = utils.setup_relu((6, 4, 3),
                                      params=None,
                                      negative_slope=0.1,
                                      bias=True,
                                      dtype=self.dtype)
        self.phi_a[0].weight.data = torch.tensor(
            [[1, -1, 2, 3, 0.5, 1], [0.5, -1, 2, 0.5, -1, 2],
             [0.5, 1.5, 2, -1, -3, -2], [0.5, 0.5, -1, -2, 1, 2]],
            dtype=self.dtype)
        self.phi_a[0].bias.data = torch.tensor([1, -1, 2, -2],
                                               dtype=self.dtype)
        self.phi_a[2].weight.data = torch.tensor(
            [[1, 2, 3, 4], [-1, -2, 0, 1], [0, 2, 1, -1]], dtype=self.dtype)
        self.phi_a[2].bias.data = torch.tensor([1, 2, 3], dtype=self.dtype)

        self.phi_b = utils.setup_relu((3, 4, 12),
                                      params=None,
                                      negative_slope=0.1,
                                      bias=True,
                                      dtype=self.dtype)
        self.phi_b[0].weight.data = torch.tensor(
            [[1, 2, 3], [2, 3, -1], [0, 1, -1], [1, 0, -2]], dtype=self.dtype)
        self.phi_b[0].bias.data = torch.tensor([1, -1, 2, 0], dtype=self.dtype)
        self.phi_b[2].weight.data = torch.tensor(
            [[1, 2, 3, 0], [-1, -3, 2, 1], [0, 1, 0, 2], [0, 1, 2, 3],
             [-1, -3, 2, 1], [0, 2, 1, -1], [1, -1, 0, 2], [-1, -2, -1, 1],
             [1, 0, 1, 2], [0, 2, -1, -3], [1, -1, 2, -2], [1, 0, 1, -3]],
            dtype=self.dtype)
        self.phi_b[2].bias.data = torch.tensor(
            [1, -1, 0, 1, 2, -3, 2, -2, 0, 1, 1, 3], dtype=self.dtype)
        self.phi_c = utils.setup_relu((3, 2, 3),
                                      params=None,
                                      negative_slope=0.1,
                                      bias=True,
                                      dtype=self.dtype)
        self.phi_c[0].weight.data = torch.tensor([[1, 2, 3], [0, -1, -2]],
                                                 dtype=self.dtype)
        self.phi_c[0].bias.data = torch.tensor([1, -2], dtype=self.dtype)
        self.phi_c[2].weight.data = torch.tensor([[1, 3], [-1, -2], [3, 1]],
                                                 dtype=self.dtype)
        self.phi_c[2].bias.data = torch.tensor([1, 3, 2], dtype=self.dtype)
        self.C = torch.tensor([[1, 2, -1, 2], [0, 1, -2, 2], [2, 1, 3, 2]],
                              dtype=self.dtype)

    def test_dynamics(self):
        x_lo = torch.tensor([-2, -2, -2, -1, -1, -1, -3, -3, -3, -2, -2, -2],
                            dtype=self.dtype)
        x_up = -x_lo
        u_lo = torch.tensor([0, 0, 0, 0], dtype=self.dtype)
        u_up = torch.tensor([2, 2, 2, 2], dtype=self.dtype)
        u_equilibrium = torch.ones((4, ), dtype=self.dtype)
        dut = mut.ControlAffineQuadrotor(x_lo, x_up, u_lo, u_up, self.phi_a,
                                         self.phi_b, self.phi_c, self.C,
                                         u_equilibrium,
                                         mip_utils.PropagateBoundsMethod.IA)

        x_samples = utils.uniform_sample_in_box(x_lo, x_up, 100)
        for i in range(x_samples.shape[0]):
            rpy = x_samples[i, 3:6]
            omega = x_samples[i, 9:12]
            f = dut.f(x_samples[i])
            zero3 = torch.zeros((3, ), dtype=self.dtype)
            f_expected = torch.cat(
                (x_samples[i, 6:9], self.phi_a(torch.cat(
                    (rpy, omega))) - self.phi_a(torch.cat(
                        (rpy, zero3))), -self.phi_b(zero3).reshape(
                            (3, 4)) @ u_equilibrium, self.phi_c(omega) -
                 self.phi_c(zero3) - self.C @ u_equilibrium))
            np.testing.assert_allclose(f.detach().numpy(),
                                       f_expected.detach().numpy())
            G = dut.G(x_samples[i])
            G_expected = torch.zeros((12, 4), dtype=self.dtype)
            G_expected[6:9, :] = self.phi_b(rpy).reshape((3, 4))
            G_expected[9:12, :] = self.C
            np.testing.assert_allclose(G.detach().numpy(),
                                       G_expected.detach().numpy())

    def mixed_integer_constraints_tester(self, dut):
        ret = dut.mixed_integer_constraints()
        mip = gurobi_torch_mip.GurobiTorchMIP(self.dtype)
        x = mip.addVars(dut.x_dim, lb=-gurobipy.GRB.INFINITY)
        f = mip.addVars(dut.x_dim, lb=-gurobipy.GRB.INFINITY)
        G = mip.addVars(dut.x_dim * dut.u_dim, lb=-gurobipy.GRB.INFINITY)
        f_slack, f_binary = mip.add_mixed_integer_linear_constraints(
            ret.mip_cnstr_f,
            x,
            f,
            "",
            "",
            "",
            "",
            "",
            binary_var_type=gurobipy.GRB.BINARY)
        G_slack, G_binary = mip.add_mixed_integer_linear_constraints(
            ret.mip_cnstr_G,
            x,
            G,
            "",
            "",
            "",
            "",
            "",
            binary_var_type=gurobipy.GRB.BINARY)
        # Sample many x, check if f and G are correct.
        torch.manual_seed(0)
        x_samples = utils.uniform_sample_in_box(dut.x_lo, dut.x_up, 100)
        mip.gurobi_model.setParam(gurobipy.GRB.Param.OutputFlag, False)
        for i in range(x_samples.shape[0]):
            for j in range(dut.x_dim):
                x[j].lb = x_samples[i][j].item()
                x[j].ub = x_samples[i][j].item()
            mip.gurobi_model.optimize()
            f_val = dut.f(x_samples[i])
            G_val = dut.G(x_samples[i])
            np.testing.assert_array_less(f_val.detach().numpy(),
                                         ret.f_up.detach().numpy() + 1E-6)
            np.testing.assert_array_less(ret.f_lo.detach().numpy(),
                                         f_val.detach().numpy() + 1e-6)
            np.testing.assert_array_less(
                G_val.reshape((-1, )).detach().numpy(),
                ret.G_flat_up.detach().numpy() + 1E-6)
            np.testing.assert_array_less(
                ret.G_flat_lo.detach().numpy(),
                G_val.reshape((-1, )).detach().numpy() + 1E-6)
            np.testing.assert_allclose(np.array([v.x for v in f]),
                                       f_val.detach().numpy())
            np.testing.assert_allclose(np.array([v.x for v in G]),
                                       G_val.reshape((-1, )).detach().numpy())

    def test_mixed_integer_constraints(self):
        x_lo = torch.tensor([-2, -2, -2, -1, -1, -1, -3, -3, -3, -2, -2, -2],
                            dtype=self.dtype)
        x_up = -x_lo
        u_lo = torch.tensor([0, 0, 0, 0], dtype=self.dtype)
        u_up = torch.tensor([2, 2, 2, 2], dtype=self.dtype)
        u_equilibrium = torch.ones((4, ), dtype=self.dtype)
        dut1 = mut.ControlAffineQuadrotor(x_lo, x_up, u_lo, u_up, self.phi_a,
                                          self.phi_b, self.phi_c, self.C,
                                          u_equilibrium,
                                          mip_utils.PropagateBoundsMethod.IA)
        self.mixed_integer_constraints_tester(dut1)
        dut2 = mut.ControlAffineQuadrotor(x_lo, x_up, u_lo, u_up, self.phi_a,
                                          self.phi_b, self.phi_c, self.C,
                                          u_equilibrium,
                                          mip_utils.PropagateBoundsMethod.LP)
        self.mixed_integer_constraints_tester(dut2)
        self.phi_b[-1].weight.data *= 0.01
        dut3 = mut.ControlAffineQuadrotor(x_lo, x_up, u_lo, u_up, self.phi_a,
                                          self.phi_b, self.phi_c, self.C,
                                          u_equilibrium,
                                          mip_utils.PropagateBoundsMethod.MIP)
        self.mixed_integer_constraints_tester(dut3)


if __name__ == "__main__":
    unittest.main()
