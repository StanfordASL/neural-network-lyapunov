import neural_network_lyapunov.control_barrier as mut
import neural_network_lyapunov.control_affine_system as control_affine_system
import neural_network_lyapunov.utils as utils
import neural_network_lyapunov.gurobi_torch_mip as gurobi_torch_mip

import torch
import numpy as np
import unittest
import gurobipy


class TestControlBarrier(unittest.TestCase):
    def setUp(self):
        self.dtype = torch.float64
        self.linear_system = control_affine_system.LinearSystem(
            torch.tensor([[1, 3], [2, -4]], dtype=self.dtype),
            torch.tensor([[1, 2, 3], [0, 1, -1]], dtype=self.dtype),
            x_lo=torch.tensor([-2, -3], dtype=self.dtype),
            x_up=torch.tensor([3, 1], dtype=self.dtype),
            u_lo=torch.tensor([-1, -3, 2], dtype=self.dtype),
            u_up=torch.tensor([2, -1, 4], dtype=self.dtype))
        self.barrier_relu1 = utils.setup_relu((2, 4, 3, 1),
                                              params=None,
                                              negative_slope=0.01,
                                              bias=True,
                                              dtype=self.dtype)
        self.barrier_relu1[0].weight.data = torch.tensor(
            [[1, -1], [0, 2], [1, 3], [-1, -2]], dtype=self.dtype)
        self.barrier_relu1[0].bias.data = torch.tensor([0, 1, -1, 2],
                                                       dtype=self.dtype)
        self.barrier_relu1[2].weight.data = torch.tensor(
            [[1, 0, -1, 2], [0, 2, -1, 1], [1, 0, 1, -2]], dtype=self.dtype)
        self.barrier_relu1[2].bias.data = torch.tensor([0, 2, 3],
                                                       dtype=self.dtype)
        self.barrier_relu1[4].weight.data = torch.tensor([[1, -3, 2]],
                                                         dtype=self.dtype)
        self.barrier_relu1[4].bias.data = torch.tensor([-1], dtype=self.dtype)

    def barrier_derivative_tester(self, dut, x):
        hdot = dut.barrier_derivative(x)
        barrier_grad = utils.relu_network_gradient(dut.barrier_relu,
                                                   x).squeeze(1)
        f = dut.system.f(x)
        G = dut.system.G(x)
        hdot_expected = barrier_grad @ f
        dhdx_times_G = barrier_grad @ G
        for i in range(dut.system.u_dim):
            for j in range(barrier_grad.shape[0]):
                hdot_expected[j] += dhdx_times_G[
                    j, i] * dut.system.u_up[i] if dhdx_times_G[
                        j, i] >= 0 else dhdx_times_G[j, i] * dut.system.u_lo[i]
        self.assertEqual(hdot.shape, hdot_expected.shape)
        np.testing.assert_allclose(hdot.detach().numpy(),
                                   hdot_expected.detach().numpy())

    def test_barrier_derivative(self):
        x_samples = utils.uniform_sample_in_box(self.linear_system.x_lo,
                                                self.linear_system.x_up, 100)
        dut1 = mut.ControlBarrier(self.linear_system, self.barrier_relu1)
        for i in range(x_samples.shape[0]):
            self.barrier_derivative_tester(dut1, x_samples[i])

        # Now test some x with multiple gradient.
        self.barrier_derivative_tester(dut1,
                                       torch.tensor([1, 1], dtype=self.dtype))
        self.barrier_derivative_tester(dut1,
                                       torch.tensor([1, 0], dtype=self.dtype))

    def compute_dhdx_times_G_tester(self, dut):
        milp = gurobi_torch_mip.GurobiTorchMIP(self.dtype)
        x = milp.addVars(dut.system.x_dim, lb=-gurobipy.GRB.INFINITY)
        barrier_mip_cnstr_return = dut.barrier_relu_free_pattern.\
            output_constraint(
                torch.from_numpy(dut.system.x_lo_all),
                torch.from_numpy(dut.system.x_up_all),
                dut.network_bound_propagate_method)
        _, barrier_relu_binary = milp.add_mixed_integer_linear_constraints(
            barrier_mip_cnstr_return, x, None, "", "", "", "", "",
            gurobipy.GRB.BINARY)
        f = milp.addVars(dut.system.x_dim, lb=-gurobipy.GRB.INFINITY)
        Gt = [None] * dut.system.u_dim
        for i in range(dut.system.u_dim):
            Gt[i] = milp.addVars(dut.system.x_dim, lb=-gurobipy.GRB.INFINITY)
        system_mip_cnstr_ret, _, _, _, _ = \
            control_affine_system.add_system_constraint(
                dut.system, milp, x, f, Gt,
                binary_var_type=gurobipy.GRB.BINARY)
        dhdx_times_G, dhdx_times_G_lo, dhdx_times_G_up = \
            dut._compute_dhdx_times_G(
                milp, x, barrier_relu_binary, Gt,
                system_mip_cnstr_ret.G_flat_lo,
                system_mip_cnstr_ret.G_flat_up, gurobipy.GRB.BINARY)
        milp.gurobi_model.setParam(gurobipy.GRB.Param.OutputFlag, False)
        torch.manual_seed(0)
        x_samples = utils.uniform_sample_in_box(dut.system.x_lo,
                                                dut.system.x_up, 100)
        for i in range(x_samples.shape[0]):
            for j in range(dut.system.x_dim):
                x[j].lb = x_samples[i][j].item()
                x[j].ub = x_samples[i][j].item()
            milp.gurobi_model.optimize()
            self.assertEqual(milp.gurobi_model.status,
                             gurobipy.GRB.Status.OPTIMAL)
            dhdx = utils.relu_network_gradient(dut.barrier_relu, x_samples[i])
            assert (dhdx.shape[0] == 1)
            G = dut.system.G(x_samples[i])
            dhdx_times_G_expected = dhdx[0][0] @ G
            np.testing.assert_allclose(np.array([v.x for v in dhdx_times_G]),
                                       dhdx_times_G_expected.detach().numpy())
            np.testing.assert_array_less(
                dhdx_times_G_expected.detach().numpy(),
                dhdx_times_G_up.detach().numpy())
            np.testing.assert_array_less(
                dhdx_times_G_lo.detach().numpy(),
                dhdx_times_G_expected.detach().numpy())

    def test_compute_dhdx_times_G(self):
        dut1 = mut.ControlBarrier(self.linear_system, self.barrier_relu1)
        self.compute_dhdx_times_G_tester(dut1)


if __name__ == "__main__":
    unittest.main()
