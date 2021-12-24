import neural_network_lyapunov.examples.pole.pole_relu_system as mut
import neural_network_lyapunov.utils as utils
import neural_network_lyapunov.mip_utils as mip_utils
import neural_network_lyapunov.gurobi_torch_mip as gurobi_torch_mip
import neural_network_lyapunov.relu_system as relu_system

import torch
import gurobipy
import numpy as np
import unittest


class TestPoleReluSystem(unittest.TestCase):
    def setUp(self):
        self.dtype = torch.float64
        self.dynamics_relu = utils.setup_relu((7, 6, 5),
                                              params=None,
                                              negative_slope=0.1,
                                              bias=True,
                                              dtype=self.dtype)
        self.dynamics_relu[0].weight.data = torch.tensor(
            [[1, -1, 2, 3, -1, 3, 1], [0, 1, -1, 2, 3, 1, 2],
             [0, 2, 3, -1, -3, 4, 2], [0, 2, 4, 1, -2, 0, 5],
             [1, 2, -1, 0, 3, 2, 1], [1, -1, 2, -2, 3, 1, 3]],
            dtype=self.dtype)
        self.dynamics_relu[0].bias.data = torch.tensor([1, 3, 2, 4, 0, 4],
                                                       dtype=self.dtype)
        self.dynamics_relu[2].weight.data = torch.tensor(
            [[1, 2, -1, 3, 2, -2], [0, 1, -1, 2, 3, 1], [0, 2, 1, 4, 3, -2],
             [2, 1, -2, 3, 2, 5], [0, 3, 2, -1, -5, 3]],
            dtype=self.dtype)
        self.dynamics_relu[2].bias.data = torch.tensor([1, 2, 3, -2, -4],
                                                       dtype=self.dtype)

    def test_step_forward(self):
        dut = mut.PoleReluSystem(x_lo=torch.tensor(
            [-0.5, -0.5, -1, -1, -2, -3, -3], dtype=self.dtype),
                                 x_up=torch.tensor([0.5, 0.5, 1, 1, 2, 3, 4],
                                                   dtype=self.dtype),
                                 u_lo=torch.tensor([-2, -2, -0.5],
                                                   dtype=self.dtype),
                                 u_up=torch.tensor([2, 2, 3.5],
                                                   dtype=self.dtype),
                                 dynamics_relu=self.dynamics_relu,
                                 dt=0.01,
                                 u_z_equilibrium=2.)
        x_samples = utils.uniform_sample_in_box(dut.x_lo, dut.x_up, 100)
        u_samples = utils.uniform_sample_in_box(dut.u_lo, dut.u_up, 100)
        x_next = dut.step_forward(x_samples, u_samples)
        self.assertEqual(x_next.shape, x_samples.shape)
        for i in range(x_samples.shape[0]):
            x_next_i = dut.step_forward(x_samples[i], u_samples[i])
            np.testing.assert_allclose(x_next[i].detach().numpy(),
                                       x_next_i.detach().numpy())
            v_next = x_samples[i, 2:] + dut.dynamics_relu(
                torch.cat(
                    (x_samples[i, :2], x_samples[i, -2:],
                     u_samples[i]))) - dut.dynamics_relu(
                         torch.cat(
                             (dut.x_equilibrium[:2], dut.x_equilibrium[-2:],
                              dut.u_equilibrium)))
            xy_AB_next = x_samples[i, :2] + (v_next[-2:] +
                                             x_samples[i, -2:]) * dut.dt / 2
            np.testing.assert_allclose(
                x_next_i.detach().numpy(),
                torch.cat((xy_AB_next, v_next)).detach().numpy())
        # At equilibrium, the next state is still the equilibrium.
        np.testing.assert_allclose(
            dut.step_forward(dut.x_equilibrium,
                             dut.u_equilibrium).detach().numpy(),
            dut.x_equilibrium.detach().numpy())

    def add_dynamics_constraint_tester(self, dut, additional_u_lo,
                                       additional_u_up):
        mip = gurobi_torch_mip.GurobiTorchMIP(self.dtype)
        x_var = mip.addVars(dut.x_dim, lb=-gurobipy.GRB.INFINITY)
        x_next_var = mip.addVars(dut.x_dim, lb=-gurobipy.GRB.INFINITY)
        u_var = mip.addVars(dut.u_dim, lb=-gurobipy.GRB.INFINITY)
        ret = dut.add_dynamics_constraint(mip, x_var, x_next_var, u_var, "",
                                          "", additional_u_lo, additional_u_up,
                                          gurobipy.GRB.BINARY)
        self.assertIsInstance(ret, relu_system.ReLUDynamicsConstraintReturn)
        # Take many samples of x and u, check if x_next is right.
        x_samples = utils.uniform_sample_in_box(dut.x_lo, dut.x_up, 100)
        u_lo = dut.u_lo if additional_u_lo is None else torch.max(
            dut.u_lo, additional_u_lo)
        u_up = dut.u_up if additional_u_up is None else torch.min(
            dut.u_up, additional_u_up)
        u_samples = utils.uniform_sample_in_box(u_lo, u_up, 100)
        mip.gurobi_model.setParam(gurobipy.GRB.Param.OutputFlag, False)
        for i in range(x_samples.shape[0]):
            for j in range(dut.x_dim):
                x_var[j].lb = x_samples[i, j].item()
                x_var[j].ub = x_samples[i, j].item()
            for j in range(dut.u_dim):
                u_var[j].lb = u_samples[i, j].item()
                u_var[j].ub = u_samples[i, j].item()
            mip.gurobi_model.optimize()
            self.assertEqual(mip.gurobi_model.status,
                             gurobipy.GRB.Status.OPTIMAL)
            np.testing.assert_allclose(
                np.array([v.x for v in x_next_var]),
                dut.step_forward(x_samples[i], u_samples[i]).detach().numpy())

    def test_add_dynamics_constraint(self):
        dut = mut.PoleReluSystem(x_lo=torch.tensor(
            [-0.5, -0.5, -1, -1, -2, -3, -3], dtype=self.dtype),
                                 x_up=torch.tensor([0.5, 0.5, 1, 1, 2, 3, 4],
                                                   dtype=self.dtype),
                                 u_lo=torch.tensor([-2, -2, -0.5],
                                                   dtype=self.dtype),
                                 u_up=torch.tensor([2, 2, 3.5],
                                                   dtype=self.dtype),
                                 dynamics_relu=self.dynamics_relu,
                                 dt=0.01,
                                 u_z_equilibrium=2.)
        self.add_dynamics_constraint_tester(dut, None, None)
        self.add_dynamics_constraint_tester(
            dut, torch.tensor([-1, -3, 0.5], dtype=self.dtype), None)
        dut.network_bound_propagate_method = \
            mip_utils.PropagateBoundsMethod.MIP
        self.add_dynamics_constraint_tester(dut, None, None)


if __name__ == "__main__":
    unittest.main()
