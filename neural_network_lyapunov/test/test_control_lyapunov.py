import neural_network_lyapunov.control_lyapunov as mut

import unittest
import torch
import numpy as np
import gurobipy

import neural_network_lyapunov.control_affine_system as control_affine_system
import neural_network_lyapunov.utils as utils
import neural_network_lyapunov.gurobi_torch_mip as gurobi_torch_mip


class TestControlLyapunov(unittest.TestCase):
    def setUp(self):
        self.dtype = torch.float64

        # Create a linear system.
        A = torch.tensor([[1, 2], [-2, 3]], dtype=self.dtype)
        B = torch.tensor([[1, 3, 1], [0, 1, 0]], dtype=self.dtype)
        self.linear_system = control_affine_system.LinearSystem(
            A,
            B,
            x_lo=torch.tensor([-2, -3], dtype=self.dtype),
            x_up=torch.tensor([3, 3], dtype=self.dtype),
            u_lo=torch.tensor([-1, -2, -3], dtype=self.dtype),
            u_up=torch.tensor([1, 2, 3], dtype=self.dtype))
        self.lyapunov_relu1 = utils.setup_relu((2, 4, 3, 1),
                                               params=None,
                                               negative_slope=0.1,
                                               bias=True,
                                               dtype=self.dtype)
        self.lyapunov_relu1[0].weight.data = torch.tensor(
            [[1, -1], [0, 2], [-1, 2], [-2, 1]], dtype=self.dtype)
        self.lyapunov_relu1[0].bias.data = torch.tensor([1, -1, 0, 2],
                                                        dtype=self.dtype)
        self.lyapunov_relu1[2].weight.data = torch.tensor(
            [[3, -2, 1, 0], [1, -1, 2, 3], [-2, -1, 0, 3]], dtype=self.dtype)
        self.lyapunov_relu1[2].bias.data = torch.tensor([1, -2, 3],
                                                        dtype=self.dtype)
        self.lyapunov_relu1[4].weight.data = torch.tensor([[1, 3, -2]],
                                                          dtype=self.dtype)
        self.lyapunov_relu1[4].bias.data = torch.tensor([2], dtype=self.dtype)

    def lyapunov_derivative_tester(self, dut, x, x_equilibrium, V_lambda,
                                   epsilon, R):
        vdot = dut.lyapunov_derivative(x,
                                       x_equilibrium,
                                       V_lambda,
                                       epsilon,
                                       R=R)

        dphi_dx = utils.relu_network_gradient(dut.lyapunov_relu, x).squeeze(1)
        dl1_dx = V_lambda * utils.l1_gradient(R @ (x - x_equilibrium)) @ R
        v = dut.lyapunov_value(x, x_equilibrium, V_lambda, R=R)
        vdot_expected = np.inf
        for i in range(dphi_dx.shape[0]):
            for j in range(dl1_dx.shape[0]):
                dV_dx = dphi_dx[i] + dl1_dx[j]
                vdot_expected_ij = dV_dx @ dut.system.f(x) + epsilon * v
                dV_dx_times_G = dV_dx @ dut.system.G(x)
                for k in range(dut.system.u_dim):
                    vdot_expected_ij += torch.minimum(
                        dV_dx_times_G[k] * dut.system.u_lo[k],
                        dV_dx_times_G[k] * dut.system.u_up[k])
                if vdot_expected_ij < vdot_expected:
                    vdot_expected = vdot_expected_ij
        self.assertAlmostEqual(vdot.item(), vdot_expected.item())

    def test_lyapunov_derivative1(self):
        # Test with linear system.
        dut = mut.ControlLyapunov(self.linear_system, self.lyapunov_relu1)
        x_equilibrium = torch.tensor([0, -1], dtype=self.dtype)
        V_lambda = 0.5
        R = torch.tensor([[1, 0], [0, 2], [1, 3]], dtype=self.dtype)
        epsilon = 0.1

        x = torch.tensor([0.5, 1.5], dtype=self.dtype)
        self.lyapunov_derivative_tester(dut, x, x_equilibrium, V_lambda,
                                        epsilon, R)
        # Some ReLU inputs equal to 0.
        x = torch.tensor([1, 2], dtype=self.dtype)
        self.lyapunov_derivative_tester(dut, x, x_equilibrium, V_lambda,
                                        epsilon, R)
        # Some l1-norm entry equals to 0.
        x = torch.tensor([0, 1], dtype=self.dtype)
        self.lyapunov_derivative_tester(dut, x, x_equilibrium, V_lambda,
                                        epsilon, R)

    def add_system_constraint_tester(self, dut, x_val: torch.Tensor,
                                     is_feasible):
        milp = gurobi_torch_mip.GurobiTorchMIP(self.dtype)
        x = milp.addVars(dut.system.x_dim, lb=x_val, ub=x_val)
        f = milp.addVars(dut.system.x_dim, lb=-gurobipy.GRB.INFINITY)
        G = [None] * dut.system.u_dim
        for i in range(dut.system.u_dim):
            G[i] = milp.addVars(dut.system.x_dim, lb=-gurobipy.GRB.INFINITY)
        ret = dut.add_system_constraint(milp, x, f, G)
        # Linear system doesn't introduce slack/binary variables.
        self.assertEqual(len(ret.slack), 0)
        self.assertEqual(len(ret.binary), 0)

        milp.gurobi_model.setParam(gurobipy.GRB.Param.OutputFlag, False)
        milp.gurobi_model.optimize()
        if is_feasible:
            self.assertEqual(milp.gurobi_model.status,
                             gurobipy.GRB.Status.OPTIMAL)
            f_val = np.array([v.x for v in f])
            G_val = [None] * dut.system.u_dim
            for i in range(dut.system.u_dim):
                G_val[i] = [v.x for v in G[i]]
            G_val = np.array(G_val).T

            f_expected = dut.system.f(x_val)
            G_expected = dut.system.G(x_val)
            np.testing.assert_allclose(f_val, f_expected.detach().numpy())
            np.testing.assert_allclose(G_val, G_expected.detach().numpy())
        else:
            self.assertEqual(milp.gurobi_model.status,
                             gurobipy.GRB.Status.INFEASIBLE)

    def test_add_system_constraint1(self):
        # Test with linear system.
        dut = mut.ControlLyapunov(self.linear_system, self.lyapunov_relu1)
        # x within [x_lo, x_up]
        self.add_system_constraint_tester(
            dut, (self.linear_system.x_lo + self.linear_system.x_up) / 2, True)
        self.add_system_constraint_tester(
            dut,
            (0.9 * self.linear_system.x_lo + 0.1 * self.linear_system.x_up),
            True)
        # x outside of [x_lo, x_up]
        self.add_system_constraint_tester(
            dut, torch.tensor([-3, 0], dtype=self.dtype), False)
        self.add_system_constraint_tester(
            dut, torch.tensor([0, 4], dtype=self.dtype), False)


if __name__ == "__main__":
    unittest.main()
