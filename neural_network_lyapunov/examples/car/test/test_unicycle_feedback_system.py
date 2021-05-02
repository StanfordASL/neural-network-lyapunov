import neural_network_lyapunov.examples.car.unicycle as unicycle
import neural_network_lyapunov.examples.car.unicycle_feedback_system as\
    unicycle_feedback_system
import neural_network_lyapunov.utils as utils
import neural_network_lyapunov.gurobi_torch_mip as gurobi_torch_mip
import neural_network_lyapunov.r_options as r_options

import unittest
import torch
import numpy as np
import gurobipy


class TestUnicycleFeedbackSystem(unittest.TestCase):
    def setUp(self):
        self.dtype = torch.float64
        self.plant = unicycle.Unicycle(self.dtype)
        forward_relu = utils.setup_relu((3, 4, 2),
                                        params=None,
                                        negative_slope=0.1,
                                        bias=True,
                                        dtype=self.dtype)
        forward_relu[0].weight.data = torch.tensor(
            [[0.2, -1.1, 0.4], [0.3, 2.1, -0.5], [0.2, 1.1, 1.5],
             [0.1, -0.2, 0.9]],
            dtype=self.dtype)
        forward_relu[0].bias.data = torch.tensor([0.4, -0.3, 1.2, 0.1],
                                                 dtype=self.dtype)
        forward_relu[2].weight.data = torch.tensor(
            [[0.2, 0.5, -2.1, 0.4], [0.3, 0.2, -1.1, 0.2]], dtype=self.dtype)
        forward_relu[2].bias.data = torch.tensor([0.2, 0.6], dtype=self.dtype)
        self.x_lo = torch.tensor([-1, -1, -0.5 * np.pi], dtype=self.dtype)
        self.x_up = torch.tensor([1, 1, 0.5 * np.pi], dtype=self.dtype)
        self.u_lo = torch.tensor([0, -np.pi * 0.25], dtype=self.dtype)
        self.u_up = torch.tensor([6, np.pi * 0.25], dtype=self.dtype)
        self.forward_system = unicycle.UnicycleReLUZeroVelModel(
            self.dtype,
            self.x_lo,
            self.x_up,
            self.u_lo,
            self.u_up,
            forward_relu,
            dt=0.01,
            thetadot_as_input=True)
        controller_network = utils.setup_relu((3, 5, 2),
                                              params=None,
                                              negative_slope=0.1,
                                              bias=True,
                                              dtype=self.dtype)
        controller_network[0].weight.data = torch.tensor(
            [[0.2, 0.5, 1.2], [-0.4, -2.1, 1.4], [0.3, 0.5, 1.2],
             [0.2, 0.4, 0.9], [1.1, -2.1, 0.5]],
            dtype=self.dtype)
        controller_network[0].bias.data = torch.tensor(
            [0.3, 0.5, -1.2, 0.3, 0.2], dtype=self.dtype)
        controller_network[2].weight.data = torch.tensor(
            [[0.2, 0.5, -0.3, -1.2, 0.4], [0.5, -2.1, 1.4, 0.3, 0.2]],
            dtype=self.dtype)
        controller_network[2].bias.data = torch.tensor([0.4, -0.1],
                                                       dtype=self.dtype)
        lambda_u = 0.4
        Ru_options = r_options.SearchRwithSVDOptions((3, 2),
                                                     a=np.array([0.1, 0.2]))
        self.dut = unicycle_feedback_system.UnicycleFeedbackSystem(
            self.forward_system, controller_network,
            self.u_lo.detach().numpy(),
            self.u_up.detach().numpy(), lambda_u, Ru_options)

    def test_compute_u(self):
        def compute_u_single(x_val):
            with torch.no_grad():
                u_pre_sat = self.dut.controller_network(
                    x_val) - self.dut.controller_network(
                        torch.zeros((3, ), dtype=self.dtype))
                u_pre_sat[0] += self.dut.lambda_u * torch.norm(
                    self.dut.Ru_options.R() @ x_val[:2], p=1)
                u = np.clip(u_pre_sat.detach().numpy(), self.dut.u_lower_limit,
                            self.dut.u_upper_limit)
                return u

        for x in (torch.tensor([0.3, 0.5, -0.2], dtype=self.dtype),
                  torch.tensor([0.9, -0.6, 1.4], dtype=self.dtype),
                  torch.tensor([-0.3, 0.8, 0.5], dtype=self.dtype)):
            np.testing.assert_allclose(
                self.dut.compute_u(x).detach().numpy(), compute_u_single(x))

        x = torch.tensor([[0.2, 0.5, 0.4], [0.1, -0.9, 0.3], [0.5, 0.3, -1.2],
                          [0.4, -0.5, 0.5]],
                         dtype=self.dtype)
        u = self.dut.compute_u(x)
        assert (u.shape == (x.shape[0], 2))
        for i in range(x.shape[0]):
            np.testing.assert_allclose(u[i].detach().numpy(),
                                       compute_u_single(x[i]))

    def add_network_controller_mip_constraint_tester(self, x_val):
        mip = gurobi_torch_mip.GurobiTorchMIP(self.dtype)
        x_var = mip.addVars(3, lb=x_val, ub=x_val, name="x")
        u_var = mip.addVars(2, lb=-gurobipy.GRB.INFINITY, name="u")

        controller_slack, controller_binary, u_lower_bound, u_upper_bound, \
            controller_pre_relu_lo, controller_pre_relu_up, _, _ =\
            self.dut._add_network_controller_mip_constraint(
                mip, x_var, u_var, "controlelr_slack", "controller_binary",
                lp_relaxation=False)
        mip.gurobi_model.setParam(gurobipy.GRB.Param.OutputFlag, False)
        mip.gurobi_model.optimize()
        if torch.all(x_val <= self.x_up) and torch.all(x_val >= self.x_lo):
            self.assertEqual(mip.gurobi_model.status,
                             gurobipy.GRB.Status.OPTIMAL)
            u_sol = np.array([v.x for v in u_var])
            np.testing.assert_allclose(
                u_sol, self.dut.compute_u(x_val).detach().numpy(), atol=1E-6)
            np.testing.assert_array_less(u_sol,
                                         u_upper_bound.detach().numpy() + 1E-6)
            np.testing.assert_array_less(u_lower_bound.detach().numpy() - 1E-6,
                                         u_sol)
        else:
            self.assertNotEqual(mip.gurobi_model.status,
                                gurobipy.GRB.Status.OPTIMAL)

    def test_add_network_controller_mip_constraint(self):
        self.add_network_controller_mip_constraint_tester(
            torch.tensor([-0.2, 0.5, 0.3], dtype=self.dtype))
        self.add_network_controller_mip_constraint_tester(
            torch.tensor([0.2, 0.9, -0.8], dtype=self.dtype))
        self.add_network_controller_mip_constraint_tester(
            torch.tensor([0.8, 0.9, -1.2], dtype=self.dtype))

    def test_add_network_controller_mip_constraint_bound(self):
        # Make sure the bounds on u is correct.
        mip = gurobi_torch_mip.GurobiTorchMILP(self.dtype)
        x_var = mip.addVars(3, lb=-gurobipy.GRB.INFINITY, name="x")
        u_var = mip.addVars(2, lb=-gurobipy.GRB.INFINITY, name="u")
        network_controller_mip_cnstr_return = \
            self.dut._add_network_controller_mip_constraint(
                mip,
                x_var,
                u_var,
                "controller_slack",
                "controller_binary",
                lp_relaxation=False)
        mip.gurobi_model.setParam(gurobipy.GRB.Param.OutputFlag, False)
        for i in range(2):
            mip.setObjective([torch.tensor([1.], dtype=self.dtype)],
                             [[u_var[i]]],
                             0.,
                             sense=gurobipy.GRB.MAXIMIZE)
            mip.gurobi_model.optimize()
            self.assertLessEqual(
                mip.gurobi_model.ObjVal,
                network_controller_mip_cnstr_return.u_upper_bound[i])
            mip.setObjective([torch.tensor([1.], dtype=self.dtype)],
                             [[u_var[i]]],
                             0.,
                             sense=gurobipy.GRB.MINIMIZE)
            mip.gurobi_model.optimize()
            self.assertGreaterEqual(
                mip.gurobi_model.ObjVal,
                network_controller_mip_cnstr_return.u_lower_bound[i])


if __name__ == "__main__":
    unittest.main()
