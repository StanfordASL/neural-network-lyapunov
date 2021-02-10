import neural_network_lyapunov.examples.quadrotor3d.quadrotor_feedback_system\
    as quadrotor_feedback_system
import neural_network_lyapunov.examples.quadrotor3d.quadrotor as quadrotor
import neural_network_lyapunov.utils as utils
import neural_network_lyapunov.gurobi_torch_mip as gurobi_torch_mip
import neural_network_lyapunov.mip_utils as mip_utils

import torch
import numpy as np
import unittest
import os
import gurobipy


class TestQuadrotorFeedbackSystem(unittest.TestCase):
    def setUp(self):
        self.dtype = torch.float64
        plant = quadrotor.Quadrotor(self.dtype)
        forward_relu = torch.load(
            os.path.dirname(os.path.realpath(__file__)) +
            "/../data/quadrotor_forward_relu14.pt")
        x_lo = torch.tensor([
            -0.05, -0.05, -0.05, -0.05 * np.pi, -0.05 * np.pi, -0.05 * np.pi,
            -0.2, -0.2, -0.2, -0.1 * np.pi, -0.1 * np.pi, -0.1 * np.pi
        ],
                            dtype=self.dtype)
        x_up = -x_lo
        u_lo = torch.zeros((4, ), dtype=self.dtype)
        u_up = torch.full((4, ), 3 * plant.hover_thrust, dtype=self.dtype)
        dt = 0.01
        forward_system = quadrotor.QuadrotorReLUSystem(self.dtype, x_lo, x_up,
                                                       u_lo, u_up,
                                                       forward_relu,
                                                       plant.hover_thrust, dt)
        torch.manual_seed(0)

        controller_network = utils.setup_relu((12, 8, 4),
                                              params=None,
                                              negative_slope=0.1,
                                              bias=True,
                                              dtype=self.dtype)
        self.closed_loop_system = \
            quadrotor_feedback_system.QuadrotorFeedbackSystem(
                forward_system, controller_network,
                u_lo.detach().numpy(), u_up.detach().numpy())

    def add_dynamics_mip_constraint_tester(self, x_val):
        mip = gurobi_torch_mip.GurobiTorchMIP(self.dtype)
        x_var = mip.addVars(12, lb=x_val, ub=x_val)
        x_next_var = mip.addVars(12,
                                 lb=-gurobipy.GRB.INFINITY,
                                 ub=gurobipy.GRB.INFINITY)
        u_var, forward_slack, controller_slack, forward_binary,\
            controller_binary =\
            self.closed_loop_system.add_dynamics_mip_constraint(
                mip, x_var, x_next_var, "u", "forward_slack",
                "forward_binary", "controller_slack", "controller_binary")
        mip.gurobi_model.setParam(gurobipy.GRB.Param.OutputFlag, False)
        mip.gurobi_model.optimize()
        if torch.all(x_val <= self.closed_loop_system.forward_system.x_up
                     ) and torch.all(
                         x_val >= self.closed_loop_system.forward_system.x_lo):
            self.assertEqual(mip.gurobi_model.status,
                             gurobipy.GRB.Status.OPTIMAL)
            # Make sure that the solution is correct.
            np.testing.assert_allclose(
                np.array([v.x for v in u_var]),
                self.closed_loop_system.compute_u(x_val).detach().numpy())
            np.testing.assert_allclose(
                np.array([v.x for v in x_next_var]),
                self.closed_loop_system.step_forward(x_val).detach().numpy(),
                atol=1E-6)
        else:
            self.assertNotEqual(mip.gurobi_model.status,
                                gurobipy.GRB.Status.OPTIMAL)

    def test_add_dynamics_mip_constraint_IA(self):
        self.closed_loop_system.forward_system.network_bound_propagate_method\
            = mip_utils.PropagateBoundsMethod.IA
        self.closed_loop_system.controller_network_bound_propagate_method\
            = mip_utils.PropagateBoundsMethod.IA
        self.add_dynamics_mip_constraint_tester(
            torch.zeros((12, ), dtype=self.dtype))
        self.add_dynamics_mip_constraint_tester(
            self.closed_loop_system.forward_system.x_lo * 0.1 +
            0.9 * self.closed_loop_system.forward_system.x_up)
        self.add_dynamics_mip_constraint_tester(
            1.1 * self.closed_loop_system.forward_system.x_up)
        self.add_dynamics_mip_constraint_tester(
            1.1 * self.closed_loop_system.forward_system.x_lo)

    def test_add_dynamics_mip_constraint_LP(self):
        self.closed_loop_system.forward_system.network_bound_propagate_method\
            = mip_utils.PropagateBoundsMethod.LP
        self.closed_loop_system.controller_network_bound_propagate_method\
            = mip_utils.PropagateBoundsMethod.LP
        self.add_dynamics_mip_constraint_tester(
            torch.zeros((12, ), dtype=self.dtype))
        self.add_dynamics_mip_constraint_tester(
            self.closed_loop_system.forward_system.x_lo * 0.1 +
            0.9 * self.closed_loop_system.forward_system.x_up)
        self.add_dynamics_mip_constraint_tester(
            self.closed_loop_system.forward_system.x_lo * 0.2 +
            0.5 * self.closed_loop_system.forward_system.x_up)
        self.add_dynamics_mip_constraint_tester(
            1.1 * self.closed_loop_system.forward_system.x_up)
        self.add_dynamics_mip_constraint_tester(
            1.1 * self.closed_loop_system.forward_system.x_lo)
        x_samples = utils.uniform_sample_in_box(
            self.closed_loop_system.forward_system.x_lo,
            self.closed_loop_system.forward_system.x_up, 30)
        for i in range(x_samples.shape[0]):
            self.add_dynamics_mip_constraint_tester(x_samples[i])


if __name__ == "__main__":
    unittest.main()
