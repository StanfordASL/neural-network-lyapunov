import torch
import gurobipy

import numpy as np

import neural_network_lyapunov.feedback_system as feedback_system
import neural_network_lyapunov.relu_system as relu_system
import neural_network_lyapunov.hybrid_linear_system as hybrid_linear_system
import neural_network_lyapunov.gurobi_torch_mip as gurobi_torch_mip
import neural_network_lyapunov.utils as utils

import unittest


class TestFeedbackSystem(unittest.TestCase):
    def setUp(self):
        self.dtype = torch.float64
        controller_network_params = torch.tensor([
            0.2, 1.2, 0.4, -0.1, 2.5, 1.3, 0.3, 0.1, 0.5, 0.3, 1.2, -0.5,
            0.2, 0.3, -0.5, 0.1, 1.2, -2.1, 0.3, 1.2, 0.5, 0.1, -0.5, 1.2,
            0.1, 1.2, -1.3, 0.3, 0.1, -0.8, 0.3, -0.5], dtype=self.dtype)
        self.controller_network = utils.setup_relu(
            (3, 3, 3, 2), controller_network_params, negative_slope=0.01,
            bias=True, dtype=self.dtype)

    def add_dynamics_mip_constraint_tester(
        self, forward_system, x_equilibrium, u_equilibrium, x_val,
            u_lower_limit, u_upper_limit):
        # This function checks if add_dynamics_mip_constraint imposes the right
        # constraint, such that x_next and u computed from MIP matches with
        # that computed from the controller network and the foward dynamics.
        dut = feedback_system.FeedbackSystem(
            forward_system, self.controller_network, x_equilibrium,
            u_equilibrium, u_lower_limit, u_upper_limit)
        milp = gurobi_torch_mip.GurobiTorchMILP(dtype=self.dtype)
        x = milp.addVars(
            3, lb=-gurobipy.GRB.INFINITY, vtype=gurobipy.GRB.CONTINUOUS,
            name="x")
        x_next = milp.addVars(
            3, lb=-gurobipy.GRB.INFINITY, vtype=gurobipy.GRB.CONTINUOUS,
            name="x_next")
        u, forward_slack, controller_slack, forward_binary, controller_binary\
            = dut.add_dynamics_mip_constraint(
                milp, x, x_next, "u", "forward_s", "forward_binary",
                "controller_s", "controller_binary")

        milp.addMConstrs(
            [torch.eye(forward_system.x_dim, dtype=self.dtype)], [x],
            sense=gurobipy.GRB.EQUAL, b=x_val)

        milp.gurobi_model.setParam(gurobipy.GRB.Param.OutputFlag, 0)
        milp.gurobi_model.optimize()

        u_val = torch.tensor([ui.X for ui in u], dtype=self.dtype)
        u_val_expected = self.controller_network(x_val) -\
            self.controller_network(x_equilibrium) + u_equilibrium
        u_val_expected = torch.max(torch.min(
            u_val_expected, torch.from_numpy(u_upper_limit)),
            torch.from_numpy(u_lower_limit))
        np.testing.assert_allclose(
            u_val.detach().numpy(), u_val_expected.detach().numpy())

        x_next_val = torch.tensor([xi.X for xi in x_next], dtype=self.dtype)
        if isinstance(forward_system, relu_system.ReLUSystem) or isinstance(
                forward_system, relu_system.ReLUSystemGivenEquilibrium):
            x_next_expected = forward_system.step_forward(x_val, u_val)
        elif isinstance(forward_system,
                        hybrid_linear_system.HybridLinearSystem):
            x_next_expected, _ = forward_system.step_forward(x_val, u_val)
        np.testing.assert_allclose(
            x_next_val.detach().numpy(), x_next_expected.detach().numpy())

    def construct_hybrid_linear_system_example(self):
        forward_system = hybrid_linear_system.HybridLinearSystem(
            3, 2, self.dtype)
        # x[n+1] = x[n] + [u[n]; 0] + [1;0;0] if [0, -1, -1] <= x[n] <= [1,1,1]
        # and [-10, -10] <= u[n] <= [10, 10]
        # x[n+1] = x[n] + [0; u[n]] + [1;0;0] if [-1,-1,-1] <= x[n] <= [0,1,1]
        # and [-10, -10] <= u[n] <= [10, 10]
        P1 = torch.zeros(10, 5, dtype=self.dtype)
        P1[:3, :3] = torch.eye(3, dtype=self.dtype)
        P1[3:6, :3] = -torch.eye(3, dtype=self.dtype)
        P1[6:8, 3:] = torch.eye(2, dtype=self.dtype)
        P1[8:, 3:] = -torch.eye(2, dtype=self.dtype)
        forward_system.add_mode(
            torch.eye(3, dtype=self.dtype),
            torch.tensor([[1, 0], [0, 1], [0, 0]], dtype=self.dtype),
            torch.tensor([1, 0, 0], dtype=self.dtype), P1,
            torch.tensor([1, 1, 1, 0, 1, 1, 10, 10, 10, 10], dtype=self.dtype))
        forward_system.add_mode(
            torch.eye(3, dtype=self.dtype),
            torch.tensor([[0, 0], [1, 0], [0, 1]], dtype=self.dtype),
            torch.tensor([1, 0, 0], dtype=self.dtype), P1,
            torch.tensor([0, 1, 1, 1, 1, 1, 10, 10, 10, 10], dtype=self.dtype))
        return forward_system

    def test_add_dynamics_mip_constraint_hybrid_linear_system(self):
        forward_system = self.construct_hybrid_linear_system_example()
        # Our network should work even if the constraint x* = f(x*, u*) is not
        # satisfied at the equilibrium.
        x_equilibrium = torch.tensor([0, 0.5, 0.3], dtype=self.dtype)
        u_equilibrium = torch.tensor([0.1, 0.2], dtype=self.dtype)
        self.add_dynamics_mip_constraint_tester(
            forward_system, x_equilibrium, u_equilibrium,
            torch.tensor([0.1, 0.2, 0.3], dtype=self.dtype),
            np.array([-1, -2.]), np.array([1, 2.]))
        self.add_dynamics_mip_constraint_tester(
            forward_system, x_equilibrium, u_equilibrium,
            torch.tensor([-0.1, 0.2, 0.3], dtype=self.dtype),
            np.array([0, 0.]), np.array([100, 100.]))
        self.add_dynamics_mip_constraint_tester(
            forward_system, x_equilibrium, u_equilibrium,
            torch.tensor([0.1, 0.0, 0.0], dtype=self.dtype),
            np.array([-100., -100.]), np.array([0.2, 0.3]))

    def construct_relu_forward_system(self):
        forward_network_params = torch.tensor(
            [0.1, 0.4, 0.5, 0.5, 0.1, 0.2, -0.1, 1.2, 1.1, -1.2, 0.5, -0.3,
             0.2, 0.1, 0.4, 1.1, 0.4, -0.5, 0.1, 0.3, 0.2, 0.2, -0.5, -0.9,
             0.8, 1.5, 0.3, 0.3, 0.5, 0.1], dtype=self.dtype)
        forward_network = utils.setup_relu(
            (5, 3, 3), forward_network_params, negative_slope=0.01, bias=True,
            dtype=self.dtype)
        forward_system = relu_system.ReLUSystem(
            self.dtype, torch.tensor([-2, -2, -2], dtype=self.dtype),
            torch.tensor([2, 2, 2], dtype=self.dtype),
            torch.tensor([-5, -5], dtype=self.dtype),
            torch.tensor([5, 5], dtype=self.dtype), forward_network)
        return forward_system

    def test_add_dynamics_mip_constraint_relu_system(self):
        # Construct a relu network as the forward dynamical system.
        forward_system = self.construct_relu_forward_system()

        x_equilibrium = torch.tensor([0, 0.5, 0.3], dtype=self.dtype)
        u_equilibrium = torch.tensor([0.1, 0.2], dtype=self.dtype)
        self.add_dynamics_mip_constraint_tester(
            forward_system, x_equilibrium, u_equilibrium,
            torch.tensor([0.1, 0.2, 0.3], dtype=self.dtype),
            np.array([-1, 0.]), np.array([1., 1.]))
        self.add_dynamics_mip_constraint_tester(
            forward_system, x_equilibrium, u_equilibrium,
            torch.tensor([-0.1, 0.2, 0.3], dtype=self.dtype),
            np.array([-10, -20.]), np.array([1, 1.]))
        self.add_dynamics_mip_constraint_tester(
            forward_system, x_equilibrium, u_equilibrium,
            torch.tensor([0.1, 0.0, 0.0], dtype=self.dtype),
            np.array([0., 0.]), np.array([10., 20.]))

    def construct_relu_forward_system_given_equilibrium(
            self, x_equilibrium, u_equilibrium):
        forward_network_params = torch.tensor(
            [0.1, 0.4, 0.5, 0.5, 0.1, 0.2, -0.1, 1.2, 1.1, -1.2, 0.5, -0.3,
             0.2, 0.1, 0.4, 1.1, 0.4, -0.5, 0.1, 0.3, 0.2, 0.2, -0.5, -0.9,
             0.8, 1.5, 0.3, 0.3, 0.5, 0.1], dtype=self.dtype)
        forward_network = utils.setup_relu(
            (5, 3, 3), forward_network_params, negative_slope=0.01, bias=True,
            dtype=self.dtype)
        forward_system = relu_system.ReLUSystemGivenEquilibrium(
            self.dtype, torch.tensor([-2, -2, -2], dtype=self.dtype),
            torch.tensor([2, 2, 2], dtype=self.dtype),
            torch.tensor([-5, -5], dtype=self.dtype),
            torch.tensor([5, 5], dtype=self.dtype), forward_network,
            x_equilibrium, u_equilibrium)
        return forward_system

    def test_add_dynamics_mip_constraint_relu_system_given_equilibrium(self):
        # Construct a ReLUSystemGivenEquilibrium as the forward dynamical
        # system.
        x_equilibrium = torch.tensor([0, 0.5, 0.3], dtype=self.dtype)
        u_equilibrium = torch.tensor([0.1, 0.2], dtype=self.dtype)
        forward_system = self.construct_relu_forward_system_given_equilibrium(
            x_equilibrium, u_equilibrium)

        self.add_dynamics_mip_constraint_tester(
            forward_system, forward_system.x_equilibrium,
            forward_system.u_equilibrium,
            torch.tensor([0.1, 0.2, 0.3], dtype=self.dtype),
            np.array([0., 0.]), np.array([1., 1.]))
        self.add_dynamics_mip_constraint_tester(
            forward_system, forward_system.x_equilibrium,
            forward_system.u_equilibrium,
            torch.tensor([-0.1, 0.2, 0.3], dtype=self.dtype),
            np.array([-10, -10.]), np.array([1., 1.]))
        self.add_dynamics_mip_constraint_tester(
            forward_system, forward_system.x_equilibrium,
            forward_system.u_equilibrium,
            torch.tensor([0.1, 0.0, 0.0], dtype=self.dtype),
            np.array([0., 0.]), np.array([10., 10.]))

    def test_compute_u(self):
        x_equilibrium = torch.tensor([0, 0.5, 0.3], dtype=self.dtype)
        u_equilibrium = torch.tensor([0.1, 0.2], dtype=self.dtype)
        forward_system = self.construct_relu_forward_system_given_equilibrium(
            x_equilibrium, u_equilibrium)
        u_lower_limit = np.array([0., -10.])
        u_upper_limit = np.array([100., 0.3])
        closed_loop_system = feedback_system.FeedbackSystem(
            forward_system, self.controller_network, x_equilibrium,
            u_equilibrium, u_lower_limit, u_upper_limit)

        x_all = (torch.tensor([0.2, 0.4, 0.9], dtype=self.dtype),
                 torch.tensor([-0.5, 1.2, 1.4], dtype=self.dtype),
                 torch.tensor([1.6, -0.4, 2.3], dtype=self.dtype))

        with torch.no_grad():
            for x in x_all:
                u = closed_loop_system.compute_u(x)
                u_expected = self.controller_network(x) - \
                    self.controller_network(x_equilibrium) + u_equilibrium
                u_expected = torch.min(torch.max(u_expected, torch.from_numpy(
                    u_lower_limit)), torch.from_numpy(u_upper_limit))
                np.testing.assert_allclose(
                    u.detach().numpy(), u_expected.detach().numpy())

    def step_forward_test(
        self, forward_system, x_equilibrium, u_equilibrium, u_lower_limit,
            u_upper_limit):
        closed_loop_system = feedback_system.FeedbackSystem(
            forward_system, self.controller_network, x_equilibrium,
            u_equilibrium, u_lower_limit, u_upper_limit)

        x = torch.tensor([-1, 0.5, 0.2], dtype=self.dtype)
        x_next = closed_loop_system.step_forward(x)
        self.assertIsInstance(x_next, torch.Tensor)
        self.assertEqual(x_next.shape, (3,))
        u = closed_loop_system.compute_u(x)
        x_next_expected = forward_system.step_forward(x, u)
        if isinstance(forward_system, hybrid_linear_system.HybridLinearSystem):
            x_next_expected = x_next_expected[0]
        np.testing.assert_allclose(
                x_next.detach().numpy(), x_next_expected.detach().numpy())

    def test_step_forward_hybrid_linear_system(self):
        forward_system = self.construct_hybrid_linear_system_example()
        x_equilibrium = torch.tensor([0, 0.5, 0.3], dtype=self.dtype)
        u_equilibrium = torch.tensor([0.1, 0.2], dtype=self.dtype)
        self.step_forward_test(
            forward_system, x_equilibrium, u_equilibrium,
            np.array([-np.inf, -np.inf]), np.array([np.inf, np.inf]))

    def test_step_forward_relu_system(self):
        forward_system = self.construct_relu_forward_system()
        x_equilibrium = torch.tensor([0, 0.5, 0.3], dtype=self.dtype)
        u_equilibrium = torch.tensor([0.1, 0.2], dtype=self.dtype)
        self.step_forward_test(
            forward_system, x_equilibrium, u_equilibrium,
            np.array([-np.inf, -np.inf]), np.array([np.inf, np.inf]))
        self.step_forward_test(
            forward_system, x_equilibrium, u_equilibrium,
            np.array([-1., 0]), np.array([10., 10.]))

    def test_step_forward_relu_system_given_equilibrium(self):
        x_equilibrium = torch.tensor([0, 0.5, 0.3], dtype=self.dtype)
        u_equilibrium = torch.tensor([0.1, 0.2], dtype=self.dtype)
        forward_system = self.construct_relu_forward_system_given_equilibrium(
            x_equilibrium, u_equilibrium)
        self.step_forward_test(
            forward_system, forward_system.x_equilibrium,
            forward_system.u_equilibrium, np.array([-np.inf, -np.inf]),
            np.array([np.inf, np.inf]))
        self.step_forward_test(
            forward_system, forward_system.x_equilibrium,
            forward_system.u_equilibrium, np.array([-10., -10.]),
            np.array([0.2, 0.3]))


if __name__ == "__main__":
    unittest.main()
