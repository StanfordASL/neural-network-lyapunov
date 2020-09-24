import torch
import gurobipy

import numpy as np

import neural_network_lyapunov.feedback_system as feedback_system
import neural_network_lyapunov.relu_system as relu_system
import neural_network_lyapunov.hybrid_linear_system as hybrid_linear_system
import neural_network_lyapunov.gurobi_torch_mip as gurobi_torch_mip

import unittest


class TestFeedbackSystem(unittest.TestCase):
    def setUp(self):
        self.dtype = torch.float64
        linear1 = torch.nn.Linear(3, 3)
        linear1.weight.data = torch.tensor([
            [0.2, 1.2, 0.4], [-0.1, 2.5, 1.3], [0.3, 0.1, 0.5]],
            dtype=self.dtype)
        linear1.bias.data = torch.tensor([0.3, 1.2, -0.5], dtype=self.dtype)
        linear2 = torch.nn.Linear(3, 3)
        linear2.weight.data = torch.tensor([
            [0.2, 0.3, -0.5], [0.1, 1.2, -2.1], [0.3, 1.2, 0.5]],
            dtype=self.dtype)
        linear2.bias.data = torch.tensor([0.1, -0.5, 1.2], dtype=self.dtype)
        linear3 = torch.nn.Linear(3, 2)
        linear3.weight.data = torch.tensor([
            [0.1, 1.2, -1.3], [0.3, 0.1, -0.8]], dtype=self.dtype)
        linear3.bias.data = torch.tensor([0.3, -0.5], dtype=self.dtype)
        self.controller_network = torch.nn.Sequential(
            linear1, torch.nn.LeakyReLU(0.01), linear2,
            torch.nn.LeakyReLU(0.01), linear3)

    def add_dynamics_mip_constraint_tester(
            self, forward_system, x_equilibrium, u_equilibrium, x_val):
        # This function checks if add_dynamics_mip_constraint imposes the right
        # constraint, such that x_next and u computed from MIP matches with
        # that computed from the controller network and the foward dynamics.
        dut = feedback_system.FeedbackSystem(
            forward_system, self.controller_network, x_equilibrium,
            u_equilibrium)
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
        np.testing.assert_allclose(
            u_val.detach().numpy(), u_val_expected.detach().numpy())

        x_next_val = torch.tensor([xi.X for xi in x_next], dtype=self.dtype)
        if isinstance(forward_system, relu_system.ReLUSystem):
            x_next_expected = forward_system.step_forward(x_val, u_val)
        elif isinstance(forward_system,
                        hybrid_linear_system.HybridLinearSystem):
            x_next_expected, _ = forward_system.step_forward(x_val, u_val)
        np.testing.assert_allclose(
            x_next_val.detach().numpy(), x_next_expected.detach().numpy())

    def test_add_dynamics_mip_constraint_hybrid_linear_system(self):
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

        # Our network should work even if the constraint x* = f(x*, u*) is not
        # satisfied at the equilibrium.
        x_equilibrium = torch.tensor([0, 0.5, 0.3], dtype=self.dtype)
        u_equilibrium = torch.tensor([0.1, 0.2], dtype=self.dtype)
        self.add_dynamics_mip_constraint_tester(
            forward_system, x_equilibrium, u_equilibrium,
            torch.tensor([0.1, 0.2, 0.3], dtype=self.dtype))
        self.add_dynamics_mip_constraint_tester(
            forward_system, x_equilibrium, u_equilibrium,
            torch.tensor([-0.1, 0.2, 0.3], dtype=self.dtype))
        self.add_dynamics_mip_constraint_tester(
            forward_system, x_equilibrium, u_equilibrium,
            torch.tensor([0.1, 0.0, 0.0], dtype=self.dtype))

    def test_add_dynamics_mip_constraint_relu_system(self):
        # Construct a relu network as the forward dynamical system.
        linear1 = torch.nn.Linear(5, 3)
        linear1.weight.data = torch.tensor([
            [0.1, 0.4, 0.5, 0.5, 0.1], [0.2, -0.1, 1.2, 1.1, -1.2],
            [0.5, -0.3, 0.2, 0.1, 0.4]], dtype=self.dtype)
        linear1.bias.data = torch.tensor([1.1, 0.4, -0.5], dtype=self.dtype)
        linear2 = torch.nn.Linear(3, 3)
        linear2.weight.data = torch.tensor([
            [0.1, 0.3, 0.2], [0.2, -0.5, -0.9], [0.8, 1.5, 0.3]],
            dtype=self.dtype)
        linear2.bias.data = torch.tensor([0.3, 0.5, 0.1])
        forward_network = torch.nn.Sequential(
            linear1, torch.nn.LeakyReLU(0.01), linear2)
        forward_system = relu_system.ReLUSystem(
            self.dtype, torch.tensor([-2, -2, -2], dtype=self.dtype),
            torch.tensor([2, 2, 2], dtype=self.dtype),
            torch.tensor([-5, -5], dtype=self.dtype),
            torch.tensor([5, 5], dtype=self.dtype), forward_network)

        x_equilibrium = torch.tensor([0, 0.5, 0.3], dtype=self.dtype)
        u_equilibrium = torch.tensor([0.1, 0.2], dtype=self.dtype)
        self.add_dynamics_mip_constraint_tester(
            forward_system, x_equilibrium, u_equilibrium,
            torch.tensor([0.1, 0.2, 0.3], dtype=self.dtype))
        self.add_dynamics_mip_constraint_tester(
            forward_system, x_equilibrium, u_equilibrium,
            torch.tensor([-0.1, 0.2, 0.3], dtype=self.dtype))
        self.add_dynamics_mip_constraint_tester(
            forward_system, x_equilibrium, u_equilibrium,
            torch.tensor([0.1, 0.0, 0.0], dtype=self.dtype))


if __name__ == "__main__":
    unittest.main()
