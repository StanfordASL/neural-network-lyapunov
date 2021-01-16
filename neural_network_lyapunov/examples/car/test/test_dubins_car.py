import neural_network_lyapunov.examples.car.dubins_car as dubins_car
import neural_network_lyapunov.utils as utils
import neural_network_lyapunov.gurobi_torch_mip as gurobi_torch_mip

import unittest

import numpy as np
import torch
import scipy.integrate

import gurobipy


class TestDubinsCar(unittest.TestCase):
    def test_dynamics(self):
        plant = dubins_car.DubinsCar(torch.float64)
        # Test with pytorch tensor.
        x = torch.tensor([2., 3., 0.5], dtype=torch.float64)
        u = torch.tensor([0.5, -0.2], dtype=torch.float64)
        xdot_torch = plant.dynamics(x, u)
        np.testing.assert_allclose(
            xdot_torch.detach().numpy(),
            np.array([u[0] * torch.cos(x[2]), u[0] * torch.sin(x[2]), u[1]]))
        xdot_np = plant.dynamics(x.detach().numpy(), u.detach().numpy())
        np.testing.assert_allclose(xdot_torch.detach().numpy(), xdot_np)

    def test_next_pose(self):
        plant = dubins_car.DubinsCar(torch.float64)
        x = torch.tensor([2., 3., 0.5], dtype=torch.float64)
        u = torch.tensor([0.5, -0.2], dtype=torch.float64)

        x_next = plant.next_pose(x, u, 0.1)
        result = scipy.integrate.solve_ivp(
            lambda t, x_val: plant.dynamics(x_val,
                                            u.detach().numpy()), [0, 0.1],
            x.detach().numpy())
        np.testing.assert_allclose(x_next, result.y[:, -1])


class TestDubinsCarReLUModel(unittest.TestCase):
    def setUp(self):
        self.dtype = torch.float64
        # Arbitrarily initialize the relu network. All the tests should pass
        # even if the network doesn't approximate the Dubin's car dynamics.
        dynamics_relu_no_thetadot = utils.setup_relu((2, 4, 3, 2),
                                                     params=None,
                                                     negative_slope=0.1,
                                                     bias=True,
                                                     dtype=self.dtype)
        dynamics_relu_no_thetadot[0].weight.data = torch.tensor(
            [[0.2, 0.5], [-1.3, 0.5], [-0.3, -0.2], [-0.4, -1.4]],
            dtype=self.dtype)
        dynamics_relu_no_thetadot[0].bias.data = torch.tensor(
            [0.4, -1.2, 0.1, 2.3], dtype=self.dtype)
        dynamics_relu_no_thetadot[2].weight.data = torch.tensor(
            [[0.4, 0.1, -1.4, 0.2], [0.1, -0.2, -0.5, -1.1],
             [0.3, 0.5, 1.1, -0.2]],
            dtype=self.dtype)
        dynamics_relu_no_thetadot[2].bias.data = torch.tensor([0.2, 0.1, -0.3],
                                                              dtype=self.dtype)
        dynamics_relu_no_thetadot[4].weight.data = torch.tensor(
            [[0.1, -0.3, 0.5], [0.3, -0.2, 2.1]], dtype=self.dtype)
        dynamics_relu_no_thetadot[4].bias.data = torch.tensor([0.4, -1.2],
                                                              dtype=self.dtype)
        self.dut_thetadot_not_input = dubins_car.DubinsCarReLUModel(
            self.dtype,
            x_lo=torch.tensor([-3, -3, -np.pi], dtype=self.dtype),
            x_up=torch.tensor([3, 3, np.pi], dtype=self.dtype),
            u_lo=torch.tensor([-2, -0.5], dtype=self.dtype),
            u_up=torch.tensor([5, 0.5], dtype=self.dtype),
            dynamics_relu=dynamics_relu_no_thetadot,
            dt=0.01,
            thetadot_as_input=False)

        dynamics_relu_thetadot = utils.setup_relu((3, 4, 3, 2),
                                                  params=None,
                                                  negative_slope=0.1,
                                                  bias=True,
                                                  dtype=self.dtype)
        dynamics_relu_thetadot[0].weight.data = torch.tensor(
            [[0.2, 0.5, 0.1], [-1.3, 0.5, -1.2], [-0.3, -0.2, 0.4],
             [-0.4, -1.4, 0.5]],
            dtype=self.dtype)
        dynamics_relu_no_thetadot[0].bias.data = torch.tensor(
            [0.4, -1.2, 0.1, 2.3], dtype=self.dtype)
        dynamics_relu_thetadot[2].weight.data = dynamics_relu_no_thetadot[
            2].weight.data
        dynamics_relu_thetadot[2].bias.data = dynamics_relu_no_thetadot[
            2].bias.data
        dynamics_relu_thetadot[4].weight.data = dynamics_relu_no_thetadot[
            4].weight.data
        dynamics_relu_thetadot[4].bias.data = dynamics_relu_thetadot[
            4].bias.data

        self.dut_thetadot_input = dubins_car.DubinsCarReLUModel(
            self.dtype,
            x_lo=torch.tensor([-3, -3, -np.pi], dtype=self.dtype),
            x_up=torch.tensor([3, 3, np.pi], dtype=self.dtype),
            u_lo=torch.tensor([-2, -0.5], dtype=self.dtype),
            u_up=torch.tensor([5, 0.5], dtype=self.dtype),
            dynamics_relu=dynamics_relu_thetadot,
            dt=0.01,
            thetadot_as_input=True)

    def step_forward_tester(self, dut):
        # First make sure that if vel = 0, then pos[n+1] = pos[n]
        x_start = torch.tensor([0.5, 0.3, -1.2], dtype=self.dtype)
        u_start = torch.tensor([0, 0.5], dtype=self.dtype)
        np.testing.assert_allclose(
            dut.step_forward(x_start, u_start)[:2].detach().numpy(),
            x_start[:2].detach().numpy())

        # First test a single x_start and u_start
        x_start = torch.tensor([0.2, 0.5, -0.1], dtype=self.dtype)
        u_start = torch.tensor([2.1, 0.3], dtype=self.dtype)
        x_next = dut.step_forward(x_start, u_start)

        def eval_next_state(x_val, u_val):
            if dut.thetadot_as_input:
                network_input = torch.tensor([x_val[2], u_val[0], u_val[1]],
                                             dtype=self.dtype)
                network_input_zero_vel = torch.tensor([x_val[2], 0, u_val[1]],
                                                      dtype=self.dtype)
            else:
                network_input = torch.tensor([x_val[2], u_val[0]],
                                             dtype=self.dtype)
                network_input_zero_vel = torch.tensor([x_val[2], 0],
                                                      dtype=self.dtype)
            position_next = x_val[:2] + \
                dut.dynamics_relu(network_input) - dut.dynamics_relu(
                    network_input_zero_vel)
            theta_next = x_val[2] + u_val[1] * dut.dt
            return np.array([
                position_next[0].item(), position_next[1].item(),
                theta_next.item()
            ])

        np.testing.assert_allclose(x_next.detach().numpy(),
                                   eval_next_state(x_start, u_start))

        # Now test a batch of x_start and u_start
        x_start = torch.tensor([[0.2, 0.5, -0.1], [0.4, 0.3, 0.5]],
                               dtype=self.dtype)
        u_start = torch.tensor([[2.1, 0.3], [-0.3, 0.4]], dtype=self.dtype)
        x_next = dut.step_forward(x_start, u_start)
        self.assertEqual(x_next.shape, (2, 3))
        for i in range(x_start.shape[0]):
            np.testing.assert_allclose(x_next[i].detach().numpy(),
                                       eval_next_state(x_start[i], u_start[i]))

    def test_step_forward_thetadot_not_input(self):
        self.step_forward_tester(self.dut_thetadot_not_input)

    def test_step_forward_thetadot_as_input(self):
        self.step_forward_tester(self.dut_thetadot_input)

    def add_dynamics_constraint_tester(self, dut):
        def tester(x_val, u_val):
            # Setup an MILP with fixed x_var and u_var, check if x_next_var is
            # solved to the right value.
            mip = gurobi_torch_mip.GurobiTorchMILP(self.dtype)
            x_var = mip.addVars(3, lb=-gurobipy.GRB.INFINITY)
            u_var = mip.addVars(2, lb=-gurobipy.GRB.INFINITY)
            x_next_var = mip.addVars(3, lb=-gurobipy.GRB.INFINITY)
            dut.add_dynamics_constraint(mip, x_var, x_next_var, u_var, "slack",
                                        "binary")
            # Fix x_var to x_val, u_var to u_val
            mip.addMConstrs([torch.eye(3, dtype=self.dtype)], [x_var],
                            sense=gurobipy.GRB.EQUAL,
                            b=x_val)
            mip.addMConstrs([torch.eye(2, dtype=self.dtype)], [u_var],
                            sense=gurobipy.GRB.EQUAL,
                            b=u_val)
            mip.gurobi_model.setParam(gurobipy.GRB.Param.OutputFlag, False)
            mip.gurobi_model.optimize()
            self.assertEqual(mip.gurobi_model.status,
                             gurobipy.GRB.Status.OPTIMAL)
            x_next_val = np.array([var.xn for var in x_next_var])
            x_next_val_expected = dut.step_forward(x_val, u_val)
            np.testing.assert_allclose(x_next_val,
                                       x_next_val_expected.detach().numpy(),
                                       atol=1e-8)

        tester(torch.tensor([0., 0., 0.], dtype=self.dtype),
               torch.tensor([0., 0.], dtype=self.dtype))
        tester(torch.tensor([0.5, -0.3, 0.4], dtype=self.dtype),
               torch.tensor([0., 0.], dtype=self.dtype))
        tester(torch.tensor([0.6, -1.3, 0.4], dtype=self.dtype),
               torch.tensor([4., 0.3], dtype=self.dtype))
        tester(torch.tensor([0.6, -1.3, 0.4], dtype=self.dtype),
               torch.tensor([-2., 0.3], dtype=self.dtype))

    def test_add_dynamics_constraint_thetadot_not_input(self):
        self.add_dynamics_constraint_tester(self.dut_thetadot_not_input)

    def test_add_dynamics_constraint_thetadot_as_input(self):
        self.add_dynamics_constraint_tester(self.dut_thetadot_input)


if __name__ == "__main__":
    unittest.main()
