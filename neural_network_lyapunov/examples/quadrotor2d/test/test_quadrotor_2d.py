import neural_network_lyapunov.examples.quadrotor2d.quadrotor_2d as\
    quadrotor_2d

import unittest
import numpy as np
import torch
import scipy.integrate
import gurobipy

import neural_network_lyapunov.utils as utils
import neural_network_lyapunov.mip_utils as mip_utils
import neural_network_lyapunov.relu_system as relu_system
import neural_network_lyapunov.gurobi_torch_mip as gurobi_torch_mip


class TestQuadrotor2D(unittest.TestCase):
    def test_dynamics_equilibrium(self):
        plant = quadrotor_2d.Quadrotor2D(torch.float64)
        u = plant.u_equilibrium
        xdot = plant.dynamics(np.zeros((6, )), u)
        np.testing.assert_allclose(xdot, np.zeros((6, )))

    def test_dynamics(self):
        plant = quadrotor_2d.Quadrotor2D(torch.float64)

        def check_dynamics(x, u):
            assert (isinstance(x, torch.Tensor))
            xdot = plant.dynamics(x, u)
            xdot_np = plant.dynamics(x.detach().numpy(), u.detach().numpy())
            np.testing.assert_allclose(xdot_np, xdot.detach().numpy())

        check_dynamics(torch.tensor([1, 2, 3, 4, 5, 6], dtype=torch.float64),
                       torch.tensor([7, 8], dtype=torch.float64))
        check_dynamics(
            torch.tensor([1, -2, 3, -4, 5, -6], dtype=torch.float64),
            torch.tensor([7, -8], dtype=torch.float64))

    def test_linearized_dynamics(self):
        plant = quadrotor_2d.Quadrotor2D(torch.float64)

        def check_linearized_dynamics(x, u):
            assert (isinstance(x, torch.Tensor))
            A, B = plant.linearized_dynamics(x, u)
            xdot = plant.dynamics(x, u)
            for i in range(6):
                if x.grad is not None:
                    x.grad.zero_()
                if u.grad is not None:
                    u.grad.zero_()
                xdot[i].backward(retain_graph=True)
                Ai_expected = x.grad.detach().numpy() if x.grad is not None\
                    else np.zeros((6,))
                np.testing.assert_allclose(A[i, :].detach().numpy(),
                                           Ai_expected)
                Bi_expected = u.grad.detach().numpy() if u.grad is not None\
                    else np.zeros((2,))
                np.testing.assert_allclose(B[i, :].detach().numpy(),
                                           Bi_expected)
            # Make sure numpy and torch input give same result.
            A_np, B_np = plant.linearized_dynamics(x.detach().numpy(),
                                                   u.detach().numpy())
            np.testing.assert_allclose(A_np, A.detach().numpy())
            np.testing.assert_allclose(B_np, B.detach().numpy())

        check_linearized_dynamics(
            torch.tensor([1, 2, 3, 4, 5, 6],
                         dtype=torch.float64,
                         requires_grad=True),
            torch.tensor([7, 8], dtype=torch.float64, requires_grad=True))
        check_linearized_dynamics(
            torch.tensor([-1, -2, 3, 4, 5, 6],
                         dtype=torch.float64,
                         requires_grad=True),
            torch.tensor([7, -8], dtype=torch.float64, requires_grad=True))

    def test_lqr_control(self):
        plant = quadrotor_2d.Quadrotor2D(torch.float64)
        x_star = np.zeros((6, ))
        u_star = plant.u_equilibrium.detach().numpy()
        Q = np.diag([10, 10, 10, 1, 1, plant.length / 2. / np.pi])
        R = np.array([[0.1, 0.05], [0.05, 0.1]])
        K, S = plant.lqr_control(Q, R, x_star, u_star)
        result = scipy.integrate.solve_ivp(
            lambda t, x: plant.dynamics(x, K @ (x - x_star) + u_star), (0, 10),
            np.array([0.1, 0.1, -0.1, 0.2, 0.2, -0.3]))
        np.testing.assert_allclose(result.y[:, -1], x_star, atol=1E-6)


class TestQuadrotorReluContinuousTime(unittest.TestCase):
    def setUp(self):
        self.dtype = torch.float64
        x_lo = torch.tensor([-1, -2, -0.5 * np.pi, -3, -3, -np.pi],
                            dtype=self.dtype)
        x_up = -x_lo
        u_lo = torch.tensor([0.1, 0.1], dtype=self.dtype)
        u_up = torch.tensor([3, 3], dtype=self.dtype)
        u_equilibrium = torch.tensor([1, 1], dtype=self.dtype)
        dynamics_relu = utils.setup_relu((3, 4, 5, 3),
                                         params=None,
                                         negative_slope=0.1,
                                         bias=True,
                                         dtype=self.dtype)
        dynamics_relu[0].weight.data = torch.tensor(
            [[1, 3, 2], [0, -1, 3], [1, 0, -2], [0, 1, -1]], dtype=self.dtype)
        dynamics_relu[0].bias.data = torch.tensor([1, -2, 0, 3],
                                                  dtype=self.dtype)
        dynamics_relu[2].weight.data = torch.tensor(
            [[3, 1, -2, 1], [0, 1, 3, 4], [-2, 1, 3, 1], [1, -2, 2, -3],
             [0, 1, 2, -3]],
            dtype=self.dtype)
        dynamics_relu[2].bias.data = torch.tensor([0, 3, 2, -1, 1],
                                                  dtype=self.dtype)
        dynamics_relu[4].weight.data = torch.tensor(
            [[1, 0, -2, 3, 2], [0, 2, -1, 3, 1], [1, 1, -2, 3, 3]],
            dtype=self.dtype)
        dynamics_relu[4].bias.data = torch.tensor([1, 1, -2], dtype=self.dtype)
        self.dut = quadrotor_2d.QuadrotorReluContinuousTime(
            self.dtype, x_lo, x_up, u_lo, u_up, dynamics_relu, u_equilibrium)

    def test_step_forward(self):
        torch.manual_seed(0)
        x_samples = utils.uniform_sample_in_box(self.dut.x_lo, self.dut.x_up,
                                                1000)
        u_samples = utils.uniform_sample_in_box(self.dut.u_lo, self.dut.u_up,
                                                x_samples.shape[0])
        xdot = self.dut.step_forward(x_samples, u_samples)
        self.assertEqual(xdot.shape, (x_samples.shape[0], self.dut.x_dim))
        for i in range(x_samples.shape[0]):
            xdot_i = self.dut.step_forward(x_samples[i], u_samples[i])
            np.testing.assert_allclose(xdot[i].detach().numpy(),
                                       xdot_i.detach().numpy())
            np.testing.assert_allclose(xdot_i[:3].detach().numpy(),
                                       x_samples[i, 3:].detach().numpy())
            np.testing.assert_allclose(
                xdot_i[3:].detach().numpy(),
                (self.dut.dynamics_relu(
                    torch.cat((x_samples[i, 2:3], u_samples[i]))) -
                 self.dut.dynamics_relu(
                     torch.cat((torch.tensor([0], dtype=self.dtype),
                                self.dut.u_equilibrium)))).detach().numpy())

    def test_add_dynamics_constraint(self):
        for method in list(mip_utils.PropagateBoundsMethod):
            mip = gurobi_torch_mip.GurobiTorchMIP(dtype=self.dtype)
            mip.gurobi_model.setParam(gurobipy.GRB.Param.OutputFlag, False)
            x_var = mip.addVars(6, lb=-gurobipy.GRB.INFINITY)
            x_next_var = mip.addVars(6, lb=-gurobipy.GRB.INFINITY)
            u_var = mip.addVars(2, lb=-gurobipy.GRB.INFINITY)

            u_input_prog = relu_system.ControlBoundProg(None, None, None)
            u_input_prog.prog = gurobi_torch_mip.GurobiTorchMILP(self.dtype)
            u_input_prog.x_var = u_input_prog.prog.addVars(
                6, lb=-gurobipy.GRB.INFINITY)
            u_input_prog.u_var = u_input_prog.prog.addVars(
                2, lb=-gurobipy.GRB.INFINITY)
            self.dut.network_bound_propagate_method = method
            result = self.dut.add_dynamics_constraint(mip, x_var, x_next_var,
                                                      u_var, "", "", None,
                                                      None,
                                                      gurobipy.GRB.BINARY,
                                                      u_input_prog)
            torch.manual_seed(0)
            x_samples = utils.uniform_sample_in_box(self.dut.x_lo,
                                                    self.dut.x_up, 100)
            u_samples = utils.uniform_sample_in_box(self.dut.u_lo,
                                                    self.dut.u_up,
                                                    x_samples.shape[0])
            xdot = self.dut.step_forward(x_samples, u_samples)
            if method == mip_utils.PropagateBoundsMethod.IA:
                xdot_lb = result.x_next_lb_IA
                xdot_ub = result.x_next_ub_IA
            else:
                xdot_lb = torch.empty((6, ), dtype=self.dtype)
                xdot_ub = torch.empty((6, ), dtype=self.dtype)
                result.x_next_bound_prog.gurobi_model.setParam(
                    gurobipy.GRB.Param.OutputFlag, False)
                result.x_next_bound_prog.gurobi_model.setParam(
                    gurobipy.GRB.Param.DualReductions, False)
                for i in range(6):
                    result.x_next_bound_prog.setObjective(
                        [torch.tensor([1], dtype=self.dtype)],
                        [[result.x_next_bound_var[i]]], 0.,
                        gurobipy.GRB.MAXIMIZE)
                    result.x_next_bound_prog.gurobi_model.optimize()
                    self.assertEqual(
                        result.x_next_bound_prog.gurobi_model.status,
                        gurobipy.GRB.Status.OPTIMAL)
                    xdot_ub[i] = result.x_next_bound_prog.gurobi_model.ObjVal
                    self.assertEqual(
                        result.x_next_bound_prog.gurobi_model.status,
                        gurobipy.GRB.Status.OPTIMAL)
                    result.x_next_bound_prog.setObjective(
                        [torch.tensor([1], dtype=self.dtype)],
                        [[result.x_next_bound_var[i]]], 0.,
                        gurobipy.GRB.MINIMIZE)
                    result.x_next_bound_prog.gurobi_model.optimize()
                    xdot_lb[i] = result.x_next_bound_prog.gurobi_model.ObjVal

            for i in range(x_samples.shape[0]):
                for j in range(self.dut.x_dim):
                    x_var[j].lb = x_samples[i, j].item()
                    x_var[j].ub = x_samples[i, j].item()
                for j in range(self.dut.u_dim):
                    u_var[j].lb = u_samples[i, j].item()
                    u_var[j].ub = u_samples[i, j].item()
                mip.gurobi_model.optimize()
                np.testing.assert_allclose([v.x for v in x_next_var],
                                           xdot[i].detach().numpy())
                np.testing.assert_array_less(xdot[i].detach().numpy(),
                                             xdot_ub.detach().numpy())
                np.testing.assert_array_less(xdot_lb.detach().numpy(),
                                             xdot[i].detach().numpy())


if __name__ == "__main__":
    unittest.main()
