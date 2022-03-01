import neural_network_lyapunov.examples.pendulum.pendulum as pendulum

import unittest

import torch
import numpy as np
import scipy.integrate
import gurobipy

import neural_network_lyapunov.mip_utils as mip_utils
import neural_network_lyapunov.utils as utils
import neural_network_lyapunov.gurobi_torch_mip as gurobi_torch_mip


class TestPendulum(unittest.TestCase):
    def test_dynamics_gradient(self):
        dtype = torch.float64
        dut = pendulum.Pendulum(dtype)

        def test_gradient(x, u):
            xdot = dut.dynamics(x, u)
            (A_expected0, B_expected0) = torch.autograd.grad(
                xdot, (x, u),
                grad_outputs=torch.tensor([1, 0], dtype=dtype),
                retain_graph=True)
            (A_expected1, B_expected1) = torch.autograd.grad(
                xdot, (x, u),
                grad_outputs=torch.tensor([0, 1], dtype=dtype),
                retain_graph=True)
            A_expected = torch.cat((A_expected0.reshape(
                (1, -1)), A_expected1.reshape((1, -1))),
                                   dim=0)
            B_expected = torch.cat((B_expected0.reshape(
                (1, -1)), B_expected1.reshape((1, -1))),
                                   dim=0)
            A, B = dut.dynamics_gradient(x)
            np.testing.assert_allclose(A_expected.detach().numpy(),
                                       A.detach().numpy())
            np.testing.assert_allclose(B_expected.detach().numpy(),
                                       B.detach().numpy())

        test_gradient(torch.tensor([0, 1], dtype=dtype, requires_grad=True),
                      torch.tensor([2], dtype=dtype, requires_grad=True))
        test_gradient(torch.tensor([3, 1], dtype=dtype, requires_grad=True),
                      torch.tensor([2], dtype=dtype, requires_grad=True))
        test_gradient(torch.tensor([3, -1], dtype=dtype, requires_grad=True),
                      torch.tensor([2], dtype=dtype, requires_grad=True))

    def test_lqr_control(self):
        dtype = torch.float64
        dut = pendulum.Pendulum(dtype)

        Q = np.diag(np.array([1, 10.]))
        R = np.eye(1)
        K = dut.lqr_control(Q, R)

        # Now start with a state close to the [pi, 0], and simulate it with the
        # lqr controller.
        x_des = np.array([np.pi, 0])

        result = scipy.integrate.solve_ivp(
            lambda t, x: dut.dynamics(x, K @ (x - x_des)), (0, 5),
            np.array([np.pi + 0.05, 0.1]))
        np.testing.assert_allclose(result.y[:, -1], x_des, atol=2E-5)

    def test_swing_up_control(self):
        # We use a energy shaping controller and an LQR controller to swing up
        # the pendulum.
        plant = pendulum.Pendulum(torch.float64)
        Q = np.diag([1, 10])
        R = np.array([[1]])
        x_des = np.array([np.pi, 0])
        lqr_gain = plant.lqr_control(Q, R)

        def controller(x):
            if (x - x_des).dot(Q @ (x - x_des)) > 0.1:
                u = plant.energy_shaping_control(x, x_des, 10)
            else:
                u = lqr_gain @ (x - x_des)
            return u

        def converged(t, y):
            return np.linalg.norm(y - x_des) - 1E-3

        converged.terminal = True

        x0s = [np.array([0, 0.]), np.array([0.1, -1]), np.array([1.5, 0.5])]
        for x0 in x0s:
            result = scipy.integrate.solve_ivp(
                lambda t, x: plant.dynamics(x, controller(x)), (0, 10),
                x0,
                events=converged)
            np.testing.assert_allclose(result.y[:, -1], x_des, atol=1E-3)


class TestPendulumReluContinuousTime(unittest.TestCase):
    def setUp(self):
        self.dtype = torch.float64
        self.dynamics_relu = utils.setup_relu((3, 2, 3, 1),
                                              params=None,
                                              negative_slope=0.1,
                                              bias=True,
                                              dtype=self.dtype)
        self.dynamics_relu[0].weight.data = torch.tensor(
            [[1, 2, -3], [0, -1, 2]], dtype=self.dtype)
        self.dynamics_relu[0].bias.data = torch.tensor([1, -2],
                                                       dtype=self.dtype)
        self.dynamics_relu[2].weight.data = torch.tensor(
            [[1, 4], [-1, 2], [3, -2]], dtype=self.dtype)
        self.dynamics_relu[2].bias.data = torch.tensor([1, 2, -3],
                                                       dtype=self.dtype)
        self.dynamics_relu[4].weight.data = torch.tensor([[1, 3, 2]],
                                                         dtype=self.dtype)
        self.dynamics_relu[4].bias.data = torch.tensor([0.5], dtype=self.dtype)

    def test_step_forward(self):
        dut = pendulum.PendulumReluContinuousTime(
            self.dtype, torch.tensor([-2, -4], dtype=self.dtype),
            torch.tensor([2, 3], dtype=self.dtype),
            torch.tensor([-1], dtype=self.dtype),
            torch.tensor([1], dtype=self.dtype), self.dynamics_relu)
        np.testing.assert_allclose(
            dut.step_forward(dut.x_equilibrium,
                             dut.u_equilibrium).detach().numpy(),
            np.zeros((2, )))
        x = torch.tensor([2, 4], dtype=self.dtype)
        u = torch.tensor([3], dtype=self.dtype)
        xdot = dut.step_forward(x, u)
        self.assertEqual(xdot[0].item(), x[1].item())
        self.assertAlmostEqual(
            xdot[1].item(),
            (self.dynamics_relu(torch.cat((x, u))) - self.dynamics_relu(
                torch.cat((dut.x_equilibrium, dut.u_equilibrium))))[0].item())

    def test_add_dynamics_constraint(self):
        x_lo = torch.tensor([-2, -4], dtype=self.dtype)
        x_up = torch.tensor([2, 3], dtype=self.dtype)
        u_lo = torch.tensor([-1], dtype=self.dtype)
        u_up = torch.tensor([1], dtype=self.dtype)
        dut = pendulum.PendulumReluContinuousTime(self.dtype, x_lo, x_up, u_lo,
                                                  u_up, self.dynamics_relu)
        for method in (mip_utils.PropagateBoundsMethod.IA,
                       mip_utils.PropagateBoundsMethod.MIP):
            dut.network_bound_propagate_method = method
            prog = gurobi_torch_mip.GurobiTorchMIP(self.dtype)
            x_var = prog.addVars(2, lb=-gurobipy.GRB.INFINITY)
            xdot_var = prog.addVars(2, lb=-gurobipy.GRB.INFINITY)
            u_var = prog.addVars(1, lb=-gurobipy.GRB.INFINITY)
            dut.add_dynamics_constraint(prog, x_var, xdot_var, u_var, "", "")
            xu_samples = utils.uniform_sample_in_box(torch.cat((x_lo, u_lo)),
                                                     torch.cat((x_up, u_up)),
                                                     1000)
            for i in range(xu_samples.shape[0]):
                x_var[0].lb = xu_samples[i, 0]
                x_var[0].ub = xu_samples[i, 0]
                x_var[1].lb = xu_samples[i, 1]
                x_var[1].ub = xu_samples[i, 1]
                u_var[0].lb = xu_samples[i, 2]
                u_var[0].ub = xu_samples[i, 2]
                prog.gurobi_model.setParam(gurobipy.GRB.Param.OutputFlag,
                                           False)
                prog.gurobi_model.optimize()
                self.assertEqual(prog.gurobi_model.status,
                                 gurobipy.GRB.Status.OPTIMAL)
                np.testing.assert_allclose(
                    np.array([v.x for v in xdot_var]),
                    dut.step_forward(xu_samples[i, :2],
                                     xu_samples[i, 2:]).detach().numpy())


if __name__ == "__main__":
    unittest.main()
