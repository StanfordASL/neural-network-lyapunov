import neural_network_lyapunov.examples.pendulum.pendulum as pendulum

import unittest

import torch
import numpy as np
import scipy.integrate
import gurobipy

import neural_network_lyapunov.mip_utils as mip_utils
import neural_network_lyapunov.utils as utils
import neural_network_lyapunov.gurobi_torch_mip as gurobi_torch_mip
import neural_network_lyapunov.relu_system as relu_system
import neural_network_lyapunov.feedback_system as feedback_system
import neural_network_lyapunov.continuous_time_lyapunov as\
    continuous_time_lyapunov


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
        self.controller_network = utils.setup_relu((2, 3, 1),
                                                   params=None,
                                                   negative_slope=0.1,
                                                   bias=True,
                                                   dtype=self.dtype)
        self.controller_network[0].weight.data = torch.tensor(
            [[1, 2], [3, -1], [0, 1]], dtype=self.dtype)
        self.controller_network[0].bias.data = torch.tensor([1, 0, -2],
                                                            dtype=self.dtype)
        self.controller_network[2].weight.data = torch.tensor([[1, 3, -1]],
                                                              dtype=self.dtype)
        self.controller_network[2].bias.data = torch.tensor([1],
                                                            dtype=self.dtype)

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
        for method in list(mip_utils.PropagateBoundsMethod):
            dut.network_bound_propagate_method = method
            prog = gurobi_torch_mip.GurobiTorchMIP(self.dtype)
            x_var = prog.addVars(2, lb=-gurobipy.GRB.INFINITY)
            xdot_var = prog.addVars(2, lb=-gurobipy.GRB.INFINITY)
            u_var = prog.addVars(1, lb=-gurobipy.GRB.INFINITY)
            additional_u_lo = torch.tensor([-0.5], dtype=self.dtype)
            additional_u_up = torch.tensor([1.5], dtype=self.dtype)

            # Create an empty control bound prog. Namely there is no
            # constraints between x_var and u_var.
            control_bound_prog = relu_system.ControlBoundProg(None, None, None)
            control_bound_prog.prog = gurobi_torch_mip.GurobiTorchMILP(
                self.dtype)
            control_bound_prog.x_var = control_bound_prog.prog.addVars(
                2, lb=-gurobipy.GRB.INFINITY)
            control_bound_prog.u_var = control_bound_prog.prog.addVars(
                1, lb=-gurobipy.GRB.INFINITY)

            result = dut.add_dynamics_constraint(prog, x_var, xdot_var, u_var,
                                                 "", "", additional_u_lo,
                                                 additional_u_up,
                                                 gurobipy.GRB.BINARY,
                                                 control_bound_prog)
            relu_at_equilibrium = self.dynamics_relu(
                torch.cat((dut.x_equilibrium, dut.u_equilibrium)))
            if method == mip_utils.PropagateBoundsMethod.IA:
                # Check x_next_lb_IA and x_next_ub_IA
                np.testing.assert_allclose(
                    result.x_next_lb_IA.detach().numpy(),
                    torch.stack((x_lo[1], result.nn_output_lo[0] -
                                 relu_at_equilibrium[0])).detach().numpy())
                np.testing.assert_allclose(
                    result.x_next_ub_IA.detach().numpy(),
                    torch.stack((x_up[1], result.nn_output_up[0] -
                                 relu_at_equilibrium[0])).detach().numpy())
            else:
                self.assertIs(result.x_next_bound_prog,
                              control_bound_prog.prog)
                self.assertIs(result.x_var, control_bound_prog.x_var)
                x_next_lb_optimization = np.empty((dut.x_dim, ))
                x_next_ub_optimization = np.empty((dut.x_dim, ))
                result.x_next_bound_prog.gurobi_model.setParam(
                    gurobipy.GRB.Param.OutputFlag, False)
                for i in range(dut.x_dim):
                    result.x_next_bound_prog.setObjective(
                        [torch.tensor([1], dtype=self.dtype)],
                        [[result.x_next_bound_var[i]]], 0.,
                        gurobipy.GRB.MINIMIZE)
                    result.x_next_bound_prog.gurobi_model.optimize()
                    x_next_lb_optimization[
                        i] = result.x_next_bound_prog.gurobi_model.ObjVal
                    result.x_next_bound_prog.setObjective(
                        [torch.tensor([1], dtype=self.dtype)],
                        [[result.x_next_bound_var[i]]], 0.,
                        gurobipy.GRB.MAXIMIZE)
                    result.x_next_bound_prog.gurobi_model.optimize()
                    x_next_ub_optimization[
                        i] = result.x_next_bound_prog.gurobi_model.ObjVal
                if result.x_next_lb_IA is not None:
                    np.testing.assert_array_less(
                        result.x_next_lb_IA.detach().numpy() - 1E-6,
                        x_next_lb_optimization)
                    np.testing.assert_array_less(
                        x_next_ub_optimization - 1E-6,
                        result.x_next_ub_IA.detach().numpy())
                if method in (mip_utils.PropagateBoundsMethod.LP,
                              mip_utils.PropagateBoundsMethod.MIP):
                    self.assertAlmostEqual(x_next_lb_optimization[1],
                                           (result.nn_output_lo[0] -
                                            relu_at_equilibrium[0]).item())
                    self.assertAlmostEqual(x_next_ub_optimization[1],
                                           (result.nn_output_up[0] -
                                            relu_at_equilibrium[0]).item())
                else:
                    # The bound in result.nn_output_lo is computed with IA_MIP,
                    # which is between the IA bound and the MIP bound.
                    self.assertGreaterEqual(x_next_lb_optimization[1],
                                            (result.nn_output_lo[0] -
                                             relu_at_equilibrium[0]).item())
                    self.assertLessEqual(x_next_ub_optimization[1],
                                         (result.nn_output_up[0] -
                                          relu_at_equilibrium[0]).item())

            xu_samples = utils.uniform_sample_in_box(
                torch.cat((x_lo, torch.max(u_lo, additional_u_lo))),
                torch.cat((x_up, torch.min(u_up, additional_u_up))), 1000)
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
                xdot_expected = dut.step_forward(
                    xu_samples[i, :2], xu_samples[i, 2:]).detach().numpy()
                np.testing.assert_allclose(np.array([v.x for v in xdot_var]),
                                           xdot_expected)
                if method == mip_utils.PropagateBoundsMethod.IA:
                    np.testing.assert_array_less(
                        xdot_expected,
                        result.x_next_ub_IA.detach().numpy())
                    np.testing.assert_array_less(
                        result.x_next_lb_IA.detach().numpy(), xdot_expected)

    def test_feedback_system(self):
        x_lo = torch.tensor([-2, -4], dtype=self.dtype)
        x_up = torch.tensor([2, 3], dtype=self.dtype)
        u_lo = torch.tensor([-1], dtype=self.dtype)
        u_up = torch.tensor([1], dtype=self.dtype)
        forward_system = pendulum.PendulumReluContinuousTime(
            self.dtype, x_lo, x_up, u_lo, u_up, self.dynamics_relu)
        dut = feedback_system.FeedbackSystem(forward_system,
                                             self.controller_network,
                                             forward_system.x_equilibrium,
                                             forward_system.u_equilibrium,
                                             u_lo.detach().numpy(),
                                             u_up.detach().numpy())
        for method in list(mip_utils.PropagateBoundsMethod):
            forward_system.network_bound_propagate_method = method
            dut.controller_network_bound_propagate_method = method
            mip = gurobi_torch_mip.GurobiTorchMIP(self.dtype)
            x_var = mip.addVars(2, lb=-gurobipy.GRB.INFINITY)
            x_next_var = mip.addVars(2, lb=-gurobipy.GRB.INFINITY)
            u_var, forward_dynamics_return, controller_mip_cnstr_return =\
                dut.add_dynamics_mip_constraint(
                    mip, x_var, x_next_var, "", "", "", "", "",
                    gurobipy.GRB.BINARY)
            x_samples = utils.uniform_sample_in_box(x_lo, x_up, 100)
            mip.gurobi_model.setParam(gurobipy.GRB.Param.OutputFlag, False)
            if method == mip_utils.PropagateBoundsMethod.IA:
                x_next_lb = forward_dynamics_return.x_next_lb_IA
                x_next_ub = forward_dynamics_return.x_next_ub_IA
            else:
                x_next_lb = torch.empty((dut.x_dim, ), dtype=self.dtype)
                x_next_ub = torch.empty((dut.x_dim, ), dtype=self.dtype)
                forward_dynamics_return.x_next_bound_prog.gurobi_model.\
                    setParam(gurobipy.GRB.Param.OutputFlag, False)
                for i in range(dut.x_dim):
                    forward_dynamics_return.x_next_bound_prog.setObjective(
                        [torch.tensor([1], dtype=self.dtype)],
                        [[forward_dynamics_return.x_next_bound_var[i]]], 0.,
                        gurobipy.GRB.MAXIMIZE)
                    forward_dynamics_return.x_next_bound_prog.gurobi_model.\
                        optimize()
                    x_next_ub[i] = forward_dynamics_return.x_next_bound_prog.\
                        gurobi_model.ObjVal
                    forward_dynamics_return.x_next_bound_prog.setObjective(
                        [torch.tensor([1], dtype=self.dtype)],
                        [[forward_dynamics_return.x_next_bound_var[i]]], 0.,
                        gurobipy.GRB.MINIMIZE)
                    forward_dynamics_return.x_next_bound_prog.gurobi_model.\
                        optimize()
                    x_next_lb[i] = forward_dynamics_return.x_next_bound_prog.\
                        gurobi_model.ObjVal
            for i in range(x_samples.shape[0]):
                for j in range(2):
                    x_var[j].lb = x_samples[i, j].item()
                    x_var[j].ub = x_samples[i, j].item()
                mip.gurobi_model.optimize()
                self.assertEqual(mip.gurobi_model.status,
                                 gurobipy.GRB.Status.OPTIMAL)
                u_val = dut.compute_u(x_samples[i])
                np.testing.assert_allclose(np.array([v.x for v in u_var]),
                                           u_val.detach().numpy())
                x_next_val = forward_system.step_forward(x_samples[i], u_val)
                np.testing.assert_allclose(np.array([v.x for v in x_next_var]),
                                           x_next_val.detach().numpy())
                np.testing.assert_array_less(x_next_val.detach().numpy(),
                                             x_next_ub.detach().numpy())
                np.testing.assert_array_less(x_next_lb.detach().numpy(),
                                             x_next_val.detach().numpy())

    def test_lyapunov(self):
        x_lo = torch.tensor([-2, -4], dtype=self.dtype)
        x_up = torch.tensor([2, 3], dtype=self.dtype)
        u_lo = torch.tensor([-1], dtype=self.dtype)
        u_up = torch.tensor([1], dtype=self.dtype)
        forward_system = pendulum.PendulumReluContinuousTime(
            self.dtype, x_lo, x_up, u_lo, u_up, self.dynamics_relu)
        closed_loop_system = feedback_system.FeedbackSystem(
            forward_system, self.controller_network,
            forward_system.x_equilibrium, forward_system.u_equilibrium,
            u_lo.detach().numpy(),
            u_up.detach().numpy())
        lyapunov_relu = utils.setup_relu((2, 3, 2, 1),
                                         params=None,
                                         negative_slope=0.1,
                                         bias=True,
                                         dtype=self.dtype)
        lyapunov_relu[0].weight.data = torch.tensor([[1, 2], [3, -1], [1, 0]],
                                                    dtype=self.dtype)
        lyapunov_relu[0].bias.data = torch.tensor([1, 2, -3], dtype=self.dtype)
        lyapunov_relu[2].weight.data = torch.tensor(
            [[3, 0.5, 1], [1, 0.5, -2]], dtype=self.dtype)
        lyapunov_relu[2].bias.data = torch.tensor([1, -2], dtype=self.dtype)
        lyapunov_relu[4].weight.data = torch.tensor([[2, -3]],
                                                    dtype=self.dtype)
        lyapunov_relu[4].bias.data = torch.tensor([1], dtype=self.dtype)
        dut = continuous_time_lyapunov.LyapunovContinuousTimeSystem(
            closed_loop_system, lyapunov_relu)
        for method in list(mip_utils.PropagateBoundsMethod):
            forward_system.network_bound_propagate_method = method
            closed_loop_system.controller_network_bound_propagate_method = \
                method
            milp = gurobi_torch_mip.GurobiTorchMILP(self.dtype)
            x = milp.addVars(closed_loop_system.x_dim,
                             lb=-gurobipy.GRB.INFINITY)
            xdot = milp.addVars(closed_loop_system.x_dim,
                                lb=-gurobipy.GRB.INFINITY)
            system_constraint_return = dut.add_system_constraint(
                milp, x, xdot, binary_var_type=gurobipy.GRB.BINARY)
            if method == mip_utils.PropagateBoundsMethod.IA:
                xdot_lb = system_constraint_return.x_next_lb_IA
                xdot_ub = system_constraint_return.x_next_ub_IA
            else:
                xdot_lb = torch.empty((closed_loop_system.x_dim, ),
                                      dtype=self.dtype)
                xdot_ub = torch.empty((closed_loop_system.x_dim, ),
                                      dtype=self.dtype)
                system_constraint_return.x_next_bound_prog.gurobi_model.\
                    setParam(gurobipy.GRB.Param.OutputFlag, False)
                for i in range(closed_loop_system.x_dim):
                    system_constraint_return.x_next_bound_prog.setObjective(
                        [torch.tensor([1], dtype=self.dtype)],
                        [[system_constraint_return.x_next_bound_var[i]]],
                        0.,
                        sense=gurobipy.GRB.MAXIMIZE)
                    system_constraint_return.x_next_bound_prog.gurobi_model.\
                        optimize()
                    xdot_ub[i] = system_constraint_return.x_next_bound_prog.\
                        gurobi_model.ObjVal
                    system_constraint_return.x_next_bound_prog.setObjective(
                        [torch.tensor([1], dtype=self.dtype)],
                        [[system_constraint_return.x_next_bound_var[i]]],
                        0.,
                        sense=gurobipy.GRB.MINIMIZE)
                    system_constraint_return.x_next_bound_prog.gurobi_model.\
                        optimize()
                    xdot_lb[i] = system_constraint_return.x_next_bound_prog.\
                        gurobi_model.ObjVal
            torch.manual_seed(0)
            x_samples = utils.uniform_sample_in_box(x_lo, x_up, 100)
            xdot_samples = closed_loop_system.step_forward(x_samples)
            milp.gurobi_model.setParam(gurobipy.GRB.Param.OutputFlag, False)
            for i in range(x_samples.shape[0]):
                for j in range(closed_loop_system.x_dim):
                    x[j].lb = x_samples[i, j].item()
                    x[j].ub = x_samples[i, j].item()
                milp.gurobi_model.optimize()
                np.testing.assert_allclose([v.x for v in xdot],
                                           xdot_samples[i].detach().numpy(),
                                           atol=1E-5)
                np.testing.assert_array_less(xdot_samples[i].detach().numpy(),
                                             xdot_ub.detach().numpy())
                np.testing.assert_array_less(xdot_lb.detach().numpy(),
                                             xdot_samples[i].detach().numpy())


if __name__ == "__main__":
    unittest.main()
