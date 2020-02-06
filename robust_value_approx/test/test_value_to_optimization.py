import robust_value_approx.value_to_optimization as value_to_optimization
import robust_value_approx.ball_paddle_hybrid_linear_system as bphls
import robust_value_approx.hybrid_linear_system as hybrid_linear_system
from robust_value_approx.utils import torch_to_numpy, get_simple_trajopt_cost
import double_integrator

import unittest
import numpy as np
import cvxpy as cp
import torch


class ValueToOptimizationTest(unittest.TestCase):
    def test_trajopt_x0xN(self):
        """
        x = [ballx, bally, paddley, paddletheta, ballvx, ballvy, paddlevy]
        u = [paddletheta_dot, paddlevy_dot]
        """
        dtype = torch.float64
        dt = .1
        x_lo = torch.Tensor(
            [-1., -1., 0., -1e3, -1e3, -1e3]).type(dtype)
        x_up = torch.Tensor(
            [1., 10., 2., 1e3, 1e3, 1e3]).type(dtype)
        u_lo = torch.Tensor([-np.pi / 2, -1e3]).type(dtype)
        u_up = torch.Tensor([np.pi / 2, 1e3]).type(dtype)
        sys = bphls.get_ball_paddle_hybrid_linear_system(
            dtype, dt, x_lo, x_up, u_lo, u_up)
        N = 10
        vf = value_to_optimization.ValueFunction(
            sys, N, x_lo, x_up, u_lo, u_up)
        [Q, R, Z, q, r, z, Qt, Rt, Zt, qt, rt, zt] = get_simple_trajopt_cost(
            sys.x_dim, sys.u_dim, sys.num_modes, dtype)
        vf.set_cost(Q=Q, R=R, Z=Z, q=q, r=r, z=z)
        vf.set_terminal_cost(Qt=Qt, Rt=Rt, Zt=Zt, qt=qt, rt=rt, zt=zt)
        # x = [ballx, bally, paddley, paddletheta, ballvx, ballvy, paddlevy]
        x0 = torch.Tensor([0., .25, .15, 0., 0., 0.])
        xN = torch.Tensor([np.nan, .5, np.nan, np.nan, np.nan, np.nan])
        vf.set_constraints(x0=x0, xN=xN)
        traj_opt = vf.traj_opt_constraint()
        (Ain1, Ain2, Ain3, rhs_in,
         Aeq1, Aeq2, Aeq3, rhs_eq,
         Q1, Q2, Q3, q1, q2, q3, c) = torch_to_numpy(traj_opt)
        x = cp.Variable(Ain1.shape[1])
        s = cp.Variable(Ain2.shape[1])
        z = cp.Variable(Ain3.shape[1], boolean=True)
        obj = cp.Minimize(.5 * cp.quad_form(x, Q1) +
                          .5 * cp.quad_form(s, Q2) +
                          .5 * cp.quad_form(z, Q3) +
                          q1.T@x + q2.T@s + q3.T@z + c)
        con = [
            Ain1@x + Ain2@s + Ain3@z <= rhs_in,
            Aeq1@x + Aeq2@s + Aeq3@z == rhs_eq,
        ]
        prob = cp.Problem(obj, con)
        prob.solve(solver=cp.GUROBI, verbose=False)
        traj = np.hstack((x.value, s.value))
        traj = np.reshape(traj, ((1 + sys.num_modes) *
                                 (sys.x_dim + sys.u_dim), N), order='F')
        self.assertAlmostEqual(traj[1, 0], x0[1])
        self.assertAlmostEqual(traj[2, 0], x0[2])
        self.assertAlmostEqual(traj[5, 0], x0[5])
        self.assertAlmostEqual(traj[1, -1], xN[1])

    def test_trajopt_obj(self):
        dtype = torch.float64
        dt = .01
        N = 5
        x_lo = torch.Tensor(
            [-1., -1., 0., -1e6, -1e6, -1e6]).type(dtype)
        x_up = torch.Tensor(
            [1., 10., 2., 1e6, 1e6, 1e6]).type(dtype)
        u_lo = torch.Tensor([-np.pi / 2, -1e5]).type(dtype)
        u_up = torch.Tensor([np.pi / 2, 1e5]).type(dtype)
        sys = bphls.get_ball_paddle_hybrid_linear_system(
            dtype, dt, x_lo, x_up, u_lo, u_up)
        vf = value_to_optimization.ValueFunction(
            sys, N, x_lo, x_up, u_lo, u_up)
        [Q, R, Z, q, r, z, Qt, Rt, Zt, qt, rt, zt] = get_simple_trajopt_cost(
            sys.x_dim, sys.u_dim, sys.num_modes, dtype)
        vf.set_cost(Q=Q, R=R, Z=Z, q=q, r=r, z=z)
        vf.set_terminal_cost(Qt=Qt, Rt=Rt, Zt=Zt, qt=qt, rt=rt, zt=zt)
        (Ain1, Ain2, Ain3, rhs_in,
         Aeq1, Aeq2, Aeq3, rhs_eq,
         Q1, Q2, Q3, q1, q2, q3, c) = vf.traj_opt_constraint()
        x0 = torch.rand(sys.x_dim).type(sys.dtype)
        x = torch.rand(sys.x_dim, N - 1).type(sys.dtype)
        u0 = torch.rand(sys.u_dim).type(sys.dtype)
        u = torch.rand(sys.u_dim, N - 1).type(sys.dtype)
        slack0 = torch.rand(
            sys.num_modes * (sys.x_dim + sys.u_dim)).type(sys.dtype)
        slack = torch.rand(sys.num_modes * (sys.x_dim +
                                            sys.u_dim), N - 1).type(sys.dtype)
        s = torch.cat((u0, slack0, torch.cat(
            (x, u, slack), 0).t().reshape(-1)))
        alpha = torch.rand(sys.num_modes, N).type(sys.dtype)
        obj_exp = .5 * x0@Q@x0.t()
        obj_exp += x0.t()@q
        obj_exp += .5 * u0@R@u0.t()
        obj_exp += u0.t()@r
        for i in range(N - 2):
            obj_exp += .5 * x[:, i].t()@Q@x[:, i]
            obj_exp += x[:, i].t()@q
            obj_exp += .5 * u[:, i].t()@R@u[:, i]
            obj_exp += u[:, i].t()@r
        i = N - 2
        obj_exp += .5 * x[:, i].t()@Qt@x[:, i]
        obj_exp += x[:, i].t()@qt
        obj_exp += .5 * u[:, i].t()@Rt@u[:, i]
        obj_exp += u[:, i].t()@rt
        for i in range(N - 1):
            obj_exp += .5 * alpha[:, i].t()@Z@alpha[:, i]
            obj_exp += alpha[:, i].t()@z
        i = N - 1
        obj_exp += .5 * alpha[:, i].t()@Zt@alpha[:, i]
        obj_exp += alpha[:, i].t()@zt
        alpha_flat = alpha.t().reshape(-1)
        obj = .5 * x0@Q1@x0.t() + .5 * s@Q2@s.t() +\
            .5 * alpha_flat@Q3@alpha_flat.t() +\
            x0@q1 + s@q2 + alpha_flat@q3 + c
        self.assertAlmostEqual(obj_exp.item(), obj.item())

    def test_trajopt_lim(self):
        dtype = torch.float64
        dt = .01
        N = 10
        x_lo = torch.Tensor(
            [-1., -1., 0., -1e6, -1e6, -1e6]).type(dtype)
        x_up = torch.Tensor(
            [1., 10., 2., 1e6, 1e6, 1e6]).type(dtype)
        u_lo = torch.Tensor([-np.pi / 2, -1e5]).type(dtype)
        u_up = torch.Tensor([np.pi / 2, 1e5]).type(dtype)
        sys = bphls.get_ball_paddle_hybrid_linear_system(
            dtype, dt, x_lo, x_up, u_lo, u_up)
        x_lo = torch.Tensor([.1, .1, .1, .1, .1, .1]).type(dtype)
        u_up = torch.Tensor([0., 0.]).type(dtype)
        vf = value_to_optimization.ValueFunction(
            sys, N, x_lo, x_up, u_lo, u_up)
        vf.set_cost(q=torch.ones(sys.x_dim), r=-torch.ones(sys.u_dim))
        traj_opt = vf.traj_opt_constraint()
        (Ain1, Ain2, Ain3, rhs_in,
         Aeq1, Aeq2, Aeq3, rhs_eq,
         Q1, Q2, Q3, q1, q2, q3, c) = torch_to_numpy(traj_opt)
        x = cp.Variable(Ain1.shape[1])
        s = cp.Variable(Ain2.shape[1])
        z = cp.Variable(Ain3.shape[1], boolean=True)
        obj = cp.Minimize(.5 * cp.quad_form(x, Q1) +
                          .5 * cp.quad_form(s, Q2) +
                          .5 * cp.quad_form(z, Q3) +
                          q1@x + q2.T@s + q3.T@z + c)
        con = [
            Ain1@x + Ain2@s + Ain3@z <= rhs_in,
            Aeq1@x + Aeq2@s + Aeq3@z == rhs_eq,
        ]
        prob = cp.Problem(obj, con)
        prob.solve(solver=cp.GUROBI, verbose=False)
        traj = np.hstack((x.value, s.value))
        traj = np.reshape(traj, ((1 + sys.num_modes) *
                                 (sys.x_dim + sys.u_dim), N), order='F')
        xtraj = traj[:sys.x_dim, :]
        utraj = traj[sys.x_dim:sys.x_dim + sys.u_dim, :]
        for i in range(N):
            for j in range(sys.x_dim):
                self.assertLessEqual(x_lo[j], xtraj[j, i])
            for j in range(sys.u_dim):
                self.assertLessEqual(utraj[j, i], u_up[j])

    def test_trajopt_obj_trajsetpoint(self):
        dtype = torch.float64
        dt = .01
        N = 5
        x_lo = torch.Tensor(
            [-1., -1., 0., -1e6, -1e6, -1e6]).type(dtype)
        x_up = torch.Tensor(
            [1., 10., 2., 1e6, 1e6, 1e6]).type(dtype)
        u_lo = torch.Tensor([-np.pi / 2, -1e5]).type(dtype)
        u_up = torch.Tensor([np.pi / 2, 1e5]).type(dtype)
        sys = bphls.get_ball_paddle_hybrid_linear_system(
            dtype, dt, x_lo, x_up, u_lo, u_up)
        vf = value_to_optimization.ValueFunction(
            sys, N, x_lo, x_up, u_lo, u_up)
        [Q, R, Z, q, r, z, Qt, Rt, Zt, qt, rt, zt] = get_simple_trajopt_cost(
            sys.x_dim, sys.u_dim, sys.num_modes, dtype)
        xtraj0 = torch.rand(sys.x_dim, 1).type(sys.dtype)
        utraj0 = torch.rand(sys.u_dim, 1).type(sys.dtype)
        utraj = torch.rand(sys.u_dim, N - 1).type(sys.dtype)
        xtraj = torch.rand(sys.x_dim, N - 1).type(sys.dtype)
        alphatraj = torch.rand(sys.num_modes, N).type(sys.dtype)
        vf.set_cost(Q=Q, R=R, Z=Z, q=q, r=r, z=z)
        vf.set_terminal_cost(Qt=Qt, Rt=Rt, Zt=Zt, qt=qt, rt=rt, zt=zt)
        vf.set_traj(xtraj=torch.cat((xtraj0, xtraj), 1), utraj=torch.cat(
            (utraj0, utraj), 1), alphatraj=alphatraj)
        (Ain1, Ain2, Ain3, rhs_in,
         Aeq1, Aeq2, Aeq3, rhs_eq,
         Q1, Q2, Q3, q1, q2, q3, c) = vf.traj_opt_constraint()
        x0 = torch.rand(sys.x_dim).type(sys.dtype)
        x = torch.rand(sys.x_dim, N - 1).type(sys.dtype)
        u0 = torch.rand(sys.u_dim).type(sys.dtype)
        u = torch.rand(sys.u_dim, N - 1).type(sys.dtype)
        slack0 = torch.rand(
            sys.num_modes * (sys.x_dim + sys.u_dim)).type(sys.dtype)
        slack = torch.rand(sys.num_modes * (sys.x_dim +
                                            sys.u_dim), N - 1).type(sys.dtype)
        s = torch.cat((u0, slack0, torch.cat(
            (x, u, slack), 0).t().reshape(-1)))
        alpha = torch.rand(sys.num_modes, N).type(sys.dtype)
        obj_exp = .5 * (x0 - xtraj0.t())@Q@(x0 - xtraj0.t()).t()
        obj_exp += (x0 - xtraj0.t())@q
        obj_exp += .5 * (u0 - utraj0.t())@R@(u0 - utraj0.t()).t()
        obj_exp += (u0 - utraj0.t())@r
        for i in range(N - 2):
            obj_exp += .5 * (x[:, i] - xtraj[:, i]
                             ).t()@Q@(x[:, i] - xtraj[:, i])
            obj_exp += (x[:, i] - xtraj[:, i]).t()@q
            obj_exp += .5 * (u[:, i] - utraj[:, i]
                             ).t()@R@(u[:, i] - utraj[:, i])
            obj_exp += (u[:, i] - utraj[:, i]).t()@r
        i = N - 2
        obj_exp += .5 * (x[:, i] - xtraj[:, i]).t()@Qt@(x[:, i] - xtraj[:, i])
        obj_exp += (x[:, i] - xtraj[:, i]).t()@qt
        obj_exp += .5 * (u[:, i] - utraj[:, i]).t()@Rt@(u[:, i] - utraj[:, i])
        obj_exp += (u[:, i] - utraj[:, i]).t()@rt
        for i in range(N - 1):
            obj_exp += .5 * (alpha[:, i] - alphatraj[:, i]
                             ).t()@Z@(alpha[:, i] - alphatraj[:, i])
            obj_exp += (alpha[:, i] - alphatraj[:, i]).t()@z
        i = N - 1
        obj_exp += .5 * (alpha[:, i] - alphatraj[:, i]
                         ).t()@Zt@(alpha[:, i] - alphatraj[:, i])
        obj_exp += (alpha[:, i] - alphatraj[:, i]).t()@zt
        alpha_flat = alpha.t().reshape(-1)
        obj = .5 * x0@Q1@x0 +\
            .5 * s@Q2@s +\
            .5 * alpha_flat@Q3@alpha_flat +\
            x0@q1 + s@q2 + alpha_flat@q3 + c
        self.assertAlmostEqual(obj_exp.item(), obj.item())

    def test_trajopt_midpoint(self):
        dtype = torch.float64
        dt = .1
        x_lo = torch.Tensor(
            [-1., -1., 0., -1e6, -1e6, -1e6]).type(dtype)
        x_up = torch.Tensor(
            [1., 10., .3, 1e6, 1e6, 1e6]).type(dtype)
        u_lo = torch.Tensor([-np.pi / 2, -1e6]).type(dtype)
        u_up = torch.Tensor([np.pi / 2, 1e6]).type(dtype)
        sys = bphls.get_ball_paddle_hybrid_linear_system(
            dtype, dt, x_lo, x_up, u_lo, u_up,
            collision_eps=1e-2, midpoint=True)
        N = 10
        vf = value_to_optimization.ValueFunction(
            sys, N, x_lo, x_up, u_lo, u_up)
        Q = torch.diag(torch.Tensor([0., 10., 0., 0., 1., 0.]).type(dtype))
        R = torch.diag(torch.Tensor([0., .01]).type(dtype))
        vf.set_cost(Q=Q, R=R)
        vf.set_terminal_cost(Qt=Q, Rt=R)
        xN = torch.Tensor([np.nan, .6, np.nan, np.nan, 0., np.nan]).type(dtype)
        vf.set_constraints(xN=xN)
        # x = [ballx, bally, paddley, paddletheta, ballvx, ballvy, paddlevy]
        x0 = torch.Tensor([0., .5, 0., 0., 0., 0.]).type(dtype)
        traj_opt = vf.traj_opt_constraint()
        (Ain1, Ain2, Ain3, rhs_in,
         Aeq1, Aeq2, Aeq3, rhs_eq,
         Q1, Q2, Q3, q1, q2, q3, c) = torch_to_numpy(traj_opt)
        s = cp.Variable(Ain2.shape[1])
        z = cp.Variable(Ain3.shape[1], boolean=True)
        obj = cp.Minimize(.5 * x0.detach().numpy()@Q1@x0.detach().numpy().T +
                          .5 * cp.quad_form(s, Q2) +
                          .5 * cp.quad_form(z, Q3) +
                          q1.T@x0.detach().numpy() + q2.T@s + q3.T@z + c)
        con = [
            Ain1@x0.detach().numpy() + Ain2@s + Ain3@z <= rhs_in,
            Aeq1@x0.detach().numpy() + Aeq2@s + Aeq3@z == rhs_eq,
        ]
        prob = cp.Problem(obj, con)
        prob.solve(solver=cp.GUROBI, verbose=False)
        s_val = torch.Tensor(s.value).type(dtype)
        traj_val = torch.cat((x0, s_val)).reshape(N, -1).t()
        xtraj_val = traj_val[:sys.x_dim, :]
        self.assertAlmostEqual(xtraj_val[1, -1], xN[1])
        self.assertAlmostEqual(xtraj_val[4, -1], xN[4])

    def test_q_fun(self):
        dtype = torch.float64
        (A_c, B_c) = double_integrator.double_integrator_dynamics(dtype)
        x_dim = A_c.shape[1]
        u_dim = B_c.shape[1]
        # continuous to discrete using forward euler
        dt = 1.
        A = torch.eye(x_dim, dtype=dtype) + dt * A_c
        B = dt * B_c
        c = torch.zeros(x_dim, dtype=dtype)
        x_lo = -10. * torch.ones(x_dim, dtype=dtype)
        x_up = 10. * torch.ones(x_dim, dtype=dtype)
        u_lo = -100. * torch.ones(u_dim, dtype=dtype)
        u_up = 100. * torch.ones(u_dim, dtype=dtype)
        P = torch.cat((-torch.eye(x_dim+u_dim),
                       torch.eye(x_dim+u_dim)), 0).type(dtype)
        q = torch.cat((-x_lo, -u_lo,
                       x_up, u_up), 0).type(dtype)
        double_int = hybrid_linear_system.HybridLinearSystem(x_dim,
                                                             u_dim,
                                                             dtype)
        double_int.add_mode(A, B, c, P, q)
        N = 6
        vf = value_to_optimization.ValueFunction(double_int, N,
                                                 x_lo, x_up,
                                                 u_lo, u_up)
        R = torch.eye(double_int.u_dim)
        vf.set_cost(R=R)
        vf.set_terminal_cost(Rt=R)
        Q = vf.get_q_function()
        x0 = torch.Tensor([1., 1.]).type(dtype)
        u0 = torch.Tensor([1.5]).type(dtype)
        self.assertTrue(Q(x0, u0)[0] is not None)
        x_lo_grid = -1. * torch.ones(x_dim, dtype=dtype)
        x_up_grid = 1. * torch.ones(x_dim, dtype=dtype)
        u_lo_grid = -1. * torch.ones(u_dim, dtype=dtype)
        u_up_grid = 1. * torch.ones(u_dim, dtype=dtype)
        x_num_breaks = [2, 2]
        u_num_breaks = [2]
        (x_samples, u_samples, v_samples) = vf.get_q_sample_grid(x_lo_grid,
                                                                 x_up_grid,
                                                                 x_num_breaks,
                                                                 u_lo_grid,
                                                                 u_up_grid,
                                                                 u_num_breaks)
        self.assertEqual(x_samples.shape[0],
                         x_num_breaks[0]*x_num_breaks[1]*u_num_breaks[0])
        self.assertEqual(u_samples.shape[0],
                         x_num_breaks[0]*x_num_breaks[1]*u_num_breaks[0])
        self.assertEqual(v_samples.shape[0],
                         x_num_breaks[0]*x_num_breaks[1]*u_num_breaks[0])

    def test_traj_cost(self):
        dtype = torch.float64
        dt = .1
        x_lo = torch.Tensor(
            [-1., -1., 0., -1e3, -1e3, -1e3]).type(dtype)
        x_up = torch.Tensor(
            [1., 10., 2., 1e3, 1e3, 1e3]).type(dtype)
        u_lo = torch.Tensor([-np.pi / 2, -1e3]).type(dtype)
        u_up = torch.Tensor([np.pi / 2, 1e3]).type(dtype)
        sys = bphls.get_ball_paddle_hybrid_linear_system(
            dtype, dt, x_lo, x_up, u_lo, u_up)
        N = 10
        vf = value_to_optimization.ValueFunction(
            sys, N, x_lo, x_up, u_lo, u_up)
        [Q, R, Z, q, r, z, Qt, Rt, Zt, qt, rt, zt] = get_simple_trajopt_cost(
            sys.x_dim, sys.u_dim, sys.num_modes, dtype)
        vf.set_cost(Q=Q, R=R, Z=Z, q=q, r=r, z=z)
        vf.set_terminal_cost(Qt=Qt, Rt=Rt, Zt=Zt, qt=qt, rt=rt, zt=zt)
        # x = [ballx, bally, paddley, paddletheta, ballvx, ballvy, paddlevy]
        x0 = torch.Tensor([0., .25, .15, 0., 0., 0.])
        xN = torch.Tensor([np.nan, .5, np.nan, np.nan, np.nan, np.nan])
        vf.set_constraints(x0=x0, xN=xN)
        traj_opt = vf.traj_opt_constraint()
        (Ain1, Ain2, Ain3, rhs_in,
         Aeq1, Aeq2, Aeq3, rhs_eq,
         Q1, Q2, Q3, q1, q2, q3, c) = torch_to_numpy(traj_opt)
        x = cp.Variable(Ain1.shape[1])
        s = cp.Variable(Ain2.shape[1])
        z = cp.Variable(Ain3.shape[1], boolean=True)
        obj = cp.Minimize(.5 * cp.quad_form(x, Q1) +
                          .5 * cp.quad_form(s, Q2) +
                          .5 * cp.quad_form(z, Q3) +
                          q1.T@x + q2.T@s + q3.T@z + c)
        con = [
            Ain1@x + Ain2@s + Ain3@z <= rhs_in,
            Aeq1@x + Aeq2@s + Aeq3@z == rhs_eq,
        ]
        prob = cp.Problem(obj, con)
        prob.solve(solver=cp.GUROBI, verbose=False)
        traj = np.hstack((x.value, s.value))
        traj = np.reshape(traj, ((1 + sys.num_modes) *
                                 (sys.x_dim + sys.u_dim), N), order='F')
        xtraj = torch.Tensor(traj[:vf.sys.x_dim, :]).type(dtype)
        utraj = torch.Tensor(
            traj[vf.sys.x_dim:vf.sys.x_dim+vf.sys.u_dim, :]).type(dtype)
        ztraj = torch.Tensor(np.reshape(
            z.value, (vf.sys.num_modes, N), order='F')).type(dtype)
        obj_ = vf.traj_cost(xtraj, utraj, ztraj)
        self.assertAlmostEqual(obj.value, obj_.item(), places=6)

    def test_step_cost(self):
        dtype = torch.float64
        dt = .1
        x_lo = torch.Tensor(
            [-1., -1., 0., -1e3, -1e3, -1e3]).type(dtype)
        x_up = torch.Tensor(
            [1., 10., 2., 1e3, 1e3, 1e3]).type(dtype)
        u_lo = torch.Tensor([-np.pi / 2, -1e3]).type(dtype)
        u_up = torch.Tensor([np.pi / 2, 1e3]).type(dtype)
        sys = bphls.get_ball_paddle_hybrid_linear_system(
            dtype, dt, x_lo, x_up, u_lo, u_up)
        N = 10
        vf = value_to_optimization.ValueFunction(
            sys, N, x_lo, x_up, u_lo, u_up)
        [Q, R, Z, q, r, z, Qt, Rt, Zt, qt, rt, zt] = get_simple_trajopt_cost(
            sys.x_dim, sys.u_dim, sys.num_modes, dtype)
        vf.set_cost(Q=Q, R=R, Z=Z, q=q, r=r, z=z)
        vf.set_terminal_cost(Qt=Qt, Rt=Rt, Zt=Zt, qt=qt, rt=rt, zt=zt)
        xtraj_ = torch.rand((sys.x_dim, N), dtype=dtype)
        utraj_ = torch.rand((sys.u_dim, N), dtype=dtype)
        ztraj_ = torch.rand((sys.num_modes, N), dtype=dtype)
        vf.set_traj(xtraj=xtraj_, utraj=utraj_, alphatraj=ztraj_)
        # x = [ballx, bally, paddley, paddletheta, ballvx, ballvy, paddlevy]
        xtraj = torch.rand((sys.x_dim, N), dtype=dtype)
        utraj = torch.rand((sys.u_dim, N), dtype=dtype)
        ztraj = torch.rand((sys.num_modes, N), dtype=dtype)
        obj = vf.traj_cost(xtraj, utraj, ztraj)
        obj_ = 0.
        for n in range(N):
            obj_ += vf.step_cost(n, xtraj[:, n], utraj[:, n], ztraj[:, n])
        self.assertAlmostEqual(obj.item(), obj_.item(), places=6)


if __name__ == '__main__':
    unittest.main()
