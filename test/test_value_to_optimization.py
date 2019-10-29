from context import value_to_optimization
from context import ball_paddle_hybrid_linear_system as bphls

import unittest
import numpy as np
import cvxpy as cp
import torch
from utils import torch_to_numpy, get_simple_trajopt_cost


class ValueToOptimizationTest(unittest.TestCase):
    def test_trajopt_x0xN(self):
        """
        x = [ballx, bally, paddley, paddletheta, ballvx, ballvy, paddlevy]
        u = [paddletheta_dot, paddlevy_dot]
        """
        dtype = torch.float64
        dt = .001
        x_lo = torch.Tensor(
            [-1., -1., 0., -np.pi / 2, -1e6, -1e6, -1e6]).type(dtype)
        x_up = torch.Tensor(
            [1., 10., 2., np.pi / 2, 1e6, 1e6, 1e6]).type(dtype)
        u_lo = torch.Tensor([-1e5, -1e5]).type(dtype)
        u_up = torch.Tensor([1e5, 1e5]).type(dtype)
        sys = bphls.get_ball_paddle_hybrid_linear_system(
            dtype, dt, x_lo, x_up, u_lo, u_up)

        N = 20
        vf = value_to_optimization.ValueFunction(
            sys, N, x_lo, x_up, u_lo, u_up)
        [Q, R, Z, q, r, z, Qt, Rt, Zt, qt, rt, zt] = get_simple_trajopt_cost(
            sys.x_dim, sys.u_dim, sys.num_modes, dtype)
        vf.set_cost(Q=Q, R=R, Z=Z, q=q, r=r, z=z)
        vf.set_terminal_cost(Qt=Qt, Rt=Rt, Zt=Zt, qt=qt, rt=rt, zt=zt)

        # x = [ballx, bally, paddley, paddletheta, ballvx, ballvy, paddlevy]
        x0 = torch.Tensor([0., .25, 0., 0., 0., 0., 0.])
        xN = torch.Tensor([np.nan, .6, 0., np.nan, np.nan, 0., np.nan])
        vf.set_constraints(x0=x0, xN=xN)

        traj_opt = vf.traj_opt_constraint()
        (Ain1, Ain2, Ain3, rhs_in,
         Aeq1, Aeq2, Aeq3, rhs_eq,
         Q2, Q3, q2, q3, c) = torch_to_numpy(traj_opt)

        x = cp.Variable(Ain1.shape[1])
        s = cp.Variable(Ain2.shape[1])
        z = cp.Variable(Ain3.shape[1], boolean=True)

        obj = cp.Minimize(.5 * cp.quad_form(s, Q2) + .5 * cp.quad_form(z, Q3)
                          + q2.T@s + q3.T@z + c)
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

    def test_trajopt_obj(self):
        dtype = torch.float64
        dt = .01
        N = 5
        x_lo = torch.Tensor(
            [-1., -1., 0., -np.pi / 2, -1e6, -1e6, -1e6]).type(dtype)
        x_up = torch.Tensor(
            [1., 10., 2., np.pi / 2, 1e6, 1e6, 1e6]).type(dtype)
        u_lo = torch.Tensor([-1e5, -1e5]).type(dtype)
        u_up = torch.Tensor([1e5, 1e5]).type(dtype)
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
         Q2, Q3, q2, q3, c) = vf.traj_opt_constraint()

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

        obj_exp = .5 * u0.t()@R@u0.t()
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
        obj = .5 * s@Q2@s.t() + .5 * alpha_flat@Q3@alpha_flat.t() + \
            s@q2 + alpha_flat@q3 + c

        # print(obj_exp.item())
        # print(obj.item())

        self.assertAlmostEqual(obj_exp.item(), obj.item())

    def test_trajopt_lim(self):
        dtype = torch.float64
        dt = .01
        N = 20
        x_lo = torch.Tensor(
            [-1., -1., 0., -np.pi / 2, -1e6, -1e6, -1e6]).type(dtype)
        x_up = torch.Tensor(
            [1., 10., 2., np.pi / 2, 1e6, 1e6, 1e6]).type(dtype)
        u_lo = torch.Tensor([-1e5, -1e5]).type(dtype)
        u_up = torch.Tensor([1e5, 1e5]).type(dtype)
        sys = bphls.get_ball_paddle_hybrid_linear_system(
            dtype, dt, x_lo, x_up, u_lo, u_up)

        x_lo = torch.rand(sys.x_dim)
        u_up = torch.rand(sys.u_dim)
        vf = value_to_optimization.ValueFunction(
            sys, N, x_lo, x_up, u_lo, u_up)
        vf.set_cost(q=torch.ones(sys.x_dim), r=-torch.ones(sys.u_dim))

        traj_opt = vf.traj_opt_constraint()
        (Ain1, Ain2, Ain3, rhs_in,
         Aeq1, Aeq2, Aeq3, rhs_eq,
         Q2, Q3, q2, q3, c) = torch_to_numpy(traj_opt)

        x = cp.Variable(Ain1.shape[1])
        s = cp.Variable(Ain2.shape[1])
        z = cp.Variable(Ain3.shape[1], boolean=True)

        obj = cp.Minimize(.5 * cp.quad_form(s, Q2) + .5 *
                          cp.quad_form(z, Q3) + q2.T@s + q3.T@z + c)
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
            [-1., -1., 0., -np.pi / 2, -1e6, -1e6, -1e6]).type(dtype)
        x_up = torch.Tensor(
            [1., 10., 2., np.pi / 2, 1e6, 1e6, 1e6]).type(dtype)
        u_lo = torch.Tensor([-1e5, -1e5]).type(dtype)
        u_up = torch.Tensor([1e5, 1e5]).type(dtype)
        sys = bphls.get_ball_paddle_hybrid_linear_system(
            dtype, dt, x_lo, x_up, u_lo, u_up)
        vf = value_to_optimization.ValueFunction(
            sys, N, x_lo, x_up, u_lo, u_up)

        [Q, R, Z, q, r, z, Qt, Rt, Zt, qt, rt, zt] = get_simple_trajopt_cost(
            sys.x_dim, sys.u_dim, sys.num_modes, dtype)
        utraj0 = torch.rand(sys.u_dim, 1).type(sys.dtype)
        utraj = torch.rand(sys.u_dim, N - 1).type(sys.dtype)
        xtraj = torch.rand(sys.x_dim, N - 1).type(sys.dtype)
        alphatraj = torch.rand(sys.num_modes, N).type(sys.dtype)
        vf.set_cost(Q=Q, R=R, Z=Z, q=q, r=r, z=z)
        vf.set_terminal_cost(Qt=Qt, Rt=Rt, Zt=Zt, qt=qt, rt=rt, zt=zt)
        torch.cat((utraj0, utraj), 1)
        vf.set_traj(xtraj=xtraj, utraj=torch.cat(
            (utraj0, utraj), 1), alphatraj=alphatraj)

        (Ain1, Ain2, Ain3, rhs_in,
         Aeq1, Aeq2, Aeq3, rhs_eq,
         Q2, Q3, q2, q3, c) = vf.traj_opt_constraint()

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

        obj_exp = .5 * (u0 - utraj0.t())@R@(u0 - utraj0.t()).t()
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
        obj = .5 * s@Q2@s.t() + .5 * alpha_flat@Q3@alpha_flat.t() + \
            s@q2 + alpha_flat@q3 + c

        # print(obj_exp.item())
        # print(obj.item())

        self.assertAlmostEqual(obj_exp.item(), obj.item())


if __name__ == '__main__':
    unittest.main()
