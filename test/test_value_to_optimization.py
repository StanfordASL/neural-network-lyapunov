from context import value_to_optimization
from context import ball_paddle_system

import unittest
import numpy as np
import cvxpy as cp
import torch
from utils import torch_to_numpy


class ValueToOptimizationTest(unittest.TestCase):
    def test_trajopt_x0xN(self):
        N = 20
        sys = ball_paddle_system.BallPaddleSystem(dt=.01)
        vf = value_to_optimization.ValueFunction(sys, N)

        [Q, R, Z, q, r, z, Qt, Rt, Zt, qt, rt, zt] =\
            sys.get_simple_trajopt_cost()
        vf.set_cost(Q=Q, R=R, Z=Z, q=q, r=r, z=z)
        vf.set_terminal_cost(Qt=Qt, Rt=Rt, Zt=Zt, qt=qt, rt=rt, zt=zt)

        x0 = torch.Tensor([0., .1, 0.])
        xN = torch.Tensor([0., .075, 0.])
        vf.set_constraints(x0=x0, xN=xN)

        traj_opt = vf.traj_opt_constraint()
        (Ain1, Ain2, Ain3, rhs_in,
         Aeq1, Aeq2, Aeq3, rhs_eq,
         Q2, Q3, q2, q3, c) = torch_to_numpy(traj_opt)

        x = cp.Variable(Ain1.shape[1])
        s = cp.Variable(Ain2.shape[1])
        z = cp.Variable(Ain3.shape[1], boolean=True)

        obj = cp.Minimize(.5*cp.quad_form(s, Q2) + .5*cp.quad_form(z, Q3)
                          + q2.T@s + q3.T@z + c)
        con = [
            Ain1@x + Ain2@s + Ain3@z <= rhs_in,
            Aeq1@x + Aeq2@s + Aeq3@z == rhs_eq,
        ]

        prob = cp.Problem(obj, con)
        prob.solve(solver=cp.GUROBI, verbose=False)

        traj = np.hstack((x.value, s.value))
        traj = np.reshape(traj, (-1, 4)).T

        xtraj = traj[:3, :]
        utraj = traj[3:, :]

        # plt.plot(xtraj[1,:])
        # plt.plot(xtraj[0,:])
        # plt.legend(['ball','paddle'])
        # plt.show()

        for i in range(3):
            self.assertAlmostEqual(xtraj[i, 0], x0[i])
            self.assertAlmostEqual(xtraj[i, -1], xN[i])

    def test_trajopt_obj(self):
        N = 3
        sys = ball_paddle_system.BallPaddleSystem(dt=.01)
        vf = value_to_optimization.ValueFunction(sys, N)

        [Q, R, Z, q, r, z, Qt, Rt, Zt, qt, rt, zt] =\
            sys.get_simple_trajopt_cost()
        vf.set_cost(Q=Q, R=R, Z=Z, q=q, r=r, z=z)
        vf.set_terminal_cost(Qt=Qt, Rt=Rt, Zt=Zt, qt=qt, rt=rt, zt=zt)

        x0 = torch.Tensor([0., .1, 0.])
        xN = torch.Tensor([0., .075, 0.])
        vf.set_constraints(x0=x0, xN=xN)

        (Ain1, Ain2, Ain3, rhs_in,
         Aeq1, Aeq2, Aeq3, rhs_eq,
         Q2, Q3, q2, q3, c) = vf.traj_opt_constraint()

        x = torch.rand(3, N-1).type(sys.dtype)
        u0 = torch.rand(1).type(sys.dtype)
        u = torch.rand(1, N-1).type(sys.dtype)
        s = torch.cat((u0, torch.cat((x, u), 0).t().reshape(-1)))
        alpha = torch.rand(1, N).type(sys.dtype)

        obj_exp = .5*u0.t()@R@u0.t()
        obj_exp += u0.t()@r
        for i in range(N-2):
            obj_exp += .5*x[:, i].t()@Q@x[:, i]
            obj_exp += x[:, i].t()@q
            obj_exp += .5*u[:, i].t()@R@u[:, i]
            obj_exp += u[:, i].t()@r
        i = N-2
        obj_exp += .5*x[:, i].t()@Qt@x[:, i]
        obj_exp += x[:, i].t()@qt
        obj_exp += .5*u[:, i].t()@Rt@u[:, i]
        obj_exp += u[:, i].t()@rt
        for i in range(N-1):
            obj_exp += .5*alpha[:, i].t()@Z@alpha[:, i]
            obj_exp += alpha[:, i].t()@z
        i = N-1
        obj_exp += .5*alpha[:, i].t()@Zt@alpha[:, i]
        obj_exp += alpha[:, i].t()@zt

        obj = .5*s@Q2@s.t() + .5*alpha@Q3@alpha.t() + s@q2 + alpha@q3 + c

        # print(obj_exp.item())
        # print(obj.item())

        self.assertAlmostEqual(obj_exp.item(), obj.item())

    def test_trajopt_lim(self):
        N = 20
        sys = ball_paddle_system.BallPaddleSystem(dt=.01)
        vf = value_to_optimization.ValueFunction(sys, N)

        vf.set_cost(q=torch.ones(3), r=-torch.ones(1))

        x_lo = torch.rand(3)
        u_up = torch.rand(1)
        vf.set_constraints(x_lo=x_lo, u_up=u_up)

        traj_opt = vf.traj_opt_constraint()
        (Ain1, Ain2, Ain3, rhs_in,
         Aeq1, Aeq2, Aeq3, rhs_eq,
         Q2, Q3, q2, q3, c) = torch_to_numpy(traj_opt)

        x = cp.Variable(Ain1.shape[1])
        s = cp.Variable(Ain2.shape[1])
        z = cp.Variable(Ain3.shape[1], boolean=True)

        obj = cp.Minimize(.5*cp.quad_form(s, Q2) + .5 *
                          cp.quad_form(z, Q3) + q2.T@s + q3.T@z + c)
        con = [
            Ain1@x + Ain2@s + Ain3@z <= rhs_in,
            Aeq1@x + Aeq2@s + Aeq3@z == rhs_eq,
        ]

        prob = cp.Problem(obj, con)
        prob.solve(solver=cp.GUROBI, verbose=False)

        traj = np.hstack((x.value, s.value))
        traj = np.reshape(traj, (-1, 4)).T

        xtraj = traj[:3, :]
        utraj = traj[3:, :]

        for i in range(N):
            for j in range(3):
                self.assertLessEqual(x_lo[j], xtraj[j, i])
            for j in range(1):
                self.assertLessEqual(utraj[j, i], u_up[j])

    def test_trajopt_obj_trajsetpoint(self):
        N = 5
        sys = ball_paddle_system.BallPaddleSystem(dt=.01)
        vf = value_to_optimization.ValueFunction(sys, N)

        [Q, R, Z, q, r, z, Qt, Rt, Zt, qt, rt, zt] =\
            sys.get_simple_trajopt_cost()
        utraj0 = torch.rand(1, 1).type(sys.dtype)
        utraj = torch.rand(1, N-1).type(sys.dtype)
        xtraj = torch.rand(3, N-1).type(sys.dtype)
        alphatraj = torch.rand(1, N).type(sys.dtype)
        vf.set_cost(Q=Q, R=R, Z=Z, q=q, r=r, z=z)
        vf.set_terminal_cost(Qt=Qt, Rt=Rt, Zt=Zt, qt=qt, rt=rt, zt=zt)
        torch.cat((utraj0, utraj), 1)
        vf.set_traj(xtraj=xtraj, utraj=torch.cat(
            (utraj0, utraj), 1), alphatraj=alphatraj)

        x0 = torch.Tensor([0., .1, 0.])
        xN = torch.Tensor([0., .075, 0.])
        vf.set_constraints(x0=x0, xN=xN)

        (Ain1, Ain2, Ain3, rhs_in,
         Aeq1, Aeq2, Aeq3, rhs_eq,
         Q2, Q3, q2, q3, c) = vf.traj_opt_constraint()

        x = torch.rand(3, N-1).type(sys.dtype)
        u0 = torch.rand(1).type(sys.dtype)
        u = torch.rand(1, N-1).type(sys.dtype)
        s = torch.cat((u0, torch.cat((x, u), 0).t().reshape(-1)))
        alpha = torch.rand(1, N).type(sys.dtype)

        obj_exp = .5*(u0-utraj0).t()@R@(u0-utraj0).t()
        obj_exp += (u0-utraj0).t()@r
        for i in range(N-2):
            obj_exp += .5*(x[:, i]-xtraj[:, i]).t()@Q@(x[:, i]-xtraj[:, i])
            obj_exp += (x[:, i]-xtraj[:, i]).t()@q
            obj_exp += .5*(u[:, i]-utraj[:, i]).t()@R@(u[:, i]-utraj[:, i])
            obj_exp += (u[:, i]-utraj[:, i]).t()@r
        i = N-2
        obj_exp += .5*(x[:, i]-xtraj[:, i]).t()@Qt@(x[:, i]-xtraj[:, i])
        obj_exp += (x[:, i]-xtraj[:, i]).t()@qt
        obj_exp += .5*(u[:, i]-utraj[:, i]).t()@Rt@(u[:, i]-utraj[:, i])
        obj_exp += (u[:, i]-utraj[:, i]).t()@rt
        for i in range(N-1):
            obj_exp += .5*(alpha[:, i]-alphatraj[:, i]
                           ).t()@Z@(alpha[:, i]-alphatraj[:, i])
            obj_exp += (alpha[:, i]-alphatraj[:, i]).t()@z
        i = N-1
        obj_exp += .5*(alpha[:, i]-alphatraj[:, i]
                       ).t()@Zt@(alpha[:, i]-alphatraj[:, i])
        obj_exp += (alpha[:, i]-alphatraj[:, i]).t()@zt

        obj = .5*s@Q2@s.t() + .5*alpha@Q3@alpha.t() + s@q2 + alpha@q3 + c

        # print(obj_exp.item())
        # print(obj.item())

        self.assertAlmostEqual(obj_exp.item(), obj.item())


if __name__ == '__main__':
    unittest.main()
