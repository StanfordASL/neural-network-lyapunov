import robust_value_approx.ball_paddle_system as ball_paddle_system

import unittest
import numpy as np
import cvxpy as cp


class BallPaddleSystemSim(unittest.TestCase):
    def test_transcription(self):
        # retrieves the dynamics
        sys = ball_paddle_system.BallPaddleSystem()
        Aeq1, Aeq2, Aeq3, Aeq4, Aeq5, rhs_eq = sys.get_dyn_eq()
        Ain1, Ain2, Ain3, Ain4, Ain5, rhs_in = sys.get_dyn_in()

        Aeq1 = Aeq1.detach().numpy()
        Aeq2 = Aeq2.detach().numpy()
        Aeq3 = Aeq3.detach().numpy()
        Aeq4 = Aeq4.detach().numpy()
        Aeq5 = Aeq5.detach().numpy()
        rhs_eq = rhs_eq.detach().numpy()
        Ain1 = Ain1.detach().numpy()
        Ain2 = Ain2.detach().numpy()
        Ain3 = Ain3.detach().numpy()
        Ain4 = Ain4.detach().numpy()
        Ain5 = Ain5.detach().numpy()
        rhs_in = rhs_in.detach().numpy()

        g = sys.g
        M = sys.M
        dt = sys.dt
        cr = sys.cr

        zp, zb, zbdot, u, zpp, zbp, zbdotp, up, a = np.random.rand(9)
        x = np.array([zp, zb, zbdot])
        xp = np.array([zpp, zbp, zbdotp])

        def eval_eq(i):
            return (Aeq1[i, :]@x + Aeq2[i, :]*u + Aeq3[i, :]@xp +
                    Aeq4[i, :]*up + Aeq5[i, :]*a - rhs_eq[i])[0]

        def eval_in(i):
            return (Ain1[i, :]@x + Ain2[i, :]*u + Ain3[i, :]@xp + Ain4[i, :]*up
                    + Ain5[i, :]*a - rhs_in[i])[0]

        self.assertAlmostEqual(zp + u*dt*.5 - zpp + up *
                               dt*.5, eval_eq(0), places=6)
        self.assertAlmostEqual(zb + zbdot*dt*.5 - zbp +
                               zbdotp*dt*.5, eval_eq(1), places=6)
        self.assertAlmostEqual(zbdot - zbdotp - M*a +
                               g*dt, eval_in(0), places=6)
        self.assertAlmostEqual(- zbdot + zbdotp - M*a -
                               g*dt, eval_in(1), places=6)
        self.assertAlmostEqual(-cr*zbdot + (cr+1.)*u -
                               zbdotp + M*a - M, eval_in(2), places=6)
        self.assertAlmostEqual(cr*zbdot - (1+cr)*u +
                               zbdotp + M*a - M, eval_in(3), places=6)
        self.assertAlmostEqual(-zp + zb + dt*zbdot - u*dt*.5 -
                               up*dt*.5 + M*a - M + .5*g*dt*dt, eval_in(4),
                               places=5)
        self.assertAlmostEqual(zp - zb, eval_in(5), places=6)

    def test_dynamics(self):
        # retrieves the dynamics
        sys = ball_paddle_system.BallPaddleSystem()
        Aeq1, Aeq2, Aeq3, Aeq4, Aeq5, rhs_eq = sys.get_dyn_eq()
        Ain1, Ain2, Ain3, Ain4, Ain5, rhs_in = sys.get_dyn_in()

        Aeq1 = Aeq1.detach().numpy()
        Aeq2 = Aeq2.detach().numpy()
        Aeq3 = Aeq3.detach().numpy()
        Aeq4 = Aeq4.detach().numpy()
        Aeq5 = Aeq5.detach().numpy()
        rhs_eq = rhs_eq.detach().numpy()
        Ain1 = Ain1.detach().numpy()
        Ain2 = Ain2.detach().numpy()
        Ain3 = Ain3.detach().numpy()
        Ain4 = Ain4.detach().numpy()
        Ain5 = Ain5.detach().numpy()
        rhs_in = rhs_in.detach().numpy()

        g = sys.g
        dt = sys.dt

        # free falling
        zp, zb, zbdot, u, zpp, zbp, zbdotp, up, a = [0., 1., 0., 0.,
                                                     0., 1.+.5*g*dt**2, g*dt,
                                                     0., 0.]
        x = np.array([zp, zb, zbdot])
        xp = np.array([zpp, zbp, zbdotp])

        def eval_eq(i):
            return (Aeq1[i, :]@x + Aeq2[i, :]*u + Aeq3[i, :]@xp + Aeq4[i, :]*up
                    + Aeq5[i, :]*a - rhs_eq[i])[0]

        def eval_in(i):
            return (Ain1[i, :]@x + Ain2[i, :]*u + Ain3[i, :]@xp + Ain4[i, :]*up
                    + Ain5[i, :]*a - rhs_in[i])[0]

        for i in range(Aeq1.shape[0]):
            self.assertAlmostEqual(eval_eq(i), 0.)
        for i in range(Ain1.shape[0]):
            self.assertLessEqual(eval_in(i), 1e-6)

        # on the paddle
        zp, zb, zbdot, u, zpp, zbp, zbdotp, up, a = [1., 1., 0., 0.,
                                                     1., 1., 0., 0., 1.]
        x = np.array([zp, zb, zbdot])
        xp = np.array([zpp, zbp, zbdotp])

        def eval_eq(i):
            return (Aeq1[i, :]@x + Aeq2[i, :]*u + Aeq3[i, :]@xp + Aeq4[i, :]*up
                    + Aeq5[i, :]*a - rhs_eq[i])[0]

        def eval_in(i):
            return (Ain1[i, :]@x + Ain2[i, :]*u + Ain3[i, :]@xp + Ain4[i, :]*up
                    + Ain5[i, :]*a - rhs_in[i])[0]

        for i in range(Aeq1.shape[0]):
            self.assertAlmostEqual(eval_eq(i), 0.)
        for i in range(Ain1.shape[0]):
            self.assertLessEqual(eval_in(i), 1e-6)

    def test_simulation(self):
        # retrieves the dynamics
        sys = ball_paddle_system.BallPaddleSystem()
        Aeq1, Aeq2, Aeq3, Aeq4, Aeq5, rhs_eq = sys.get_dyn_eq()
        Ain1, Ain2, Ain3, Ain4, Ain5, rhs_in = sys.get_dyn_in()

        Aeq1 = Aeq1.detach().numpy()
        Aeq2 = Aeq2.detach().numpy()
        Aeq3 = Aeq3.detach().numpy()
        Aeq4 = Aeq4.detach().numpy()
        Aeq5 = Aeq5.detach().numpy()
        rhs_eq = rhs_eq.detach().numpy()
        Ain1 = Ain1.detach().numpy()
        Ain2 = Ain2.detach().numpy()
        Ain3 = Ain3.detach().numpy()
        Ain4 = Ain4.detach().numpy()
        Ain5 = Ain5.detach().numpy()
        rhs_in = rhs_in.detach().numpy()

        N = 50
        x = cp.Variable((3, N))
        a = cp.Variable((1, N), boolean=True)
        u = 0.

        con = [x[:, 0] == np.array([0., 0.1, 0.])]
        for i in range(N-1):
            con += [
                Aeq1@x[:, i] + Aeq2@np.array([u]) + Aeq3@x[:, i+1] +
                Aeq4@np.array([u]) + Aeq5@a[:, i] == rhs_eq[:, 0],
                Ain1@x[:, i] + Ain2@np.array([u]) + Ain3@x[:, i+1] +
                Ain4@np.array([u]) + Ain5@a[:, i] <= rhs_in[:, 0],
            ]
        obj = cp.Minimize(0.)
        prob = cp.Problem(obj, con)
        prob.solve(solver=cp.GUROBI)

        # plt.plot((x.value)[1,:])
        # plt.show()

        self.assertTrue(x.value is not None)


if __name__ == '__main__':
    unittest.main()
