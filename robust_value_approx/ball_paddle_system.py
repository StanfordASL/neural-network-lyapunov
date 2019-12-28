import torch
import robust_value_approx.constants as constants


class BallPaddleSystem:

    def __init__(self, dt=.01, cr=.8, M=100., dtype=torch.float64):
        """
        Class to capture the dynamics of the ball paddle system in the form
        Aeq1 x[n] + Aeq2 u[n] + Aeq3 x[n+1] + Aeq4 u[n+1] + Aeq5 α[n] =
            rhs_eq_dyn
        Ain1 x[n] + Ain2 u[n] + Ain3 x[n+1] + Ain4 u[n+1] + Ain5 α[n] ≤
            rhs_in_dyn
        where
        x[n] = [zp[n], zb[n], zbdot[n]]
        u[n] = zpdot[n]
        and α is a binary variable

        @param dt The time step size
        @param cr The coefficient of restitution for contact between the ball
        and paddle
        @param M the value used the "big-M" formulation of the piecewise
        affine dynamics
        @param dtype The datatype used to store the coefficients
        """
        self.dt = dt
        self.cr = cr
        self.M = M
        self.dtype = dtype

        self._init_eq()
        self._init_in()

        g = constants.G
        M = self.M
        # 0 == zp[n] + u[n]*dt*.5 - zp[n+1] + u[n+1]*dt*.5
        self._add_eq([1., 0., 0.], [.5*dt], [-1., 0., 0.], [.5*dt], [0.], [0.])
        # 0 == zb[n] + zbdot[n]*dt*.5 - zb[n+1] + zbdot[n+1]*dt*.5
        self._add_eq([0., 1., .5*dt], [0.], [0., -1., .5*dt], [0.], [0.], [0.])
        # zbdot[n] - zbdot[n+1] - M*a[n] <= -constants.G*dt
        self._add_in([0., 0., 1.], [0.], [0., 0., -1.], [0.], [-M], [-g*dt])
        # - zbdot[n] + zbdot[n+1] - M*a[n] <= constants.G*dt
        self._add_in([0., 0., -1.], [0.], [0., 0., 1.], [0.], [-M], [g*dt])
        # -cr*zbdot[n] + (cr+1)*u[n] - zbdot[n+1] + M*a[n] <= M
        self._add_in([0., 0., -cr], [1.+cr], [0., 0., -1.], [0.], [M], [M])
        # cr*zbdot[n] -(1+cr)*u[n] + zbdot[n+1] + M*a[n] <= M
        self._add_in([0., 0., cr], [-1.-cr], [0., 0., 1.], [0.], [M], [M])
        # - zp[n] + zb[n] + dt*zbdot[n] - u[n]*dt*.5 - u[n+1]*dt*.5 + M*a[n]
        # <= M - .5*g*dt*dt
        self._add_in([-1., 1., dt], [-.5*dt], [0., 0., 0.],
                     [-.5*dt], [M], [M-.5*g*dt*dt])
        # zp[n] - zb[n] <= 0
        self._add_in([1., -1., 0.], [0.], [0., 0., 0.], [0.], [0.], [0.])

    def _init_eq(self):
        self.Aeq1 = torch.empty((0, 3), dtype=self.dtype)
        self.Aeq2 = torch.empty((0, 1), dtype=self.dtype)
        self.Aeq3 = torch.empty((0, 3), dtype=self.dtype)
        self.Aeq4 = torch.empty((0, 1), dtype=self.dtype)
        self.Aeq5 = torch.empty((0, 1), dtype=self.dtype)
        self.rhs_eq = torch.empty((0, 1), dtype=self.dtype)

    def _init_in(self):
        self.Ain1 = torch.empty((0, 3), dtype=self.dtype)
        self.Ain2 = torch.empty((0, 1), dtype=self.dtype)
        self.Ain3 = torch.empty((0, 3), dtype=self.dtype)
        self.Ain4 = torch.empty((0, 1), dtype=self.dtype)
        self.Ain5 = torch.empty((0, 1), dtype=self.dtype)
        self.rhs_in = torch.empty((0, 1), dtype=self.dtype)

    def _add_eq(self, a1, a2, a3, a4, a5, rhsi):
        self.Aeq1 = torch.cat(
            (self.Aeq1, torch.Tensor(a1).type(self.dtype).unsqueeze(0)), 0)
        self.Aeq2 = torch.cat(
            (self.Aeq2, torch.Tensor(a2).type(self.dtype).unsqueeze(0)), 0)
        self.Aeq3 = torch.cat(
            (self.Aeq3, torch.Tensor(a3).type(self.dtype).unsqueeze(0)), 0)
        self.Aeq4 = torch.cat(
            (self.Aeq4, torch.Tensor(a4).type(self.dtype).unsqueeze(0)), 0)
        self.Aeq5 = torch.cat(
            (self.Aeq5, torch.Tensor(a5).type(self.dtype).unsqueeze(0)), 0)
        self.rhs_eq = torch.cat(
            (self.rhs_eq, torch.Tensor(rhsi).type(self.dtype).unsqueeze(0)), 0)

    def _add_in(self, a1, a2, a3, a4, a5, rhsi):
        self.Ain1 = torch.cat(
            (self.Ain1, torch.Tensor(a1).type(self.dtype).unsqueeze(0)), 0)
        self.Ain2 = torch.cat(
            (self.Ain2, torch.Tensor(a2).type(self.dtype).unsqueeze(0)), 0)
        self.Ain3 = torch.cat(
            (self.Ain3, torch.Tensor(a3).type(self.dtype).unsqueeze(0)), 0)
        self.Ain4 = torch.cat(
            (self.Ain4, torch.Tensor(a4).type(self.dtype).unsqueeze(0)), 0)
        self.Ain5 = torch.cat(
            (self.Ain5, torch.Tensor(a5).type(self.dtype).unsqueeze(0)), 0)
        self.rhs_in = torch.cat(
            (self.rhs_in, torch.Tensor(rhsi).type(self.dtype).unsqueeze(0)), 0)

    def get_dyn_eq(self):
        """
        The equality part of the piecewise linear dynamics in the form
        Aeq1 x[n] + Aeq2 u[n] + Aeq3 x[n+1] + Aeq4 u[n+1] + Aeq5 α[n] =
            rhs_eq_dyn

        @return Aeq1, Aeq2, Aeq3, Aeq4, Aeq5, rhs_eq_dyn The coefficients as
        torch Tensor of self.dtype
        """
        return(self.Aeq1, self.Aeq2, self.Aeq3, self.Aeq4, self.Aeq5,
               self.rhs_eq)

    def get_dyn_in(self):
        """
        The inequality part of the piecewise linear dynamics in the form
        Ain1 x[n] + Ain2 u[n] + Ain3 x[n+1] + Ain4 u[n+1] + Ain5 α[n] <=
            rhs_in_dyn

        @return Ain1, Ain2, Ain3, Ain4, Ain5, rhs_in_dyn The coefficients as
        torch Tensor of self.dtype
        """
        return(self.Ain1, self.Ain2, self.Ain3, self.Ain4, self.Ain5,
               self.rhs_in)

    def get_simple_trajopt_cost(self):
        """
        Returns a set of tensors that represent the a simple cost for a
        trajectory optimization problem. This is useful to write tests for
        example

        @return Q, R, Z, q, r, z, Qt, Rt, Zt, qt, rt, zt
        where
        min ∑(.5 xᵀ[n] Q x[n] + .5 uᵀ[n] R u[n] + .5 αᵀ[n] Z α[n] + qᵀx[n]
              + rᵀu[n] + zᵀα[n])
                + .5 xᵀ[N] Qt x[N] + .5 uᵀ[N] Rt u[N] + .5 αᵀ[N] Zt α[N]
                + qtᵀx[N] + rtᵀu[N] + ztᵀα[N]
        """

        Q = torch.eye(3, dtype=self.dtype)*0.1
        q = torch.ones(3, dtype=self.dtype)*0.2
        R = torch.eye(1, dtype=self.dtype)*3.
        r = torch.ones(1, dtype=self.dtype)*0.4
        Z = torch.eye(1, dtype=self.dtype)*0.5
        z = torch.ones(1, dtype=self.dtype)*0.6

        Qt = torch.eye(3, dtype=self.dtype)*0.7
        qt = torch.ones(3, dtype=self.dtype)*0.8
        Rt = torch.eye(1, dtype=self.dtype)*9.
        rt = torch.ones(1, dtype=self.dtype)*0.11
        Zt = torch.eye(1, dtype=self.dtype)*0.12
        zt = torch.ones(1, dtype=self.dtype)*0.13

        return(Q, R, Z, q, r, z, Qt, Rt, Zt, qt, rt, zt)
