import robust_value_approx.value_nlp as value_nlp
import plotly.graph_objs as go
import numpy as np
import torch
import jax


class Pendulum:
    def __init__(self, dtype):
        self.l = 1.
        self.m = 1.
        self.b = .5
        self.I = self.m*self.l**2
        self.x_lo = [torch.Tensor([-1e9, -1e9]).type(dtype)]
        self.x_up = [torch.Tensor([1e9, 1e9]).type(dtype)]
        self.u_lo = [torch.Tensor([-100.]).type(dtype)]
        self.u_up = [torch.Tensor([100.]).type(dtype)]
        self.g = 9.81
        self.x_dim = [2]
        self.u_dim = [1]

    def dx(self, x0, u0, arraylib=np):
        theta = x0[0]
        theta_dot = x0[1]
        theta_ddot = (1./self.I)*(u0[0] - self.b*theta_dot -\
            self.m*self.g*self.l*arraylib.sin(theta))
        dx0 = arraylib.array([theta_dot, theta_ddot])
        return dx0
    
    def dyn(self, var, arraylib=np):
        x_dim = self.x_dim[0]
        u_dim = self.u_dim[0]
        x0 = var[:x_dim]
        u0 = var[x_dim:x_dim+u_dim]
        dt0 = var[x_dim+u_dim:x_dim+u_dim+1]
        x1 = var[x_dim+u_dim+1:x_dim+u_dim+1+x_dim]
        u1 = var[x_dim+u_dim+1+x_dim:x_dim+u_dim+1+x_dim+u_dim]
        dx0 = self.dx(x0, u0, arraylib=arraylib)
        dx1 = self.dx(x1, u1, arraylib=arraylib)
        return x0 + .5 * (dx0 + dx1) * dt0 - x1

    def quad_cost(self, var,
                  Q=None, R=None, x_desired=None, u_desired=None,
                  arraylib=np):
        x_dim = self.x_dim[0]
        u_dim = self.u_dim[0]
        x = var[:x_dim]
        u = var[x_dim:x_dim+u_dim]
        dt = var[x_dim+u_dim:x_dim+u_dim+1]
        if Q is None:
            Q = arraylib.zeros((x_dim, x_dim))
        if R is None:
            R = arraylib.zeros((u_dim, u_dim))
        if x_desired is None:
            x_desired = arraylib.zeros(x_dim)
        if u_desired is None:
            u_desired = arraylib.zeros(u_dim)
        cost = dt[0]*((x - x_desired)@Q@(x - x_desired) + (u - u_desired)@R@(u - u_desired))
        return cost

    def plot_result(self, result):
        fig = go.Figure()
        names = ['theta', 'theta_dot']
        x = torch.cat([x.unsqueeze(0) for x in result['x_traj']]).t()
        t = torch.cat([torch.zeros(1, dtype=x.dtype), torch.cumsum(
                result['dt_traj'], 0).squeeze()[:-1]])
        for i in range(self.x_dim[0]):
            fig.add_trace(go.Scatter(
                x=t,
                y=x[i, :],
                name=names[i],
                ))
        return fig


def get_value_function(N):
    sys = Pendulum(torch.float64)
    dt_lo = .2
    dt_up = .2
    vf = value_nlp.NLPValueFunction(
        sys.x_lo, sys.x_up, sys.u_lo, sys.u_up, dt_lo, dt_up)
    vf.add_segment(N-1, sys.dyn, lambda x: sys.dyn(x, arraylib=jax.numpy))
    Q = np.diag([1., .1])
    R = np.diag([.01])
    x_desired = np.array([np.pi, 0.])
    for n in range(vf.N-1):
        fun = lambda x: sys.quad_cost(
            x, Q=Q, R=R, x_desired=x_desired, arraylib=np)
        fun_jax = lambda x: sys.quad_cost(
            x, Q=Q, R=R, x_desired=x_desired, arraylib=jax.numpy)
        vf.add_step_cost(n, fun, fun_jax)
    Qt = np.diag([100., 10.])
    Rt = np.diag([.01])
    fun = lambda x: sys.quad_cost(
        x, Q=Qt, R=Rt, x_desired=x_desired, arraylib=np)
    fun_jax = lambda x: sys.quad_cost(
        x, Q=Qt, R=Rt, x_desired=x_desired, arraylib=jax.numpy)
    vf.add_step_cost(vf.N-1, fun, fun_jax)
    return(vf, sys)