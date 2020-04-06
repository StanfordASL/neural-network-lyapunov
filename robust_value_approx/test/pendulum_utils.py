import robust_value_approx.nonlinear_system as nonlinear_system
import robust_value_approx.value_nlp as value_nlp
import numpy as np
import torch
import jax


class Pendulum(nonlinear_system.NonlinearSystem):
    def __init__(self, dtype):
        self.dtype = dtype
        self.l = 1.
        self.m = 1.
        self.b = .5
        self.I = self.m*self.l**2
        self.x_lo = [torch.Tensor([-10, -10]).type(dtype)]
        self.x_up = [torch.Tensor([10, 10]).type(dtype)]
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

    def plot_result(self, result):
        names = ['theta', 'theta_dot']
        return self.plot_result_named(result, names)


def get_value_function(N):
    sys = Pendulum(torch.float64)
    dt_lo = .1
    dt_up = .1
    vf = value_nlp.NLPValueFunction(
        sys.x_lo, sys.x_up, sys.u_lo, sys.u_up, dt_lo, dt_up)
    vf.add_segment(N-1, sys.dyn, lambda x: sys.dyn(x, arraylib=jax.numpy))
    Q = np.diag([1., 1.])
    R = np.diag([.1])
    x_desired = np.array([np.pi, 0.])
    for n in range(vf.N-1):
        fun = lambda x: sys.quad_cost(
            x, Q=Q, R=R, x_desired=x_desired, arraylib=np)
        fun_jax = lambda x: sys.quad_cost(
            x, Q=Q, R=R, x_desired=x_desired, arraylib=jax.numpy)
        vf.add_step_cost(n, fun, fun_jax)
    Qt = np.diag([1., 1.])
    Rt = np.diag([.1])
    fun = lambda x: sys.quad_cost(
        x, Q=Qt, R=Rt, x_desired=x_desired, arraylib=np)
    fun_jax = lambda x: sys.quad_cost(
        x, Q=Qt, R=Rt, x_desired=x_desired, arraylib=jax.numpy)
    vf.add_step_cost(vf.N-1, fun, fun_jax)
    return(vf, sys)