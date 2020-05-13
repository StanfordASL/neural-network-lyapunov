import robust_value_approx.nonlinear_system as nonlinear_system
import robust_value_approx.value_nlp as value_nlp
import numpy as np
import torch
import jax


class Manipulator(nonlinear_system.NonlinearSystem):
    def __init__(self, dtype):
        """
        x = [x, y, theta, xdot, ydot, thetadot]
        u = [xddot, yddot]
        """
        self.dtype = dtype
        self.x_lo = [torch.Tensor([-1e6, -1e6, -1e6, -1e6, -1e6, -1e6]).type(dtype)]
        self.x_up = [torch.Tensor([1e6, 1e6, 1e6, 1e6, 1e6, 1e6]).type(dtype)]
        self.u_lo = [torch.Tensor([-1e3, -1e3, -1e3]).type(dtype)]
        self.u_up = [torch.Tensor([1e3, 1e3, 1e3]).type(dtype)]
        self.x_dim = [6, 6]
        self.u_dim = [2, 2]

    def dx_contact(self, x0, u0, arraylib=np):
        pass

    def dx_nocontact(self, x0, u0, arraylib=np):
        xddot = u0
        dx0 = arraylib.array([x0[3], x0[4], x0[5],
            x_ddot[0], x_ddot[1], x_ddot[2]])
        return dx0

    def plot_result(self, result):
        names = ['x', 'y', 'theta', 'xdot', 'ydot', 'thetadot']
        return self.plot_result_named(result, names)


def get_value_function(N, dt, dtype):
    sys = Manipulator(dtype)
    dt_lo = dt
    dt_up = dt
    
    vf = value_nlp.NLPValueFunction(
        sys.x_lo, sys.x_up, sys.u_lo, sys.u_up, dt_lo, dt_up)

    vf.add_segment(N-1, sys.dyn, lambda x: sys.dyn(x, arraylib=jax.numpy))

    if Q is None:
        Q = np.diag([.5, .5, .1, .1])
    if R is None:
        R = np.diag([.001])
    x_desired = np.array([np.pi, 0., 0., 0.])
    cost_exp = 0
    for n in range(vf.N-1):
        fun = lambda x: sys.quad_cost(
            x, Q=Q * (n+1)**cost_exp, R=R, x_desired=x_desired,
            arraylib=np)
        fun_jax = lambda x: sys.quad_cost(
            x, Q=Q * (n+1)**cost_exp, R=R, x_desired=x_desired,
            arraylib=jax.numpy)
        vf.add_step_cost(n, fun, fun_jax)

    return(vf, sys)