import robust_value_approx.nonlinear_system as nonlinear_system
import robust_value_approx.value_nlp as value_nlp
import numpy as np
import torch
import jax


class Acrobot(nonlinear_system.NonlinearSystem):
    def __init__(self, dtype):
        self.dtype = dtype
        self.l1 = 1.
        self.l2 = 1.
        self.m1 = 1.
        self.m2 = 1.
        self.lc1 = .5
        self.lc2 = .5
        self.I = 1.
        self.b1 = .01
        self.b2 = .01
        self.x_lo = [torch.Tensor([-1e6, -1e6, -1e6, -1e6]).type(dtype)]
        self.x_up = [torch.Tensor([1e6, 1e6, 1e6, 1e6]).type(dtype)]
        self.u_lo = [torch.Tensor([-100.]).type(dtype)]
        self.u_up = [torch.Tensor([100.]).type(dtype)]
        self.g = 9.81
        self.x_dim = [4]
        self.u_dim = [1]

    def dx(self, x0, u0, arraylib=np):
        I = self.I
        m1 = self.m1
        m2 = self.m2
        l1 = self.l1
        lc1 = self.lc1
        lc2 = self.lc2
        b1 = self.b1
        b2 = self.b2
        g = self.g
        x_dim = self.x_dim[0]
        u_dim = self.u_dim[0]
        theta1 = x0[0]
        theta2 = x0[1]
        theta1_dot = x0[2]
        theta2_dot = x0[3]
        s1 = arraylib.sin(theta1)
        c1 = arraylib.cos(theta1)
        s2 = arraylib.sin(theta2)
        c2 = arraylib.cos(theta2)
        s12 = arraylib.sin(theta1 + theta2)
        H = arraylib.array([[I+I+m2*l1**2+2*m2*l1*lc2*c2, I+m2*l1*lc2*c2],
                            [I+m2*l1*lc2*c2, I]])
        C = arraylib.array([[-2*m2*l1*lc2*s2*theta2_dot, -m2*l1*lc2*s2*theta2_dot],
                            [m2*l1*lc2*s2*theta1_dot, 0.]])
        G = arraylib.array([-m1*g*lc1*s1 - m2*g*(l1*s1 + lc2*s12) - b1*theta1_dot,
                            -m2*g*lc2*s12 - b2*theta2_dot])
        B = arraylib.array([[0.],
                            [1.]])        
        Hdet = H[0,0]*H[1,1] - H[0,1]*H[1,0]
        Hinv = (1./Hdet)*arraylib.array([[H[1,1], -H[0,1]], [-H[1,0], H[0,0]]])
        x_ddot = Hinv@(G + B@u0 - C@x0[2:])
        dx0 = arraylib.array([x0[2], x0[3], x_ddot[0], x_ddot[1]])
        return dx0

    def plot_result(self, result):
        names = ['theta1', 'theta2', 'theta1_dot', 'theta2_dot']
        return self.plot_result_named(result, names)


def get_value_function(N):
    sys = Acrobot(torch.float64)
    dt_lo = .2
    dt_up = .2
    vf = value_nlp.NLPValueFunction(
        sys.x_lo, sys.x_up, sys.u_lo, sys.u_up, dt_lo, dt_up)
    vf.add_segment(N-1, sys.dyn, lambda x: sys.dyn(x, arraylib=jax.numpy))
    Q = np.diag([.1, .1, .1, .1])
    R = np.diag([.01])
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
    Qt = np.diag([.1, .1, .1, .1]) * vf.N**cost_exp
    Rt = np.diag([.01])
    fun = lambda x: sys.quad_cost(
        x, Q=Qt, R=Rt, x_desired=x_desired, arraylib=np)
    fun_jax = lambda x: sys.quad_cost(
        x, Q=Qt, R=Rt, x_desired=x_desired, arraylib=jax.numpy)
    vf.add_step_cost(vf.N-1, fun, fun_jax)
    return(vf, sys)