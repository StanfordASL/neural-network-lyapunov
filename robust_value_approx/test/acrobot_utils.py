import robust_value_approx.value_to_optimization as value_to_optimization
from pydrake.solvers.mathematicalprogram import MathematicalProgram
from pydrake.solvers.snopt import SnoptSolver
import numpy as np
import scipy
import time
import os
import inspect
import torch
import jax


class AcrobotNLP:
    def __init__(self):
        self.l1 = 1.
        self.l2 = 1.
        self.m1 = 1.
        self.m2 = 1.
        self.lc1 = .5
        self.lc2 = .5
        self.I = 1.
        self.b1 = .1
        self.b2 = .1
        self.x_lo = [np.array([-1e9, -1e9, -1e9, -1e9])]
        self.x_up = [np.array([1e9, 1e9, 1e9, 1e9])]
        self.u_lo = [np.array([-1000.])]
        self.u_up = [np.array([1000.])]
        self.g = 9.81
        self.x_dim = 4
        self.u_dim = 1

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
        x_dim = self.x_dim
        u_dim = self.u_dim
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
    
    def dyn(self, var, arraylib=np):
        x_dim = self.x_dim
        u_dim = self.u_dim
        x0 = var[:x_dim]
        u0 = var[x_dim:x_dim+u_dim]
        dt0 = var[x_dim+u_dim:x_dim+u_dim+1]
        x1 = var[x_dim+u_dim+1:x_dim+u_dim+1+x_dim]
        u1 = var[x_dim+u_dim+1+x_dim:x_dim+u_dim+1+x_dim+u_dim]
        dx0 = self.dx(x0, u0, arraylib=arraylib)
        dx1 = self.dx(x1, u1, arraylib=arraylib)
        return x0 + .5 * (dx0 + dx1) * dt0 - x1

    def dyn_jax(self, var):
        return self.dyn(var, arraylib=jax.numpy)

    def get_nlp_value_function(self, N):
        Q = np.diag([1., 1., 0.01, 0.01])
        R = np.diag([0.01])
        dt_lo = .2
        dt_up = .2
        x_desired = np.array([np.pi, 0., 0., 0.])
        vf = value_to_optimization.NLPValueFunction(self, self.x_lo, self.x_up,
            self.u_lo, self.u_up, init_mode=0, dt_lo=dt_lo, dt_up=dt_up,
            Q=[Q], x_desired=[x_desired], R=[R])
        vf.add_mode(N-1, self.dyn, self.dyn_jax)
        vf.add_init_state_constraint()
        return vf


def get_value_function(N):
	acro = AcrobotNLP()
	return acro.get_nlp_value_function(N=N)
