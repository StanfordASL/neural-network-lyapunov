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


class PendulumNLP:
    def __init__(self):
        self.l = 1.
        self.m = 1.
        self.b = .5
        self.x_lo = [np.array([-1e9, -1e9])]
        self.x_up = [np.array([1e9, 1e9])]
        self.u_lo = [np.array([-10.])]
        self.u_up = [np.array([10.])]
        self.g = 9.81
        self.x_dim = 2
        self.u_dim = 1
        self.I = self.m*self.l**2

    def dx(self, x0, u0, arraylib=np):
        theta = x0[0]
        theta_dot = x0[1]
        theta_ddot = (1./self.I)*(u0[0] - self.b*theta_dot -\
            self.m*self.g*self.l*arraylib.sin(theta))
        dx0 = arraylib.array([theta_dot, theta_ddot])
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
        Q = np.diag([1., .1])
        R = np.diag([0.01])
        dt_lo = .2
        dt_up = .2
        x_desired = np.array([np.pi, 0.])
        vf = value_to_optimization.NLPValueFunction(self, self.x_lo, self.x_up,
            self.u_lo, self.u_up, init_mode=0, dt_lo=dt_lo, dt_up=dt_up,
            Q=[Q], x_desired=[x_desired], R=[R])
        vf.add_mode(N-1, self.dyn, self.dyn_jax)
        vf.add_init_state_constraint()
        return vf


def get_value_function(N):
	acro = PendulumNLP()
	return acro.get_nlp_value_function(N=N)
