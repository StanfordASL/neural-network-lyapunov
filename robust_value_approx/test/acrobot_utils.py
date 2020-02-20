import robust_value_approx.value_to_optimization as value_to_optimization
from pydrake.solvers.mathematicalprogram import MathematicalProgram
from pydrake.solvers.snopt import SnoptSolver
import numpy as np
import matplotlib.pyplot as plt
import pybullet as p
import time
import os
import inspect
import torch


class AcrobotNLP:
    def __init__(self):
        """
        https://ocw.mit.edu/courses/electrical-engineering-and-computer-\
        science/6-832-underactuated-robotics-spring-2009/readings/\
        MIT6_832s09_read_ch03.pdf
        """
        self.l1 = 1.
        self.l2 = 1.
        self.m1 = 1.
        self.m2 = 1.
        self.l1_com = .5
        self.l2_com = .5
        self.I = 1.
        self.x_lo = [np.array([-1e4, -1e4, -1e4, -1e4])]
        self.x_up = [np.array([1e4, 1e4, 1e4, 1e4])]
        self.u_lo = [np.array([-50.])]
        self.u_up = [np.array([50.])]

    def dyn(self, var):
        x_dim = 4
        u_dim = 1
        x0 = var[:x_dim]
        u0 = var[x_dim:x_dim+u_dim]
        dt0 = var[x_dim+u_dim:x_dim+u_dim+1]
        x1 = var[x_dim+u_dim+1:x_dim+u_dim+1+x_dim]        
        theta1 = x0[0]
        theta2 = x0[1]
        theta1_dot = x0[2]
        theta2_dot = x0[3]
        s1 = np.sin(theta1)
        c1 = np.cos(theta1)
        s2 = np.sin(theta2)
        c2 = np.cos(theta2)
        s12 = np.sin(theta1 + theta2)
        I = self.I
        m1 = self.m1
        m2 = self.m2
        l1 = self.l1
        l2 = self.l2
        lc1 = self.l1_com
        lc2 = self.l2_com
        g = -9.81
        H = np.array([[I+I+m2*l1**2+2*m2*l1*lc2*c2, I+m2*l1*lc2*c2],
                      [I+m2*l1*lc2*c2, I]])
        C = np.array([[-2*m2*l1*lc2*s2*theta2_dot, -m2*l1*lc2*s2*theta2_dot],
                      [m2*l1*lc2*s2*theta1_dot, 0.]])
        G = np.array([(m1*lc1+m2*l1)*g*s1 + m2*g*l2*s12, m2*g*l2*s12])
        B = np.array([[0.], [1.]])
        Hdet = H[0,0]*H[1,1] - H[0,1]*H[1,0]
        Hinv = (1./Hdet)*np.array([[H[1,1], -H[0,1]], [-H[1,0], H[0,0]]])
        x_ddot = Hinv@(G + B@u0 - C@x0[2:])
        dx0 = np.array([x0[2], x0[3], x_ddot[0], x_ddot[1]])
        return x0 + dt0 * dx0 - x1

    def get_nlp_value_function(self, N):
        Q = np.diag([10., 10., 1., 1.])
        R = np.diag([1.])
        dt_lo = .1
        dt_up = .1
        x_desired = np.array([np.pi, 0., 0., 0.])
        vf = value_to_optimization.NLPValueFunction(self.x_lo, self.x_up,
            self.u_lo, self.u_up, init_mode=0, dt_lo=dt_lo, dt_up=dt_up,
            Q=[Q], x_desired=[x_desired], R=[R])
        vf.add_mode(N, self.dyn)
        return vf

    def plot_traj(self, x_traj):
        plt.plot(x_traj)
        plt.legend(['theta1', 'theta2', 'theta1_dot', 'theta2_dot'])
        plt.show()

    def vis_traj(self, x_traj, dt_traj):
        p.connect(p.GUI)
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(.001)
        currentdir = os.path.dirname(os.path.abspath(
        inspect.getfile(inspect.currentframe())))
        p.setAdditionalSearchPath(currentdir)
        acrobot_id = p.loadURDF(
            "test/acrobot_description/acrobot.urdf", [0, 0, 2])
        for n in range(len(x_traj)):
            x = x_traj[n]
            dt = dt_traj[n]
            joint_poses = [x[0], x[1]]
            for i in range(len(joint_poses)):
                p.resetJointState(acrobot_id, i, joint_poses[i], 0.)
            p.stepSimulation()
            time.sleep(dt*10)


def get_value_function(N):
	acro = AcrobotNLP()
	return acro.get_nlp_value_function(N=N)