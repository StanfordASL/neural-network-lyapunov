# -*- coding: utf-8 -*-
import ReLUToOptimization
import ValueToOptimization

import cvxpy as cp
import numpy as np
import torch


class ModelBounds:
    """
    Generates problems that produce bounds on the error of a learned cost-to-go
    """

    def __init__(self, model, sys,
                 Q, R, Z, q, r, z,
                 Qt, Rt, Zt, qt, rt, zt,
                 N, xN,
                 x_lo_traj, x_up_traj, u_lo, u_up):
        self.relu_opt = ReLUToOptimization.ReLUFreePattern(model, sys.dtype)
        self.sys = sys
        self.value_fun = ValueToOptimization.ValueFunction(sys)
        self.value_opt = self.value_fun.traj_opt_x0xN(Q, R, Z, q, r, z, 
                          Qt, Rt, Zt, qt, rt, zt,
                          N, xN,
                          x_lo_traj, x_up_traj, u_lo, u_up)

    def upper_bound_opt(self, model, x_lo, x_up):
        """
        returns an MIQP in standard form that corresponds to the upper bound problem.
        The solution to the returned problem corresponds to ε̅, and has the property that
        
        η(x) ≤ V(x) + ε̅, ∀ x_lo ≤ x ≤ x_up

        where η is the output of the neural network, V is the optimal cost-to-go
        
        y = [x, s, z]
        γ = [α, β]
        
        min     .5 yᵀ Q1 y + .5 γᵀ Q2 γ + yᵀ q1 + γᵀ q2 + k
        s.t.    A1 y + A2 γ = b
                G1 y + G2 γ <= h
                γ ∈ {0,1}
                
        @return Q1, Q2, q1, q2, k, A1, A2, b, G1, G2, h
        """
        Pin1, Pin2, Pin3, qrhs_in, Peq1, Peq2, Peq3, qrhs_eq, a_out, b_out, z_lo, z_up = self.relu_opt.output_constraint(model, x_lo, x_up)
        Q2_val, Q3_val, q2_val, q3_val, Ain1, Ain2, Ain3, brhs_in, Aeq1, Aeq2, Aeq3, brhs_eq = self.value_opt

        # x size equal in both program
        assert(Pin1.shape[1] == Ain1.shape[1])
        
        num_x = Ain1.shape[1]
        num_s = Ain2.shape[1]
        num_alpha = Ain3.shape[1]
        num_z = Pin2.shape[1]
        num_beta = Pin3.shape[1]
        num_y = num_x + num_s + num_z
        num_gamma = num_alpha + num_beta

        num_Ain = brhs_in.shape[0]
        num_Pin = qrhs_in.shape[0]
        num_in = num_Pin + num_Ain + 2*num_x

        num_Aeq = brhs_eq.shape[0]
        num_Peq = qrhs_eq.shape[0]
        num_eq = num_Peq + num_Aeq

        x_index = torch.arange(0,num_x)
        s_index = torch.arange(num_x,num_x+num_s)
        z_index = torch.arange(num_x+num_s,num_x+num_s+num_z)

        alpha_index = torch.arange(0,num_alpha)
        beta_index = torch.arange(num_alpha,num_alpha+num_beta)

        G1 = torch.zeros(num_in, num_y, dtype=self.sys.dtype)
        G1[0:num_Ain,x_index] = Ain1
        G1[0:num_Ain,s_index] = Ain2
        G1[num_Ain:num_Ain+num_Pin,x_index] = Pin1
        G1[num_Ain:num_Ain+num_Pin,z_index] = Pin2

        G2 = torch.zeros(num_in, num_gamma, dtype=self.sys.dtype)
        G2[0:num_Ain,alpha_index] = Ain3
        G2[num_Ain:num_Ain+num_Pin,beta_index] = Pin3
        G2[num_Ain+num_Pin:num_Ain+num_Pin+num_x,x_index] = torch.eye(num_x, dtype=self.sys.dtype)
        G2[num_Ain+num_Pin+num_x:num_Ain+num_Pin+2*num_x,x_index] = -torch.eye(num_x, dtype=self.sys.dtype)

        h = torch.cat((brhs_in, qrhs_in.squeeze(), x_up, -x_lo), 0)

        A1 = torch.zeros(num_eq, num_y, dtype=self.sys.dtype)
        A1[0:num_Aeq,x_index] = Aeq1
        A1[0:num_Aeq,s_index] = Aeq2
        A1[num_Aeq:num_Aeq+num_Peq,x_index] = Peq1
        A1[num_Aeq:num_Aeq+num_Peq,z_index] = Peq2

        A2 = torch.zeros(num_eq, num_gamma, dtype=self.sys.dtype)
        A2[0:num_Aeq,alpha_index] = Aeq3
        A2[num_Aeq:num_Aeq+num_Peq,beta_index] = Peq3

        b = torch.cat((brhs_eq, qrhs_eq.squeeze()), 0)

        Q1 = torch.zeros(num_y, num_y, dtype=self.sys.dtype)
        Q1[min(s_index):max(s_index)+1,min(s_index):max(s_index)+1] = Q2_val
        
        Q2 = torch.zeros(num_gamma, num_gamma, dtype=self.sys.dtype)
        Q2[min(alpha_index):max(alpha_index)+1,min(alpha_index):max(alpha_index)+1] = Q3_val

        q1 = torch.zeros(num_y, dtype=self.sys.dtype)
        q1[s_index] = q2_val
        q1[z_index] = -a_out.squeeze()
        
        q2 = torch.zeros(num_gamma, dtype=self.sys.dtype)
        q2[alpha_index] = q3_val

        k = -b_out

        return(Q1,Q2,q1,q2,k,G1,G2,h,A1,A2,b)
