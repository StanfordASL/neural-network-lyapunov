# -*- coding: utf-8 -*-
import ReLUToOptimization
import ValueToOptimization

import cvxpy as cp
import numpy as np


class ModelBounds:
    """
    Generates problems that produce bounds on the error of a learned cost-to-go
    """

    def __init__(self, opt_problem, model):
        self.relu_opt = ReLUToOptimization.ReLUFreePattern(self.model)
        # (Ain1, Ain2, Ain3, brhs_in, Aeq1, Aeq2, Aeq3, brhs_eq, Q2, Q3, q2, q3) = self.value_opt.output_constraint(x_lo, x_up)

    def upper_bound_program(self, model, x_lo, x_up, alpha=None, beta=None):
        """
        returns an [MI]QP in standard form that corresponds to the upper bound problem
        y = [x, s, alpha, z, beta]
        min     .5 yᵀ Q y + qᵀ y + k
        s.t.    A y = b
                G y <= h
                yᵢ ∈ Z, i = int_vars
        return Q,q,k,G,h,A,b,int_vars,x_index
        """

        (Q2, Q3, q2, q3, Ain1, Ain2, Ain3, brhs_in, Aeq1, Aeq2, Aeq3, brhs_eq) = self.value_opt.value_program(x_lo, x_up)
        (Pin1, Pin2, Pin3, qrhs_in, Peq1, Peq2, Peq3, qrhs_eq, a_out, b_out, z_lo, z_up) = self.relu_opt.output_constraint(model, x_lo, x_up)

        # x size equal in both program
        assert(Pin1.shape[1] == Ain1.shape[1])
        num_x = Ain1.shape[1]
        num_s = Ain2.shape[1]
        num_alpha = Ain3.shape[1]
        num_z = Pin2.shape[1]
        num_beta = Pin3.shape[1]
        num_vars = num_x + num_s + num_alpha + num_z + num_beta

        num_Ain = brhs_in.shape[0]
        num_Pin = qrhs_in.shape[0]
        num_in = num_Pin + num_Ain

        num_Aeq = brhs_eq.shape[0]
        num_Peq = qrhs_eq.shape[0]
        num_eq = num_Peq + num_Aeq

        x_index = torch.arange(0,num_x)
        s_index = torch.arange(num_x,num_x+num_s)
        alpha_index = torch.arange(num_x+num_s:num_x+num_s+num_alpha)
        z_index = torch.arange(num_x+num_s+num_alpha:num_x+num_s+num_alpha+num_z)
        beta_index = torch.arange(num_x+num_s+num_alpha+num_z:num_x+num_s+num_alpha+num_z+num_beta)

        G = torch.zeros(num_in, num_vars)
        G[0:num_Ain,x_index] = Ain1
        G[0:num_Ain,s_index] = Ain2
        G[0:num_Ain,alpha_index] = Ain3
        G[num_Ain:num_Ain+num_Pin,x_index] = Pin1
        G[num_Ain:num_Ain+num_Pin,z_index] = Pin2
        G[num_Ain:num_Ain+num_Pin,beta_index] = Pin3

        h = torch.cat((brhs_in, qrhs_in), 0)

        A = torch.zeros(num_eq, num_vars)
        A[0:num_Aeq,x_index] = Aeq1
        A[0:num_Aeq,s_index] = Aeq2
        A[0:num_Aeq,alpha_index] = Aeq3
        A[num_Aeq:num_Aeq+num_Peq,x_index] = Peq1
        A[num_Aeq:num_Aeq+num_Peq,z_index] = Peq2
        A[num_Aeq:num_Aeq+num_Peq,beta_index] = Peq3

        b = torch.cat((brhs_eq, qrhs_eq), 0)

        Q = torch.zeros(num_vars, num_vars)
        Q[s_index,s_index] = Q2
        Q[alpha_index,alpha_index] = Q3

        q = torch.zeros(num_vars)
        q[s_index] = q2
        q[alpha_index] = q3
        q[z_index] = -a_out

        k = -b_out

        int_vars = torch.cat((alpha_index,beta_index))

        return(Q,q,k,G,h,A,b,int_vars,x_index)
