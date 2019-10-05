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

        x_index_s = 0
        x_index_e = num_x
        s_index_s = num_x
        s_index_e = num_x+num_s
        z_index_s = num_x+num_s
        z_index_e = num_x+num_s+num_z

        alpha_index_s = 0
        alpha_index_e = num_alpha
        beta_index_s = num_alpha
        beta_index_e = num_alpha+num_beta

        G1 = torch.zeros(num_in, num_y, dtype=self.sys.dtype)
        G1[0:num_Ain,x_index_s:x_index_e] = Ain1
        G1[0:num_Ain,s_index_s:s_index_e] = Ain2
        G1[num_Ain:num_Ain+num_Pin,x_index_s:x_index_e] = Pin1
        G1[num_Ain:num_Ain+num_Pin,z_index_s:z_index_e] = Pin2

        G1[num_Ain+num_Pin:num_Ain+num_Pin+num_x,x_index_s:x_index_e] = torch.eye(num_x, dtype=self.sys.dtype)
        G1[num_Ain+num_Pin+num_x:num_Ain+num_Pin+2*num_x,x_index_s:x_index_e] = -torch.eye(num_x, dtype=self.sys.dtype)

        G2 = torch.zeros(num_in, num_gamma, dtype=self.sys.dtype)
        G2[0:num_Ain,alpha_index_s:alpha_index_e] = Ain3
        G2[num_Ain:num_Ain+num_Pin,beta_index_s:beta_index_e] = Pin3

        h = torch.cat((brhs_in, qrhs_in.squeeze(), x_up, -x_lo), 0)

        A1 = torch.zeros(num_eq, num_y, dtype=self.sys.dtype)
        A1[0:num_Aeq,x_index_s:x_index_e] = Aeq1
        A1[0:num_Aeq,s_index_s:s_index_e] = Aeq2
        A1[num_Aeq:num_Aeq+num_Peq,x_index_s:x_index_e] = Peq1
        A1[num_Aeq:num_Aeq+num_Peq,z_index_s:z_index_e] = Peq2

        A2 = torch.zeros(num_eq, num_gamma, dtype=self.sys.dtype)
        A2[0:num_Aeq,alpha_index_s:alpha_index_e] = Aeq3
        A2[num_Aeq:num_Aeq+num_Peq,beta_index_s:beta_index_e] = Peq3

        b = torch.cat((brhs_eq, qrhs_eq.squeeze()), 0)

        Q1 = torch.zeros(num_y, num_y, dtype=self.sys.dtype)
        Q1[s_index_s:s_index_e,s_index_s:s_index_e] = Q2_val
        
        Q2 = torch.zeros(num_gamma, num_gamma, dtype=self.sys.dtype)
        Q2[alpha_index_s:alpha_index_e,alpha_index_s:alpha_index_e] = Q3_val

        q1 = torch.zeros(num_y, dtype=self.sys.dtype)
        q1[s_index_s:s_index_e] = q2_val
        q1[z_index_s:z_index_e] = -a_out.squeeze()
        
        q2 = torch.zeros(num_gamma, dtype=self.sys.dtype)
        q2[alpha_index_s:alpha_index_e] = q3_val

        k = -b_out

        return(Q1,Q2,q1,q2,k,G1,G2,h,A1,A2,b)
                
    def lower_bound_opt(self, model, x_lo, x_up, x0):
        """
        1) seperate the value constraints that don't depend on x
        2) now write the dual of the inner maximization
        
        3) compute an activation path (from a sample) - check that it wasn't already checked
        4) compute the P that corresponds to this activation sample
        
        5) Assemble the nonconvex MIQP
        6) solve it with CPLEX
        
        OR
        
        5) guess alpha and s
        6) solve (21) that gets you lambda
        7) solve (20) that get you alpha and s
        8) repeat (6-7) until satisfied (same bound as the lower bound?)
        """
        activation_pattern = ReLUToOptimization.ComputeReLUActivationPattern(model, x0)
        a, k, P, q = ReLUToOptimization.ReLUGivenActivationPattern(model, x0.shape[0], activation_pattern, self.sys.dtype)
        
        Q1, Q2, c1, c2, G1_, G2_, G3_, h_, A1_, A2_, A3_, b_ = self.value_opt
        
        # x size equal in both program
        assert(P.shape[1] == A1_.shape[1])
        
        index_x = torch.sum(G1_,1) != 0.
        
        G1_x = G1_[index_x,:]
        G1 = G1_[~index_x,:]

        G2_x = G2_[index_x,:]
        G2 = G2_[~index_x,:]
        
        G3_x = G3_[index_x,:]
        G3 = G3_[~index_x,:]
        
        h_x = h_[index_x]
        h = h_[~index_x]
        
        index_x = torch.sum(A1_,1) != 0.
        
        A1_x = A1_[index_x,:]
        A1 = A1_[~index_x,:]
        
        A2_x = A2_[index_x,:]
        A2 = A2_[~index_x,:]
        
        A3_x = A3_[index_x,:]
        A3 = A3_[~index_x,:]
        
        b_x = b_[index_x]
        b = b_[~index_x]
        
        Q1 = Q1.detach().numpy()
        Q2 = Q2.detach().numpy()
        c1 = c1.detach().numpy()
        c2 = c2.detach().numpy()
        A1_x = A1_x.detach().numpy()
        A1 = A1.detach().numpy()
        A2_x = A2_x.detach().numpy()
        A2 = A2.detach().numpy()
        A3_x = A3_x.detach().numpy()
        A3 = A3.detach().numpy()
        b_x = b_x.squeeze().detach().numpy()
        b = b.squeeze().detach().numpy()
        G1_x = G1_x.detach().numpy()
        G1 = G1.detach().numpy()
        G2_x = G2_x.detach().numpy()
        G2 = G2.detach().numpy()
        G3_x = G3_x.detach().numpy()
        G3 = G3.detach().numpy()
        h_x = h_x.squeeze().detach().numpy()
        h = h.squeeze().detach().numpy()
        a = a.squeeze().detach().numpy()
        k = k.squeeze().detach().numpy()
        P = P.detach().numpy()
        q = q.detach().numpy()       
        
        num_s = Q1.shape[0]
        num_alpha = Q2.shape[0]
        num_lambda1 = G2_x.shape[0]
        num_lambda2 = q.shape[0]
        num_nu = A2_x.shape[0]
        
        s_val = np.random.rand(num_s)
        alpha_val = np.round(np.random.rand(num_alpha))
        lambda1_val = np.random.rand(num_lambda1)
        lambda2_val = np.random.rand(num_lambda2)
        nu_val = np.random.rand(num_nu)

        s = cp.Variable(num_s)
        alpha = cp.Variable(num_alpha, boolean=True)
        lambda1 = cp.Variable(num_lambda1)
        lambda2 = cp.Variable(num_lambda2)
        nu = cp.Variable(num_nu)

        mul_prob_con = [
            A1_x.T@nu + G1_x.T@lambda1 + P.T@lambda2 + a.T == 0.,
            lambda1 >= 0.,
            lambda2 >= 0.,
        ]
        
        val_prob_con = [
            A2@s + A3@alpha == b,
            G2@s + G3@alpha <= h,
        ]        

        max_iter = 5
        for iter in range(max_iter):
            # solve for the multipliers: lambda1, lambda2, nu
            mul_prob_obj = cp.Minimize(
                (h_x - G2_x@s_val - G3_x@alpha_val).T@lambda1 +
                q.T@lambda2 +
                (b_x - A2_x@s_val - A3_x@alpha_val).T@nu - k
            )
            mul_prob = cp.Problem(mul_prob_obj,mul_prob_con)
            mul_prob.solve(solver=cp.GUROBI, verbose=False)
            lambda1_val = lambda1.value
            lambda2_val = lambda2.value
            nu_val = nu.value
            
            # # otherwise tiny numerical errors can make future iterations unbounded
            # lambda1_val = np.maximum(lambda1_val, 0.)
            # lambda2_val = np.maximum(lambda2_val, 0.)
            
            # solve for the value variables
            val_prob_obj = cp.Minimize(
                .5*cp.quad_form(s,Q1) + .5*cp.quad_form(alpha,Q2) +
                c1.T@s + c2.T@alpha +
                (h_x - G2_x@s - G3_x@alpha).T@lambda1_val +
                q.T@lambda2_val +
                (b_x - A2_x@s - A3_x@alpha).T@nu_val - k
            )
            val_prob = cp.Problem(val_prob_obj, val_prob_con)
            val_prob.solve(solver=cp.GUROBI, verbose=False)
            s_val = s.value
            alpha_val = alpha.value
            
            # compute epsilon
            epsilon = (.5*s_val.T@Q1@s_val + .5*alpha_val.T@Q2@alpha_val +
                c1.T@s_val + c2.T@alpha_val +
                (h_x - G2_x@s_val - G3_x@alpha_val).T@lambda1_val +
                q.T@lambda2_val +
                (b_x - A2_x@s_val - A3_x@alpha_val).T@nu_val - k)

            print(epsilon)

            # check for convergence