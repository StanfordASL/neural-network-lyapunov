# -*- coding: utf-8 -*-
import robust_value_approx.relu_to_optimization as relu_to_optimization

import torch


class ModelBounds:
    """
    Generates problems that produce bounds on the error of a learned cost-to-go
    """

    def __init__(self, model, value_fun):
        """
        can solve lower and upper bound problems that bound
        a ReLU neural network from above and below with respect to a true
        optimal cost-to-go.

        @param model The ReLU neural network to be verified
        @param valu_fun An instance of ValueFunction that corresponds
        to the value function to be verified
        """
        self.model = model
        self.value_fun = value_fun
        self.relu_opt = relu_to_optimization.ReLUFreePattern(
            model, value_fun.dtype)
        self.traj_opt = value_fun.traj_opt_constraint()
        self.dtype = value_fun.dtype

    def upper_bound_opt(self, model, x_lo, x_up):
        """
        returns an MIQP in standard form that corresponds to the upper bound
        problem.
        The solution to the returned problem corresponds to ε̅, and has the
        property that

        η(x) ≤ V(x) + ε̅, ∀ x_lo ≤ x ≤ x_up

        where η is the output of the neural network, V is the optimal
        cost-to-go

        y = [x, s, z]
        γ = [α, β]

        min     .5 yᵀ Q1 y + .5 γᵀ Q2 γ + yᵀ q1 + γᵀ q2 + k
        s.t.    A1 y + A2 γ = b
                G1 y + G2 γ <= h
                γ ∈ {0,1}

        @param model: the ReLU network to verify
        @param x_lo: lower bound for the input to the neural net (x0)
        @param x_up: upper bound for the input to the neural net (x0)
        @return Q1, Q2, q1, q2, k, G1, G2, h, A1, A2, b
        """
        (Pin1, Pin2, Pin3, qrhs_in,
         Peq1, Peq2, Peq3, qrhs_eq,
         a_out, b_out, z_lo, z_up) = self.relu_opt.output_constraint(
             model, x_lo, x_up)
        (Ain1, Ain2, Ain3, rhs_in,
         Aeq1, Aeq2, Aeq3, rhs_eq,
         Q2_val, Q3_val, q2_val, q3_val, c) = self.traj_opt

        # x size equal in both program
        assert(Pin1.shape[1] == Ain1.shape[1])

        num_x = Ain1.shape[1]
        num_s = Ain2.shape[1]
        num_alpha = Ain3.shape[1]
        num_z = Pin2.shape[1]
        num_beta = Pin3.shape[1]
        num_y = num_x + num_s + num_z
        num_gamma = num_alpha + num_beta

        num_Ain = rhs_in.shape[0]
        num_Pin = qrhs_in.shape[0]
        num_in = num_Pin + num_Ain + 2*num_x

        num_Aeq = rhs_eq.shape[0]
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

        G1 = torch.zeros(num_in, num_y, dtype=self.dtype)
        G1[0:num_Ain, x_index_s:x_index_e] = Ain1
        G1[0:num_Ain, s_index_s:s_index_e] = Ain2
        G1[num_Ain:num_Ain+num_Pin, x_index_s:x_index_e] = Pin1
        G1[num_Ain:num_Ain+num_Pin, z_index_s:z_index_e] = Pin2

        G1[num_Ain+num_Pin:num_Ain+num_Pin+num_x,
            x_index_s:x_index_e] = torch.eye(num_x, dtype=self.dtype)
        G1[num_Ain+num_Pin+num_x:num_Ain+num_Pin+2*num_x,
            x_index_s:x_index_e] = -torch.eye(num_x, dtype=self.dtype)

        G2 = torch.zeros(num_in, num_gamma, dtype=self.dtype)
        G2[0:num_Ain, alpha_index_s:alpha_index_e] = Ain3
        G2[num_Ain:num_Ain+num_Pin, beta_index_s:beta_index_e] = Pin3

        h = torch.cat((rhs_in, qrhs_in.squeeze(), x_up, -x_lo), 0)

        A1 = torch.zeros(num_eq, num_y, dtype=self.dtype)
        A1[0:num_Aeq, x_index_s:x_index_e] = Aeq1
        A1[0:num_Aeq, s_index_s:s_index_e] = Aeq2
        A1[num_Aeq:num_Aeq+num_Peq, x_index_s:x_index_e] = Peq1
        A1[num_Aeq:num_Aeq+num_Peq, z_index_s:z_index_e] = Peq2

        A2 = torch.zeros(num_eq, num_gamma, dtype=self.dtype)
        A2[0:num_Aeq, alpha_index_s:alpha_index_e] = Aeq3
        A2[num_Aeq:num_Aeq+num_Peq, beta_index_s:beta_index_e] = Peq3

        b = torch.cat((rhs_eq, qrhs_eq.squeeze()), 0)

        Q1 = torch.zeros(num_y, num_y, dtype=self.dtype)
        Q1[s_index_s:s_index_e, s_index_s:s_index_e] = Q2_val

        Q2 = torch.zeros(num_gamma, num_gamma, dtype=self.dtype)
        Q2[alpha_index_s:alpha_index_e, alpha_index_s:alpha_index_e] = Q3_val

        q1 = torch.zeros(num_y, dtype=self.dtype)
        q1[s_index_s:s_index_e] = q2_val
        q1[z_index_s:z_index_e] = -a_out.squeeze()

        q2 = torch.zeros(num_gamma, dtype=self.dtype)
        q2[alpha_index_s:alpha_index_e] = q3_val

        k = c - b_out

        return(Q1, Q2, q1, q2, k, G1, G2, h, A1, A2, b)

    def lower_bound_opt(self, model, x_lo, x_up):
        """
        returns an MIQP in standard form that corresponds to the lower bound
        problem. The solution to the returned problem corresponds 
        to -ε̲, and has the property that

        V(x) - η(x) ≤ ε̲, ∀ x_lo ≤ x ≤ x_up

        where η is the output of the neural network, V is the optimal
        cost-to-go

        y = [x, λ, ν, z]
        γ = [α, β]
                
        min     yᵀ Q1 y + γᵀ Q2 γ + yᵀ Q3 γ + yᵀ q1 + γᵀ q2 + k
        s.t.    A1 y + A2 γ = b
                G1 y + G2 γ <= h
                γ ∈ {0,1}

        NOTE: there are no .5 multipliers in front of the quadratic terms
        @param model: the ReLU network to verify
        @param x_lo: lower bound for the input to the neural net (x0)
        @param x_up: upper bound for the input to the neural net (x0)
        @return Q1, Q2, Q3, q1, q2, k, G1, G2, h, A1, A2, b
        """
        (Pin1, Pin2, Pin3, qrhs_in,
         Peq1, Peq2, Peq3, qrhs_eq,
         a_out, b_out, z_lo, z_up) = self.relu_opt.output_constraint(model, 
                                                                     x_lo, 
                                                                     x_up)
        (Ain1_all, Ain2_all, Ain3_all, rhs_in_all,
         Aeq1_all, Aeq2_all, Aeq3_all, rhs_eq_all,
         Q2_val, Q3_val, q2_val, q3_val, c_val) = self.traj_opt

        # TODO(blandry) support nonzero Q3
        assert(torch.all(Q3_val == 0.))
        # x size equal in both program
        assert(Pin1.shape[1] == Ain1_all.shape[1])
        
        s_in = torch.any(Ain2_all != 0., dim=1)
        s_eq = torch.any(Aeq2_all != 0., dim=1)

        Ain1_s = Ain1_all[s_in, :]
        Ain2_s = Ain2_all[s_in, :]
        Ain3_s = Ain3_all[s_in, :]
        rhs_in_s = rhs_in_all[s_in]
        Aeq1_s = Aeq1_all[s_eq, :]
        Aeq2_s = Aeq2_all[s_eq, :]
        Aeq3_s = Aeq3_all[s_eq, :]
        rhs_eq_s = rhs_eq_all[s_eq]

        Ain1 = Ain1_all[~s_in, :]
        Ain3 = Ain3_all[~s_in, :]
        rhs_in = rhs_in_all[~s_in]
        Aeq1 = Aeq1_all[~s_eq, :]
        Aeq3 = Aeq3_all[~s_eq, :]
        rhs_eq = rhs_eq_all[~s_eq]

        num_x = Ain1.shape[1]
        num_lambda = Ain2_s.shape[0]
        num_nu = Aeq2_s.shape[0]
        num_z = Pin2.shape[1]
        
        num_alpha = Ain3.shape[1]
        num_beta = Pin3.shape[1]

        num_y = num_x + num_lambda + num_nu + num_z
        num_gamma = num_alpha + num_beta

        x_index_s = 0
        x_index_e = num_x
        lambda_index_s = num_x
        lambda_index_e = num_x+num_lambda
        nu_index_s = num_x+num_lambda
        nu_index_e = num_x+num_lambda+num_nu
        z_index_s = num_x+num_lambda+num_nu
        z_index_e = num_x+num_lambda+num_nu+num_z

        alpha_index_s = 0
        alpha_index_e = num_alpha
        beta_index_s = num_alpha
        beta_index_e = num_alpha+num_beta

        # slack variables have no cost,
        # so we take the pseudoinverse to ignore them
        Q1_inv = torch.pinverse(Q2_val)
        
        Q1 = torch.zeros(num_y, num_y, dtype=self.dtype)
        Q1[lambda_index_s:lambda_index_e,
            lambda_index_s:lambda_index_e] = .5 * Ain2_s @ Q1_inv @ Ain2_s.t()
        Q1[nu_index_s:nu_index_e,
            nu_index_s:nu_index_e] = .5 * Aeq2_s @ Q1_inv @ Aeq2_s.t()
        Q1[lambda_index_s:lambda_index_e,
            nu_index_s:nu_index_e] = .5 * Ain2_s @ Q1_inv @ Aeq2_s.t()
        Q1[nu_index_s:nu_index_e,
            lambda_index_s:lambda_index_e] = .5 * Aeq2_s @ Q1_inv @ Ain2_s.t()
        
        Q1[x_index_s:x_index_e,
            lambda_index_s:lambda_index_e] = -.5 * Ain1_s.t()
        Q1[lambda_index_s:lambda_index_e,
            x_index_s:x_index_e] = -.5 * Ain1_s
        Q1[x_index_s:x_index_e, nu_index_s:nu_index_e] = -.5 * Aeq1_s.t()
        Q1[nu_index_s:nu_index_e, x_index_s:x_index_e] = -.5 * Aeq1_s
                
        Q2 = torch.zeros(num_gamma, num_gamma, dtype=self.dtype)
        
        Q3 = torch.zeros(num_y, num_gamma, dtype=self.dtype)
        Q3[lambda_index_s:lambda_index_e, alpha_index_s:alpha_index_e] = -Ain3_s
        Q3[nu_index_s:nu_index_e, alpha_index_s:alpha_index_e] = -Aeq3_s

        q1 = torch.zeros(num_y, dtype=self.dtype)
        q1[lambda_index_s:lambda_index_e] = rhs_in_s + Ain2_s @ Q1_inv @ q2_val
        q1[nu_index_s:nu_index_e] = rhs_eq_s + Aeq2_s @ Q1_inv @ q2_val
        q1[z_index_s:z_index_e] = a_out

        q2 = torch.zeros(num_gamma, dtype=self.dtype)
        q2[alpha_index_s:alpha_index_e] = -q3_val

        k = .5*q2_val.t()@Q1_inv@q2_val - c_val + b_out

        # continuous variables that have no quadratic cost
        no_quad_cost = (torch.diag(Q2_val) == 0.).nonzero()
        no_quad_cost_lambda = (torch.diag(Q1[lambda_index_s:lambda_index_e,lambda_index_s:lambda_index_e]) == 0.).nonzero()
        no_quad_cost_nu = (torch.diag(Q1[nu_index_s:nu_index_e,nu_index_s:nu_index_e]) == 0.).nonzero()

        num_Ain = rhs_in.shape[0]
        num_Pin = qrhs_in.shape[0]
        num_in = num_Pin + num_Ain + num_lambda + 2*num_x + 2*num_z + num_lambda

        num_Aeq = rhs_eq.shape[0]
        num_Peq = qrhs_eq.shape[0]
        num_eq = num_Peq + num_Aeq + len(no_quad_cost)
        
        G1 = torch.zeros(num_in, num_y, dtype=self.dtype)
        G1[0:num_Ain, x_index_s:x_index_e] = Ain1
        G1[num_Ain:num_Ain+num_Pin, x_index_s:x_index_e] = Pin1
        G1[num_Ain:num_Ain+num_Pin, z_index_s:z_index_e] = Pin2
        G1[num_Ain+num_Pin:num_Ain+num_Pin+num_lambda,
            lambda_index_s:lambda_index_e] = -torch.eye(num_lambda,
                                                        dtype=self.dtype)
        G1[num_Ain+num_Pin+num_lambda:num_Ain+num_Pin+num_lambda+num_x,
            x_index_s:x_index_e] = torch.eye(num_x, dtype=self.dtype)
        G1[num_Ain+num_Pin+num_lambda+num_x:num_Ain+num_Pin+num_lambda+2*num_x,
            x_index_s:x_index_e] = -torch.eye(num_x, dtype=self.dtype)
        G1[num_Ain+num_Pin+num_lambda+2*num_x:num_Ain+num_Pin+\
                num_lambda+2*num_x+num_z,
                z_index_s:z_index_e] = torch.eye(num_z, dtype=self.dtype)
        G1[num_Ain+num_Pin+num_lambda+2*num_x+num_z:num_Ain+num_Pin+\
                num_lambda+2*num_x+2*num_z,
                z_index_s:z_index_e] = -torch.eye(num_z, dtype=self.dtype)
                
        G1[num_Ain+num_Pin+num_lambda+2*num_x+2*num_z:num_Ain+num_Pin+num_lambda+2*num_x+2*num_z+num_lambda,lambda_index_s:lambda_index_e] = torch.eye(num_lambda, dtype=self.dtype)

        G2 = torch.zeros(num_in, num_gamma, dtype=self.dtype)
        G2[0:num_Ain, alpha_index_s:alpha_index_e] = Ain3
        G2[num_Ain:num_Ain+num_Pin, beta_index_s:beta_index_e] = Pin3

        h = torch.cat((rhs_in, qrhs_in.squeeze(),
                       torch.zeros(num_lambda, dtype=self.dtype),
                       x_up, -x_lo,
                       torch.clamp(z_up,0), -torch.clamp(z_lo,0),
                       100000000.*torch.ones(num_lambda, dtype=self.dtype)),0)

        A1 = torch.zeros(num_eq, num_y, dtype=self.dtype)
        A1[0:num_Aeq, x_index_s:x_index_e] = Aeq1
        A1[num_Aeq:num_Aeq+num_Peq, x_index_s:x_index_e] = Peq1
        A1[num_Aeq:num_Aeq+num_Peq, z_index_s:z_index_e] = Peq2

        A2 = torch.zeros(num_eq, num_gamma, dtype=self.dtype)
        A2[0:num_Aeq, alpha_index_s:alpha_index_e] = Aeq3
        A2[num_Aeq:num_Aeq+num_Peq, beta_index_s:beta_index_e] = Peq3

        b = torch.cat((rhs_eq, 
                       qrhs_eq.squeeze(), 
                       torch.zeros(len(no_quad_cost), dtype=self.dtype),
                       torch.zeros(len(no_quad_cost_nu), dtype=self.dtype)), 0)
        
        for i in range(len(no_quad_cost)):
            A1[num_Aeq+num_Peq+i,lambda_index_s:lambda_index_e] =\
                Ain2_s[:,no_quad_cost[i]].t()
            A1[num_Aeq+num_Peq+i,nu_index_s:nu_index_e] =\
                Aeq2_s[:,no_quad_cost[i]].t()
            b[num_Aeq+num_Peq+i] = -q2_val[no_quad_cost[i]]
            
        # i_s = num_Aeq+num_Peq+len(no_quad_cost)
        # for i in range(len(no_quad_cost_nu)):
        #     A1[i_s+i,x_index_s:x_index_e] = Aeq1_s[no_quad_cost_nu[i],:]
        #     A2[i_s+i,alpha_index_s:alpha_index_e] = Aeq3_s[no_quad_cost_nu[i],:]
        #     b[i_s+i] = rhs_eq_s[no_quad_cost_nu[i]]
        
        return(Q1, Q2, Q3, q1, q2, k, G1, G2, h, A1, A2, b)

    def lower_bound_opt_activation(self, model, x_lo, x_up, x_act):
        """
        """
        activation_pattern = relu_to_optimization.ComputeReLUActivationPattern(
                                                                model, x_act)
        
        (a_out, b_out, Pin1, qrhs_in) = relu_to_optimization.ReLUGivenActivationPattern(model,
                                                        len(x_act),
                                                        activation_pattern,
                                                        self.dtype)

        (Ain1_all, Ain2_all, Ain3_all, rhs_in_all,
         Aeq1_all, Aeq2_all, Aeq3_all, rhs_eq_all,
         Q2_val, Q3_val, q2_val, q3_val, c_val) = self.traj_opt

        # TODO(blandry) support nonzero Q3
        assert(torch.all(Q3_val == 0.))
        # x size equal in both program
        assert(Pin1.shape[1] == Ain1_all.shape[1])
        
        s_in = torch.any(Ain2_all != 0., dim=1)
        s_eq = torch.any(Aeq2_all != 0., dim=1)

        Ain1_s = Ain1_all[s_in, :]
        Ain2_s = Ain2_all[s_in, :]
        Ain3_s = Ain3_all[s_in, :]
        rhs_in_s = rhs_in_all[s_in]
        Aeq1_s = Aeq1_all[s_eq, :]
        Aeq2_s = Aeq2_all[s_eq, :]
        Aeq3_s = Aeq3_all[s_eq, :]
        rhs_eq_s = rhs_eq_all[s_eq]

        Ain1 = Ain1_all[~s_in, :]
        Ain3 = Ain3_all[~s_in, :]
        rhs_in = rhs_in_all[~s_in]
        Aeq1 = Aeq1_all[~s_eq, :]
        Aeq3 = Aeq3_all[~s_eq, :]
        rhs_eq = rhs_eq_all[~s_eq]

        num_x = Ain1.shape[1]
        num_lambda = Ain2_s.shape[0]
        num_nu = Aeq2_s.shape[0]
        
        num_alpha = Ain3.shape[1]

        num_y = num_x + num_lambda + num_nu
        num_gamma = num_alpha

        x_index_s = 0
        x_index_e = num_x
        lambda_index_s = num_x
        lambda_index_e = num_x+num_lambda
        nu_index_s = num_x+num_lambda
        nu_index_e = num_x+num_lambda+num_nu

        alpha_index_s = 0
        alpha_index_e = num_alpha

        # slack variables have no cost,
        # so we take the pseudoinverse to ignore them
        Q1_inv = torch.pinverse(Q2_val)
        
        Q1 = torch.zeros(num_y, num_y, dtype=self.dtype)
        Q1[lambda_index_s:lambda_index_e,
            lambda_index_s:lambda_index_e] = .5 * Ain2_s @ Q1_inv @ Ain2_s.t()
        Q1[nu_index_s:nu_index_e,
            nu_index_s:nu_index_e] = .5 * Aeq2_s @ Q1_inv @ Aeq2_s.t()
        Q1[lambda_index_s:lambda_index_e,
            nu_index_s:nu_index_e] = .5 * Ain2_s @ Q1_inv @ Aeq2_s.t()
        Q1[nu_index_s:nu_index_e,
            lambda_index_s:lambda_index_e] = .5 * Aeq2_s @ Q1_inv @ Ain2_s.t()
        
        Q1[x_index_s:x_index_e,
            lambda_index_s:lambda_index_e] = -.5 * Ain1_s.t()
        Q1[lambda_index_s:lambda_index_e,
            x_index_s:x_index_e] = -.5 * Ain1_s
        Q1[x_index_s:x_index_e, nu_index_s:nu_index_e] = -.5 * Aeq1_s.t()
        Q1[nu_index_s:nu_index_e, x_index_s:x_index_e] = -.5 * Aeq1_s
        
        # Q1[x_index_s:x_index_e,x_index_s:x_index_e] = torch.eye(num_x, dtype=self.dtype)*10.
                
        Q2 = torch.zeros(num_gamma, num_gamma, dtype=self.dtype)
        
        Q3 = torch.zeros(num_y, num_gamma, dtype=self.dtype)
        Q3[lambda_index_s:lambda_index_e, alpha_index_s:alpha_index_e] = -Ain3_s
        Q3[nu_index_s:nu_index_e, alpha_index_s:alpha_index_e] = -Aeq3_s

        q1 = torch.zeros(num_y, dtype=self.dtype)
        q1[x_index_s:x_index_e] = a_out.squeeze()
        q1[lambda_index_s:lambda_index_e] = rhs_in_s + Ain2_s @ Q1_inv @ q2_val
        q1[nu_index_s:nu_index_e] = rhs_eq_s + Aeq2_s @ Q1_inv @ q2_val

        q2 = torch.zeros(num_gamma, dtype=self.dtype)
        q2[alpha_index_s:alpha_index_e] = -q3_val

        k = .5*q2_val.t()@Q1_inv@q2_val - c_val + b_out.item()

        # continuous variables that have no quadratic cost
        no_quad_cost = (torch.diag(Q2_val) == 0.).nonzero()
        no_quad_cost_lambda = (torch.diag(Q1[lambda_index_s:lambda_index_e,lambda_index_s:lambda_index_e]) == 0.).nonzero()
        no_quad_cost_nu = (torch.diag(Q1[nu_index_s:nu_index_e,nu_index_s:nu_index_e]) == 0.).nonzero()

        num_Ain = rhs_in.shape[0]
        num_Pin = qrhs_in.shape[0]
        num_in = num_Pin + num_Ain + num_lambda + 2*num_x + num_lambda

        num_Aeq = rhs_eq.shape[0]
        num_eq =  num_Aeq + len(no_quad_cost)
        
        G1 = torch.zeros(num_in, num_y, dtype=self.dtype)
        G1[0:num_Ain, x_index_s:x_index_e] = Ain1
        G1[num_Ain:num_Ain+num_Pin, x_index_s:x_index_e] = Pin1
        G1[num_Ain+num_Pin:num_Ain+num_Pin+num_lambda,
            lambda_index_s:lambda_index_e] = -torch.eye(num_lambda,
                                                        dtype=self.dtype)
        G1[num_Ain+num_Pin+num_lambda:num_Ain+num_Pin+num_lambda+num_x,
            x_index_s:x_index_e] = torch.eye(num_x, dtype=self.dtype)
        G1[num_Ain+num_Pin+num_lambda+num_x:num_Ain+num_Pin+num_lambda+2*num_x,
            x_index_s:x_index_e] = -torch.eye(num_x, dtype=self.dtype)

        G2 = torch.zeros(num_in, num_gamma, dtype=self.dtype)
        G2[0:num_Ain, alpha_index_s:alpha_index_e] = Ain3
        
        G1[num_Ain+num_Pin+num_lambda+2*num_x:num_Ain+num_Pin+num_lambda+2*num_x+num_lambda,
           lambda_index_s:lambda_index_e] = torch.eye(num_lambda, dtype=self.dtype)

        h = torch.cat((rhs_in, qrhs_in.squeeze(),
                       torch.zeros(num_lambda, dtype=self.dtype),
                       x_up, -x_lo,
                       1000000.*torch.ones(num_lambda, dtype=self.dtype)),0)

        print(lambda_index_s)
        print(lambda_index_e)

        # i_s = num_Ain+num_Pin+num_lambda+2*num_x
        # for i in range(len(no_quad_cost_lambda)):
        #     G1[i_s+i,x_index_s:x_index_e] = Ain1_s[no_quad_cost_lambda[i],:]
        #     G2[i_s+i,alpha_index_s:alpha_index_e] = Ain3_s[no_quad_cost_lambda[i],:]
        #     h[i_s+i] = rhs_in_s[no_quad_cost_lambda[i]]

        A1 = torch.zeros(num_eq, num_y, dtype=self.dtype)
        A1[0:num_Aeq, x_index_s:x_index_e] = Aeq1

        A2 = torch.zeros(num_eq, num_gamma, dtype=self.dtype)
        A2[0:num_Aeq, alpha_index_s:alpha_index_e] = Aeq3

        b = torch.cat((rhs_eq,
                       torch.zeros(len(no_quad_cost), dtype=self.dtype),
                       torch.zeros(len(no_quad_cost_nu), dtype=self.dtype)), 0)
        
        for i in range(len(no_quad_cost)):
            A1[num_Aeq+i,lambda_index_s:lambda_index_e] =\
                Ain2_s[:,no_quad_cost[i]].t()
            A1[num_Aeq+i,nu_index_s:nu_index_e] =\
                Aeq2_s[:,no_quad_cost[i]].t()
            b[num_Aeq+i] = -q2_val[no_quad_cost[i]]
            
        # i_s = num_Aeq+len(no_quad_cost)
        # for i in range(len(no_quad_cost_nu)):
        #     A1[i_s+i,x_index_s:x_index_e] = Aeq1_s[no_quad_cost_nu[i],:]
        #     A2[i_s+i,alpha_index_s:alpha_index_e] = Aeq3_s[no_quad_cost_nu[i],:]
        #     b[i_s+i] = rhs_eq_s[no_quad_cost_nu[i]]
        
        return(Q1, Q2, Q3, q1, q2, k, G1, G2, h, A1, A2, b)