# -*- coding: utf-8 -*-
import neural_network_lyapunov.relu_to_optimization as relu_to_optimization
import neural_network_lyapunov.value_to_optimization as value_to_optimization
import neural_network_lyapunov.mip_utils as mip_utils

import torch


class ModelBounds:
    def __init__(self, vf, model):
        """
        Generates problems that help bound the difference between
        a model and the true cost-to-go it approximates

        @param model The ReLU neural network to be verified
        @param vf An instance of ValueFunction that corresponds
        to the value function to be verified
        """
        assert (isinstance(vf, value_to_optimization.ValueFunction))
        self.vf = vf
        self.model = model
        self.dtype = vf.dtype
        self.traj_opt = vf.traj_opt_constraint()
        self.relu_opt = relu_to_optimization.ReLUFreePattern(model, self.dtype)

    def epsilon_opt(self, model, x_lo, x_up):
        """
        This function returns the coefficients of an optimization problem
        (an MIQP in standard form) such that solving its objective
        corresponds to ε(x), where

        ε(x) = V(x) - η(x)

        where η is the output of the neural network, V is the optimal
        cost-to-go. Note that minimizing this problem with for a fixed
        x results in evaluating the value of ε at that x. However x can
        also be kept as a decision variable, and minimizing the resulting
        problem gives a global lower bound on the error between the value
        function and the neural network, which effectively bounds the neural
        network to never be above V by more than ε for any x.

        x = input of nn, initial state of opt control problem
        y = [s, z]
        γ = [α, β]

        min     .5 xᵀ Q0 x + .5 yᵀ Q1 y + .5 γᵀ Q2 γ +
                xᵀ q0 + yᵀ q1 + γᵀ q2 + k
        s.t.    A0 x + A1 y + A2 γ = b
                G0 x + G1 y + G2 γ <= h
                γ ∈ {0,1}

        @param model: the ReLU network to compute ε for
        @param x_lo: lower bound for the input to the neural net (x0)
        @param x_up: upper bound for the input to the neural net (x0)
        @return Q1, Q2, q1, q2, k, G0, G1, G2, h, A0, A1, A2, b
        """
        relu_output_return = self.relu_opt.output_constraint(
            x_lo, x_up, mip_utils.PropagateBoundsMethod.IA)
        (Ain1, Ain2, Ain3, rhs_in, Aeq1, Aeq2, Aeq3, rhs_eq, Q1_val, Q2_val,
         Q3_val, q1_val, q2_val, q3_val, c) = self.traj_opt

        # x size equal in both program
        assert (relu_output_return.Ain_input.shape[1] == Ain1.shape[1])

        num_x = Ain1.shape[1]
        num_s = Ain2.shape[1]
        num_alpha = Ain3.shape[1]
        num_z = relu_output_return.Ain_slack.shape[1]
        num_beta = relu_output_return.Ain_binary.shape[1]

        num_y = num_s + num_z
        num_gamma = num_alpha + num_beta

        num_Ain = rhs_in.shape[0]
        num_Pin = relu_output_return.rhs_in.shape[0]
        num_in = num_Pin + num_Ain + 2 * num_x

        num_Aeq = rhs_eq.shape[0]
        num_Peq = relu_output_return.rhs_eq.shape[0]
        num_eq = num_Peq + num_Aeq

        s_index_s = 0
        s_index_e = num_s
        z_index_s = num_s
        z_index_e = num_s + num_z

        alpha_index_s = 0
        alpha_index_e = num_alpha
        beta_index_s = num_alpha
        beta_index_e = num_alpha + num_beta

        G0 = torch.zeros(num_in, num_x, dtype=self.dtype)
        G0[0:num_Ain, :] = Ain1
        G0[num_Ain:num_Ain + num_Pin, :] = relu_output_return.Ain_input
        G0[num_Ain + num_Pin:num_Ain + num_Pin + num_x, :] = torch.eye(
            num_x, dtype=self.dtype)
        G0[num_Ain+num_Pin+num_x:num_Ain+num_Pin+2*num_x, :] = - \
            torch.eye(num_x, dtype=self.dtype)

        G1 = torch.zeros(num_in, num_y, dtype=self.dtype)
        G1[0:num_Ain, s_index_s:s_index_e] = Ain2
        G1[num_Ain:num_Ain+num_Pin, z_index_s:z_index_e] =\
            relu_output_return.Ain_slack

        G2 = torch.zeros(num_in, num_gamma, dtype=self.dtype)
        G2[0:num_Ain, alpha_index_s:alpha_index_e] = Ain3
        G2[num_Ain:num_Ain+num_Pin, beta_index_s:beta_index_e] =\
            relu_output_return.Ain_binary

        h = torch.cat(
            (rhs_in, relu_output_return.rhs_in.squeeze(), x_up, -x_lo), 0)

        A0 = torch.zeros(num_eq, num_x, dtype=self.dtype)
        A0[0:num_Aeq, :] = Aeq1
        A0[num_Aeq:num_Aeq + num_Peq, :] = relu_output_return.Aeq_input

        A1 = torch.zeros(num_eq, num_y, dtype=self.dtype)
        A1[0:num_Aeq, s_index_s:s_index_e] = Aeq2
        A1[num_Aeq:num_Aeq+num_Peq, z_index_s:z_index_e] =\
            relu_output_return.Aeq_slack

        A2 = torch.zeros(num_eq, num_gamma, dtype=self.dtype)
        A2[0:num_Aeq, alpha_index_s:alpha_index_e] = Aeq3
        A2[num_Aeq:num_Aeq+num_Peq, beta_index_s:beta_index_e] =\
            relu_output_return.Aeq_binary

        b = torch.cat((rhs_eq, relu_output_return.rhs_eq.squeeze()), 0)

        Q0 = Q1_val.clone()

        Q1 = torch.zeros(num_y, num_y, dtype=self.dtype)
        Q1[s_index_s:s_index_e, s_index_s:s_index_e] = Q2_val

        Q2 = torch.zeros(num_gamma, num_gamma, dtype=self.dtype)
        Q2[alpha_index_s:alpha_index_e, alpha_index_s:alpha_index_e] = Q3_val

        q0 = q1_val.clone()

        q1 = torch.zeros(num_y, dtype=self.dtype)
        q1[s_index_s:s_index_e] = q2_val
        q1[z_index_s:z_index_e] = -relu_output_return.Aout_slack.squeeze()

        q2 = torch.zeros(num_gamma, dtype=self.dtype)
        q2[alpha_index_s:alpha_index_e] = q3_val

        k = c - relu_output_return.Cout

        return (Q0, Q1, Q2, q0, q1, q2, k, G0, G1, G2, h, A0, A1, A2, b)
