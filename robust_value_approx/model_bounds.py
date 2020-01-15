# -*- coding: utf-8 -*-
import robust_value_approx.relu_to_optimization as relu_to_optimization
import robust_value_approx.gurobi_torch_mip as gurobi_torch_mip

import torch
import gurobipy


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

        η(x) ≤ V(x) - ε̅, ∀ x_lo ≤ x ≤ x_up

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
         a_out, b_out, z_lo, z_up, _, _) = self.relu_opt.output_constraint(
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

    def bound_opt(self, model, x_lo, x_up):
        """
        This function returns an optimization problem (an MIQP in standard
        form) such that given an x, solving that optimization problems
        computes

        ε(x) = V(x) - η(x)

        where η is the output of the neural network, V is the optimal
        cost-to-go

        y = [s, z]
        γ = [α, β]

        min     .5 yᵀ Q1 y + .5 γᵀ Q2 γ + yᵀ q1 + γᵀ q2 + k
        s.t.    A0 x + A1 y + A2 γ = b
                G0 x + G1 y + G2 γ <= h
                γ ∈ {0,1}

        @param model: the ReLU network to verify
        @param x_lo: lower bound for the input to the neural net (x0)
        @param x_up: upper bound for the input to the neural net (x0)
        @return Q1, Q2, q1, q2, k, G0, G1, G2, h, A0, A1, A2, b
        """
        (Q1_, Q2, q1_, q2, k,
         G1_, G2, h,
         A1_, A2, b) = self.upper_bound_opt(model, x_lo, x_up)

        x_dim = self.value_fun.sys.x_dim

        Q1 = Q1_[x_dim:, x_dim:]
        q1 = q1_[x_dim:]
        A0 = A1_[:, :x_dim]
        A1 = A1_[:, x_dim:]
        G0 = G1_[:, :x_dim]
        G1 = G1_[:, x_dim:]

        return(Q1, Q2, q1, q2, k, G0, G1, G2, h, A0, A1, A2, b)

    def bound_fun(self, model, x_lo, x_up, x, penalty=1e-8):
        """
        returns a function that take x as input and returns the value of ε,
        everything returned should be a tensor, with the right gradients!

        @param model: the ReLU network to verify
        @param x_lo: lower bound for the input to the neural net (x0)
        @param x_up: upper bound for the input to the neural net (x0)
        """
        (Q1, Q2, q1, q2, k,
         G0, G1, G2, h_,
         A0, A1, A2, b_) = self.bound_opt(model, x_lo, x_up)
        b = b_ - A0@x
        h = h_ - G0@x
        prob = gurobi_torch_mip.GurobiTorchMIQP(x.dtype)
        prob.gurobi_model.setParam(gurobipy.GRB.Param.OutputFlag, False)
        y = prob.addVars(Q1.shape[0], lb=-gurobipy.GRB.INFINITY,
                         vtype=gurobipy.GRB.CONTINUOUS, name="y")
        gamma = prob.addVars(Q2.shape[0], vtype=gurobipy.GRB.BINARY,
                             name="gamma")
        prob.setObjective([.5 * Q1, .5 * Q2],
                          [(y, y), (gamma, gamma)],
                          [q1, q2], [y, gamma], k,
                          gurobipy.GRB.MINIMIZE)
        for i in range(G1.shape[0]):
            prob.addLConstr([G1[i, :], G2[i, :]], [y, gamma],
                            gurobipy.GRB.LESS_EQUAL, h[i])
        for i in range(A1.shape[0]):
            prob.addLConstr([A1[i, :], A2[i, :]], [y, gamma],
                            gurobipy.GRB.EQUAL, b[i])
        prob.gurobi_model.update()
        prob.gurobi_model.optimize()
        epsilon = prob.compute_objective_from_mip_data_and_solution(
            penalty=penalty)
        return epsilon
