# -*- coding: utf-8 -*-
import robust_value_approx.model_bounds as model_bounds
import robust_value_approx.gurobi_torch_mip as gurobi_torch_mip

import gurobipy
import torch


class AdversarialSampleGenerator:
    def __init__(self, vf, model, x0_lo, x0_up):
        """
        Generates adversarial samples for the value function
        approximator

        @param model The ReLU neural network to be verified
        @param vf An instance of ValueFunction that corresponds
        to the value function to be verified
        @param x0_lo a Tensor that is the lower bound of initial states to
        check
        @param x0_up a Tensor that is the upper bound of initial states to
        check
        """
        self.vf = vf
        self.V = vf.get_value_function()
        self.dtype = self.vf.dtype
        self.mb = model_bounds.ModelBounds(vf, model)
        self.x0_lo = x0_lo
        self.x0_up = x0_up

    def setup_eps_opt(self, eps_opt_coeffs, x_val=None):
        """
        Assembles the ε optimization problem (V - η)

        @param eps_opt_coeffs A tuple with the coeffs returned by
        ModelBounds.epsilon_opt
        @param (optional) x_val a tensor with the value of x if it is known
        @return prob an instance of GurobiTorchMIQP corresponding to the
        epsilon problem (V - η)
        @return x, y, gamma, lists of variables of the problem (see
        ModelBounds for full description of those variables)
        """
        (Q1, Q2, q1, q2, k, G0, G1, G2, h, A0, A1, A2, b) = eps_opt_coeffs
        prob = gurobi_torch_mip.GurobiTorchMIQP(self.dtype)
        prob.gurobi_model.setParam(gurobipy.GRB.Param.OutputFlag, False)
        if x_val is None:
            x = prob.addVars(A0.shape[1], lb=-gurobipy.GRB.INFINITY,
                             vtype=gurobipy.GRB.CONTINUOUS, name="x")
        else:
            x = x_val
        y = prob.addVars(Q1.shape[0], lb=-gurobipy.GRB.INFINITY,
                         vtype=gurobipy.GRB.CONTINUOUS, name="y")
        gamma = prob.addVars(Q2.shape[0], vtype=gurobipy.GRB.BINARY,
                             name="gamma")
        prob.setObjective([.5 * Q1, .5 * Q2],
                          [(y, y), (gamma, gamma)],
                          [q1, q2], [y, gamma], k,
                          gurobipy.GRB.MINIMIZE)
        if x_val is None:
            for i in range(G0.shape[0]):
                prob.addLConstr([G0[i, :], G1[i, :], G2[i, :]], [x, y, gamma],
                                gurobipy.GRB.LESS_EQUAL, h[i])
            for i in range(A0.shape[0]):
                prob.addLConstr([A0[i, :], A1[i, :], A2[i, :]], [x, y, gamma],
                                gurobipy.GRB.EQUAL, b[i])
        else:
            h_ = h - G0@x_val
            b_ = b - A0@x_val
            for i in range(G1.shape[0]):
                prob.addLConstr([G1[i, :], G2[i, :]], [y, gamma],
                                gurobipy.GRB.LESS_EQUAL, h_[i])
            for i in range(A1.shape[0]):
                prob.addLConstr([A1[i, :], A2[i, :]], [y, gamma],
                                gurobipy.GRB.EQUAL, b_[i])
        prob.gurobi_model.update()
        return(prob, x, y, gamma)

    def get_upper_bound_sample(self, model, requires_grad=False, penalty=1e-8):
        """
        Checks that the model is upper bounded by some margin
        above the true optimal cost-to-go, i.e. η(x) ≤ V(x) + ε
        This is done by minimizing, V(x) - n(x) (over x), which is an MIQP
        that we solve with GUROBI. Since this problem is convex,
        the bound returned is valid globally

        @param requires_grad A boolean that says whether the gradients of
        the returned values are needed or not
        @param penalty a float for the penalty when getting the gradient
        of the eps opt problem (see
        gurobi_torch_mip/compute_objective_from_mip_data_and_solution)
        """
        if requires_grad:
            eps_opt_coeffs = self.mb.epsilon_opt(model, self.x0_lo, self.x0_up)
            (prob, x, y, gamma) = self.setup_eps_opt(eps_opt_coeffs)
            prob.gurobi_model.optimize()
            epsilon = prob.compute_objective_from_mip_data_and_solution(
                penalty=penalty)
            # TODO(blandry) return the gradient of the adv example as well
            # (just need the prob to return the primal as well)
            x_adv = torch.Tensor([v.x for v in x]).type(self.dtype)
        else:
            with torch.no_grad():
                eps_opt_coeffs = self.mb.epsilon_opt(model,
                                                     self.x0_lo, self.x0_up)
                (prob, x, y, gamma) = self.setup_eps_opt(eps_opt_coeffs)
                prob.gurobi_model.optimize()
                epsilon = torch.Tensor(
                    [prob.gurobi_model.objVal]).type(self.dtype)
                x_adv = torch.Tensor([v.x for v in x]).type(self.dtype)
        return(epsilon, x_adv)

    def get_lower_bound_sample(self, model, num_iter=10, learning_rate=.01,
                               x_adv0=None, requires_grad=False, penalty=1e-8):
        """
        Checks that the model is lower bounded by some margin
        below the true optimal cost-to-go, i.e. η(x) ≥ V(x) - ε
        This is done by maximizing, V(x) - n(x) (over x), which is max-min
        problem that we solve using bilevel nonlinear optimization. Since this
        problem is nonconvex, the bound returned is valid LOCALLY

        @param num_iter (optional) Integer number of gradient ascent to do
        @param learning_rate (optional) Float learning rate of the
        gradient ascent
        @param x_adv0 (optional) Tensor which is initial guess for the
        adversarial example
        @param (optional) requires_grad A boolean that says whether the
        gradients of those are needed or not
        @param penalty (optional) a float for the penalty when getting the
        gradient of the eps opt problem (see
        compute_objective_from_mip_data_and_solution)
        # TODO(blandry) consider using BFGS
        # TODO(blandry) return the iterates, they can be used as well!
        """
        if x_adv0 is None:
            x_adv_params = torch.zeros(self.vf.sys.x_dim, dtype=self.dtype)
        else:
            assert(isinstance(x_adv0, torch.Tensor))
            assert(len(x_adv0) == self.vf.sys.x_dim)
            x_adv_params = x_adv0.clone()
        x_adv_params.requires_grad = True
        x_adv = torch.max(torch.min(x_adv_params, self.x0_up), self.x0_lo)
        optimizer = torch.optim.SGD([x_adv_params], lr=learning_rate)
        if not requires_grad:
            with torch.no_grad():
                eps_opt_coeffs = self.mb.epsilon_opt(model,
                                                     self.x0_lo, self.x0_up)
        for i in range(num_iter):
            if requires_grad:
                eps_opt_coeffs = self.mb.epsilon_opt(model,
                                                     self.x0_lo, self.x0_up)
            (prob, x, y, gamma) = self.setup_eps_opt(eps_opt_coeffs,
                                                     x_val=x_adv)
            prob.gurobi_model.optimize()
            epsilon = prob.compute_objective_from_mip_data_and_solution(
                penalty=penalty)
            objective = -epsilon
            optimizer.zero_grad()
            objective.backward()
            optimizer.step()
            x_adv = torch.max(torch.min(x_adv_params, self.x0_up), self.x0_lo)
        if requires_grad:
            (prob, x, y, gamma) = self.setup_eps_opt(eps_opt_coeffs,
                                                     x_val=x_adv)
            prob.gurobi_model.optimize()
            epsilon = prob.compute_objective_from_mip_data_and_solution(
                penalty=penalty)
        else:
            with torch.no_grad():
                epsilon = self.V(x_adv)[0] - model(x_adv)
        return(epsilon, x_adv)
