# -*- coding: utf-8 -*-
import robust_value_approx.model_bounds as model_bounds
import robust_value_approx.gurobi_torch_mip as gurobi_torch_mip
import robust_value_approx.value_to_optimization as value_to_optimization
import robust_value_approx.utils as utils

import gurobipy
import torch
import cvxpy as cp


class DiffValueFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, x0,
                prob_mi, alpha_mi, obj_mi, con_mi,
                prob_con, alpha_con, obj_con, con_con,
                G0, A0):
        x0.value = x.detach().numpy()
        prob_mi.solve(solver=cp.GUROBI, verbose=False, warm_start=True)
        if obj_mi.value is None:
            ctx.success = False
            return torch.Tensor([float('nan')])
        alpha_con.value = alpha_mi.value
        prob_con.solve(solver=cp.GUROBI, verbose=False, warm_start=True)
        if obj_con.value is None:
            ctx.success = False
            return torch.Tensor([float('nan')])
        assert(abs(obj_mi.value - obj_con.value) <= 1e-5)
        ctx.success = True
        ctx.lambda_G = torch.Tensor(con_con[0].dual_value).type(x.dtype)
        ctx.lambda_A = torch.Tensor(con_con[1].dual_value).type(x.dtype)
        ctx.G0 = G0
        ctx.A0 = A0
        return torch.Tensor([obj_con.value]).type(x.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        if not ctx.success:
            grad = grad_output.clone()
            grad *= float('nan')
            return grad
        dy = (ctx.lambda_A.t()@ctx.A0 + ctx.lambda_G.t()@ctx.G0)
        grad_input = (grad_output.unsqueeze(1) @ dy.unsqueeze(0)).squeeze()
        return(grad_input, *([None]*11))


class AdversarialSampleGenerator:
    def __init__(self, vf, x0_lo, x0_up):
        """
        Generates adversarial samples for value function approximators

        @param vf An instance of ValueFunction that corresponds
        to the value function to be verified
        @param x0_lo Tensor that is the lower bound of initial states to check
        @param x0_up Tensor that is the upper bound of initial states to check
        """
        assert(isinstance(vf, value_to_optimization.ValueFunction))
        self.vf = vf
        self.V = vf.get_value_function()

        self.vf = vf
        (G0, G1, G2, h,
         A0, A1, A2, b,
         Q1, Q2, q1, q2, k) = utils.torch_to_numpy(vf.traj_opt_constraint())
        self.G0_t = torch.Tensor(G0).type(vf.dtype)
        self.A0_t = torch.Tensor(A0).type(vf.dtype)
        self.x0 = cp.Parameter(G0.shape[1])
        self.s = cp.Variable(G1.shape[1])
        self.alpha_mi = cp.Variable(G2.shape[1], boolean=True)
        self.obj_mi = cp.Minimize(.5 * cp.quad_form(self.s, Q1) +
                                  .5 * cp.quad_form(self.alpha_mi, Q2) +
                                  q1.T@self.s + q2.T@self.alpha_mi + k)
        self.con_mi = [G1@self.s + G2@self.alpha_mi <= h - G0@self.x0,
                       A1@self.s + A2@self.alpha_mi == b - A0@self.x0]
        self.prob_mi = cp.Problem(self.obj_mi, self.con_mi)
        self.alpha_con = cp.Parameter(G2.shape[1], boolean=False)
        self.obj_con = cp.Minimize(.5 * cp.quad_form(self.s, Q1) +
                                   .5 * cp.quad_form(self.alpha_con, Q2) +
                                   q1.T@self.s + q2.T@self.alpha_con + k)
        self.con_con = [G1@self.s + G2@self.alpha_con <= h - G0@self.x0,
                        A1@self.s + A2@self.alpha_con == b - A0@self.x0]
        self.prob_con = cp.Problem(self.obj_con, self.con_con)
        self.V_with_grad = lambda x: DiffValueFunction.apply(x,
                                                             self.x0,
                                                             self.prob_mi,
                                                             self.alpha_mi,
                                                             self.obj_mi,
                                                             self.con_mi,
                                                             self.prob_con,
                                                             self.alpha_con,
                                                             self.obj_con,
                                                             self.con_con,
                                                             self.G0_t,
                                                             self.A0_t)
        self.traj_opt_coeffs = vf.traj_opt_constraint()
        self.dtype = vf.dtype
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

    def get_upper_bound_global(self, model, requires_grad=False, penalty=1e-8):
        """
        Checks that the model is upper bounded by some margin
        above the true optimal cost-to-go, i.e. η(x) ≤ V(x) + ε
        This is done by minimizing, V(x) - n(x) (over x), which is an MIQP
        that we solve with GUROBI. Since this problem is (MI-)convex,
        the bound returned is valid GLOBALLY

        @param model A piecewise LINEAR (linear layers + (leaky)ReLU) neural
        network
        @param requires_grad A boolean that says whether the gradients of
        the returned values are needed or not
        @param penalty a float for the penalty when getting the gradient
        of the eps opt problem (see
        gurobi_torch_mip/compute_objective_from_mip_data_and_solution)
        """
        if requires_grad:
            mb = model_bounds.ModelBounds(self.vf, model)
            eps_opt_coeffs = mb.epsilon_opt(model, self.x0_lo, self.x0_up)
            (prob, x, y, gamma) = self.setup_eps_opt(eps_opt_coeffs)
            prob.gurobi_model.optimize()
            epsilon = prob.compute_objective_from_mip_data_and_solution(
                penalty=penalty)
            # TODO(blandry) return the gradient of the adv example as well
            # (just need the prob to return the primal as well)
            x_adv = torch.Tensor([v.x for v in x]).type(self.dtype)
        else:
            with torch.no_grad():
                mb = model_bounds.ModelBounds(self.vf, model)
                eps_opt_coeffs = mb.epsilon_opt(model, self.x0_lo, self.x0_up)
                (prob, x, y, gamma) = self.setup_eps_opt(eps_opt_coeffs)
                prob.gurobi_model.optimize()
                epsilon = torch.Tensor(
                    [prob.gurobi_model.objVal]).type(self.dtype)
                x_adv = torch.Tensor([v.x for v in x]).type(self.dtype)
        return(epsilon, x_adv)

    def setup_val_opt(self, x_val=None):
        """
        Assembles optimization problem V

        @param (optional) x_val a tensor with the value of x if it is known
        @return prob an instance of GurobiTorchMIQP corresponding to the
        value problem
        @return x, s, alpha, lists of variables of the problem (see
        ValueFunction for full description of those variables)
        """
        (G0, G1, G2, h,
         A0, A1, A2, b,
         Q1, Q2, q1, q2, k) = self.traj_opt_coeffs
        prob = gurobi_torch_mip.GurobiTorchMIQP(self.dtype)
        prob.gurobi_model.setParam(gurobipy.GRB.Param.OutputFlag, False)
        if x_val is None:
            x = prob.addVars(A0.shape[1], lb=-gurobipy.GRB.INFINITY,
                             vtype=gurobipy.GRB.CONTINUOUS, name="x")
        s = prob.addVars(Q1.shape[0], lb=-gurobipy.GRB.INFINITY,
                         vtype=gurobipy.GRB.CONTINUOUS, name="s")
        alpha = prob.addVars(Q2.shape[0], vtype=gurobipy.GRB.BINARY,
                             name="alpha")
        prob.setObjective([.5 * Q1, .5 * Q2],
                          [(s, s), (alpha, alpha)],
                          [q1, q2], [s, alpha], k,
                          gurobipy.GRB.MINIMIZE)
        if x_val is None:
            for i in range(G0.shape[0]):
                prob.addLConstr([G0[i, :], G1[i, :], G2[i, :]], [x, s, alpha],
                                gurobipy.GRB.LESS_EQUAL, h[i])
            for i in range(A0.shape[0]):
                prob.addLConstr([A0[i, :], A1[i, :], A2[i, :]], [x, s, alpha],
                                gurobipy.GRB.EQUAL, b[i])
        else:
            h_ = h - G0@x_val
            b_ = b - A0@x_val
            for i in range(G1.shape[0]):
                prob.addLConstr([G1[i, :], G2[i, :]], [s, alpha],
                                gurobipy.GRB.LESS_EQUAL, h_[i])
            for i in range(A1.shape[0]):
                prob.addLConstr([A1[i, :], A2[i, :]], [s, alpha],
                                gurobipy.GRB.EQUAL, b_[i])
        prob.gurobi_model.update()
        if x_val is None:
            return(prob, x, s, alpha)
        return(prob, x_val, s, alpha)

    def get_upper_bound_sample(self, model, num_iter=10, learning_rate=.01,
                               x_adv0=None, penalty=1e-8):
        """
        Checks that the model is upper bounded by some margin
        above the true optimal cost-to-go, i.e. η(x) ≤ V(x) + ε
        This is done by minimizing, V(x) - n(x) (over x), which is a min-min
        problem that we solve using bilevel nonlinear optimization. In general,
        this problem is not convex, and so the bound returned is valid
        LOCALLY (use get_upper_bound_sample_global if you want a global bound)

        @param model the model to verify
        @param num_iter (optional) Integer number of gradient ascent to do
        @param learning_rate (optional) Float learning rate of the
        gradient ascent
        @param x_adv0 (optional) Tensor which is initial guess for the
        adversarial example
        @param penalty (optional) a float for the penalty when getting the
        gradient of the eps opt problem (see
        compute_objective_from_mip_data_and_solution)
        @return epsilon_buff, the ε for each iterate
        @return x_adv_buff, each iterate of the optimization
        @return V_buff, the value of each iterate
        # TODO(blandry) consider using BFGS
        """
        if x_adv0 is None:
            x_adv_params = torch.zeros(self.vf.sys.x_dim, dtype=self.dtype)
        else:
            assert(isinstance(x_adv0, torch.Tensor))
            assert(len(x_adv0) == self.vf.sys.x_dim)
            x_adv_params = x_adv0.clone()
        x_adv_params.requires_grad = True
        x_adv = torch.max(torch.min(x_adv_params, self.x0_up), self.x0_lo)
        optimizer = torch.optim.Adam([x_adv_params], lr=learning_rate)
        epsilon_buff = torch.Tensor(num_iter, 1).type(self.dtype)
        x_adv_buff = torch.Tensor(num_iter, self.vf.sys.x_dim).type(self.dtype)
        V_buff = torch.Tensor(num_iter, 1).type(self.dtype)
        for i in range(num_iter):
            (prob, x, s, alpha) = self.setup_val_opt(x_val=x_adv)
            prob.gurobi_model.optimize()
            Vx = prob.compute_objective_from_mip_data_and_solution(
                penalty=penalty)
            nx = model(x_adv)
            epsilon = Vx - nx
            epsilon_buff[i, 0] = epsilon.clone().detach()
            x_adv_buff[i, :] = x_adv.clone().detach()
            V_buff[i, 0] = Vx.clone().detach()
            if i < (num_iter-1):
                objective = epsilon
                optimizer.zero_grad()
                objective.backward()
                optimizer.step()
                x_adv = torch.max(torch.min(x_adv_params, self.x0_up),
                                  self.x0_lo)
        return(epsilon_buff, x_adv_buff, V_buff)

    def get_lower_bound_sample(self, model, num_iter=10, learning_rate=.01,
                               x_adv0=None, penalty=1e-8):
        """
        Checks that the model is lower bounded by some margin
        below the true optimal cost-to-go, i.e. η(x) ≥ V(x) - ε
        This is done by maximizing, V(x) - n(x) (over x), which is a max-min
        problem that we solve using bilevel nonlinear optimization. Since this
        problem is always nonconvex, the bound returned is valid LOCALLY

        @param model the model to verify
        @param num_iter (optional) Integer number of gradient ascent to do
        @param learning_rate (optional) Float learning rate of the
        gradient ascent
        @param x_adv0 (optional) Tensor which is initial guess for the
        adversarial example
        @param penalty (optional) a float for the penalty when getting the
        gradient of the eps opt problem (see
        compute_objective_from_mip_data_and_solution)
        @return epsilon_buff, the ε for each iterate
        @return x_adv_buff, each iterate of the optimization
        @return V_buff, the value of each iterate
        # TODO(blandry) consider using BFGS
        """
        if x_adv0 is None:
            x_adv_params = torch.zeros(self.vf.sys.x_dim, dtype=self.dtype)
        else:
            assert(isinstance(x_adv0, torch.Tensor))
            assert(len(x_adv0) == self.vf.sys.x_dim)
            x_adv_params = x_adv0.clone()
        x_adv_params.requires_grad = True
        x_adv = torch.max(torch.min(x_adv_params, self.x0_up), self.x0_lo)
        optimizer = torch.optim.Adam([x_adv_params], lr=learning_rate)
        epsilon_buff = torch.Tensor(num_iter, 1).type(self.dtype)
        x_adv_buff = torch.Tensor(num_iter, self.vf.sys.x_dim).type(self.dtype)
        V_buff = torch.Tensor(num_iter, 1).type(self.dtype)
        for i in range(num_iter):
            (prob, x, s, alpha) = self.setup_val_opt(x_val=x_adv)
            prob.gurobi_model.optimize()
            Vx = prob.compute_objective_from_mip_data_and_solution(
                penalty=penalty)
            nx = model(x_adv)
            epsilon = Vx - nx
            epsilon_buff[i, 0] = epsilon.clone().detach()
            x_adv_buff[i, :] = x_adv.clone().detach()
            V_buff[i, 0] = Vx.clone().detach()
            if i < (num_iter-1):
                objective = -epsilon
                optimizer.zero_grad()
                objective.backward()
                optimizer.step()
                x_adv = torch.max(torch.min(x_adv_params, self.x0_up),
                                  self.x0_lo)
        return(epsilon_buff, x_adv_buff, V_buff)

    def get_squared_bound_sample(self, model, max_iter=10, conv_tol=1e-5,
                                 learning_rate=.001, x_adv0=None, penalty=1e-8,
                                 optimizer_state=None):
        """
        Checks that the squared model error is upper bounded by some margin
        around the true optimal cost-to-go, i.e. (V(x) - η(x))^2 ≤ ε
        This is done by maximizing, (V(x) - n(x))^2 (over x), which is a
        max-min problem that we solve using bilevel nonlinear optimization.
        Since this problem is always nonconvex, the bound returned is
        valid LOCALLY

        @param model the model to verify
        @param max_iter (optional) Integer maximum number of gradient ascent
        to do
        @param conv_tol (optional) float when the change in x is lower
        than this returns the samples
        @param learning_rate (optional) Float learning rate of the
        gradient ascent
        @param x_adv0 (optional) Tensor which is initial guess for the
        adversarial example
        @param penalty (optional) a float for the penalty when getting the
        gradient of the eps opt problem (see
        compute_objective_from_mip_data_and_solution)
        @param optimizer_state (optional) a dictionnary of optimizer states to
        reinitialize the optimizer to
        @return epsilon_buff, the ε for each iterate
        @return x_adv_buff, each iterate of the optimization
        @return V_buff, the value of each iterate
        @return optimizer_state, the state of the optimizer at the end
        """
        if x_adv0 is None:
            x_adv_params = torch.zeros(self.vf.sys.x_dim, dtype=self.dtype)
        else:
            assert(isinstance(x_adv0, torch.Tensor))
            assert(len(x_adv0) == self.vf.sys.x_dim)
            x_adv_params = x_adv0.clone()
        x_adv_params.requires_grad = True
        x_adv = torch.max(torch.min(x_adv_params, self.x0_up), self.x0_lo)
        optimizer = torch.optim.Adam([x_adv_params], lr=learning_rate)
        if optimizer_state is not None:
            optimizer.load_state_dict(optimizer_state)
        epsilon_buff = torch.Tensor(0, 1).type(self.dtype)
        x_adv_buff = torch.Tensor(0, self.vf.sys.x_dim).type(self.dtype)
        V_buff = torch.Tensor(0, 1).type(self.dtype)
        for i in range(max_iter):
            Vx = self.V_with_grad(x_adv)
            nx = model(x_adv)
            epsilon = torch.pow(Vx - nx, 2)
            epsilon_buff = torch.cat(
                (epsilon_buff, epsilon.clone().detach().unsqueeze(1)),
                axis=0)
            x_adv_buff = torch.cat(
                (x_adv_buff, x_adv.clone().detach().unsqueeze(0)),
                axis=0)
            V_buff = torch.cat(
                (V_buff, Vx.clone().detach().unsqueeze(1)),
                axis=0)
            if i == (max_iter-1):
                break
            objective = -epsilon
            optimizer.zero_grad()
            objective.backward()
            optimizer.step()
            x_adv = torch.max(torch.min(x_adv_params, self.x0_up),
                              self.x0_lo)
            if torch.all(torch.abs(x_adv - x_adv_buff[-1, :]) <= conv_tol):
                break
        optimizer_state = optimizer.state_dict()
        return(epsilon_buff, x_adv_buff, V_buff, optimizer_state)
