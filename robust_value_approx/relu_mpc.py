# -*- coding: utf-8 -*-
import robust_value_approx.relu_to_optimization as relu_to_optimization

import torch
import cvxpy as cp
from heapq import heappush, heappop
import copy


class RandomShootingMPC:
    def __init__(self, vf, model, num_samples):
        """
        Class that uses random shooting and a learned model in order to
        compute a control action. For now does not support value function
        with feedforward terms
        @param vf A ValueFunction object containing the optimization
        representation that is approximated by model
        @param model A pytorch model to be used to approximated the optimal
        cost to go (model input is state, output is optimal cost to go)
        @param num_samples The number of samples to take in the random
        shooting. Note that infeasible samples (control action that take the
        system outside of its state constraints) still count towards that total
        TODO(blandry) @param horizon An integer which corresponds to
        how many forward simulation steps are to be used in the rollout before
        using the learned optimal cost-to-go (model)
        """
        self.vf = vf
        self.model = model
        self.num_samples = num_samples
        self.u_range = (self.vf.u_up - self.vf.u_lo).repeat(num_samples, 1)
        self.u_lo_samples = self.vf.u_lo.repeat(num_samples, 1)
        assert(self.vf.xtraj is None)
        assert(self.vf.utraj is None)
        assert(self.vf.alphatraj is None)

    def step_cost(self, x_val, u_val, alpha_val):
        """
        Computes the cost of a single step with the value function.
        Note that the step should not be the terminal one (i.e. not
        correspond to Qt, Rt and Zt.
        @param x_val A tensor with the value of the state
        @param u_val A tensor with the value of the control input
        @param alpha_val A tensor with the value of the discrete variables
        """
        cost = 0.
        if self.vf.Q is not None:
            cost += .5 * x_val @ self.vf.Q @ x_val
        if self.vf.R is not None:
            cost += .5 * u_val @ self.vf.R @ u_val
        if self.vf.Z is not None:
            cost += .5 * alpha_val @ self.vf.Z @ alpha_val
        if self.vf.q is not None:
            cost += x_val @ self.vf.q
        if self.vf.r is not None:
            cost += u_val @ self.vf.r
        if self.vf.z is not None:
            cost += alpha_val @ self.vf.z
        return cost

    def get_ctrl(self, x0):
        """
        Uses random shooting in order to return the optimal control
        @param x0 A tensor that is the current/starting state
        """
        u_samples = torch.rand((self.num_samples, self.vf.sys.u_dim),
                               dtype=self.vf.dtype) * self.u_range +\
            self.u_lo_samples
        v_opt = float("Inf")
        u_opt = torch.zeros(self.vf.sys.u_dim, dtype=self.vf.dtype)
        for k in range(self.num_samples):
            (xn, mode) = self.vf.sys.step_forward(x0, u_samples[k, :])
            if ~isinstance(xn, type(None)):
                step_cost = self.step_cost(x0, u_samples[k, :], mode)
                v = step_cost + torch.clamp(self.model(xn), 0.)
                if v < v_opt:
                    v_opt = v
                    u_opt = u_samples[k, :]
        return u_opt


class ReLUMPC:
    def __init__(self, vf, model):
        """
        Controller that computes optimal control actions by optimizing
        over the learned value function directly
        @param vf A ValueFunction object containing the optimization
        representation that is approximated by model
        @param model A pytorch model to be used to approximated the optimal
        cost to go (model input is state, output is optimal cost to go)
        @param x_lo A tensor that is the lower bound of the input to the
        controller
        @param x_up A tensor that is the upper bound of the input to the
        controller
        TODO(blandry) @param horizon An integer which corresponds to
        how many forward simulation steps are to be used in the rollout before
        using the learned optimal cost-to-go (model)
        """
        self.vf = vf
        self.model = model
        relu = relu_to_optimization.ReLUFreePattern(model, vf.dtype)
        relu_con, _, _, _, _ = relu.output_constraint(vf.x_lo, vf.x_up)
        Pin1 = relu_con.Ain_input.detach().numpy()
        Pin2 = relu_con.Ain_slack.detach().numpy()
        Pin3 = relu_con.Ain_binary.detach().numpy()
        Prhs_in = relu_con.rhs_in.detach().numpy()
        Peq1 = relu_con.Aeq_input.detach().numpy()
        Peq2 = relu_con.Aeq_slack.detach().numpy()
        Peq3 = relu_con.Aeq_binary.detach().numpy()
        Prhs_eq = relu_con.rhs_eq.detach().numpy()
        a_out = relu_con.Aout_slack.detach().numpy()
        b_out = relu_con.Cout.detach().numpy()
        mip_constr_return = vf.sys.mixed_integer_constraints()
        Aeq_slack = mip_constr_return.Aout_slack.detach().numpy()
        Aeq_alpha = mip_constr_return.Aout_binary.detach().numpy()
        Ain_xu = mip_constr_return.Ain_input.detach().numpy()
        Ain_x = Ain_xu[:, :vf.sys.x_dim]
        Ain_u = Ain_xu[:, vf.sys.x_dim:]
        Ain_slack = mip_constr_return.Ain_slack.detach().numpy()
        Ain_alpha = mip_constr_return.Ain_binary.detach().numpy()
        Arhs_in = mip_constr_return.rhs_in.detach().numpy()
        self.x0 = cp.Parameter(self.vf.sys.x_dim)
        self.u0 = cp.Variable(self.vf.sys.u_dim)
        self.slack = cp.Variable(Ain_slack.shape[1])
        self.alpha = cp.Variable(Aeq_alpha.shape[1], boolean=True)
        self.z = cp.Variable(Pin2.shape[1])
        self.beta = cp.Variable(Pin3.shape[1], boolean=True)
        self.x1 = Aeq_slack @ self.slack + Aeq_alpha @ self.alpha
        vf_obj = 0.
        if not isinstance(self.vf.Q, type(None)):
            Q = self.vf.Q.detach().numpy()
            if not isinstance(self.vf.xtraj, type(None)):
                xt = self.vf.xtraj[:, 0].detach().numpy()
                vf_obj += .5*cp.quad_form(self.x1 - xt, Q)
            else:
                vf_obj += .5*cp.quad_form(self.x1, Q)
        if not isinstance(self.vf.R, type(None)):
            R = self.vf.R.detach().numpy()
            if not isinstance(self.vf.utraj, type(None)):
                ut = self.vf.utraj[:, 0].detach().numpy()
                vf_obj += .5*cp.quad_form(self.u0 - ut, R)
            else:
                vf_obj += .5*cp.quad_form(self.u0, R)
        if not isinstance(self.vf.Z, type(None)):
            Z = self.vf.Z.detach().numpy()
            if not isinstance(self.vf.alphatraj, type(None)):
                alphat = self.vf.alphatraj[:, 0].detach().numpy()
                vf_obj += .5*cp.quad_form(self.alpha - alphat, Z)
            else:
                vf_obj += .5*cp.quad_form(self.alpha, Z)
        if not isinstance(self.vf.q, type(None)):
            q = self.vf.q.detach().numpy()
            if not isinstance(self.vf.xtraj, type(None)):
                xt = self.vf.xtraj[:, 0].detach().numpy()
                vf_obj += (self.x1 - xt).T@q
            else:
                vf_obj += self.x1.T@q
        if not isinstance(self.vf.r, type(None)):
            r = self.vf.r.detach().numpy()
            if not isinstance(self.vf.utraj, type(None)):
                ut = self.vf.utraj[:, 0].detach().numpy()
                vf_obj += (self.u0 - ut).T@r
            else:
                vf_obj += self.u0.T@r
        if not isinstance(self.vf.z, type(None)):
            z = self.vf.z.detach().numpy()
            if not isinstance(self.vf.alphatraj, type(None)):
                zt = self.vf.alphatraj[:, 0].detach().numpy()
                vf_obj += (self.alpha - zt).T@z
            else:
                vf_obj += self.alpha.T@z
        self.obj = cp.Minimize(vf_obj + a_out @ self.z + b_out)
        self.cons = []
        if len(Prhs_in) > 0:
            self.cons.append(Pin1 @ self.x1 + Pin2 @ self.z +
                             Pin3 @ self.beta <= Prhs_in.squeeze())
        if len(Prhs_eq) > 0:
            self.cons.append(Peq1 @ self.x1 + Peq2 @ self.z +
                             Peq3 @ self.beta == Prhs_eq.squeeze())
        if len(Arhs_in) > 0:
            self.cons.append(Ain_x @ self.x0 + Ain_u @ self.u0 +
                             Ain_slack@self.slack +
                             Ain_alpha@self.alpha <= Arhs_in.squeeze())
        self.cons.append(cp.sum(self.alpha) == 1.)
        self.cons.append(self.x1 >= self.vf.x_lo)
        self.cons.append(self.x1 <= self.vf.x_up)
        self.cons.append(self.u0 >= self.vf.u_lo)
        self.cons.append(self.u0 <= self.vf.u_up)
        self.prob = cp.Problem(self.obj, self.cons)

    def get_ctrl(self, x0):
        """
        Solves an MIQP to return the optimal control action corresponding
        to one step integration and the value function model evaluated
        at the resulting state
        @param x0 A tensor that is the current/starting state
        """
        assert(isinstance(x0, torch.Tensor))
        self.x0.value = x0.detach().numpy()
        self.prob.solve(solver=cp.GUROBI, warm_start=True)
        if self.u0.value is None:
            return (None, None)
        u0_opt = torch.Tensor(self.u0.value).type(self.vf.dtype)
        x1_opt = torch.Tensor(self.x1.value).type(self.vf.dtype)
        return(u0_opt, x1_opt)


class QReLUMPC:
    def __init__(self, model, x_lo, x_up, u_lo, u_up):
        """
        Controller that computes optimal control actions by optimizing
        over the learned Q-function directly
        @param model A pytorch model to be used to approximated the q-function
        (model input is [state,input], output is optimal cost to go)
        @param x_lo A tensor that is the lower bound of the state input to the
        controller
        @param x_up A tensor that is the upper bound of the state input to the
        controller
        @param u_lo A tensor that is the lower bound of the input to the
        controller
        @param u_up A tensor that is the upper bound of the input to the
        controller
        """
        self.dtype = x_lo.dtype
        self.x_dim = x_lo.shape[0]
        self.u_dim = u_lo.shape[0]
        self.model = model
        relu = relu_to_optimization.ReLUFreePattern(model, self.dtype)
        xu_lo = torch.cat((x_lo, u_lo), 0)
        xu_up = torch.cat((x_up, u_up), 0)
        relu_con, _, _, _, _ = relu.output_constraint(xu_lo, xu_up)
        Pin1 = relu_con.Ain_input.detach().numpy()
        Pin2 = relu_con.Ain_slack.detach().numpy()
        Pin3 = relu_con.Ain_binary.detach().numpy()
        Prhs_in = relu_con.rhs_in.detach().numpy()
        Peq1 = relu_con.Aeq_input.detach().numpy()
        Peq2 = relu_con.Aeq_slack.detach().numpy()
        Peq3 = relu_con.Aeq_binary.detach().numpy()
        Prhs_eq = relu_con.rhs_eq.detach().numpy()
        a_out = relu_con.Aout_slack.detach().numpy()
        b_out = relu_con.Cout.detach().numpy()
        self.x0 = cp.Parameter(self.x_dim)
        self.u0 = cp.Variable(self.u_dim)
        self.z = cp.Variable(Pin2.shape[1])
        self.beta = cp.Variable(Pin3.shape[1], boolean=True)
        self.obj = cp.Minimize(a_out @ self.z + b_out)
        self.cons = []
        if len(Prhs_in) > 0:
            self.cons.append(Pin1[:, :self.x_dim] @ self.x0 +
                             Pin1[:, self.x_dim:] @ self.u0 +
                             Pin2 @ self.z +
                             Pin3 @ self.beta <= Prhs_in.squeeze())
        if len(Prhs_eq) > 0:
            self.cons.append(Peq1[:, :self.x_dim] @ self.x0 +
                             Peq1[:, self.x_dim:] @ self.u0 +
                             Peq2 @ self.z +
                             Peq3 @ self.beta == Prhs_eq.squeeze())
        self.prob = cp.Problem(self.obj, self.cons)

    def get_ctrl(self, x0):
        """
        Solves an MIQP to return the optimal control action corresponding
        to the learned q function
        @param x0 A tensor that is the current/starting state
        """
        assert(isinstance(x0, torch.Tensor))
        self.x0.value = x0.detach().numpy()
        self.prob.solve(solver=cp.GUROBI, warm_start=True)
        if self.u0.value is None:
            return None
        u0_opt = torch.Tensor(self.u0.value).type(self.dtype)
        return u0_opt


class InformedSearchMPC:
    def __init__(self, vf, model, num_samples):
        """
        Class that uses searh and a learned model in order to
        compute a control action.
        @param vf A ValueFunction object containing the optimization
        representation that is approximated by model
        @param model A pytorch model to be used to approximated the optimal
        cost to go (model input is state, output is optimal cost to go)
        @param num_samples The number of samples to take in the random
        shooting. Note that infeasible samples (control action that take the
        system outside of its state constraints) still count towards that total
        """
        self.vf = vf
        self.model = model
        self.num_samples = num_samples
        self.u_range = (self.vf.u_up - self.vf.u_lo).repeat(num_samples, 1)
        self.u_lo_samples = self.vf.u_lo.repeat(num_samples, 1)

    def get_ctrl(self, x0):
        """
        Uses search in order to return the optimal control
        @param x0 A tensor that is the current/starting state
        """
        assert(isinstance(x0, torch.Tensor))
        assert(x0.shape[0] == self.vf.sys.x_dim)
        search_nodes = []
        # (hn = vn + cum_cost, cum_cost, n, x_traj, u_traj)
        heappush(search_nodes, (self.model(x0), 0., 0, [x0], []))
        x_traj = []
        u_traj = []
        while True:
            hn, cum_cost, n, x_traj, u_traj = heappop(search_nodes)
            if n == self.vf.N:
                break
            xn = x_traj[-1]
            u_samples = torch.rand((self.num_samples, self.vf.sys.u_dim),
                                   dtype=self.vf.dtype) * self.u_range +\
                self.u_lo_samples
            xn_ = torch.Tensor(
                self.num_samples, self.vf.sys.x_dim).type(self.vf.dtype)
            cost = torch.Tensor(self.num_samples).type(self.vf.dtype)
            for k in range(self.num_samples):
                (xn_k, mode_k) = self.vf.sys.step_forward(xn, u_samples[k, :])
                if xn_k is None:
                    cost[k] = torch.Tensor(float("inf")).type(self.vf.dtype)
                else:
                    alpha_k = torch.zeros(
                        self.vf.sys.num_modes, dtype=self.vf.dtype)
                    alpha_k[mode_k] = 1.
                    cost[k] = self.vf.step_cost(
                        n, xn, u_samples[k, :], alpha_k)
                    xn_[k, :] = xn_k
            with torch.no_grad():
                vn_ = self.model(xn_)
            for k in range(self.num_samples):
                x_traj_ = copy.deepcopy(x_traj)
                x_traj_.append(xn_[k, :])
                u_traj_ = copy.deepcopy(u_traj)
                u_traj_.append(u_samples[k, :])
                node = (cum_cost + cost[k] + vn_[k, 0],
                        cum_cost + cost[k], n + 1, x_traj_, u_traj_)
                heappush(search_nodes, node)
        x_opt = torch.cat([x.unsqueeze(1) for x in x_traj[:-1]], axis=1)
        u_opt = torch.cat([u.unsqueeze(1) for u in u_traj], axis=1)
        return (u_opt, x_opt)
