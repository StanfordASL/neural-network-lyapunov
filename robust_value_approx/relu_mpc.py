# -*- coding: utf-8 -*-
import robust_value_approx.relu_to_optimization as relu_to_optimization
import robust_value_approx.utils as utils

import torch
import cvxpy as cp


class RandomShootingMPC:
    def __init__(self, vf, model, num_samples):
        """
        Class that uses random shooting and a learned model in order to
        compute a control action
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
                step_cost = self.vf.step_cost(x0, u_samples[k, :], mode)
                v = step_cost + torch.clamp(self.model(xn), 0.)
                if v < v_opt:
                    v_opt = v
                    u_opt = u_samples[k, :]
        return u_opt


class ReLUMPC:
    def __init__(self, vf, model, x_lo, x_up):
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
        relu_con = relu.output_constraint(model, x_lo, x_up)
        (Pin1, Pin2, Pin3, Prhs_in,
         Peq1, Peq2, Peq3, Prhs_eq,
         a_out, b_out,
         z_pre_relu_lo, z_pre_relu_up, _, _) = utils.torch_to_numpy(
             relu_con, squeeze=False)
        (Aeq_slack, Aeq_alpha,
         Ain_x, Ain_u, Ain_slack,
         Ain_alpha, Arhs_in) = utils.torch_to_numpy(
            vf.sys.mixed_integer_constraints(), squeeze=False)
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
        self.x0.value = x0.detach().numpy().squeeze()
        self.prob.solve(solver=cp.GUROBI, warm_start=True)
        u0_opt = torch.Tensor(self.u0.value).type(self.vf.dtype)
        x1_opt = torch.Tensor(self.x1.value).type(self.vf.dtype)
        return(u0_opt, x1_opt)
