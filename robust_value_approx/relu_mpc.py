# -*- coding: utf-8 -*-

import torch


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
        Uses random shooting in order to return the optimal input control
        trajectory as well as the corresponding cost
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
