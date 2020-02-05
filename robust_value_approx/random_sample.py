import torch
import numpy as np


class RandomSampleGenerator:
    def __init__(self, vf, x0_lo, x0_up):
        self.vf = vf
        self.x0_lo = x0_lo
        self.x0_up = x0_up
        self.x_dim = x0_lo.shape[0]
        self.dtype = x0_lo.dtype
        self.V = vf.get_value_function()
        self.N = vf.N

    def get_random_x0(self):
        """
        @returns a Tensor between the initial state bounds of the training
        """
        return torch.rand(self.x_dim, dtype=self.dtype) *\
            (self.x0_up - self.x0_lo) + self.x0_lo

    def project_x0(self, x0_rand):
        """
        Projects a state sample to a random state boundary
        """
        projected_x0_rand = x0_rand.clone()
        dim = np.random.choice(self.x_dim, 1)
        if np.random.rand(1) >= .5:
            projected_x0_rand[dim] = self.x0_lo[dim]
        else:
            projected_x0_rand[dim] = self.x0_up[dim]
        return projected_x0_rand

    def get_random_samples(self, n, project=False):
        """
        @param n Integer number of samples
        @param project Boolean on whether or not to project the sample
        to a random state/input space limit
        @return rand_data Tensor with random initial states
        @return rand_label Tensor with corresponding labels
        """
        rand_data = torch.zeros(n, self.x_dim*(self.N-1), dtype=self.dtype)
        rand_label = torch.zeros(n, self.N-1, dtype=self.dtype)
        k = 0
        while k < n:
            rand_x0 = self.get_random_x0()
            if project:
                rand_x0 = self.project_x0(rand_x0)
            rand_v, rand_s, rand_alpha = self.V(rand_x0)
            if rand_v is not None:
                (x_traj_val,
                 u_traj_val,
                 alpha_traj_val) = self.vf.sol_to_traj(
                    rand_x0, rand_s, rand_alpha)
                x_traj_flat = x_traj_val[:, :-1].t().reshape((1, -1))
                step_costs = [self.vf.step_cost(
                    j,
                    x_traj_val[:, j],
                    u_traj_val[:, j],
                    alpha_traj_val[:, j]).item() for j in range(self.N)]
                cost_to_go = torch.Tensor(
                    list(np.cumsum(step_costs[::-1]))[::-1]).type(self.dtype)
                rand_data[k, :] = x_traj_flat
                rand_label[k, :] = cost_to_go[:-1]
                k += 1
        return(rand_data, rand_label)
