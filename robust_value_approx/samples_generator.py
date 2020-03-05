import torch
import numpy as np
import cvxpy as cp
import robust_value_approx.value_approximation as value_approximation
import robust_value_approx.value_to_optimization as value_to_optimization
import robust_value_approx.utils as utils


class SampleGenerator:
    def __init__(self, vf, x0_lo, x0_up):
        """
        Super class to generate samples from a value function
        @param vf ValueFunction instance from which to generate the samples
        @param x0_lo tensor size x_dim lower bound on initial state for the
        samples generation
        @param x0_up tensor size x_dim upper bound on initial state for the
        samples generation
        """
        assert(isinstance(vf, value_to_optimization.ValueFunction))
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


class RandomSampleGenerator(SampleGenerator):
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

    def generate_samples(self, n, project=False):
        """
        @param n Integer number of samples
        @param project Boolean on whether or not to project the sample
        to a random state/input space limit
        @return rand_data Tensor with random initial states
        @return rand_label Tensor with corresponding labels
        """
        rand_data = []
        rand_label = []
        while len(rand_data) < n:
            rand_x0 = self.get_random_x0()
            if project:
                rand_x0 = self.project_x0(rand_x0)
            rand_v, rand_res = self.V(rand_x0)
            if rand_v is not None:
                x_traj_flat = torch.cat(rand_res['x_traj'][:-1])
                cost_to_go = self.vf.result_to_costtogo(rand_res)
                rand_data.append(x_traj_flat.unsqueeze(0))
                rand_label.append(cost_to_go.unsqueeze(0))
        return(torch.cat(rand_data, axis=0), torch.cat(rand_label, axis=0))


class AdversarialSampleGenerator(SampleGenerator):
    def __init__(self, vf, x0_lo, x0_up,
                 max_iter=10, conv_tol=1e-5, learning_rate=.01, penalty=1e-8):
        assert(isinstance(vf, value_to_optimization.ValueFunction))
        self.max_iter = max_iter
        self.conv_tol = conv_tol
        self.learning_rate = learning_rate
        self.penalty = penalty
        self.V_with_grad = vf.get_differentiable_value_function()
        self.x_dim = x0_lo.shape[0]
        self.dtype = x0_lo.dtype
        self.N = vf.N
        super().__init__(vf, x0_lo, x0_up)

    def generate_samples(self, n, value_approx):
        adv_data = []
        adv_label = []
        k = 0
        while k < n:
            max_iter = min(self.max_iter, n - k)
            x_adv0 = self.get_random_x0()
            (eps, cost_to_go_buff, x_adv_buff) = self.get_squared_bound_sample(
                value_approx, x_adv0=x_adv0, max_iter=max_iter)
            print(eps)
            adv_data.append(x_adv_buff)
            adv_label.append(cost_to_go_buff)
            k += x_adv_buff.shape[0]
        return(torch.cat(adv_data, axis=0), torch.cat(adv_label, axis=0))

    def get_squared_bound_sample(self, value_approx,
                                 x_adv0=None, max_iter=None):
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
        @return cost_to_go_buff, the value of each iterate
        """
        assert(isinstance(value_approx,
            value_approximation.ValueFunctionApproximation))
        if x_adv0 is None:
            x_adv_params = torch.zeros(self.x_dim, dtype=self.dtype)
        else:
            assert(isinstance(x_adv0, torch.Tensor))
            assert(len(x_adv0) == self.x_dim)
            x_adv_params = x_adv0.clone()
        if max_iter is None:
            max_iter = self.max_iter
        x_adv_params.requires_grad = True
        x_adv = torch.max(torch.min(x_adv_params, self.x0_up), self.x0_lo)
        optimizer = torch.optim.Adam([x_adv_params], lr=self.learning_rate)
        epsilon_buff = []
        cost_to_go_buff = []
        x_adv_buff = []
        for i in range(max_iter):
            cost_to_go, x_traj_flat = self.V_with_grad(x_adv)
            if cost_to_go is None:
                break
            Vx = cost_to_go[0]
            nx = torch.clamp(value_approx.eval(x_adv.unsqueeze(0), n=0), 0.)
            epsilon = torch.pow(Vx - nx, 2)
            epsilon_buff.append(epsilon.clone().detach().unsqueeze(0))
            cost_to_go_buff.append(cost_to_go.clone().detach().unsqueeze(0))
            x_adv_buff.append(x_traj_flat.clone().detach().unsqueeze(0))
            if i == (max_iter-1):
                break
            objective = -epsilon
            optimizer.zero_grad()
            objective.backward()
            optimizer.step()
            x_adv = torch.max(torch.min(x_adv_params, self.x0_up), self.x0_lo)
            if torch.all(torch.abs(
                x_adv - x_adv_buff[-1][0, :self.x_dim]) <= self.conv_tol):
                break
        return(torch.cat(epsilon_buff, axis=0),
            torch.cat(cost_to_go_buff, axis=0),
            torch.cat(x_adv_buff, axis=0))
