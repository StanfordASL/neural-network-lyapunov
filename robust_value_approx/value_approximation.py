# -*- coding: utf-8 -*-
import torch


class FiniteHorizonValueFunctionApproximation:
    def __init__(self, vf, x0_lo, x0_up, nn_width, nn_depth):
        """
        Contains an approximation for a finite-horizon value function
        the last state is the terminal state so there is no "cost-to-go" there.
        Therefore a value function of length N should have N-1 models
        """
        assert(vf.N > 1)
        self.vf = vf
        self.x0_lo = x0_lo
        self.x0_up = x0_up
        self.nn_width = nn_width
        self.nn_depth = nn_depth
        self.N = vf.N
        self.x_dim = vf.sys.x_dim
        self.u_dim = vf.sys.u_dim
        self.dtype = vf.dtype
        self.models = []
        self.optimizers = []
        for n in range(self.N-1):
            nn_layers = [torch.nn.Linear(self.x_dim, nn_width),
                         torch.nn.ReLU()]
            for i in range(nn_depth):
                nn_layers += [torch.nn.Linear(nn_width, nn_width),
                              torch.nn.ReLU()]
            nn_layers += [torch.nn.Linear(nn_width, 1)]
            model = torch.nn.Sequential(*nn_layers).double()
            self.models.append(model)
            self.optimizers.append(torch.optim.Adam(model.parameters()))

    def eval(self, n, x):
        assert(isinstance(x, torch.Tensor))
        assert(x.shape[1] == self.x_dim)
        return self.models[n](x)

    def train_step(self, x_traj_samples, v_labels):
        assert(isinstance(x_traj_samples, torch.Tensor))
        assert(isinstance(v_labels, torch.Tensor))
        assert(x_traj_samples.shape[1] == self.x_dim*(self.N-1))
        loss_log = []
        for n in range(self.N-1):
            v_predicted = self.eval(
                n, x_traj_samples[:, n*self.x_dim:(n+1)*self.x_dim])
            loss = torch.nn.functional.mse_loss(
                v_predicted.squeeze(), v_labels[:, n])
            self.optimizers[n].zero_grad()
            loss.backward()
            self.optimizers[n].step()
            loss_log.append(loss.item())
        return torch.Tensor(loss_log).type(self.dtype)

    def validation_loss(self, x_traj_samples, v_labels):
        assert(isinstance(x_traj_samples, torch.Tensor))
        assert(isinstance(v_labels, torch.Tensor))
        assert(x_traj_samples.shape[1] == self.x_dim*(self.N-1))
        loss_log = []
        with torch.no_grad():
            for n in range(self.N-1):
                v_predicted = self.eval(
                    n, x_traj_samples[:, n*self.x_dim:(n+1)*self.x_dim])
                loss = torch.nn.functional.mse_loss(
                    v_predicted.squeeze(), v_labels[:, n])
                loss_log.append(loss.item())
        return torch.Tensor(loss_log).type(self.dtype)
