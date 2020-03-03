# -*- coding: utf-8 -*-
import robust_value_approx.value_to_optimization as value_to_optimization
from pydrake.solvers.mathematicalprogram import MathematicalProgram
from pydrake.solvers.snopt import SnoptSolver
import torch
import copy
import numpy as np
import scipy


class QuadraticModel(torch.nn.Module):
    def __init__(self, dim, dtype, Q=None, q=None, c=None):
        super(QuadraticModel, self).__init__()
        if Q is None:
            self.sqrtQ = torch.nn.Parameter(
                torch.rand((dim, dim), dtype=dtype))
        else:
            self.sqrtQ = torch.Tensor(scipy.linalg.sqrtm(Q)).type(dtype)
        if q is None:
            self.q = torch.nn.Parameter(torch.rand(dim, dtype=dtype))
        else:
            self.q = q
        if c is None:
            self.c = torch.nn.Parameter(torch.rand(1, dtype=dtype))
        else:
            self.c = c

    def forward(self, x):
        if len(x.shape) == 1:
            return x@(self.sqrtQ.t()@self.sqrtQ)@x + x@self.q + self.c
        else:
            return (torch.diag(x@(self.sqrtQ.t()@self.sqrtQ)@x.t()) +\
                x@self.q + self.c).unsqueeze(1)


class InfiniteHorizonValueFunctionApproximation:
    def __init__(self, x0_lo, x0_up, nn_width, nn_depth):
        """
        Contains an approximation for a infinite-horizon value function
        """
        self.x0_lo = x0_lo
        self.x0_up = x0_up
        self.nn_width = nn_width
        self.nn_depth = nn_depth
        self.dtype = x0_lo.dtype
        self.x_dim = x0_lo.shape[0]
        nn_layers = [torch.nn.Linear(self.x_dim, nn_width),
                     torch.nn.ReLU()]
        for i in range(nn_depth):
            nn_layers += [torch.nn.Linear(nn_width, nn_width),
                          torch.nn.ReLU()]
        nn_layers += [torch.nn.Linear(nn_width, 1)]
        self.model = torch.nn.Sequential(*nn_layers).double()
        # self.model = QuadraticModel(self.x_dim, self.dtype)
        self.optimizer = torch.optim.Adam(self.model.parameters())

    def eval(self, n, x):
        """
        n for compatibility with finite horizon
        """
        return self.model(x)

    def train_step(self, x_traj_samples, v_labels):
        assert(isinstance(x_traj_samples, torch.Tensor))
        assert(isinstance(v_labels, torch.Tensor))
        assert(x_traj_samples.shape[1] >= self.x_dim)
        loss_log = []
        v_predicted = self.model(x_traj_samples[:, :self.x_dim])
        loss = torch.nn.functional.mse_loss(
            v_predicted.squeeze(), v_labels[:, 0])
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        loss_log.append(loss.item())
        return torch.Tensor(loss_log).type(self.dtype)

    def validation_loss(self, x_traj_samples, v_labels):
        assert(isinstance(x_traj_samples, torch.Tensor))
        assert(isinstance(v_labels, torch.Tensor))
        assert(x_traj_samples.shape[1] >= self.x_dim)
        loss_log = []
        with torch.no_grad():
            v_predicted = self.model(x_traj_samples[:, :self.x_dim])
            loss = torch.nn.functional.mse_loss(
                v_predicted.squeeze(), v_labels[:, 0])
            loss_log.append(loss.item())
        return torch.Tensor(loss_log).type(self.dtype)


class FiniteHorizonValueFunctionApproximation:
    def __init__(self, N, x0_lo, x0_up, nn_width, nn_depth):
        """
        Contains an approximation for a finite-horizon value function
        the last state is the terminal state so there is no "cost-to-go" there.
        Therefore a value function of length N should have N-1 models
        """
        self.N = N
        self.x0_lo = x0_lo
        self.x0_up = x0_up
        self.nn_width = nn_width
        self.nn_depth = nn_depth
        self.dtype = x0_lo.dtype
        self.x_dim = x0_lo.shape[0]
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


