# -*- coding: utf-8 -*-
import robust_value_approx.value_to_optimization as value_to_optimization
from pydrake.solvers.mathematicalprogram import MathematicalProgram
from pydrake.solvers.snopt import SnoptSolver
import torch
import copy
import numpy as np
import scipy


class QuadraticModel(torch.nn.Module):
    def __init__(self, dtype, dim, Q=None, q=None, c=None, scaling=1.):
        super(QuadraticModel, self).__init__()
        if Q is None:
            Q_init = 2. * (torch.rand((dim, dim), dtype=dtype) - .5)
            Q_init = .5 * Q_init.t()@Q_init
            self.Q = torch.nn.Parameter(Q_init)
        else:
            self.Q = torch.nn.Parameter(Q)
        if q is None:
            q_init = 2. * (torch.rand(dim, dtype=dtype) - .5)
            self.q = torch.nn.Parameter(q_init)
        else:
            self.q = torch.nn.Parameter(q)
        if c is None:
            c_init = 2. * (torch.rand(1, dtype=dtype) - .5)
            self.c = torch.nn.Parameter(c_init)
        else:
            self.c = torch.nn.Parameter(c)
        self.scaling = scaling

    def forward(self, x):
        if len(x.shape) == 1:
            return (x@self.Q@x + x@self.q + self.c) * self.scaling
        else:
            return (torch.diag(x@self.Q@x.t()) +\
                x@self.q + self.c).unsqueeze(1) * self.scaling


class NeuralNetworkModel():
    def __init__(self, dtype, dim, nn_width, nn_depth,
                 activation=torch.nn.Tanh, scaling=1.):
        super(NeuralNetworkModel, self).__init__()
        self.nn_layers = [torch.nn.Linear(dim, nn_width), activation()]
        for i in range(nn_depth):
            self.nn_layers += [torch.nn.Linear(nn_width, nn_width),
                               activation()]
        nn_layers += [torch.nn.Linear(nn_width, 1)]
        self.nn = torch.nn.Sequential(*nn_layers).type(dtype)
        self.scaling = scaling

    def forward(self, x):
        return self.nn(x) * self.scaling


class ValueFunctionApproximation:
    def __init__(self):
        raise(NotImplementedError)


class InfiniteHorizonValueFunctionApproximation(ValueFunctionApproximation):
    def __init__(self, dtype, x_dim, model,
                 learning_rate=1e-3, weight_decay=0.):
        """
        Contains an approximation for a infinite-horizon value function
        uses a quadratic model if no neural network parameters are provided
        """
        self.dtype = dtype
        self.x_dim = x_dim
        self.model = model
        self.optimizer = torch.optim.Adam(self.model.parameters(),
            lr=learning_rate, weight_decay=weight_decay)

    def eval(self, x, n=0):
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


# class FiniteHorizonValueFunctionApproximation(ValueFunctionApproximation):
#     def __init__(self, dtype, x_dim, N, nn_width, nn_depth):
#         """
#         Contains an approximation for a finite-horizon value function
#         the last state is the terminal state so there is no "cost-to-go" there.
#         Therefore a value function of length N should have N-1 models
#         """
#         self.N = N
#         self.x0_lo = x0_lo
#         self.x0_up = x0_up
#         self.nn_width = nn_width
#         self.nn_depth = nn_depth
#         self.dtype = x0_lo.dtype
#         self.x_dim = x0_lo.shape[0]
#         self.models = []
#         self.optimizers = []
#         for n in range(self.N-1):
#             nn_layers = [torch.nn.Linear(self.x_dim, nn_width),
#                          torch.nn.ReLU()]
#             for i in range(nn_depth):
#                 nn_layers += [torch.nn.Linear(nn_width, nn_width),
#                               torch.nn.ReLU()]
#             nn_layers += [torch.nn.Linear(nn_width, 1)]
#             model = torch.nn.Sequential(*nn_layers).double()
#             self.models.append(model)
#             self.optimizers.append(torch.optim.Adam(model.parameters()))

#     def eval(self, n, x):
#         assert(isinstance(x, torch.Tensor))
#         assert(x.shape[1] == self.x_dim)
#         return self.models[n](x)

#     def train_step(self, x_traj_samples, v_labels):
#         assert(isinstance(x_traj_samples, torch.Tensor))
#         assert(isinstance(v_labels, torch.Tensor))
#         assert(x_traj_samples.shape[1] == self.x_dim*(self.N-1))
#         loss_log = []
#         for n in range(self.N-1):
#             v_predicted = self.eval(
#                 n, x_traj_samples[:, n*self.x_dim:(n+1)*self.x_dim])
#             loss = torch.nn.functional.mse_loss(
#                 v_predicted.squeeze(), v_labels[:, n])
#             self.optimizers[n].zero_grad()
#             loss.backward()
#             self.optimizers[n].step()
#             loss_log.append(loss.item())
#         return torch.Tensor(loss_log).type(self.dtype)

#     def validation_loss(self, x_traj_samples, v_labels):
#         assert(isinstance(x_traj_samples, torch.Tensor))
#         assert(isinstance(v_labels, torch.Tensor))
#         assert(x_traj_samples.shape[1] == self.x_dim*(self.N-1))
#         loss_log = []
#         with torch.no_grad():
#             for n in range(self.N-1):
#                 v_predicted = self.eval(
#                     n, x_traj_samples[:, n*self.x_dim:(n+1)*self.x_dim])
#                 loss = torch.nn.functional.mse_loss(
#                     v_predicted.squeeze(), v_labels[:, n])
#                 loss_log.append(loss.item())
#         return torch.Tensor(loss_log).type(self.dtype)
