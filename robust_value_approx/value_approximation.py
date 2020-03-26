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
        self.dtype = dtype
        self.dim = dim
        if Q is None:
            Q_init = torch.rand((dim, dim), dtype=dtype)
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
        self.reg = torch.eye(dim) * 1e-6

    def forward(self, x):
        if len(x.shape) == 1:
            return (x@(.5*self.Q.t()@self.Q + self.reg)@x +\
                x@self.q + self.c) * self.scaling
        else:
            val = (torch.sum(x.t()*((.5*self.Q.t()@self.Q +\
                self.reg)@x.t()), dim=0) + x@self.q + self.c)
            return val.unsqueeze(1) * self.scaling


class NeuralNetworkModel(torch.nn.Module):
    def __init__(self, dtype, dim, nn_width, nn_depth,
                 activation=torch.nn.Tanh, scaling=1.):
        super(NeuralNetworkModel, self).__init__()
        self.dtype = dtype
        self.dim = dim
        self.nn_layers = [torch.nn.Linear(dim, nn_width), activation()]
        for i in range(nn_depth):
            self.nn_layers += [torch.nn.Linear(nn_width, nn_width),
                               activation()]
        self.nn_layers += [torch.nn.Linear(nn_width, 1)]
        self.nn = torch.nn.Sequential(*self.nn_layers).type(dtype)
        self.scaling = scaling

    def forward(self, x):
        return self.nn(x) * self.scaling


class ValueFunctionApproximation:
    def __init__(self, model, learning_rate=1e-3, weight_decay=0.):
        self.dtype = model.dtype
        self.dim = model.dim
        self.model = model
        self.optimizer = torch.optim.Adam(self.model.parameters(),
            lr=learning_rate, weight_decay=weight_decay)

    def eval(self, x):
        return self.model(x)

    def train_step(self, samples, labels):
        loss_log = []
        predicted = self.model(samples)
        loss = torch.nn.functional.mse_loss(predicted.squeeze(), labels[:, 0])
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        loss_log.append(loss.item())
        return torch.tensor(loss_log, dtype=self.dtype)

    def validation_loss(self, samples, labels):
        loss_log = []
        with torch.no_grad():
            predicted = self.model(samples)
            loss = torch.nn.functional.mse_loss(
                predicted.squeeze(), labels[:, 0])
            loss_log.append(loss.item())
        return torch.tensor(loss_log, dtype=self.dtype)