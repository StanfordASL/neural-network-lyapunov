# -*- coding: utf-8 -*-
import robust_value_approx.value_to_optimization as value_to_optimization
from pydrake.solvers.mathematicalprogram import MathematicalProgram
from pydrake.solvers.snopt import SnoptSolver
import torch
import copy
import numpy as np


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

    def get_inifinite_horizon_ctrl(self, dyn_con, vf):
        x_lo = vf.x_lo[0]
        x_up = vf.x_up[0]
        u_lo = vf.u_lo[0]
        u_up = vf.u_up[0]
        dt_lo = vf.dt_lo
        dt_up = vf.dt_up
        Q = vf.Q[0]
        x_desired = vf.x_desired[0]
        R = vf.R[0]
        x_dim = x_lo.shape[0]
        u_dim = u_lo.shape[0]
        dtype = x_lo.dtype
        prog = MathematicalProgram()
        ctrl_model = copy.deepcopy(self.models[0])
        x0 = prog.NewContinuousVariables(x_dim, "x0")
        u0 = prog.NewContinuousVariables(u_dim, "u0")
        dt0 = prog.NewContinuousVariables(1, "dt0")
        x1 = prog.NewContinuousVariables(x_dim, "x1")
        u1 = prog.NewContinuousVariables(u_dim, "u1")
        dt1 = prog.NewContinuousVariables(1, "dt1")
        x0_constraint = prog.AddBoundingBoxConstraint(x_lo, x_up, x0)
        prog.AddBoundingBoxConstraint(u_lo, u_up, u0)
        prog.AddBoundingBoxConstraint(dt_lo, dt_up, dt0)
        prog.AddBoundingBoxConstraint(x_lo, x_up, x1)
        prog.AddBoundingBoxConstraint(u_lo, u_up, u1)
        prog.AddBoundingBoxConstraint(dt_lo, dt_up, dt1)
        prog.AddConstraint(dyn_con,
            lb=np.zeros(x_dim), ub=np.zeros(x_dim),
            vars=np.concatenate((x0, u0, dt0, x1, u1, dt1)))
        if Q is not None:
            if x_desired is not None:
                prog.AddQuadraticErrorCost(
                    Q=Q, x_desired=x_desired, vars=x0)
            else:
                prog.AddQuadraticCost(
                    Q=Q, b=np.zeros(x_dim), c=0., vars=x0)
        if R is not None:
            prog.AddQuadraticCost(
                Q=R, b=np.zeros(u_dim), c=0., vars=u0)
        a_nn = prog.NewContinuousVariables(x_dim, "a_nn")
        b_nn = prog.NewContinuousVariables(1, "b_nn")
        a_nn_constraint = prog.AddBoundingBoxConstraint(
            np.zeros(x_dim), np.zeros(x_dim), a_nn)
        b_nn_constraint = prog.AddBoundingBoxConstraint(
            np.zeros(1), np.zeros(1), b_nn)
        def nn_cost(vars):
            a_nn_start = 0
            a_nn_end = a_nn_start + x_dim
            b_nn_start = a_nn_end
            b_nn_end = b_nn_start + 1
            x0_start = b_nn_end
            x0_end = x0_start + x_dim
            x1_start = x0_end
            x1_end = x1_start + x_dim
            a_nn = vars[a_nn_start:a_nn_end]
            b_nn = vars[b_nn_start:b_nn_end]
            x0 = vars[x0_start:x0_end]
            x1 = vars[x1_start:x1_end]
            return np.dot(a_nn, x1 - x0) + b_nn[0]
        prog.AddCost(nn_cost, vars=np.concatenate((a_nn, b_nn, x0, x1)))
        solver = SnoptSolver()
        def ctrl(x):
            assert(isinstance(x, torch.Tensor))
            dtype = x.dtype
            x.requires_grad = True
            b = ctrl_model(x)
            a = torch.autograd.grad(b, x)[0]
            b_np = b.detach().numpy()
            a_np = a.detach().numpy()
            a_nn_constraint.evaluator().set_bounds(a_np, a_np)
            b_nn_constraint.evaluator().set_bounds(b_np, b_np)
            x = x.detach().numpy()
            x0_constraint.evaluator().set_bounds(x, x)
            result = solver.Solve(prog, np.zeros(prog.num_vars()), None)
            if not result.is_success():
                return(None, None, None)
            u0_opt = result.GetSolution(u0)
            u1_opt = result.GetSolution(u1)
            x_opt = result.GetSolution(x1)
            return(torch.Tensor(u0_opt).type(dtype),
                torch.Tensor(u1_opt).type(dtype),
                torch.Tensor(x_opt).type(dtype))
        return ctrl
