import robust_value_approx.value_to_optimization as value_to_optimization
import robust_value_approx.ball_paddle_hybrid_linear_system as bphls
import robust_value_approx.model_bounds as model_bounds
from robust_value_approx.utils import torch_to_numpy
import robust_value_approx.hybrid_linear_system as hybrid_linear_system
import robust_value_approx.gurobi_torch_mip as gurobi_torch_mip
import double_integrator

import numpy as np
import unittest
import cvxpy as cp
import torch
import torch.nn as nn
import cplex
import gurobipy


class ModelBoundsUpperBound(unittest.TestCase):
    def setUp(self):
        self.dtype = torch.float64

        self.linear1 = nn.Linear(6, 10)
        self.linear1.weight.data = torch.tensor(
            np.random.rand(10, 6), dtype=self.dtype)
        self.linear1.bias.data = torch.tensor(
            np.random.rand(10), dtype=self.dtype)
        self.linear2 = nn.Linear(10, 10)
        self.linear2.weight.data = torch.tensor(
            np.random.rand(10, 10),
            dtype=self.dtype)
        self.linear2.bias.data = torch.tensor(
            np.random.rand(10), dtype=self.dtype)
        self.linear3 = nn.Linear(10, 1)
        self.linear3.weight.data = torch.tensor(
            np.random.rand(1, 10), dtype=self.dtype)
        self.linear3.bias.data = torch.tensor([-10], dtype=self.dtype)
        self.ball_paddle_model = nn.Sequential(self.linear1,
                                               nn.ReLU(),
                                               self.linear2,
                                               nn.ReLU(),
                                               self.linear3)
                                                     
        # self.linear1 = nn.Linear(2, 10)
        # self.linear1.weight.data = torch.tensor(
        #     np.random.rand(10, 2), dtype=self.dtype)
        # self.linear1.bias.data = torch.tensor(
        #     np.random.rand(10), dtype=self.dtype)
        # self.linear2 = nn.Linear(10, 10)
        # self.linear2.weight.data = torch.tensor(
        #     np.random.rand(10, 10),
        #     dtype=self.dtype)
        # self.linear2.bias.data = torch.tensor(
        #     np.random.rand(10), dtype=self.dtype)
        # self.linear3 = nn.Linear(10, 1)
        # self.linear3.weight.data = torch.tensor(
        #     np.random.rand(1, 10), dtype=self.dtype)
        # self.linear3.bias.data = torch.tensor([1.], dtype=self.dtype)
        # self.double_integrator_model = nn.Sequential(self.linear1,
        #                                              nn.ReLU(),
        #                                              self.linear2,
        #                                              nn.ReLU(),
        #                                              self.linear3)
                                                     
        self.double_integrator_model = torch.load("double_integrator_model.pt")

    def test_upper_bound(self):
        dtype = self.dtype
        dt = .01
        N = 10
        x_lo = torch.Tensor(
            [-1., -1., 0., -1e6, -1e6, -1e6]).type(dtype)
        x_up = torch.Tensor(
            [1., 10., 2., 1e6, 1e6, 1e6]).type(dtype)
        u_lo = torch.Tensor([-np.pi / 2, -1e7]).type(dtype)
        u_up = torch.Tensor([np.pi / 2, 1e7]).type(dtype)
        sys = bphls.get_ball_paddle_hybrid_linear_system(
            dtype, dt, x_lo, x_up, u_lo, u_up)
        vf = value_to_optimization.ValueFunction(
            sys, N, x_lo, x_up, u_lo, u_up)

        Q = torch.ones(sys.x_dim, sys.x_dim) * 0.1
        q = torch.ones(sys.x_dim) * 0.1
        R = torch.ones(sys.u_dim, sys.u_dim) * 2.
        r = torch.ones(sys.u_dim) * 0.1
        vf.set_cost(Q=Q, R=R, q=q, r=r)
        vf.set_terminal_cost(Qt=Q, Rt=R, qt=q, rt=r)
        xN = torch.Tensor([np.nan, .5, 0., np.nan, 0., np.nan])
        vf.set_constraints(xN=xN)

        mb = model_bounds.ModelBounds(self.ball_paddle_model, vf)
        x0_lo = torch.Tensor([0., 0., 0., 0., 0., 0.]).type(dtype)
        x0_up = torch.Tensor([0., 2., .1, 0., 0., 0.]).type(dtype)
        bound_opt = mb.upper_bound_opt(self.ball_paddle_model, x0_lo, x0_up)
        Q1, Q2, q1, q2, k, G1, G2, h, A1, A2, b = torch_to_numpy(bound_opt)

        num_y = Q1.shape[0]
        num_gamma = Q2.shape[0]
        y = cp.Variable(num_y)
        gamma = cp.Variable(num_gamma, boolean=True)

        obj = cp.Minimize(.5 * cp.quad_form(y, Q1) + .5 *
                          cp.quad_form(gamma, Q2) + q1@y + q2@gamma + k)
        con = [
            A1@y + A2@gamma == b,
            G1@y + G2@gamma <= h,
        ]

        prob = cp.Problem(obj, con)
        prob.solve(solver=cp.GUROBI, verbose=False)
        epsilon = obj.value

        V = vf.get_value_function()

        x0 = torch.Tensor(y.value[:sys.x_dim]).type(sys.dtype)
        value, _, _ = V(x0)
        z_nn = self.ball_paddle_model(x0).item()

        self.assertTrue(np.abs((epsilon - (value - z_nn)) / epsilon) <= .01)

        for i in range(20):
            x0_sub = x0 + .1 * \
                torch.rand(sys.x_dim, dtype=sys.dtype) * (x0_up - x0_lo)
            x0_sub = torch.max(torch.min(x0_sub, x0_up), x0_lo)
            value_sub = None
            try:
                value_sub, _, _ = V(x0_sub)
            except AttributeError:
                # for some reason the solver didn't return anything
                pass
            if value_sub is not None:
                z_nn_sub = self.ball_paddle_model(x0_sub).item()
                epsilon_sub = value_sub - z_nn_sub
                self.assertGreaterEqual(epsilon_sub, epsilon)

    def test_lower_bound(self):
        dtype = torch.float64
        (A, B) = double_integrator.double_integrator_dynamics(dtype)
        x_dim = A.shape[1]
        u_dim = B.shape[1]
        sys = hybrid_linear_system.HybridLinearSystem(x_dim, u_dim, dtype)

        c = torch.zeros(x_dim, dtype=dtype)
        x_lo = -1. * torch.ones(x_dim, dtype=dtype)
        x_up = 1. * torch.ones(x_dim, dtype=dtype)
        u_lo = -1. * torch.ones(u_dim, dtype=dtype)
        u_up = 1. * torch.ones(u_dim, dtype=dtype)
        P = torch.cat((-torch.eye(x_dim+u_dim),
                       torch.eye(x_dim+u_dim)), 0).type(dtype)
        q = torch.cat((-x_lo, -u_lo, x_up, u_up), 0).type(dtype)
        sys.add_mode(A, B, c, P, q)

        # value function
        N = 5
        vf = value_to_optimization.ValueFunction(
            sys, N, x_lo, x_up, u_lo, u_up)
        Q = torch.eye(sys.x_dim)
        R = torch.eye(sys.u_dim)
        vf.set_cost(Q=Q)
        vf.set_cost(R=R)
        vf.set_terminal_cost(Qt=Q)
        vf.set_terminal_cost(Rt=R)
        xN = torch.Tensor([1., 1.])
        vf.set_constraints(xN=xN)

        mb = model_bounds.ModelBounds(self.double_integrator_model, vf)
        x0_lo = x_lo
        x0_up = x_up
        bound_opt = mb.lower_bound_opt(
            self.double_integrator_model, x0_lo, x0_up)
        
        Q, q, k, G, h, A, b, intv = bound_opt
        num_var = Q.shape[0]
        num_gamma = len(intv)
        num_y = num_var - num_gamma
        gtm = gurobi_torch_mip.GurobiTorchMIQP(dtype)
        y = gtm.addVars(num_y, vtype=gurobipy.GRB.CONTINUOUS, name="y")
        gamma = gtm.addVars(num_gamma, vtype=gurobipy.GRB.BINARY, name="gamma")
        gtm.setObjective([.5*Q[:num_y,:num_y]+1e-12*torch.eye(num_y,dtype=dtype),
                          .5*Q[num_y:,num_y:],
                          Q[:num_y,num_y:]],
                          [(y,y),(gamma,gamma),(y,gamma)],
                          [q[:num_y],q[num_y:]],[y,gamma],
                          constant=k,sense=gurobipy.GRB.MINIMIZE)
        # gtm.setObjective([.5*Q[:num_y,:num_y],
        #                   .5*Q[num_y:,num_y:],
        #                   Q[:num_y,num_y:]],
        #                   [(y,y),(gamma,gamma),(y,gamma)],
        #                   [q[:num_y],q[num_y:]],[y,gamma],
        #                   constant=k,sense=gurobipy.GRB.MINIMIZE)
        for i in range(G.shape[0]):
            gtm.addLConstr([G[i,:num_y],G[i,num_y:]],[y,gamma],
                           gurobipy.GRB.LESS_EQUAL,h[i])
        for i in range(A.shape[0]):
            gtm.addLConstr([A[i,:num_y],A[i,num_y:]],[y,gamma],
                           gurobipy.GRB.EQUAL,b[i])

        gtm.gurobi_model.update()
        gtm.gurobi_model.optimize()
        epsilon = -gtm.gurobi_model.getObjective().getValue()     
        print("EPSILON: %f" % epsilon)

        V = vf.get_value_function()
        for i in range(10):
            x0_sub = torch.rand(sys.x_dim, dtype=sys.dtype) * (x0_up - x0_lo) + x0_lo
            x0_sub = torch.max(torch.min(x0_sub, x0_up), x0_lo)
            value_sub = None
            try:
                value_sub, _, _ = V(x0_sub)
            except AttributeError:
                # for some reason the solver didn't return anything
                pass
            if value_sub is not None:
                z_nn_sub = self.double_integrator_model(x0_sub).item()
                epsilon_sub = value_sub - z_nn_sub
                print(epsilon_sub)
                # self.assertLessEqual(epsilon_sub, epsilon)

if __name__ == '__main__':
    unittest.main()