import ValueToOptimization
import ReLUToOptimization
import BallPaddleSystem
import ModelBounds

import unittest
import numpy as np
import cvxpy as cp
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


class ModelBoundsUpperBound(unittest.TestCase):
    def setUp(self):
        self.dtype = torch.float64
        self.linear1 = nn.Linear(3, 3)
        self.linear1.weight.data = torch.tensor(
            [[1, 2, 2.5], [3, 4, 4.5], [5, 6, 6.5]], dtype=self.dtype)
        self.linear1.bias.data = torch.tensor(
            [-11, 13, 4], dtype=self.dtype)
        self.linear2 = nn.Linear(3, 4)
        self.linear2.weight.data = torch.tensor(
            [[-1, 0.5, 1.5], [2, 5, 6], [-2, -3, -4], [1, 4, 6]],
            dtype=self.dtype)
        self.linear2.bias.data = torch.tensor(
            [4, -1, -2, 3], dtype=self.dtype)
        self.linear3 = nn.Linear(4, 1)
        self.linear3.weight.data = torch.tensor(
            [[4, 5, 6, 7]], dtype=self.dtype)
        self.linear3.bias.data = torch.tensor([-10], dtype=self.dtype)
        self.model = nn.Sequential(self.linear1, nn.ReLU(), self.linear2,
                                   nn.ReLU(),
                                   self.linear3)
                                   
    def test_upper_bound(self):
        sys = BallPaddleSystem.BallPaddleSystem(dt=.01)
    
        Q = torch.ones(3,3,dtype=sys.dtype)*0.1*0.
        q = torch.ones(3,dtype=sys.dtype)*0.1*0.
        R = torch.ones(1,1,dtype=sys.dtype)*2.*0.
        r = torch.ones(1,dtype=sys.dtype)*0.1*0.
        Z = torch.ones(1,1,dtype=sys.dtype)*0.*0.
        z = torch.ones(1,dtype=sys.dtype)*0.1*0.
    
        Qt = torch.ones(3,3,dtype=sys.dtype)*0.1*0.
        qt = torch.ones(3,dtype=sys.dtype)*0.1*0.
        Rt = torch.ones(1,1,dtype=sys.dtype)*2.*0.
        rt = torch.ones(1,dtype=sys.dtype)*0.1*0.
        Zt = torch.ones(1,1,dtype=sys.dtype)*0.*0.
        zt = torch.ones(1,dtype=sys.dtype)*0.1*0.
    
        x_lo_traj = torch.ones(3,dtype=sys.dtype)*-100.
        x_up_traj = torch.ones(3,dtype=sys.dtype)*100.
        u_lo = torch.ones(1,dtype=sys.dtype)*-100.
        u_up = torch.ones(1,dtype=sys.dtype)*100.
    
        xN = torch.Tensor([0.,.075,0.]).type(sys.dtype)
    
        N = 10
        
        x_lo = torch.Tensor([0.,0.,-10.]).type(sys.dtype)
        x_up = torch.Tensor([1.,1.5,10.]).type(sys.dtype)  
    
        mb = ModelBounds.ModelBounds(self.model, sys,
                                     Q, R, Z, q, r, z,
                                     Qt, Rt, Zt, qt, rt, zt,
                                     N, xN,
                                     x_lo_traj, x_up_traj, u_lo, u_up)
        Q1, Q2, q1, q2, k, G1, G2, h, A1, A2, b = mb.upper_bound_opt(self.model, x_lo, x_up)
        Q1 = Q1.detach().numpy()
        Q2 = Q2.detach().numpy()
        q1 = q1.detach().numpy()
        q2 = q2.detach().numpy()
        G1 = G1.detach().numpy()
        G2 = G2.detach().numpy()
        h = h.detach().numpy()
        A1 = A1.detach().numpy()
        A2 = A2.detach().numpy()
        b = b.detach().numpy()
        num_y = Q1.shape[0]
        num_gamma = Q2.shape[0]
        y = cp.Variable(num_y)
        gamma = cp.Variable(num_gamma, boolean=True)
        obj = cp.Minimize(.5*cp.quad_form(y,Q1) + .5*cp.quad_form(gamma,Q2) + q1@y + q2@gamma + k)
        # obj = cp.Minimize(0.)
        con = [
            A1@y + A2@gamma == b,
            G1@y + G2@gamma <= h,
        ]
        prob = cp.Problem(obj,con)
        prob.solve(solver=cp.GUROBI, verbose=False)
        epsilon = obj.value
        # epsilon = q1[N*4:]@y.value[N*4:] + q2[N:]@gamma.value[N:] + k
        print(epsilon)
        
        x0 = torch.Tensor(y.value[:3]).type(sys.dtype)
    
        vf = ValueToOptimization.ValueFunction(sys)
        (Q2_traj, Q3_traj, q2_traj, q3_traj,
        Ain1_traj, Ain2_traj, Ain3_traj, brhs_in_traj, 
        Aeq1_traj, Aeq2_traj, Aeq3_traj, brhs_eq_traj) = vf.traj_opt_x0xN(
            Q, R, Z, q, r, z, 
            Qt, Rt, Zt, qt, rt, zt, 
            N, xN,
            x_lo_traj, x_up_traj, u_lo, u_up,
            x0=x0)        
        Q2_traj = Q2_traj.detach().numpy()
        q2_traj = q2_traj.detach().numpy()
        Q3_traj = Q3_traj.detach().numpy()
        q3_traj = q3_traj.detach().numpy()
        Ain1_traj = Ain1_traj.detach().numpy()
        Ain2_traj = Ain2_traj.detach().numpy()
        Ain3_traj = Ain3_traj.detach().numpy()
        brhs_in_traj = brhs_in_traj.detach().numpy()    
        Aeq1_traj = Aeq1_traj.detach().numpy()
        Aeq2_traj = Aeq2_traj.detach().numpy()
        Aeq3_traj = Aeq3_traj.detach().numpy()
        brhs_eq_traj = brhs_eq_traj.detach().numpy()
        x = cp.Variable(Ain1_traj.shape[1])
        s = cp.Variable(Ain2_traj.shape[1])
        alpha = cp.Variable(Ain3_traj.shape[1],boolean=True)
        obj = cp.Minimize(.5*cp.quad_form(s,Q2_traj) + q2_traj*s + .5*cp.quad_form(alpha,Q3_traj) + q3_traj*alpha)
        con = [
            Ain1_traj@x + Ain2_traj@s + Ain3_traj@alpha <= brhs_in_traj.squeeze(),
            Aeq1_traj@x + Aeq2_traj@s + Aeq3_traj@alpha == brhs_eq_traj.squeeze()
        ]
        prob = cp.Problem(obj,con)
        prob.solve(solver=cp.GUROBI, verbose=False)
        value = obj.value
        print(value)
        
        relu_opt = ReLUToOptimization.ReLUFreePattern(self.model, sys.dtype)
        Ain1_nn, Ain2_nn, Ain3_nn, rhs_in_nn, Aeq1_nn, Aeq2_nn, Aeq3_nn, rhs_eq_nn, a_out_nn, b_out_nn, z_lo_nn, z_up_nn = relu_opt.output_constraint(self.model, x_lo, x_up)
        Ain1_nn = Ain1_nn.detach().numpy()
        Ain2_nn = Ain2_nn.detach().numpy()
        Ain3_nn = Ain3_nn.detach().numpy()
        rhs_in_nn = rhs_in_nn.squeeze().detach().numpy()
        Aeq1_nn = Aeq1_nn.detach().numpy()
        Aeq2_nn = Aeq2_nn.detach().numpy()
        Aeq3_nn = Aeq3_nn.detach().numpy()
        rhs_eq_nn = rhs_eq_nn.squeeze().detach().numpy()
        a_out_nn = a_out_nn.squeeze().detach().numpy()
        z = cp.Variable(Ain2_nn.shape[1])
        beta = cp.Variable(Ain3_nn.shape[1], boolean=True)
        # obj = cp.Minimize(0.)
        obj = cp.Minimize(-a_out_nn@z - b_out_nn)
        con = [
            Ain1_nn@x0.detach().numpy() + Ain2_nn@z + Ain3_nn@beta <= rhs_in_nn,
            Aeq1_nn@x0.detach().numpy() + Aeq2_nn@z + Aeq3_nn@beta == rhs_eq_nn,
        ]
        prob = cp.Problem(obj, con)
        prob.solve(solver=cp.GUROBI, verbose=False)
        z_opt_nn = a_out_nn@z.value + b_out_nn
        print(z_opt_nn)

        z_nn = self.model(x0).item()
        print(z_nn)
        
        # print(value - z_opt_nn)
        # print(value - z_nn)
        # print(epsilon)
        

if __name__ == '__main__':
    unittest.main()