import ValueToOptimization
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
    
        Q = torch.ones(3,3,dtype=sys.dtype)*0.1
        q = torch.ones(3,dtype=sys.dtype)*0.1
        R = torch.ones(1,1,dtype=sys.dtype)*2.
        r = torch.ones(1,dtype=sys.dtype)*0.1
        Z = torch.ones(1,1,dtype=sys.dtype)*0.
        z = torch.ones(1,dtype=sys.dtype)*0.1
    
        Qt = torch.ones(3,3,dtype=sys.dtype)*0.1
        qt = torch.ones(3,dtype=sys.dtype)*0.1
        Rt = torch.ones(1,1,dtype=sys.dtype)*2.
        rt = torch.ones(1,dtype=sys.dtype)*0.1
        Zt = torch.ones(1,1,dtype=sys.dtype)*0.
        zt = torch.ones(1,dtype=sys.dtype)*0.1        
    
        x_lo_traj = torch.ones(3,dtype=sys.dtype)*-1000.
        x_up_traj = torch.ones(3,dtype=sys.dtype)*1000.
        u_lo = torch.ones(1,dtype=sys.dtype)*-1000.
        u_up = torch.ones(1,dtype=sys.dtype)*1000.
    
        xN = torch.Tensor([0.,.075,0.]).type(sys.dtype)
    
        N = 20

        mb = ModelBounds.ModelBounds(self.model, sys,
                     Q, R, Z, q, r, z,
                     Qt, Rt, Zt, qt, rt, zt,
                     N, xN,
                     x_lo_traj, x_up_traj, u_lo, u_up)
                     
        x_lo = torch.Tensor([0.,0.,-10.]).type(sys.dtype)
        x_up = torch.Tensor([1.,1.5,10.]).type(sys.dtype)        
        Q1,Q2,q1,q2,k,G1,G2,h,A1,A2,b = mb.upper_bound_opt(self.model, x_lo, x_up)

if __name__ == '__main__':
    unittest.main()