import sys
sys.path.append("..")
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


class ModelBoundsTests(unittest.TestCase):
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
        con = [
            A1@y + A2@gamma == b,
            G1@y + G2@gamma <= h,
        ]
        prob = cp.Problem(obj,con)
        prob.solve(solver=cp.GUROBI, verbose=False)
        epsilon = obj.value
    
        def value_fun_wrapper(x0):
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
            return value
    
        x0 = torch.Tensor(y.value[:3]).type(sys.dtype)
        value = value_fun_wrapper(x0)
        z_nn = self.model(x0).item()
    
        self.assertAlmostEqual(epsilon, value - z_nn, places=4)
    
        for i in range(200):
            if i < 50:
                x0_sub = torch.rand(3, dtype=sys.dtype)*(x_up - x_lo) + x_lo
            else:
                x0_sub = x0 + .2*torch.rand(3, dtype=sys.dtype)*(x_up - x_lo)
                x0_sub = torch.max(torch.min(x0_sub,x_up),x_lo)
            value_sub = None
            try:
                value_sub = value_fun_wrapper(x0_sub)
            except:
                pass
            if type(value_sub) != type(None):
                z_nn_sub = self.model(x0_sub).item()
                epsilon_sub = value_sub - z_nn_sub
                self.assertLessEqual(-epsilon_sub, -epsilon)

    def test_lower_bound(self):
        sys = BallPaddleSystem.BallPaddleSystem(dt=.05)
    
        # Q = torch.ones(3,3,dtype=sys.dtype)*0.1
        Q = torch.eye(3,dtype=sys.dtype)*0.1
        q = torch.ones(3,dtype=sys.dtype)*0.1
        # R = torch.ones(1,1,dtype=sys.dtype)*2.
        R = torch.eye(1,dtype=sys.dtype)*2.
        r = torch.ones(1,dtype=sys.dtype)*0.1
        # Z = torch.ones(1,1,dtype=sys.dtype)*0.01
        Z = torch.eye(1,dtype=sys.dtype)*0.01
        z = torch.ones(1,dtype=sys.dtype)*0.1
    
        # Qt = torch.ones(3,3,dtype=sys.dtype)*0.1
        Qt = torch.eye(3,dtype=sys.dtype)*0.1
        qt = torch.ones(3,dtype=sys.dtype)*0.1
        # Rt = torch.ones(1,1,dtype=sys.dtype)*2.
        Rt = torch.eye(1,dtype=sys.dtype)*2.
        rt = torch.ones(1,dtype=sys.dtype)*0.1
        # Zt = torch.ones(1,1,dtype=sys.dtype)*0.01
        Zt = torch.eye(1,dtype=sys.dtype)*0.01
        zt = torch.ones(1,dtype=sys.dtype)*0.1
    
        x_lo_traj = torch.ones(3,dtype=sys.dtype)*-10000.
        x_up_traj = torch.ones(3,dtype=sys.dtype)*10000.
        u_lo = torch.ones(1,dtype=sys.dtype)*-10000.
        u_up = torch.ones(1,dtype=sys.dtype)*10000.
    
        xN = torch.Tensor([0.,.1,0.]).type(sys.dtype)
    
        N = 10
    
        x_lo = torch.Tensor([0.,0.,-10.]).type(sys.dtype)
        x_up = torch.Tensor([1.,1.5,10.]).type(sys.dtype)
        
        x0 = torch.Tensor([0., .01, 0.]).type(sys.dtype)
        activation_pattern = ReLUToOptimization.ComputeReLUActivationPattern(self.model, x0)
        
        mb = ModelBounds.ModelBounds(self.model, sys,
                                     Q, R, Z, q, r, z,
                                     Qt, Rt, Zt, qt, rt, zt,
                                     N, xN,
                                     x_lo_traj, x_up_traj, u_lo, u_up) 

        def value_fun_wrapper(x0):
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
            return value, s.value, alpha.value
            
        _,s_,alpha_ = value_fun_wrapper(x0)
        epsilon = mb.lower_bound(self.model, x_lo, x_up, activation_pattern, s_, np.maximum(0.,alpha_))

        # print("epsilon: %f" % epsilon)
        # 
        # checked_sampled = 0
        # for i in range(100):
        #     x0_sub = x0 + .25*torch.rand(3, dtype=sys.dtype)*(x_up - x_lo)
        #     x0_sub = torch.max(torch.min(x0_sub,x_up),x_lo)
        #     activation_pattern_sub = ReLUToOptimization.ComputeReLUActivationPattern(self.model, x0_sub)
        #     if activation_pattern_sub != activation_pattern:
        #         continue
        #     value_sub = None
        #     try:
        #         value_sub,_,_ = value_fun_wrapper(x0_sub)
        #     except:
        #         pass
        #     if type(value_sub) != type(None):
        #         z_nn_sub = self.model(x0_sub).item()
        #         epsilon_sub = z_nn_sub - value_sub
        #         print("sampled epsilon %f" % epsilon_sub)
        #         self.assertLessEqual(epsilon_sub, epsilon)
        #         checked_sampled += 1
        # 
        # self.assertLessEqual(5, checked_sampled)
    
    
if __name__ == '__main__':
    unittest.main()