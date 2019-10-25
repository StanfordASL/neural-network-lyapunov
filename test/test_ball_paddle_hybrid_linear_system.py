from context import ball_paddle_hybrid_linear_system as ball_paddle
from context import utils

import unittest
import numpy as np
import torch
import cvxpy as cp
from utils import torch_to_numpy

import hybrid_linear_system

class BallPaddleHybridLinearSystemTest(unittest.TestCase):
    def setUp(self):
        """
        x = [ballx, bally, paddley, paddletheta, ballvx, ballvy, paddlevy]
        u = [paddletheta_dot, paddlevy_dot]
        """
        self.dtype = torch.float64
        self.dt = .01
        self.x_lo = torch.Tensor([-1.,0.1,0.,-np.pi/2,-100.,-100.,-100.]).type(self.dtype)
        self.x_up = torch.Tensor([1.,2.,1.,np.pi/2,100.,100.,100.]).type(self.dtype)
        self.u_lo = torch.Tensor([-100.,-1000.]).type(self.dtype)
        self.u_up = torch.Tensor([100.,1000.]).type(self.dtype)
        self.sys = ball_paddle.BallPaddleHybridLinearSystem(self.dtype, self.dt, self.x_lo, self.x_up, self.u_lo, self.u_up)

    def test_ball_paddle_dynamics(self):
        # hls = self.sys.get_hybrid_linear_system()
        # hls_dyn = hls.mixed_integer_constraints(self.x_lo, self.x_up, self.u_lo, self.u_up)
        # (Aeq_slack, Aeq_alpha, Ain_x, Ain_u, Ain_slack, Ain_alpha, rhs_in) = torch_to_numpy(hls_dyn)
        # 
        # if len(Aeq_slack.shape) == 1:
        #     Aeq_slack = Aeq_slack.reshape((-1,1))
        # num_slack = Aeq_slack.shape[1]
        # if len(Aeq_alpha.shape) == 1:
        #     Aeq_alpha = Aeq_alpha.reshape((-1,1))
        # num_alpha = Aeq_alpha.shape[1]
        # 
        # slack = cp.Variable(num_slack)
        # alpha = cp.Variable(num_alpha, boolean=True)
        # 
        # x0 = np.array([0.,.25,0.,0.,0.,0.,0.])
        # u0 = np.array([0.,0.])
        # 
        # obj = cp.Minimize(0.)
        # # # for i in range(10):
        # con = [Ain_x@x0 + Ain_u@u0 + Ain_slack@slack + Ain_alpha@alpha <= rhs_in]
        # prob = cp.Problem(obj,con)
        # prob.solve(solver=cp.CPLEX, verbose=True)
        # 
        # xn = Aeq_slack@slack.value + Aeq_alpha@alpha.value
        # print(xn)
        # 
        # import pdb; pdb.set_trace()
        dtype = torch.float64
        hls = hybrid_linear_system.HybridLinearSystem(4, 4, dtype)
        
        # free falling mode
        A = torch.eye(4,dtype=dtype)
        B = torch.zeros(4,4,dtype=dtype)
        c = torch.zeros(4,dtype=dtype)
        
        P_xu_lo = -torch.eye(8, dtype=dtype)
        P_xu_up = torch.eye(8, dtype=dtype)
        P_lim = torch.cat((P_xu_lo, P_xu_up),dim=0)
        q_lim = torch.ones(16, dtype=dtype)
        
        hls.add_mode(A, B, c, P_lim, q_lim)
        
        lim = 10.*torch.ones(4,dtype=dtype)
        hls_dyn = hls.mixed_integer_constraints(-lim,lim,-lim,lim)
        (Aeq_slack, Aeq_alpha, Ain_x, Ain_u, Ain_slack, Ain_alpha, rhs_in) = torch_to_numpy(hls_dyn)
        
        if len(Aeq_slack.shape) == 1:
            Aeq_slack = Aeq_slack.reshape((-1,1))
        num_slack = Aeq_slack.shape[1]
        if len(Aeq_alpha.shape) == 1:
            Aeq_alpha = Aeq_alpha.reshape((-1,1))
        num_alpha = Aeq_alpha.shape[1]
        
        slack = cp.Variable(num_slack)
        alpha = cp.Variable(num_alpha,boolean=True)
        
        x0 = torch.Tensor([.5,0.,0.,0.]).type(dtype)
        u0 = torch.Tensor([0.,0.,0.,0.]).type(dtype)
        
        obj = cp.Maximize(0.)
        con = [Ain_x@x0.detach().numpy() + Ain_u@u0.detach().numpy() + Ain_slack@slack + Ain_alpha@alpha <= rhs_in]
        prob = cp.Problem(obj,con)
        prob.solve(solver=cp.GUROBI, verbose=True)        
        
        xn = Aeq_slack@slack.value + Aeq_alpha@alpha.value
        print(xn)        
        
        # import pdb;pdb.set_trace()
        

if __name__ == '__main__':
    unittest.main()