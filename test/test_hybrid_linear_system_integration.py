from context import utils

import unittest
import numpy as np
import torch
import cvxpy as cp
import hybrid_linear_system
from utils import torch_to_numpy

class HybridLinearSystemIntegrationTest(unittest.TestCase):
    def test_integration(self):
        dtype = torch.float64

        x_dim = 2
        u_dim = 2
        x_min = -torch.ones(x_dim,dtype)
        x_max = torch.ones(x_dim,dtype)
        u_min = -torch.ones(u_dim,dtype)
        u_max = torch.ones(u_dim,dtype)
        
        hls = hybrid_linear_system.HybridLinearSystem(x_dim, u_dim, dtype)
        
        # free falling mode
        A = torch.eye(x_dim,dtype=dtype)
        B = torch.zeros(x_dim,u_dim,dtype=dtype)
        c = torch.zeros(x_dim,dtype=dtype)
        
        P_xu_lo = -torch.eye(x_dim+u_dim, dtype=dtype)
        P_xu_up = torch.eye(x_dim+u_dim, dtype=dtype)
        P_lim = torch.cat((P_xu_lo, P_xu_up),dim=0)
        q_lim = torch.cat((-x_min,-u_min,x_max,u_max),dim=0)
        
        hls.add_mode(A, B, c, P_lim, q_lim)
        
        hls_dyn = hls.mixed_integer_constraints(x_min,x_max,u_min,u_max)
        (Aeq_slack, Aeq_alpha, Ain_x, Ain_u, Ain_slack, Ain_alpha, rhs_in) = torch_to_numpy(hls_dyn)
        
        if len(Aeq_slack.shape) == 1:
            Aeq_slack = Aeq_slack.reshape((-1,1))
        num_slack = Aeq_slack.shape[1]
        if len(Aeq_alpha.shape) == 1:
            Aeq_alpha = Aeq_alpha.reshape((-1,1))
        num_alpha = Aeq_alpha.shape[1]
        
        slack = cp.Variable(num_slack)
        alpha = cp.Variable(num_alpha,boolean=True)
        
        x0 = np.array([.1,.2,.3,.4])
        u0 = np.array([0.,0.,0.,0.])
        
        obj = cp.Minimize(0.)
        con = [Ain_x@x0 + Ain_u@u0 + Ain_slack@slack + Ain_alpha@alpha <= rhs_in]
        prob = cp.Problem(obj,con)
        prob.solve(solver=cp.GUROBI, verbose=True)        
        
        xn = Aeq_slack@slack.value + Aeq_alpha@alpha.value
        print(xn) 

if __name__ == '__main__':
    unittest.main()