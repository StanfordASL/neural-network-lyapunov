import ValueToOptimization
import BallPaddleSystem

import unittest
import numpy as np
import cvxpy as cp
import torch
import matplotlib.pyplot as plt


class ValueToOptimizationTest(unittest.TestCase):
    def test_trajopt_x0xN(self):
        sys = BallPaddleSystem.BallPaddleSystem(dt=.01)
        vf = ValueToOptimization.ValueFunction(sys)

        Q = torch.ones(3,3,dtype=sys.dtype)*0.
        q = torch.ones(3,dtype=sys.dtype)*0.
        R = torch.ones(1,1,dtype=sys.dtype)*2.
        r = torch.ones(1,dtype=sys.dtype)*0.
        Z = torch.ones(1,1,dtype=sys.dtype)*0.
        z = torch.ones(1,dtype=sys.dtype)*0.
        
        Qt = torch.ones(3,3,dtype=sys.dtype)*0.
        qt = torch.ones(3,dtype=sys.dtype)*0.
        Rt = torch.ones(1,1,dtype=sys.dtype)*2.
        rt = torch.ones(1,dtype=sys.dtype)*0.
        Zt = torch.ones(1,1,dtype=sys.dtype)*0.
        zt = torch.ones(1,dtype=sys.dtype)*0.        
        
        x_lo = torch.ones(3,dtype=sys.dtype)*-1000.
        x_up = torch.ones(3,dtype=sys.dtype)*1000.
        u_lo = torch.ones(1,dtype=sys.dtype)*-1000.
        u_up = torch.ones(1,dtype=sys.dtype)*1000.
        
        x0 = torch.Tensor([0.,.1,0.]).type(sys.dtype)
        xN = torch.Tensor([0.,.075,0.]).type(sys.dtype)
        
        N = 20
        
        (Q2, Q3, q2, q3,
         Ain1, Ain2, Ain3, brhs_in, 
         Aeq1, Aeq2, Aeq3, brhs_eq) = vf.traj_opt_x0xN(
            Q, R, Z, q, r, z, 
            Qt, Rt, Zt, qt, rt, zt, 
            N, x0, xN,
            x_lo, x_up, u_lo, u_up)

        Q2 = Q2.detach().numpy()
        q2 = q2.detach().numpy()
        Q3 = Q3.detach().numpy()
        q3 = q3.detach().numpy()
        Ain1 = Ain1.detach().numpy()
        Ain2 = Ain2.detach().numpy()
        Ain3 = Ain3.detach().numpy()
        brhs_in = brhs_in.detach().numpy()    
        Aeq1 = Aeq1.detach().numpy()
        Aeq2 = Aeq2.detach().numpy()
        Aeq3 = Aeq3.detach().numpy()
        brhs_eq = brhs_eq.detach().numpy()
                                      
        x = cp.Variable(Ain1.shape[1])
        s = cp.Variable(Ain2.shape[1])
        z = cp.Variable(Ain3.shape[1],boolean=True)
        
        con = [
            Ain1@x + Ain2@s + Ain3@z <= brhs_in.squeeze(),
            Aeq1@x + Aeq2@s + Aeq3@z == brhs_eq.squeeze()
        ]
        
        obj = cp.Minimize(.5*cp.quad_form(s,Q2) + q2*s + .5*cp.quad_form(z,Q3) + q3*z)
        
        prob = cp.Problem(obj,con)
        
        prob.solve(solver=cp.GUROBI, verbose=True)
        
        traj = np.hstack((x.value,s.value))
        traj = np.reshape(traj,(-1,4)).T
        
        xtraj = traj[:3,:]
        utraj = traj[3:,:]
        
        # plt.plot(xtraj[1,:])
        # plt.plot(xtraj[0,:])
        # plt.legend(['ball','paddle'])
        # plt.show()

        for i in range(3):
            self.assertAlmostEqual(xtraj[i,0],x0[i])
            self.assertAlmostEqual(xtraj[i,-1],xN[i])

if __name__ == '__main__':
    unittest.main()