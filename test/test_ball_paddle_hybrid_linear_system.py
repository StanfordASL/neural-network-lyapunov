from context import ball_paddle_hybrid_linear_system as ball_paddle
from context import utils

import unittest
import numpy as np
import torch


class BallPaddleHybridLinearSystemTest(unittest.TestCase):
    def setUp(self):
        """
        x = [ballx, bally, paddley, paddletheta, ballvx, ballvy, paddlevy]
        u = [paddletheta_dot, paddlevy_dot]
        """
        dtype = torch.float64
        dt = .01
        x_lo = torch.Tensor([-1.,0.,0.,-np.pi/2,-100.,-100.,-100.]).type(dtype)
        x_up = torch.Tensor([1.,2.,1.,np.pi/2,100.,100.,100.]).type(dtype)
        u_lo = torch.Tensor([-100.,-1000.]).type(dtype)
        u_up = torch.Tensor([100.,1000.]).type(dtype)
        self.sys = ball_paddle.BallPaddleHybridLinearSystem(dtype, dt, x_lo, x_up, u_lo, u_up)

    def test_ball_paddle_dynamics(self):
        hls = self.sys.get_hybrid_linear_system()
        
        print(hls)
        

if __name__ == '__main__':
    unittest.main()