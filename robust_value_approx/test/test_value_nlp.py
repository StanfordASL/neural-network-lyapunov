import robust_value_approx.value_nlp as value_nlp
import pendulum_utils

import unittest
import numpy as np
import torch


class ValueNLPTest(unittest.TestCase):
    def test_grad(self):
        N = 5
        vf, sys = pendulum_utils.get_value_function(N)
        x0_lo = -1 * torch.ones(sys.x_dim[0], dtype=vf.dtype)
        x0_up = 1 * torch.ones(sys.x_dim[0], dtype=vf.dtype)
        V_with_grad = vf.get_differentiable_value_function()
        eps = 1e-3
        for k in range(5):
            x0 = torch.rand(vf.x_dim[0], dtype=vf.dtype) *\
                (x0_up - x0_lo) + x0_lo
            x0.requires_grad = True
            c2g, traj = V_with_grad(x0)
            obj_grad = torch.autograd.grad(c2g[0], x0)[0]
            for i in range(len(x0)):
                x0_ = x0.clone()
                x0_[i] += eps
                c2g_, traj_ = V_with_grad(x0_)
                obj_grad_1 = (c2g_[0] - c2g[0]) / eps
                x0_ = x0.clone()
                x0_[i] -= eps
                c2g_, traj_ = V_with_grad(x0_)
                obj_grad_2 = (c2g_[0] - c2g[0]) / -eps
                obj_grad_ = .5 * (obj_grad_1 + obj_grad_2)
                self.assertAlmostEqual(obj_grad_.item(), obj_grad[i].item(),
                                       places=2)
                # print(obj_grad[i].item())
                # print(obj_grad_.item())
                # print("====")


if __name__ == '__main__':
    unittest.main()