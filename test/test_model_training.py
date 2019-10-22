from context import value_to_optimization
from context import ball_paddle_system

import unittest
import torch
import torch.nn as nn
from utils import train_model


class ModelTrainingTest(unittest.TestCase):
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

    def test_model_training(self):
        sys = ball_paddle_system.BallPaddleSystem(dt=.01)

        vf = value_to_optimization.ValueFunction(sys, 10)
        Q = torch.ones(3, 3)*0.1
        q = torch.ones(3)*0.1
        R = torch.ones(1, 1)*2.
        r = torch.ones(1)*0.1
        vf.set_cost(Q=Q, R=R, q=q, r=r)
        vf.set_terminal_cost(Qt=Q, Rt=R, qt=q, rt=r)
        xN = torch.Tensor([0., .075, 0.]).type(sys.dtype)
        vf.set_constraints(xN=xN)

        x_lo = torch.Tensor([0., 0., -10.]).type(sys.dtype)
        x_up = torch.Tensor([.5, 1., 10.]).type(sys.dtype)
        dims = [3, 3, 3]
        x_samples, v_samples = vf.get_sample_grid(x_lo, x_up, dims)

        # check loss pre training
        res = self.model(x_samples) - v_samples
        loss_before = (res.t() @ res).item()

        train_model(self.model, x_samples, v_samples,
                    num_epoch=1000, batch_size=10)

        res = self.model(x_samples) - v_samples
        loss_after = (res.t() @ res).item()

        self.assertLess(loss_after, loss_before)


if __name__ == '__main__':
    unittest.main()
