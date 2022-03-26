import neural_network_lyapunov.barrier as barrier

import gurobipy
import torch
import numpy as np

import unittest
import neural_network_lyapunov.utils as utils
import neural_network_lyapunov.gurobi_torch_mip as gurobi_torch_mip
import neural_network_lyapunov.relu_system as relu_system


class TestBarrier(unittest.TestCase):
    def setUp(self):
        self.dtype = torch.float64
        self.barrier_relu = utils.setup_relu((3, 4, 5, 1),
                                             params=None,
                                             negative_slope=0.1,
                                             bias=True,
                                             dtype=self.dtype)
        self.barrier_relu[0].weight.data = torch.tensor(
            [[4, -2, 0.5], [-0.1, 1.5, -0.2], [0.3, 1, 2], [0.5, 1, -2]],
            dtype=self.dtype)
        self.barrier_relu[0].bias.data = torch.tensor([1, -2, 0.5, 1],
                                                      dtype=self.dtype)
        self.barrier_relu[2].weight.data = torch.tensor(
            [[-1, 2, 0.4, -2], [0.5, -1, 3, -0.5], [0.2, -0.1, 1.5, 0],
             [2, -1, 3, 2], [0.5, -1, 1.5, 2.1]],
            dtype=self.dtype)
        self.barrier_relu[2].bias.data = torch.tensor([1, -2, 0.5, -0.1, 1.5],
                                                      dtype=self.dtype)
        self.barrier_relu[4].weight.data = torch.tensor([[1, -2, 3, 1, 0.5]],
                                                        dtype=self.dtype)
        self.barrier_relu[4].bias.data = torch.tensor([0.5], dtype=self.dtype)
        dynamics_relu = utils.setup_relu((3, 2, 3),
                                         params=None,
                                         negative_slope=0.1,
                                         bias=True,
                                         dtype=self.dtype)
        x_lo = torch.tensor([-2, 1, -3], dtype=self.dtype)
        x_up = torch.tensor([3, 4, -1], dtype=self.dtype)
        self.system = relu_system.AutonomousReLUSystem(self.dtype, x_lo, x_up,
                                                       dynamics_relu)

    def test_barrier_value(self):
        dut = barrier.Barrier(self.system, self.barrier_relu)
        x = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=self.dtype)
        x_star = torch.tensor([-2, 3, 1], dtype=self.dtype)
        c = 0.5
        h = dut.barrier_value(x, x_star, c)
        self.assertEqual(h.shape, (x.shape[0], ))
        for i in range(x.shape[0]):
            h_i = dut.barrier_value(x[i], x_star, c)
            self.assertAlmostEqual(h_i.item(), h[i].item())
            self.assertAlmostEqual(h_i.item(),
                                   (self.barrier_relu(x[i]) -
                                    self.barrier_relu(x_star) + c).item())

    def test_barrier_value_as_milp(self):
        region = gurobi_torch_mip.MixedIntegerConstraintsReturn()
        region.Ain_input = torch.tensor([[1, 0, 0]], dtype=self.dtype)
        region.rhs_in = torch.tensor([0.5], dtype=self.dtype)

        dut = barrier.Barrier(self.system, self.barrier_relu)
        x_star = torch.tensor([1.5, 2, -2], dtype=self.dtype)
        c = 0.5
        milp_safe, x_safe = dut.barrier_value_as_milp(x_star,
                                                      c,
                                                      region,
                                                      safe_flag=True)
        milp_safe.gurobi_model.setParam(gurobipy.GRB.Param.OutputFlag, False)
        milp_safe.gurobi_model.optimize()
        self.assertEqual(milp_safe.gurobi_model.status,
                         gurobipy.GRB.Status.OPTIMAL)
        x_safe_val = torch.tensor([v.x for v in x_safe], dtype=self.dtype)
        self.assertAlmostEqual(
            milp_safe.gurobi_model.ObjVal,
            -dut.barrier_value(x_safe_val, x_star, c).item())
        x_samples = utils.uniform_sample_in_box(self.system.x_lo,
                                                self.system.x_up, 1000)
        in_region = (region.Ain_input @ x_samples.T <= region.rhs_in).squeeze()
        x_region_samples = x_samples[torch.arange(
            x_samples.shape[0])[in_region]]
        np.testing.assert_array_less(
            -dut.barrier_value(x_region_samples, x_star, c).detach().numpy(),
            milp_safe.gurobi_model.ObjVal)

        milp_unsafe, x_unsafe = dut.barrier_value_as_milp(x_star,
                                                          c,
                                                          region,
                                                          safe_flag=False)
        milp_unsafe.gurobi_model.setParam(gurobipy.GRB.Param.OutputFlag, False)
        milp_unsafe.gurobi_model.optimize()
        self.assertEqual(milp_unsafe.gurobi_model.status,
                         gurobipy.GRB.Status.OPTIMAL)
        x_unsafe_val = torch.tensor([v.x for v in x_unsafe], dtype=self.dtype)
        self.assertAlmostEqual(
            milp_unsafe.gurobi_model.ObjVal,
            dut.barrier_value(x_unsafe_val, x_star, c).item())
        np.testing.assert_array_less(
            dut.barrier_value(x_region_samples, x_star, c).detach().numpy(),
            milp_unsafe.gurobi_model.ObjVal)


if __name__ == "__main__":
    unittest.main()
