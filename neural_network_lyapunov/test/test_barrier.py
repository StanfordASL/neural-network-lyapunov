import neural_network_lyapunov.barrier as barrier

import gurobipy
import torch
import numpy as np

import unittest
import neural_network_lyapunov.utils as utils
import neural_network_lyapunov.gurobi_torch_mip as gurobi_torch_mip
import neural_network_lyapunov.relu_system as relu_system
import neural_network_lyapunov.mip_utils as mip_utils


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
        dynamics_relu[0].weight.data = torch.tensor(
            [[1, 2, -0.5], [0.2, 0.5, 1]], dtype=self.dtype)
        dynamics_relu[0].bias.data = torch.tensor([0.1, 0.3], dtype=self.dtype)
        dynamics_relu[2].weight.data = torch.tensor(
            [[0.2, -0.4], [0.5, -1.2], [0.3, -0.8]], dtype=self.dtype)
        dynamics_relu[2].bias.data = torch.tensor([0.2, 0.1, -0.5],
                                                  dtype=self.dtype)
        x_lo = torch.tensor([-2, 1, -3], dtype=self.dtype)
        x_up = torch.tensor([3, 4, -1], dtype=self.dtype)
        self.system = relu_system.AutonomousReLUSystemGivenEquilibrium(
            self.dtype,
            x_lo,
            x_up,
            dynamics_relu, (x_lo + x_up) / 2,
            discrete_time_flag=True)

    def test_value(self):
        dut = barrier.Barrier(self.system, self.barrier_relu)
        x = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=self.dtype)
        x_star = torch.tensor([-2, 3, 1], dtype=self.dtype)
        c = 0.5
        h = dut.value(x, x_star, c)
        self.assertEqual(h.shape, (x.shape[0], ))
        for i in range(x.shape[0]):
            h_i = dut.value(x[i], x_star, c)
            self.assertAlmostEqual(h_i.item(), h[i].item())
            self.assertAlmostEqual(h_i.item(),
                                   (self.barrier_relu(x[i]) -
                                    self.barrier_relu(x_star) + c).item())

    def test_value_as_milp(self):
        region = gurobi_torch_mip.MixedIntegerConstraintsReturn()
        region.Ain_input = torch.tensor([[1, 0, 0]], dtype=self.dtype)
        region.rhs_in = torch.tensor([0.5], dtype=self.dtype)

        dut = barrier.Barrier(self.system, self.barrier_relu)
        x_star = torch.tensor([1.5, 2, -2], dtype=self.dtype)
        c = 0.5
        milp_safe, x_safe = dut.value_as_milp(x_star,
                                              c,
                                              region,
                                              safe_flag=True)
        milp_safe.gurobi_model.setParam(gurobipy.GRB.Param.OutputFlag, False)
        milp_safe.gurobi_model.optimize()
        self.assertEqual(milp_safe.gurobi_model.status,
                         gurobipy.GRB.Status.OPTIMAL)
        x_safe_val = torch.tensor([v.x for v in x_safe], dtype=self.dtype)
        self.assertAlmostEqual(milp_safe.gurobi_model.ObjVal,
                               -dut.value(x_safe_val, x_star, c).item())
        x_samples = utils.uniform_sample_in_box(self.system.x_lo,
                                                self.system.x_up, 1000)
        in_region = (region.Ain_input @ x_samples.T <= region.rhs_in).squeeze()
        x_region_samples = x_samples[torch.arange(
            x_samples.shape[0])[in_region]]
        np.testing.assert_array_less(
            -dut.value(x_region_samples, x_star, c).detach().numpy(),
            milp_safe.gurobi_model.ObjVal)

        milp_unsafe, x_unsafe = dut.value_as_milp(x_star,
                                                  c,
                                                  region,
                                                  safe_flag=False)
        milp_unsafe.gurobi_model.setParam(gurobipy.GRB.Param.OutputFlag, False)
        milp_unsafe.gurobi_model.optimize()
        self.assertEqual(milp_unsafe.gurobi_model.status,
                         gurobipy.GRB.Status.OPTIMAL)
        x_unsafe_val = torch.tensor([v.x for v in x_unsafe], dtype=self.dtype)
        self.assertAlmostEqual(milp_unsafe.gurobi_model.ObjVal,
                               dut.value(x_unsafe_val, x_star, c).item())
        np.testing.assert_array_less(
            dut.value(x_region_samples, x_star, c).detach().numpy(),
            milp_unsafe.gurobi_model.ObjVal)


class TestDiscreteTimeBarrier(TestBarrier):
    def test_derivative(self):
        dut = barrier.DiscreteTimeBarrier(self.system, self.barrier_relu)

        x_samples = utils.uniform_sample_in_box(self.system.x_lo,
                                                self.system.x_up, 100)
        x_star = torch.tensor([0.5, 2, -2], dtype=self.dtype)
        c = 0.5
        epsilon = 0.1
        hdot = dut.derivative(x_samples, x_star, c, epsilon)
        self.assertEqual(hdot.shape, (x_samples.shape[0], ))
        for i in range(x_samples.shape[0]):
            hdot_i = dut.derivative(x_samples[i], x_star, c, epsilon)
            self.assertAlmostEqual(hdot[i].item(), hdot_i.item())
            x_next = dut.system.step_forward(x_samples[i])
            self.assertAlmostEqual(
                hdot_i.item(),
                (-dut.barrier_relu(x_next) +
                 (1 - epsilon) * dut.barrier_relu(x_samples[i]) + epsilon *
                 (dut.barrier_relu(x_star) - c)).item())

    def test_derivative_as_milp(self):
        dut = barrier.DiscreteTimeBarrier(self.system, self.barrier_relu)

        x_star = torch.tensor([0.5, 2, -2], dtype=self.dtype)
        c = 0.5
        epsilon = 0.1
        x_samples = utils.uniform_sample_in_box(self.system.x_lo,
                                                self.system.x_up, 1000)
        x_next = dut.system.step_forward(x_samples)
        valid_x_next_mask = torch.all(torch.logical_and(
            x_next <= dut.system.x_up, x_next >= dut.system.x_lo),
                                      dim=1)
        x_valid_samples = x_samples[torch.arange(
            x_samples.shape[0])[valid_x_next_mask]]
        for method in list(mip_utils.PropagateBoundsMethod):
            dut.network_bound_propagate_method = method
            dut.system.network_bound_propagate_method = method
            ret = dut.derivative_as_milp(x_star, c, epsilon)
            ret.milp.gurobi_model.setParam(gurobipy.GRB.Param.OutputFlag,
                                           False)
            ret.milp.gurobi_model.optimize()
            self.assertEqual(ret.milp.gurobi_model.status,
                             gurobipy.GRB.Status.OPTIMAL)
            x_val = torch.tensor([v.x for v in ret.x], dtype=self.dtype)
            self.assertAlmostEqual(
                ret.milp.gurobi_model.ObjVal,
                dut.derivative(x_val, x_star, c, epsilon).item())
            self.assertGreaterEqual(
                ret.milp.gurobi_model.ObjVal,
                torch.max(dut.derivative(x_valid_samples, x_star, c,
                                         epsilon)).item())

    def test_derivative_loss_at_samples_and_next_states(self):
        dut = barrier.DiscreteTimeBarrier(self.system, self.barrier_relu)
        torch.random.manual_seed(0)
        x_samples = utils.uniform_sample_in_box(dut.system.x_lo,
                                                dut.system.x_up, 100)
        x_next = dut.system.step_forward(x_samples)
        x_star = (dut.system.x_lo + dut.system.x_up) / 2
        c = 0.5
        margin = 0.2
        epsilon = 0.1
        loss_all = torch.maximum(
            -dut.value(x_next, x_star, c) +
            (1 - epsilon) * dut.value(x_samples, x_star, c) + margin,
            torch.tensor(0., dtype=self.dtype))
        for reduction in ("mean", "max", "4norm"):
            loss = dut.derivative_loss_at_samples_and_next_states(
                x_star,
                c,
                epsilon,
                x_samples,
                x_next,
                margin=margin,
                reduction=reduction)
            if reduction == "mean":
                loss_expected = torch.mean(loss_all)
            elif reduction == "max":
                loss_expected = torch.max(loss_all)
            elif reduction == "4norm":
                loss_expected = torch.norm(loss_all, p=4)
            self.assertAlmostEqual(loss.item(), loss_expected.item())


if __name__ == "__main__":
    unittest.main()
