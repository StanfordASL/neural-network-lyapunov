import neural_network_lyapunov.barrier as mut
import neural_network_lyapunov.control_affine_system as control_affine_system
import neural_network_lyapunov.utils as utils
import neural_network_lyapunov.gurobi_torch_mip as gurobi_torch_mip

import torch
import numpy as np
import unittest
import gurobipy


class TestInfNormTerm(unittest.TestCase):
    def test_from_bounding_box(self):
        dtype = torch.float64
        x_lo = torch.tensor([-3, -2, 1], dtype=dtype)
        x_up = torch.tensor([-1, 2, 3], dtype=dtype)
        scale = 0.5
        dut = mut.InfNormTerm.from_bounding_box(x_lo, x_up, scale)
        x_samples = utils.uniform_sample_on_box_boundary(x_lo, x_up, 100)
        np.testing.assert_allclose(
            torch.norm(dut.R @ x_samples.T - dut.p.unsqueeze(1),
                       p=float("inf"),
                       dim=0).detach().numpy(),
            np.ones((x_samples.shape[0], )) * scale)


class TestBarrier(unittest.TestCase):
    def setUp(self):
        self.dtype = torch.float64
        self.linear_system = control_affine_system.LinearSystem(
            torch.tensor([[1, 3], [2, -4]], dtype=self.dtype),
            torch.tensor([[1, 2, 3], [0, 1, -1]], dtype=self.dtype),
            x_lo=torch.tensor([-2, -3], dtype=self.dtype),
            x_up=torch.tensor([3, 1], dtype=self.dtype),
            u_lo=torch.tensor([-1, -3, 2], dtype=self.dtype),
            u_up=torch.tensor([2, -1, 4], dtype=self.dtype))
        self.barrier_relu1 = utils.setup_relu((2, 4, 3, 1),
                                              params=None,
                                              negative_slope=0.01,
                                              bias=True,
                                              dtype=self.dtype)
        self.barrier_relu1[0].weight.data = torch.tensor(
            [[1, -1], [0, 2], [1, 3], [-1, -2]], dtype=self.dtype)
        self.barrier_relu1[0].bias.data = torch.tensor([0, 1, -1, 2],
                                                       dtype=self.dtype)
        self.barrier_relu1[2].weight.data = torch.tensor(
            [[1, 0, -1, 2], [0, 2, -1, 1], [1, 0, 1, -2]], dtype=self.dtype)
        self.barrier_relu1[2].bias.data = torch.tensor([0, 2, 3],
                                                       dtype=self.dtype)
        self.barrier_relu1[4].weight.data = torch.tensor([[1, -3, 2]],
                                                         dtype=self.dtype)
        self.barrier_relu1[4].bias.data = torch.tensor([-1], dtype=self.dtype)

    def test_barrier_value(self):
        dut = mut.Barrier(self.linear_system, self.barrier_relu1)
        x = torch.tensor([2, 3], dtype=self.dtype)
        # Test a single state.
        x_star = torch.tensor([-0.2, 1], dtype=self.dtype)
        c = 0.5
        self.assertEqual(
            dut.barrier_value(x, x_star, c).item(),
            (dut.barrier_relu(x) - dut.barrier_relu(x_star) + c).item())
        # Test a batch of states.
        x = torch.tensor([[-1, 1], [2, -1], [0, 0.5]], dtype=self.dtype)
        val = dut.barrier_value(x, x_star, c)
        self.assertEqual(val.shape, (x.shape[0], 1))
        for i in range(x.shape[0]):
            self.assertEqual(val[i].item(),
                             (dut.barrier_relu(x[i]) -
                              dut.barrier_relu(x_star) + c).item())
        # Test with inf_norm_term
        inf_norm_term = mut.InfNormTerm(R=torch.tensor(
            [[1, 3], [-1, 2], [0, 1]], dtype=self.dtype),
                                        p=torch.tensor([1, 2, -3],
                                                       dtype=self.dtype))
        val = dut.barrier_value(x, x_star, c, inf_norm_term)
        self.assertEqual(val.shape, (x.shape[0], 1))
        for i in range(x.shape[0]):
            self.assertEqual(
                val[i].item(),
                (dut.barrier_relu(x[i]) - dut.barrier_relu(x_star) + c -
                 torch.max(torch.abs(inf_norm_term.R @ x[i] -
                                     inf_norm_term.p))).item())
            self.assertEqual(
                val[i].item(),
                dut.barrier_value(x[i], x_star, c, inf_norm_term).item())

    def barrier_value_as_milp_tester(self, dut, x_star, c, region_cnstr,
                                     inf_norm_term):
        ret = dut.barrier_value_as_milp(x_star,
                                        c,
                                        region_cnstr,
                                        inf_norm_term=inf_norm_term)
        milp = ret.milp
        x = ret.x
        milp.gurobi_model.setParam(gurobipy.GRB.Param.OutputFlag, False)
        milp.gurobi_model.optimize()
        assert (milp.gurobi_model.status == gurobipy.GRB.Status.OPTIMAL)
        # First check if the objective value is same as h(x)
        x_optimal = torch.tensor([v.x for v in x], dtype=self.dtype)
        self.assertAlmostEqual(
            milp.gurobi_model.ObjVal,
            dut.barrier_value(x_optimal, x_star, c, inf_norm_term).item())
        # Now verify that x_optimal is actually in the region. We do
        # this by creating an MILP that only contains region_cnstr.
        milp_region = gurobi_torch_mip.GurobiTorchMIP(self.dtype)
        x_region = milp_region.addVars(
            dut.system.x_dim,
            lb=torch.from_numpy(dut.system.x_lo_all),
            ub=torch.from_numpy(dut.system.x_up_all))
        milp_region.add_mixed_integer_linear_constraints(
            region_cnstr, x_region, None, "region_s", "region_binary", "", "",
            "", gurobipy.GRB.BINARY)
        for j in range(dut.system.x_dim):
            x_region[j].lb = x_optimal[j].item()
            x_region[j].ub = x_optimal[j].item()
        milp_region.gurobi_model.setParam(gurobipy.GRB.Param.OutputFlag, False)
        milp_region.gurobi_model.optimize()
        self.assertEqual(milp_region.gurobi_model.status,
                         gurobipy.GRB.Status.OPTIMAL)
        # Now sample many states. If they are in the region, then their
        # barrier function should be no larger than the MILP optimal cost.
        torch.manual_seed(0)
        x_samples = utils.uniform_sample_in_box(
            torch.from_numpy(dut.system.x_lo_all),
            torch.from_numpy(dut.system.x_up_all), 1000)
        for i in range(x_samples.shape[0]):
            for j in range(dut.system.x_dim):
                x_region[j].lb = x_samples[i][j].item()
                x_region[j].ub = x_samples[i][j].item()
            milp_region.gurobi_model.optimize()
            if milp_region.gurobi_model.status == gurobipy.GRB.Status.OPTIMAL:
                self.assertLessEqual(
                    dut.barrier_value(x_samples[i], x_star, c,
                                      inf_norm_term).item(),
                    milp.gurobi_model.ObjVal)

    def test_barrier_value_as_milp(self):
        dut = mut.Barrier(self.linear_system, self.barrier_relu1)

        # The unsafe region is just x[0] <= 0
        unsafe_region_cnstr1 = gurobi_torch_mip.MixedIntegerConstraintsReturn()
        unsafe_region_cnstr1.Ain_input = torch.tensor([[1, 0]],
                                                      dtype=self.dtype)
        unsafe_region_cnstr1.rhs_in = torch.tensor([0], dtype=self.dtype)

        x_star = torch.tensor([0.2, 0.1], dtype=self.dtype)
        c = 0.5

        self.barrier_value_as_milp_tester(dut,
                                          x_star,
                                          c,
                                          unsafe_region_cnstr1,
                                          inf_norm_term=None)

        # The unsafe region is x[0] <= 0 or x[1] >= 1
        # Formulated as the mixed-integer linear constraint
        # x[0] <= x_up[0] * z
        # x[1] >= 1 - (1-x_lo[1])(1-z)
        unsafe_region_cnstr2 = gurobi_torch_mip.MixedIntegerConstraintsReturn()
        unsafe_region_cnstr2.Ain_input = torch.tensor([[1, 0], [0, -1]],
                                                      dtype=self.dtype)
        unsafe_region_cnstr2.Ain_binary = torch.tensor(
            [[dut.system.x_up_all[0]], [1 - dut.system.x_lo_all[1]]],
            dtype=self.dtype)
        unsafe_region_cnstr2.rhs_in = torch.tensor(
            [0, -dut.system.x_lo_all[1]], dtype=self.dtype)
        self.barrier_value_as_milp_tester(dut,
                                          x_star,
                                          c,
                                          unsafe_region_cnstr2,
                                          inf_norm_term=None)
        self.barrier_value_as_milp_tester(dut,
                                          x_star,
                                          c,
                                          unsafe_region_cnstr2,
                                          inf_norm_term=mut.InfNormTerm(
                                              torch.tensor([[1, 3], [2, -1]],
                                                           dtype=self.dtype),
                                              torch.tensor([1, 3],
                                                           dtype=self.dtype)))

    def test_add_inf_norm_term(self):
        dut = mut.Barrier(self.linear_system, self.barrier_relu1)
        milp = gurobi_torch_mip.GurobiTorchMIP(self.dtype)
        x = milp.addVars(dut.system.x_dim, lb=-gurobipy.GRB.INFINITY)
        inf_norm_term = mut.InfNormTerm(
            torch.tensor([[1, 3], [-2, 4], [3, 1]], dtype=self.dtype),
            torch.tensor([1, -2, 3], dtype=self.dtype))
        inf_norm, inf_norm_binary = dut._add_inf_norm_term(
            milp, x, inf_norm_term)
        self.assertEqual(len(inf_norm), 1)
        self.assertEqual(len(inf_norm_binary), inf_norm_term.R.shape[0] * 2)
        x_samples = utils.uniform_sample_in_box(dut.system.x_lo,
                                                dut.system.x_up, 100)
        milp.gurobi_model.setParam(gurobipy.GRB.Param.OutputFlag, False)
        for i in range(x_samples.shape[0]):
            for j in range(dut.system.x_dim):
                x[j].lb = x_samples[i, j].item()
                x[j].ub = x_samples[i, j].item()
            milp.gurobi_model.optimize()
            self.assertEqual(milp.gurobi_model.status,
                             gurobipy.GRB.Status.OPTIMAL)
            self.assertAlmostEqual(
                inf_norm[0].x,
                torch.norm(inf_norm_term.R @ x_samples[i] - inf_norm_term.p,
                           p=float("inf")).item())
            inf_norm_binary_expected = np.zeros(
                (2 * inf_norm_term.R.shape[0], ))
            inf_norm_binary_expected[torch.argmax(
                torch.cat(
                    (inf_norm_term.R @ x_samples[i] - inf_norm_term.p,
                     -inf_norm_term.R @ x_samples[i] + inf_norm_term.p)))] = 1
            np.testing.assert_allclose(
                np.array([v.x for v in inf_norm_binary]),
                inf_norm_binary_expected)

    def test_barrier_gradient(self):
        dut = mut.Barrier(self.linear_system, self.barrier_relu1)
        inf_norm_term = mut.InfNormTerm(
            torch.tensor([[1, 3], [-2, 4], [3, 1]], dtype=self.dtype),
            torch.tensor([1, -2, 3], dtype=self.dtype))

        torch.manual_seed(0)
        x_samples = utils.uniform_sample_in_box(dut.system.x_lo,
                                                dut.system.x_up, 100)
        for i in range(x_samples.shape[0]):
            dhdx = dut._barrier_gradient(x_samples[i],
                                         inf_norm_term=None,
                                         zero_tol=0.)
            assert (dhdx.shape[0] == 1)
            x_clone = x_samples[i].clone()
            x_clone.requires_grad = True
            dut.barrier_value(x_clone,
                              torch.zeros_like(x_clone, dtype=self.dtype),
                              c=100.,
                              inf_norm_term=None).backward()
            dhdx_expected = x_clone.grad
            np.testing.assert_allclose(dhdx[0].detach().numpy(),
                                       dhdx_expected.detach().numpy())

            dhdx = dut._barrier_gradient(x_samples[i],
                                         inf_norm_term,
                                         zero_tol=0.)
            assert (dhdx.shape[0] == 1)
            x_clone.grad.zero_()
            dut.barrier_value(x_clone,
                              torch.zeros_like(x_clone, dtype=self.dtype),
                              c=100.,
                              inf_norm_term=inf_norm_term).backward()
            dhdx_expected = x_clone.grad
            np.testing.assert_allclose(dhdx[0].detach().numpy(),
                                       dhdx_expected.detach().numpy())

        # Now try x with multiple sub-gradient.
        x = torch.tensor([2, 1], dtype=self.dtype)
        dhdx = dut._barrier_gradient(x, inf_norm_term, zero_tol=0.)
        self.assertEqual(dhdx.shape, (2, dut.system.x_dim))
        x_perturb1 = x + torch.tensor([1E-6, 0], dtype=self.dtype)
        x_perturb2 = x - torch.tensor([1E-6, 0], dtype=self.dtype)
        dhdx1 = dut._barrier_gradient(x_perturb1, inf_norm_term, zero_tol=0.)
        dhdx2 = dut._barrier_gradient(x_perturb2, inf_norm_term, zero_tol=0.)
        self.assertTrue(
            torch.norm(dhdx - torch.cat((dhdx1, dhdx2), dim=0)) < 1E-10
            or torch.norm(dhdx - torch.cat((dhdx2, dhdx1), dim=0)) < 1E-10)

    def barrier_gradient_batch_tester(self, dut, x, inf_norm_term,
                                      create_graph):
        dhdx = dut._barrier_gradient_batch(x,
                                           inf_norm_term,
                                           create_graph=create_graph)
        self.assertEqual(dhdx.shape, (x.shape[0], dut.system.x_dim))
        dhdx_expected = torch.zeros_like(x, dtype=self.dtype)
        for i in range(x.shape[0]):
            dhdx_sample = dut._barrier_gradient(x[i],
                                                inf_norm_term,
                                                zero_tol=0.)
            assert (dhdx_sample.shape[0] == 1)
            dhdx_expected[i] = dhdx_sample
        np.testing.assert_allclose(dhdx.detach().numpy(),
                                   dhdx_expected.detach().numpy())
        return dhdx, dhdx_expected

    def test_barrier_gradient_batch(self):
        dut = mut.Barrier(self.linear_system, self.barrier_relu1)
        inf_norm_term = mut.InfNormTerm(
            torch.tensor([[1, 3], [-2, 4], [3, 1]], dtype=self.dtype),
            torch.tensor([1, -2, 3], dtype=self.dtype))

        torch.manual_seed(0)
        x_samples = utils.uniform_sample_in_box(dut.system.x_lo,
                                                dut.system.x_up, 100)

        self.barrier_gradient_batch_tester(dut,
                                           x_samples,
                                           inf_norm_term=None,
                                           create_graph=False)
        self.barrier_gradient_batch_tester(dut,
                                           x_samples,
                                           inf_norm_term=inf_norm_term,
                                           create_graph=False)

        for v in dut.barrier_relu.parameters():
            v.requires_grad = True
        dhdx, dhdx_expected = self.barrier_gradient_batch_tester(
            dut, x_samples, inf_norm_term=inf_norm_term, create_graph=True)
        torch.sum(dhdx).backward()
        grad = [
            v.grad.clone() for v in dut.barrier_relu.parameters()
            if v.grad is not None
        ]
        dut.barrier_relu.zero_grad()
        torch.sum(dhdx_expected).backward()
        grad_expected = [
            v.grad.clone() for v in dut.barrier_relu.parameters()
            if v.grad is not None
        ]
        for (v1, v2) in zip(grad, grad_expected):
            np.testing.assert_allclose(v1.detach().numpy(),
                                       v2.detach().numpy())
        dut.barrier_relu.zero_grad()


if __name__ == "__main__":
    unittest.main()
