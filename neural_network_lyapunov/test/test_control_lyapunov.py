import neural_network_lyapunov.control_lyapunov as mut

import unittest
import torch
import numpy as np
import gurobipy

import neural_network_lyapunov.control_affine_system as control_affine_system
import neural_network_lyapunov.utils as utils
import neural_network_lyapunov.mip_utils as mip_utils
import neural_network_lyapunov.gurobi_torch_mip as gurobi_torch_mip
import neural_network_lyapunov.lyapunov as lyapunov


class TestControlLyapunov(unittest.TestCase):
    def setUp(self):
        self.dtype = torch.float64

        # Create a linear system.
        A = torch.tensor([[1, 2], [-2, 3]], dtype=self.dtype)
        B = torch.tensor([[1, 3, 1], [0, 1, 0]], dtype=self.dtype)
        self.linear_system = control_affine_system.LinearSystem(
            A,
            B,
            x_lo=torch.tensor([-2, -3], dtype=self.dtype),
            x_up=torch.tensor([3, 3], dtype=self.dtype),
            u_lo=torch.tensor([-1, -2, -3], dtype=self.dtype),
            u_up=torch.tensor([2, 3, 4], dtype=self.dtype))
        self.lyapunov_relu1 = utils.setup_relu((2, 4, 3, 1),
                                               params=None,
                                               negative_slope=0.1,
                                               bias=True,
                                               dtype=self.dtype)
        self.lyapunov_relu1[0].weight.data = torch.tensor(
            [[1, -1], [0, 2], [-1, 2], [-2, 1]], dtype=self.dtype)
        self.lyapunov_relu1[0].bias.data = torch.tensor([1, -1, 0, 2],
                                                        dtype=self.dtype)
        self.lyapunov_relu1[2].weight.data = torch.tensor(
            [[3, -2, 1, 0], [1, -1, 2, 3], [-2, -1, 0, 3]], dtype=self.dtype)
        self.lyapunov_relu1[2].bias.data = torch.tensor([1, -2, 3],
                                                        dtype=self.dtype)
        self.lyapunov_relu1[4].weight.data = torch.tensor([[1, 3, -2]],
                                                          dtype=self.dtype)
        self.lyapunov_relu1[4].bias.data = torch.tensor([2], dtype=self.dtype)

        phi_a = utils.setup_relu((2, 3, 1),
                                 params=None,
                                 negative_slope=0.01,
                                 bias=True,
                                 dtype=self.dtype)
        phi_a[0].weight.data = torch.tensor([[1, 1], [3, -1], [-2, -1]],
                                            dtype=self.dtype)
        phi_a[0].bias.data = torch.tensor([0, 2, -1], dtype=self.dtype)
        phi_a[2].weight.data = torch.tensor([[2, -1, 1]], dtype=self.dtype)
        phi_a[2].bias.data = torch.tensor([3], dtype=self.dtype)
        phi_b = utils.setup_relu((2, 2, 3),
                                 params=None,
                                 negative_slope=0.01,
                                 bias=True,
                                 dtype=self.dtype)
        phi_b[0].weight.data = torch.tensor([[-1, 1], [3, 1]],
                                            dtype=self.dtype)
        phi_b[0].bias.data = torch.tensor([1, 3], dtype=self.dtype)
        phi_b[2].weight.data = torch.tensor([[1, -1], [2, 3], [-1, -2]],
                                            dtype=self.dtype)
        phi_b[2].bias.data = torch.tensor([0, 1, -3], dtype=self.dtype)

        self.second_order_system = \
            control_affine_system.ReluSecondOrderControlAffineSystem(
                x_lo=torch.tensor([-2, -3], dtype=self.dtype),
                x_up=torch.tensor([1, -1], dtype=self.dtype),
                u_lo=torch.tensor([-1, -3, 1], dtype=self.dtype),
                u_up=torch.tensor([1, -1, 2], dtype=self.dtype),
                phi_a=phi_a,
                phi_b=phi_b,
                method=mip_utils.PropagateBoundsMethod.IA)
        self.lyapunov_relu2 = utils.setup_relu((2, 2, 1),
                                               params=None,
                                               negative_slope=0.1,
                                               bias=True,
                                               dtype=self.dtype)
        self.lyapunov_relu2[0].weight.data = torch.tensor([[1, 3], [3, 4]],
                                                          dtype=self.dtype)
        self.lyapunov_relu2[0].bias.data = torch.tensor([1, 2],
                                                        dtype=self.dtype)
        self.lyapunov_relu2[2].weight.data = torch.tensor([[1, 3]],
                                                          dtype=self.dtype)
        self.lyapunov_relu2[2].bias.data = torch.tensor([1], dtype=self.dtype)

        self.lyapunov_relu3 = utils.setup_relu((2, 2, 1),
                                               params=None,
                                               negative_slope=0.1,
                                               bias=True,
                                               dtype=self.dtype)
        self.lyapunov_relu3[0].weight.data = torch.tensor([[-1, -3], [-3, -4]],
                                                          dtype=self.dtype)
        self.lyapunov_relu3[0].bias.data = torch.tensor([1, 2],
                                                        dtype=self.dtype)
        self.lyapunov_relu3[2].weight.data = torch.tensor([[1, 3]],
                                                          dtype=self.dtype)
        self.lyapunov_relu3[2].bias.data = torch.tensor([1], dtype=self.dtype)

    def lyapunov_derivative_tester(self,
                                   dut,
                                   x,
                                   x_equilibrium,
                                   V_lambda,
                                   epsilon,
                                   R,
                                   subgradient_rule,
                                   zero_tol=0.):
        vdot = dut.lyapunov_derivative(x,
                                       x_equilibrium,
                                       V_lambda,
                                       epsilon,
                                       R=R,
                                       subgradient_rule=subgradient_rule,
                                       zero_tol=zero_tol)

        dphi_dx = utils.relu_network_gradient(dut.lyapunov_relu,
                                              x,
                                              zero_tol=zero_tol).squeeze(1)
        dl1_dx = V_lambda * utils.l1_gradient(
            R @ (x - x_equilibrium),
            zero_tol=zero_tol,
            subgradient_samples=dut.l1_subgradient_policy.subgradient_samples
        ) @ R
        v = dut.lyapunov_value(x, x_equilibrium, V_lambda, R=R)
        if subgradient_rule == "max_sample":
            vdot_expected = -np.inf
        elif subgradient_rule == "min_sample":
            vdot_expected = np.inf
        elif subgradient_rule == "all":
            subgradient_rule = []
        for i in range(dphi_dx.shape[0]):
            for j in range(dl1_dx.shape[0]):
                dV_dx = dphi_dx[i] + dl1_dx[j]
                vdot_expected_ij = dV_dx @ dut.system.f(x) + epsilon * v
                dV_dx_times_G = dV_dx @ dut.system.G(x)
                for k in range(dut.system.u_dim):
                    vdot_expected_ij += torch.minimum(
                        dV_dx_times_G[k] * dut.system.u_lo[k],
                        dV_dx_times_G[k] * dut.system.u_up[k])
                if subgradient_rule == "max_sample" and vdot_expected_ij > \
                        vdot_expected:
                    vdot_expected = vdot_expected_ij
                elif subgradient_rule == "min_sample" and vdot_expected_ij < \
                        vdot_expected:
                    vdot_expected = vdot_expected_ij
                elif subgradient_rule == "all":
                    vdot_expected.append(vdot_expected_ij)
        if subgradient_rule in ("min_sample", "max_sample"):
            self.assertAlmostEqual(vdot[0].item(), vdot_expected.item())
        elif subgradient_rule == "all":
            # We don't care the order in vdot.
            self.assertEqual(vdot.shape[0] == len(vdot_expected))
            for i in range(vdot.shape[0]):
                found_match = False
                for j in len(vdot_expected):
                    if np.abs(vdot[i].item() - vdot_expected[j]) < 1E-10:
                        found_match = True
                        break
                self.assertTrue(found_match)

    def test_lyapunov_derivative1(self):
        # Test with linear system.
        dut = mut.ControlLyapunov(self.linear_system, self.lyapunov_relu1,
                                  mut.SubgradientPolicy(None))
        x_equilibrium = torch.tensor([0, -1], dtype=self.dtype)
        V_lambda = 0.5
        R = torch.tensor([[1, 0], [0, 2], [1, 3]], dtype=self.dtype)
        epsilon = 0.1

        for subgradient_rule in ("min_sample", "max_sample", "all"):
            x = torch.tensor([0.5, 1.5], dtype=self.dtype)
            self.lyapunov_derivative_tester(dut, x, x_equilibrium, V_lambda,
                                            epsilon, R, subgradient_rule)
            # Some ReLU inputs equal to 0.
            x = torch.tensor([1, 2], dtype=self.dtype)
            self.lyapunov_derivative_tester(dut, x, x_equilibrium, V_lambda,
                                            epsilon, R, subgradient_rule)
            # Some l1-norm entry equals to 0.
            x = torch.tensor([0, 1], dtype=self.dtype)
            self.lyapunov_derivative_tester(dut, x, x_equilibrium, V_lambda,
                                            epsilon, R, subgradient_rule)

            zero_tol = 1E-10
            # Test with a non-zero zero_tol.
            # Some relu input is almost zero.
            x = torch.tensor([0.5 + 0.01 * zero_tol, 2 - 0.01 * zero_tol],
                             dtype=self.dtype)
            self.lyapunov_derivative_tester(dut, x, x_equilibrium, V_lambda,
                                            epsilon, R, subgradient_rule,
                                            zero_tol)
            # Some l1-norm entry is almost zero.
            x = torch.tensor([0.01 * zero_tol, 1], dtype=self.dtype)
            self.lyapunov_derivative_tester(dut, x, x_equilibrium, V_lambda,
                                            epsilon, R, subgradient_rule,
                                            zero_tol)

    def test_lyapunov_derivative2(self):
        # Test with subgradient_samples
        dut = mut.ControlLyapunov(self.linear_system,
                                  self.lyapunov_relu1,
                                  l1_subgradient_policy=mut.SubgradientPolicy(
                                      np.array([-0.5, 0, 0.5])))
        x_equilibrium = torch.tensor([0, -1], dtype=self.dtype)
        V_lambda = 0.5
        R = torch.tensor([[1, 0], [0, 2], [1, 3]], dtype=self.dtype)
        epsilon = 0.1

        x = torch.tensor([0.5, 1], dtype=self.dtype)
        self.lyapunov_derivative_tester(dut,
                                        x,
                                        x_equilibrium,
                                        V_lambda,
                                        epsilon,
                                        R,
                                        subgradient_rule="max_sample",
                                        zero_tol=0.)

        # Some l1 norm entry equals to 0.
        x = torch.tensor([0, 1], dtype=self.dtype)
        self.lyapunov_derivative_tester(dut,
                                        x,
                                        x_equilibrium,
                                        V_lambda,
                                        epsilon,
                                        R,
                                        subgradient_rule="max_sample",
                                        zero_tol=0.)
        x = x_equilibrium
        self.lyapunov_derivative_tester(dut,
                                        x,
                                        x_equilibrium,
                                        V_lambda,
                                        epsilon,
                                        R,
                                        subgradient_rule="max_sample",
                                        zero_tol=0.)

    def add_system_constraint_tester(self, dut, x_val: torch.Tensor,
                                     is_feasible):
        milp = gurobi_torch_mip.GurobiTorchMIP(self.dtype)
        x = milp.addVars(dut.system.x_dim, lb=x_val, ub=x_val)
        f = milp.addVars(dut.system.x_dim, lb=-gurobipy.GRB.INFINITY)
        Gt = [None] * dut.system.u_dim
        for i in range(dut.system.u_dim):
            Gt[i] = milp.addVars(dut.system.x_dim, lb=-gurobipy.GRB.INFINITY)
        ret = dut.add_system_constraint(milp, x, f, Gt)
        # Linear system doesn't introduce slack/binary variables.
        self.assertEqual(len(ret.slack), 0)
        self.assertEqual(len(ret.binary), 0)

        milp.gurobi_model.setParam(gurobipy.GRB.Param.OutputFlag, False)
        milp.gurobi_model.optimize()
        if is_feasible:
            self.assertEqual(milp.gurobi_model.status,
                             gurobipy.GRB.Status.OPTIMAL)
            f_val = np.array([v.x for v in f])
            Gt_val = [None] * dut.system.u_dim
            for i in range(dut.system.u_dim):
                Gt_val[i] = [v.x for v in Gt[i]]
            G_val = np.array(Gt_val).T

            f_expected = dut.system.f(x_val)
            G_expected = dut.system.G(x_val)
            np.testing.assert_allclose(f_val, f_expected.detach().numpy())
            np.testing.assert_allclose(G_val, G_expected.detach().numpy())
        else:
            self.assertEqual(milp.gurobi_model.status,
                             gurobipy.GRB.Status.INFEASIBLE)

    def test_add_system_constraint1(self):
        # Test with linear system.
        dut = mut.ControlLyapunov(self.linear_system, self.lyapunov_relu1,
                                  mut.SubgradientPolicy(None))
        # x within [x_lo, x_up]
        self.add_system_constraint_tester(
            dut, (self.linear_system.x_lo + self.linear_system.x_up) / 2, True)
        self.add_system_constraint_tester(
            dut,
            (0.9 * self.linear_system.x_lo + 0.1 * self.linear_system.x_up),
            True)
        # x outside of [x_lo, x_up]
        self.add_system_constraint_tester(
            dut, torch.tensor([-3, 0], dtype=self.dtype), False)
        self.add_system_constraint_tester(
            dut, torch.tensor([0, 4], dtype=self.dtype), False)

    def test_add_dl1dx_times_f(self):
        dut = mut.ControlLyapunov(self.linear_system, self.lyapunov_relu1,
                                  mut.SubgradientPolicy(None))
        x_equilibrium = torch.tensor([0.1, 0.2], dtype=dut.system.dtype)
        R = torch.tensor([[1, 2], [-1, 1], [0, 4]], dtype=dut.system.dtype)
        milp = gurobi_torch_mip.GurobiTorchMILP(dut.system.dtype)
        x = milp.addVars(dut.system.x_dim, lb=-gurobipy.GRB.INFINITY)
        l1_slack, l1_binary = dut.add_state_error_l1_constraint(milp,
                                                                x_equilibrium,
                                                                x,
                                                                R=R)
        f = milp.addVars(dut.system.x_dim, lb=-gurobipy.GRB.INFINITY)
        f_lo = torch.tensor([-2, 3], dtype=self.dtype)
        f_up = torch.tensor([1., 5], dtype=self.dtype)
        Rf_lo, Rf_up = mip_utils.compute_range_by_IA(
            R, torch.zeros(R.shape[0], dtype=self.dtype), f_lo, f_up)
        V_lambda = 0.5
        l1_subgradient_binary = None
        Vdot_coeff = []
        Vdot_vars = []
        dl1dx_times_f = dut._add_dl1dx_times_f(milp, x, l1_binary,
                                               l1_subgradient_binary, f, R,
                                               Rf_lo, Rf_up, V_lambda,
                                               Vdot_coeff, Vdot_vars)
        self.assertEqual(len(Vdot_coeff), len(Vdot_vars))

        torch.manual_seed(0)
        # Fix x and f to some values, compute Vdot_coeff * Vdot_vars.
        x_samples = utils.uniform_sample_in_box(dut.system.x_lo,
                                                dut.system.x_up, 100)
        f_samples = utils.uniform_sample_in_box(f_lo, f_up, x_samples.shape[0])
        milp.gurobi_model.setParam(gurobipy.GRB.Param.OutputFlag, False)
        for i in range(x_samples.shape[0]):
            for j in range(dut.system.x_dim):
                x[j].lb = x_samples[i][j]
                x[j].ub = x_samples[i][j]
                f[j].lb = f_samples[i][j]
                f[j].ub = f_samples[i][j]
            milp.gurobi_model.optimize()
            self.assertEqual(milp.gurobi_model.status,
                             gurobipy.GRB.Status.OPTIMAL)
            l1_binary_sol = torch.tensor([v.x for v in l1_binary],
                                         dtype=self.dtype)
            # Check l1_binary_sol is correct.
            l1_terms = R @ (x_samples[i] - x_equilibrium)
            for j in range(R.shape[0]):
                if torch.abs(l1_binary_sol[j] - 1) < 1E-5:
                    self.assertGreaterEqual(l1_terms[j].item(), -1E-6)
                elif torch.abs(l1_binary_sol[j] + 1) < 1E-5:
                    self.assertLessEqual(l1_terms[j].item(), 1E-6)
            # Check dl1dx_times_f is correct.
            dl1dx = V_lambda * (
                utils.l1_gradient(R @ (x_samples[i] - x_equilibrium)) @ R)
            assert (dl1dx.shape[0] == 1)
            dl1dx_times_f_expected = (dl1dx[0] @ f_samples[i]).squeeze()
            self.assertAlmostEqual(dl1dx_times_f.x,
                                   dl1dx_times_f_expected.item())
            Vdot_sol = torch.sum(
                torch.stack([
                    Vdot_coeff[i] @ torch.tensor([v.x for v in Vdot_vars[i]],
                                                 dtype=self.dtype)
                    for i in range(len(Vdot_coeff))
                ]))

            x_samples_i = x_samples[i].clone()
            x_samples_i.requires_grad = True
            (V_lambda *
             torch.norm(R @ (x_samples_i - x_equilibrium), p=1)).backward()
            Vdot_expected = x_samples_i.grad @ f_samples[i]
            np.testing.assert_allclose(Vdot_sol.item(), Vdot_expected.item())

    def add_dVdx_times_G_tester(self, dut, x_equilibrium, V_lambda, R):
        milp = gurobi_torch_mip.GurobiTorchMILP(dut.system.dtype)
        x = milp.addVars(dut.system.x_dim, lb=-gurobipy.GRB.INFINITY)
        l1_slack, l1_binary = dut.add_state_error_l1_constraint(milp,
                                                                x_equilibrium,
                                                                x,
                                                                R=R)
        _, relu_beta, _, _, _ = dut.add_lyap_relu_output_constraint(milp, x)
        Gt = [None] * dut.system.u_dim
        for i in range(dut.system.u_dim):
            Gt[i] = milp.addVars(dut.system.x_dim, lb=-gurobipy.GRB.INFINITY)
        mip_cnstr_return = dut.system.mixed_integer_constraints()
        RG_lo, RG_up = dut._compute_RG_bounds_IA(R, mip_cnstr_return.G_flat_lo,
                                                 mip_cnstr_return.G_flat_up)
        l1_subgradient_binary = None
        Vdot_coeff = []
        Vdot_vars = []
        dVdx_times_G_ret, dVdx_times_G_binary = \
            dut._add_dVdx_times_G(
                milp, x, l1_binary, relu_beta, l1_subgradient_binary, Gt, R,
                mip_cnstr_return.G_flat_lo, mip_cnstr_return.G_flat_up,
                RG_lo, RG_up, V_lambda, Vdot_coeff, Vdot_vars)
        self.assertEqual(len(Vdot_coeff), len(Vdot_vars))
        # Now sample many x and G, then solve the optimization problem, make
        # sure that Vdot_coeff * Vdot_vars evaluates to
        # ∂V/∂x * G * (u_lo + u_up)/2
        #  -|∑ᵢ∂V/∂x * G.col(i) * (u_up(i) - u_lo(i))/2|₁
        # Which is the same as min_u ∂V/∂x * G * u
        #                      s.t   u_lo <= u <= u_up
        torch.manual_seed(0)
        num_samples = 100
        x_samples = utils.uniform_sample_in_box(dut.system.x_lo,
                                                dut.system.x_up, num_samples)
        for k in range(num_samples):
            for j in range(dut.system.x_dim):
                x[j].lb = x_samples[k][j]
                x[j].ub = x_samples[k][j]
            G_sample = dut.system.G(x_samples[k])
            for j in range(dut.system.x_dim):
                for i in range(dut.system.u_dim):
                    Gt[i][j].lb = G_sample[j, i]
                    Gt[i][j].ub = G_sample[j, i]
            milp.gurobi_model.setParam(gurobipy.GRB.Param.OutputFlag, False)
            milp.setObjective(Vdot_coeff,
                              Vdot_vars,
                              0.,
                              sense=gurobipy.GRB.MINIMIZE)
            milp.gurobi_model.optimize()
            self.assertEqual(milp.gurobi_model.status,
                             gurobipy.GRB.Status.OPTIMAL)
            # Check the value of the returned variables.
            dphi_dx = utils.relu_network_gradient(dut.lyapunov_relu,
                                                  x_samples[k])
            assert (dphi_dx.shape[0] == 1)
            dl1_dx = utils.l1_gradient(R @ (x_samples[k] - x_equilibrium))
            assert (dl1_dx.shape[0] == 1)
            dVdx = dphi_dx[0][0] + V_lambda * dl1_dx[0] @ R
            dVdx_times_G_expected = dVdx @ G_sample
            dVdx_times_G_binary_expected = dVdx_times_G_expected >= 0
            dVdx_times_G_binary_sol = np.array(
                [v.x for v in dVdx_times_G_binary])
            np.testing.assert_allclose(
                dVdx_times_G_binary_sol,
                dVdx_times_G_binary_expected.detach().numpy())
            np.testing.assert_allclose(np.array(
                [v.x for v in dVdx_times_G_ret.dVdx_times_G]),
                                       dVdx_times_G_expected.detach().numpy(),
                                       atol=1E-6)

            # Now check if Vdot_coeff * Vdot_vars =
            # ∂V/∂x * G * (u_lo + u_up)/2
            # - |∑ᵢ∂V/∂x * G.col(i) * (u_up(i) - u_lo(i))/2|₁
            Vdot_sol = 0.
            for i in range(len(Vdot_vars)):
                Vdot_sol += Vdot_coeff[i] @ torch.tensor(
                    [v.x for v in Vdot_vars[i]], dtype=self.dtype)
            Vdot_expected = dVdx_times_G_expected @ (
                dut.system.u_lo + dut.system.u_up) / 2 - torch.norm(
                    dVdx_times_G_expected *
                    (dut.system.u_up - dut.system.u_lo) / 2,
                    p=1)
            np.testing.assert_allclose(Vdot_sol.item(), Vdot_expected.item())
            self.assertAlmostEqual(milp.gurobi_model.ObjVal,
                                   Vdot_expected.item())
            # Vdot_coeff * Vdot_vars should be equal to
            # min_u ∂V/∂x * G * u
            # s.t   u_lo <= u <= u_up
            u_coeff = dVdx @ G_sample
            Vdot_expected_alternative = 0
            for j in range(dut.system.u_dim):
                if u_coeff[j] >= 0:
                    Vdot_expected_alternative += \
                        u_coeff[j] * dut.system.u_lo[j]
                else:
                    Vdot_expected_alternative += \
                        u_coeff[j] * dut.system.u_up[j]
            np.testing.assert_allclose(Vdot_sol.item(), Vdot_expected.item())

    def test_add_dVdx_times_G(self):
        dut1 = mut.ControlLyapunov(self.linear_system, self.lyapunov_relu1,
                                   mut.SubgradientPolicy(None))
        x_equilibrium = torch.tensor([0.1, 0.2], dtype=self.dtype)
        R = torch.tensor([[1, 2], [-1, 1], [0, 4]], dtype=self.dtype)
        V_lambda = 0.5
        self.add_dVdx_times_G_tester(dut1, x_equilibrium, V_lambda, R)
        dut2 = mut.ControlLyapunov(self.linear_system, self.lyapunov_relu2,
                                   mut.SubgradientPolicy(None))
        self.add_dVdx_times_G_tester(dut2, x_equilibrium, V_lambda, R)
        dut3 = mut.ControlLyapunov(self.linear_system, self.lyapunov_relu3,
                                   mut.SubgradientPolicy(None))
        self.add_dVdx_times_G_tester(dut3, x_equilibrium, V_lambda, R)

    def lyapunov_derivative_as_milp_tester(self, dut, x_equilibrium, V_lambda,
                                           epsilon, eps_type, R,
                                           lyapunov_lower, lyapunov_upper,
                                           binary_var_type, is_feasible):
        lyap_deriv_return = dut.lyapunov_derivative_as_milp(
            x_equilibrium,
            V_lambda,
            epsilon,
            eps_type,
            R=R,
            lyapunov_lower=lyapunov_lower,
            lyapunov_upper=lyapunov_upper,
            x_warmstart=None,
            binary_var_type=binary_var_type)
        lyap_deriv_return.milp.gurobi_model.setParam(
            gurobipy.GRB.Param.OutputFlag, False)
        lyap_deriv_return.milp.gurobi_model.setParam(
            gurobipy.GRB.Param.DualReductions, False)
        lyap_deriv_return.milp.gurobi_model.optimize()
        if eps_type in (lyapunov.ConvergenceEps.ExpLower,
                        lyapunov.ConvergenceEps.Asymp):
            subgradient_rule = "max_sample"
        else:
            subgradient_rule = "min_sample"
        if is_feasible:
            self.assertEqual(lyap_deriv_return.milp.gurobi_model.status,
                             gurobipy.GRB.Status.OPTIMAL)
            # First evaluate the optimal solution.
            x_sol = torch.tensor([v.x for v in lyap_deriv_return.x],
                                 dtype=self.dtype)
            Vdot_optimal = lyap_deriv_return.milp.gurobi_model.ObjVal
            Vdot_sol = dut.lyapunov_derivative(
                x_sol,
                x_equilibrium,
                V_lambda,
                0.,
                R=R,
                subgradient_rule=subgradient_rule,
                zero_tol=1E-8)
            V_sol = dut.lyapunov_value(x_sol, x_equilibrium, V_lambda, R=R)
            if eps_type == lyapunov.ConvergenceEps.ExpLower:
                np.testing.assert_allclose((Vdot_sol + epsilon * V_sol).item(),
                                           Vdot_optimal,
                                           atol=2E-5)
            elif eps_type == lyapunov.ConvergenceEps.ExpUpper:
                np.testing.assert_allclose(
                    (-Vdot_sol - epsilon * V_sol).item(),
                    Vdot_optimal,
                    atol=2E-5)
            elif eps_type == lyapunov.ConvergenceEps.Asymp:
                np.testing.assert_allclose(
                    (Vdot_sol + epsilon *
                     torch.norm(R @ (x_sol - x_equilibrium), p=1)).item(),
                    Vdot_optimal,
                    atol=2E-5)

            # Sample many x_val within the bounds. make sure the objective
            # value is smaller than the optimal one.
            torch.manual_seed(0)
            x_samples = utils.uniform_sample_in_box(dut.system.x_lo,
                                                    dut.system.x_up, 1000)
            V_samples = dut.lyapunov_value(x_samples,
                                           x_equilibrium,
                                           V_lambda,
                                           R=R)
            acceptable_samples = torch.tensor([True] * x_samples.shape[0])
            if lyapunov_lower is not None:
                acceptable_samples = V_samples >= lyapunov_lower
            if lyapunov_upper is not None:
                acceptable_samples = torch.logical_and(
                    acceptable_samples, V_samples <= lyapunov_upper)
            # Now set x to x_samples[i], compute the milp objective. Check if
            # the objective matches with Vdot + epsilon * V. Also this
            # objective should be smaller than the previous objective value
            # (which maximizes over x).
            for i in range(x_samples.shape[0]):
                for j in range(dut.system.x_dim):
                    lyap_deriv_return.x[j].lb = x_samples[i][j]
                    lyap_deriv_return.x[j].ub = x_samples[i][j]
                lyap_deriv_return.milp.gurobi_model.optimize()
                if acceptable_samples[i]:
                    self.assertEqual(
                        lyap_deriv_return.milp.gurobi_model.status,
                        gurobipy.GRB.Status.OPTIMAL)
                    Vdot_sample = dut.lyapunov_derivative(
                        x_samples[i],
                        x_equilibrium,
                        V_lambda,
                        0.,
                        R=R,
                        subgradient_rule=subgradient_rule,
                        zero_tol=1E-7)
                    V_sample = dut.lyapunov_value(x_samples[i],
                                                  x_equilibrium,
                                                  V_lambda,
                                                  R=R)
                    if eps_type == lyapunov.ConvergenceEps.ExpLower:
                        self.assertAlmostEqual(
                            lyap_deriv_return.milp.gurobi_model.ObjVal,
                            (Vdot_sample + epsilon * V_sample).item(),
                            places=6)
                    elif eps_type == lyapunov.ConvergenceEps.ExpUpper:
                        self.assertAlmostEqual(
                            lyap_deriv_return.milp.gurobi_model.ObjVal,
                            (-Vdot_sample - epsilon * V_sample).item(),
                            places=6)
                    elif eps_type == lyapunov.ConvergenceEps.Asymp:
                        self.assertAlmostEqual(
                            lyap_deriv_return.milp.gurobi_model.ObjVal,
                            (Vdot_sample + epsilon *
                             torch.norm(R @ (x_samples[i] - x_equilibrium),
                                        p=1)).item(),
                            places=6)
                    self.assertLessEqual(
                        lyap_deriv_return.milp.gurobi_model.ObjVal,
                        Vdot_optimal)

                else:
                    self.assertEqual(
                        lyap_deriv_return.milp.gurobi_model.status,
                        gurobipy.GRB.Status.INFEASIBLE)

        else:
            self.assertEqual(lyap_deriv_return.milp.gurobi_model.status,
                             gurobipy.GRB.Status.INFEASIBLE)

    def test_lyapunov_derivative_as_milp1(self):
        # V_lambda = 0 and epsilon = 0, only compute ϕdot.
        dut = mut.ControlLyapunov(self.linear_system, self.lyapunov_relu1,
                                  mut.SubgradientPolicy(None))
        x_equilibrium = torch.tensor([0.1, 0.2], dtype=self.dtype)
        V_lambda = 0.
        epsilon = 0.
        R = torch.tensor([[1., 0.], [0.5, 1.], [1., -1.]], dtype=self.dtype)
        self.lyapunov_derivative_as_milp_tester(
            dut,
            x_equilibrium,
            V_lambda,
            epsilon,
            lyapunov.ConvergenceEps.ExpLower,
            R,
            None,
            None,
            gurobipy.GRB.BINARY,
            is_feasible=True)

    def test_lyapunov_derivative_as_milp2(self):
        # V_lambda != 0 and epsilon = 0
        dut = mut.ControlLyapunov(self.linear_system, self.lyapunov_relu1,
                                  mut.SubgradientPolicy(None))
        x_equilibrium = torch.tensor([0.1, 0.2], dtype=self.dtype)
        V_lambda = 0.5
        epsilon = 0.
        R = torch.tensor([[1., 0.], [0.5, 1.], [1., -1.]], dtype=self.dtype)
        self.lyapunov_derivative_as_milp_tester(
            dut,
            x_equilibrium,
            V_lambda,
            epsilon,
            lyapunov.ConvergenceEps.ExpLower,
            R,
            None,
            None,
            gurobipy.GRB.BINARY,
            is_feasible=True)

    def test_lyapunov_derivative_as_milp3(self):
        # V_lambda != 0 and epsilon != 0
        dut = mut.ControlLyapunov(self.linear_system, self.lyapunov_relu1,
                                  mut.SubgradientPolicy(None))
        x_equilibrium = torch.tensor([0.1, 0.2], dtype=self.dtype)
        V_lambda = 0.5
        epsilon = 0.2
        R = torch.tensor([[1., 0.], [0.5, 1.], [1., -1.]], dtype=self.dtype)
        for eps_type in list(lyapunov.ConvergenceEps):
            self.lyapunov_derivative_as_milp_tester(dut,
                                                    x_equilibrium,
                                                    V_lambda,
                                                    epsilon,
                                                    eps_type,
                                                    R,
                                                    None,
                                                    None,
                                                    gurobipy.GRB.BINARY,
                                                    is_feasible=True)

    def test_lyapunov_derivative_as_milp4(self):
        # V_lambda != 0 and epsilon != 0 and l1 subgradient samples is
        # non-empty.
        dut = mut.ControlLyapunov(self.linear_system,
                                  self.lyapunov_relu1,
                                  l1_subgradient_policy=mut.SubgradientPolicy(
                                      np.array([-0.5, 0, 0.5])))
        x_equilibrium = torch.tensor([0.1, 0.2], dtype=self.dtype)
        V_lambda = 0.5
        epsilon = 0.2
        R = torch.tensor([[1., 0.], [0.5, 1.], [1., -1.]], dtype=self.dtype)
        for eps_type in list(lyapunov.ConvergenceEps):
            self.lyapunov_derivative_as_milp_tester(dut,
                                                    x_equilibrium,
                                                    V_lambda,
                                                    epsilon,
                                                    eps_type,
                                                    R,
                                                    None,
                                                    None,
                                                    gurobipy.GRB.BINARY,
                                                    is_feasible=True)
        # Now fix x to x_equilibrium, where l1-norm has many subgradient. Make
        # sure that the MILP solution is correct.

        lyap_deriv_milp_ret = dut.lyapunov_derivative_as_milp(
            x_equilibrium,
            V_lambda,
            epsilon,
            lyapunov.ConvergenceEps.ExpLower,
            R=R)
        for i in range(dut.system.x_dim):
            lyap_deriv_milp_ret.x[i].lb = x_equilibrium[i].item()
            lyap_deriv_milp_ret.x[i].ub = x_equilibrium[i].item()
        lyap_deriv_milp_ret.milp.gurobi_model.setParam(
            gurobipy.GRB.Param.OutputFlag, False)
        lyap_deriv_milp_ret.milp.gurobi_model.optimize()
        self.assertEqual(lyap_deriv_milp_ret.milp.gurobi_model.status,
                         gurobipy.GRB.Status.OPTIMAL)
        self.assertAlmostEqual(
            lyap_deriv_milp_ret.milp.gurobi_model.ObjVal,
            dut.lyapunov_derivative(x_equilibrium,
                                    x_equilibrium,
                                    V_lambda,
                                    epsilon,
                                    R=R,
                                    zero_tol=1E-14).item())

    def compute_dVdx_times_G_tester(self, dut, x_equilibrium, R, G_flat_lo,
                                    G_flat_up, RG_lo, RG_up, V_lambda):
        milp = gurobi_torch_mip.GurobiTorchMIP(self.dtype)
        x = milp.addVars(dut.system.x_dim, lb=-gurobipy.GRB.INFINITY)
        l1_slack, l1_binary = dut.add_state_error_l1_constraint(
            milp,
            x_equilibrium,
            x,
            R=R,
            binary_for_zero_input=dut.l1_subgradient_policy.search_subgradient)
        _, relu_beta, _, _, _ = dut.add_lyap_relu_output_constraint(milp, x)
        l1_subgradient_binary = dut._add_l1_subgradient_binary(milp, l1_binary)
        Gt = [None] * dut.system.u_dim
        for i in range(dut.system.u_dim):
            Gt[i] = milp.addVars(dut.system.x_dim, lb=-gurobipy.GRB.INFINITY)
        ret = dut._compute_dVdx_times_G(milp, x, relu_beta, l1_binary,
                                        l1_subgradient_binary, Gt, G_flat_lo,
                                        G_flat_up, RG_lo, RG_up, R, V_lambda)
        # Now take many samples of x. Compute dVdx_times_G for each sample.
        torch.manual_seed(0)
        x_samples = utils.uniform_sample_in_box(dut.system.x_lo,
                                                dut.system.x_up, 100)
        milp.gurobi_model.setParam(gurobipy.GRB.Param.OutputFlag, False)
        for i in range(x_samples.shape[0]):
            for j in range(dut.system.x_dim):
                x[j].lb = x_samples[i][j]
                x[j].ub = x_samples[i][j]
            milp.gurobi_model.optimize()
            self.assertEqual(milp.gurobi_model.status,
                             gurobipy.GRB.Status.OPTIMAL)
            G = dut.system.G(x_samples[i])
            dphidx = utils.relu_network_gradient(dut.lyapunov_relu,
                                                 x_samples[i])
            assert (dphidx.shape[0] == 1)
            dphidx_times_G_expected = dphidx[0][0] @ G
            dphidx_times_G_sol = np.array([v.x for v in ret.dphidx_times_G])
            np.testing.assert_allclose(
                dphidx_times_G_sol,
                dphidx_times_G_expected.detach().numpy())
            dl1dx = V_lambda * (
                utils.l1_gradient(R @ (x_samples[i] - x_equilibrium)) @ R)
            assert (dl1dx.shape[0] == 1)
            np.testing.assert_allclose(
                np.array([v.x for v in ret.dl1dx_times_G]),
                (dl1dx @ G).squeeze())

            dVdx = dphidx[0][0] + dl1dx[0]
            dVdx_times_G_expected = dVdx @ G
            dVdx_times_G_sol = np.array([v.x for v in ret.dVdx_times_G])
            np.testing.assert_allclose(dVdx_times_G_sol,
                                       dVdx_times_G_expected.detach().numpy())
            # Check the bounds
            np.testing.assert_array_less(
                dVdx_times_G_sol,
                ret.dVdx_times_G_up.detach().numpy() + 1E-10)
            np.testing.assert_array_less(
                ret.dVdx_times_G_lo.detach().numpy() - 1E-10, dVdx_times_G_sol)

    def test_compute_dVdx_times_G1(self):
        # Test with no subgradient
        dut = mut.ControlLyapunov(self.linear_system, self.lyapunov_relu1,
                                  mut.SubgradientPolicy(None))
        mip_cnstr_return = self.linear_system.mixed_integer_constraints()
        R = torch.tensor([[1., 2.], [0., 1.], [1., -2.]], dtype=self.dtype)
        RG_lo, RG_up = dut._compute_RG_bounds_IA(R, mip_cnstr_return.G_flat_lo,
                                                 mip_cnstr_return.G_flat_up)
        x_equilibrium = torch.tensor([0.5, -1.], dtype=self.dtype)
        V_lambda = 0.5
        self.compute_dVdx_times_G_tester(dut, x_equilibrium, R,
                                         mip_cnstr_return.G_flat_lo,
                                         mip_cnstr_return.G_flat_up, RG_lo,
                                         RG_up, V_lambda)

    def test_compute_dVdx_times_G2(self):
        # Test with no subgradient
        l1_subgradient_samples = np.array([-0.5, 0, 0.5])
        dut = mut.ControlLyapunov(
            self.linear_system, self.lyapunov_relu1,
            mut.SubgradientPolicy(l1_subgradient_samples))
        mip_cnstr_return = self.linear_system.mixed_integer_constraints()
        R = torch.tensor([[1., 2.], [0., 1.], [1., -2.]], dtype=self.dtype)
        RG_lo, RG_up = dut._compute_RG_bounds_IA(R, mip_cnstr_return.G_flat_lo,
                                                 mip_cnstr_return.G_flat_up)
        x_equilibrium = torch.tensor([0.5, -1.], dtype=self.dtype)
        V_lambda = 0.5
        self.compute_dVdx_times_G_tester(dut, x_equilibrium, R,
                                         mip_cnstr_return.G_flat_lo,
                                         mip_cnstr_return.G_flat_up, RG_lo,
                                         RG_up, V_lambda)

    def test_compute_dVdx_times_G3(self):
        # Now test the input x with subgradient.
        l1_subgradient_samples = np.array([-0.5, 0, 0.5])
        dut = mut.ControlLyapunov(
            self.linear_system, self.lyapunov_relu1,
            mut.SubgradientPolicy(l1_subgradient_samples))
        mip_cnstr_return = self.linear_system.mixed_integer_constraints()
        R = torch.tensor([[1., 2.], [0., 1.], [1., -2.]], dtype=self.dtype)
        RG_lo, RG_up = dut._compute_RG_bounds_IA(R, mip_cnstr_return.G_flat_lo,
                                                 mip_cnstr_return.G_flat_up)
        x_equilibrium = torch.tensor([0.5, -1.], dtype=self.dtype)
        V_lambda = 0.5
        milp = gurobi_torch_mip.GurobiTorchMILP(self.dtype)
        x = milp.addVars(dut.system.x_dim, lb=-gurobipy.GRB.INFINITY)
        l1_slack, l1_binary = dut.add_state_error_l1_constraint(
            milp,
            x_equilibrium,
            x,
            R=R,
            binary_for_zero_input=dut.l1_subgradient_policy.search_subgradient)
        _, relu_beta, _, _, _ = dut.add_lyap_relu_output_constraint(milp, x)
        l1_subgradient_binary = dut._add_l1_subgradient_binary(milp, l1_binary)
        Gt = [None] * dut.system.u_dim
        for i in range(dut.system.u_dim):
            Gt[i] = milp.addVars(dut.system.x_dim, lb=-gurobipy.GRB.INFINITY)
        ret = dut._compute_dVdx_times_G(milp, x, relu_beta, l1_binary,
                                        l1_subgradient_binary, Gt,
                                        mip_cnstr_return.G_flat_lo,
                                        mip_cnstr_return.G_flat_up, RG_lo,
                                        RG_up, R, V_lambda)

        def tester(x_sample, l1_subgradient_index_map):
            # subgradient_index_map set
            # l1_subgradient_binary[i][subgradient_index_map[i]] to 1.
            for i in range(dut.system.x_dim):
                x[i].lb = x_sample[i].item()
                x[i].ub = x_sample[i].item()
            for key, value in l1_subgradient_index_map.items():
                l1_subgradient_binary[key][value].lb = 1.
                l1_subgradient_binary[key][value].ub = 1.
            milp.gurobi_model.setParam(gurobipy.GRB.Param.OutputFlag, False)
            milp.gurobi_model.optimize()
            self.assertEqual(milp.gurobi_model.status,
                             gurobipy.GRB.Status.OPTIMAL)
            dphidx_expected = utils.relu_network_gradient(
                dut.lyapunov_relu, x_sample)
            # I don't handle subgradient of network yet.
            assert (dphidx_expected.shape[0] == 1)
            dphidx_expected = dphidx_expected[0]
            dl1dx_expected = torch.zeros(R.shape[0], dtype=self.dtype)
            Rx = R @ (x_sample - x_equilibrium)
            for i in range(R.shape[0]):
                if Rx[i] > 0:
                    dl1dx_expected[i] = 1
                elif Rx[i] < 0:
                    dl1dx_expected[i] = -1
            for key, value in l1_subgradient_index_map.items():
                dl1dx_expected[key] = l1_subgradient_samples[value]
            dl1dx_expected = V_lambda * dl1dx_expected @ R
            G = dut.system.G(x_sample)
            dl1dx_times_G_expected = dl1dx_expected @ G
            for i in range(dut.system.u_dim):
                self.assertAlmostEqual(ret.dl1dx_times_G[i].x,
                                       dl1dx_times_G_expected[i].item())
                self.assertAlmostEqual(ret.dphidx_times_G[i].x,
                                       (dphidx_expected @ G[:, i]).item())
                self.assertAlmostEqual(
                    ret.dVdx_times_G[i].x,
                    ret.dphidx_times_G[i].x + ret.dl1dx_times_G[i].x)
            np.testing.assert_array_less(
                np.array([v.x for v in ret.dVdx_times_G]),
                ret.dVdx_times_G_up.detach().numpy() + 1E-10)
            np.testing.assert_array_less(
                ret.dVdx_times_G_lo.detach().numpy() - 1E-10,
                np.array([v.x for v in ret.dVdx_times_G]))

            # Now reset l1_subgradient_binary
            for i in range(len(l1_subgradient_binary)):
                for v in l1_subgradient_binary[i]:
                    v.lb = 0.
                    v.ub = 1.

        tester(torch.tensor([1, -1], dtype=self.dtype), {1: 0})
        tester(torch.tensor([1, -1], dtype=self.dtype), {1: 1})
        tester(torch.tensor([1, -1], dtype=self.dtype), {1: 2})
        tester(torch.tensor([0, -0.75], dtype=self.dtype), {0: 0})
        tester(torch.tensor([0, -0.75], dtype=self.dtype), {0: 1})
        tester(torch.tensor([0, -0.75], dtype=self.dtype), {0: 2})

    def test_lyapunov_derivative_loss_at_samples_and_next_states(self):
        # Test with the linear system.
        dut = mut.ControlLyapunov(self.linear_system, self.lyapunov_relu1,
                                  mut.SubgradientPolicy(None))
        V_lambda = 0.5
        epsilon = 0.1
        torch.manual_seed(0)
        state_samples = utils.uniform_sample_in_box(dut.system.x_lo,
                                                    dut.system.x_up, 100)
        x_equilibrium = torch.tensor([0.2, 0.5], dtype=self.dtype)
        R = torch.tensor([[1., 2.], [-1., 1.], [2, 0.]], dtype=self.dtype)
        margin = 1.5

        loss1 = dut.lyapunov_derivative_loss_at_samples_and_next_states(
            V_lambda,
            epsilon,
            state_samples,
            None,
            x_equilibrium,
            lyapunov.ConvergenceEps.ExpLower,
            R=R,
            margin=margin,
            reduction="mean")
        Vdot_samples = torch.stack([
            dut.lyapunov_derivative(state_samples[i],
                                    x_equilibrium,
                                    V_lambda,
                                    epsilon,
                                    R=R,
                                    subgradient_rule="max_sample",
                                    zero_tol=1e-6)
            for i in range(state_samples.shape[0])
        ])
        V_samples = dut.lyapunov_value(state_samples,
                                       x_equilibrium,
                                       V_lambda,
                                       R=R)
        np.testing.assert_allclose(
            torch.mean(
                torch.maximum(Vdot_samples + epsilon * V_samples + margin,
                              torch.zeros_like(Vdot_samples,
                                               dtype=self.dtype))).item(),
            loss1.item())

        loss2 = dut.lyapunov_derivative_loss_at_samples_and_next_states(
            V_lambda,
            epsilon,
            state_samples,
            None,
            x_equilibrium,
            lyapunov.ConvergenceEps.Asymp,
            R=R,
            margin=margin,
            reduction="max")
        np.testing.assert_allclose(
            torch.max(
                torch.maximum(
                    Vdot_samples + epsilon * torch.norm(
                        R @ (state_samples - x_equilibrium).T, p=1, dim=0) +
                    margin, torch.zeros_like(Vdot_samples,
                                             dtype=self.dtype))).item(),
            loss2.item())

    def test_add_l1_subgradient_binary(self):
        # Test with no subgradient.
        dut = mut.ControlLyapunov(self.linear_system, self.lyapunov_relu1,
                                  mut.SubgradientPolicy(None))
        milp = gurobi_torch_mip.GurobiTorchMIP(self.dtype)
        l1_binary = l1_binary = milp.addVars(3, vtype=gurobipy.GRB.BINARY)
        l1_subgradient_binary = dut._add_l1_subgradient_binary(milp, l1_binary)
        self.assertIsNone(l1_subgradient_binary)

        # Test with subgradient.
        dut = mut.ControlLyapunov(
            self.linear_system, self.lyapunov_relu1,
            mut.SubgradientPolicy(np.array([-0.5, 0, 0.1, 0.5])))
        milp = gurobi_torch_mip.GurobiTorchMIP(self.dtype)
        l1_binary = [None] * 2
        for i in range(2):
            l1_binary[i] = milp.addVars(3, vtype=gurobipy.GRB.BINARY)
        l1_subgradient_binary = dut._add_l1_subgradient_binary(milp, l1_binary)
        self.assertIsInstance(l1_subgradient_binary, list)
        self.assertEqual(len(l1_subgradient_binary), 2)
        for i in range(2):
            self.assertEqual(len(l1_subgradient_binary[i]), 4)
            for j in range(4):
                self.assertEqual(l1_subgradient_binary[i][j].vtype,
                                 gurobipy.GRB.BINARY)
        # Set two l1_subgradient_binary to be active simultaneously, the
        # problem is infeasible.
        l1_subgradient_binary[0][0].lb = 1
        l1_subgradient_binary[0][0].ub = 1
        l1_subgradient_binary[0][1].lb = 1
        l1_subgradient_binary[0][1].ub = 1
        milp.gurobi_model.setParam(gurobipy.GRB.Param.OutputFlag, False)
        milp.gurobi_model.optimize()
        self.assertEqual(milp.gurobi_model.status,
                         gurobipy.GRB.Status.INFEASIBLE)

    def test_calc_vdot_allsubgradient(self):
        dut = mut.ControlLyapunov(self.linear_system,
                                  self.lyapunov_relu1,
                                  l1_subgradient_policy=mut.SubgradientPolicy(
                                      np.array([0.])))
        x_equilibrium = torch.tensor([0., 0.], dtype=self.dtype)
        V_lambda = 0.1
        R = torch.tensor([[1, 2], [0, 1], [1, -1]], dtype=self.dtype)
        zero_tol = 0.

        # First test x with unique gradient.
        x = torch.tensor([3, 2], dtype=self.dtype)
        vdot, _ = dut.calc_vdot_allsubgradient(x, x_equilibrium, V_lambda, R,
                                               zero_tol)
        self.assertEqual(
            vdot.item(),
            dut.lyapunov_derivative(x, x_equilibrium, V_lambda, 0.,
                                    R=R).item())

        # The l1-norm has one entry with subgradient.
        def tester(x):
            vdot, u_sol = dut.calc_vdot_allsubgradient(x, x_equilibrium,
                                                       V_lambda, R, zero_tol)
            np.testing.assert_array_less(
                u_sol,
                dut.system.u_up.detach().numpy() + 1E-10)
            np.testing.assert_array_less(
                dut.system.u_lo.detach().numpy() - 1E-10, u_sol)
            dphi_dx = utils.relu_network_gradient(dut.lyapunov_relu,
                                                  x,
                                                  zero_tol=zero_tol)
            dl1_dx = V_lambda * utils.l1_gradient(R @ (x - x_equilibrium),
                                                  zero_tol=zero_tol) @ R
            all_subgradient = dphi_dx.squeeze(1).repeat(
                (1, dl1_dx.shape[0])).reshape(
                    (-1, dut.system.x_dim)) + dl1_dx.repeat(
                        (dphi_dx.shape[0], 1))
            f = dut.system.f(x)
            G = dut.system.G(x)
            self.assertAlmostEqual(
                torch.max(
                    all_subgradient @ f +
                    all_subgradient @ G @ torch.from_numpy(u_sol)).item(),
                vdot.item())
            # Sample many u inside [u_lo, u_up], evaluate vdot on these u,
            # they should all be larger than vdot evaluated at u_sol.
            torch.manual_seed(0)
            u_samples = utils.uniform_sample_in_box(dut.system.u_lo,
                                                    dut.system.u_up, 100)
            self.assertGreaterEqual(
                torch.min(
                    torch.max((all_subgradient @ f).reshape((-1, 1)).repeat(
                        (1, u_samples.shape[0])) +
                              all_subgradient @ G @ u_samples.T,
                              dim=0)[0]).item(), vdot.item())
            self.assertGreaterEqual(
                vdot.item(),
                dut.lyapunov_derivative(x, x_equilibrium, V_lambda, 0.,
                                        R=R).item())

        tester(torch.tensor([3, 2], dtype=self.dtype))
        tester(torch.tensor([1, 1], dtype=self.dtype))
        tester(torch.tensor([2, -1], dtype=self.dtype))
        tester(torch.tensor([0, 0], dtype=self.dtype))


def set_l1_binary_val(R, x, x_equilibrium, l1_binary, idx):
    """
    Set the value of l1_binary[idx] based on R * (x - x_equilibrium)
    """
    Rx = R @ (x - x_equilibrium)
    for i in idx:
        if isinstance(l1_binary[i], gurobipy.Var):
            if Rx[i] > 0:
                l1_binary[i].lb = 1
                l1_binary[i].ub = 1
            elif Rx[i] < 0:
                l1_binary[i].lb = 0
                l1_binary[i].ub = 0
            else:
                raise Exception("Can't set l1_binary value")
        elif isinstance(l1_binary[i], list):
            if Rx[i] > 0:
                l1_binary[i][0].lb = 0
                l1_binary[i][0].ub = 0
                l1_binary[i][1].lb = 0
                l1_binary[i][1].ub = 0
                l1_binary[i][2].lb = 1
                l1_binary[i][2].ub = 1
            elif Rx[i] < 0:
                l1_binary[i][0].lb = 1
                l1_binary[i][0].ub = 1
                l1_binary[i][1].lb = 0
                l1_binary[i][1].ub = 0
                l1_binary[i][2].lb = 0
                l1_binary[i][2].ub = 0
            else:
                raise Exception("Can't set l1_binary value")
        else:
            raise Exception("Unknow l1_binary value")


class TestComputeDl1dxTimesY(unittest.TestCase):
    def test1(self):
        # test with search_subgradient=False
        dtype = torch.float64
        milp = gurobi_torch_mip.GurobiTorchMIP(dtype)
        R = torch.tensor([[1, -2], [0, 1], [-1, 1]], dtype=dtype)
        Ry_lo = torch.tensor([1, -3, -5], dtype=dtype)
        Ry_up = torch.tensor([9, -1, 5], dtype=dtype)
        V_lambda = 0.5
        l1_binary = milp.addVars(3, vtype=gurobipy.GRB.BINARY)
        y = milp.addVars(2, lb=-gurobipy.GRB.INFINITY)
        dl1dx_times_y_var = mut._compute_dl1dx_times_y(milp, l1_binary, y, R,
                                                       Ry_lo, Ry_up, V_lambda,
                                                       "dl1dx_times_y", False)
        # Take many sample x and y, compute dl1dx_times_y by hand.
        x_equilibrium = torch.tensor([0, 0], dtype=dtype)
        torch.manual_seed(0)
        x_samples = utils.uniform_sample_in_box(
            torch.tensor([-3, -3], dtype=dtype),
            torch.tensor([3, 3], dtype=dtype), 100)
        for i in range(x_samples.shape[0]):
            dl1dx = utils.l1_gradient(R @ (x_samples[i] - x_equilibrium))
            # We will check non-unique subgradient separately.
            if dl1dx.shape[0] > 1:
                continue
            dl1dx = V_lambda * (dl1dx @ R)
            # Now sample y
            found_y = False
            while not found_y:
                y_sample = 10 * torch.rand(2, dtype=dtype) - 5
                Ry = R @ y_sample
                if torch.all(Ry <= Ry_up) and torch.all(Ry >= Ry_lo):
                    found_y = True
            dl1dx_times_y = dl1dx @ y_sample
            # Now solve the MIP
            for j in range(2):
                y[j].lb = y_sample[j].item()
                y[j].ub = y_sample[j].item()
            set_l1_binary_val(R, x_samples[i], x_equilibrium, l1_binary,
                              [0, 1, 2])
            milp.gurobi_model.setParam(gurobipy.GRB.Param.OutputFlag, False)
            milp.gurobi_model.optimize()
            self.assertEqual(milp.gurobi_model.status,
                             gurobipy.GRB.Status.OPTIMAL)
            self.assertAlmostEqual(dl1dx_times_y_var.x, dl1dx_times_y.item())

        # Test input x such that R * (x-x*) has non-unique gradient.
        x_sample = torch.tensor([2, 1], dtype=dtype)
        dl1dx = V_lambda * (
            utils.l1_gradient(R @ (x_sample - x_equilibrium)) @ R)
        dl1dx_times_y = dl1dx @ y_sample
        for alpha0_val in (0, 1):
            l1_binary[0].lb = alpha0_val
            l1_binary[0].ub = alpha0_val
            set_l1_binary_val(R, x_sample, x_equilibrium, l1_binary, [1, 2])
            for j in range(2):
                y[j].lb = y_sample[j].item()
                y[j].ub = y_sample[j].item()
            milp.gurobi_model.optimize()
            self.assertEqual(milp.gurobi_model.status,
                             gurobipy.GRB.Status.OPTIMAL)
            self.assertLess(
                torch.min(torch.abs(dl1dx_times_y_var.x - dl1dx_times_y)),
                1E-6)

    def subgradient_tester(self, milp, y, l1_binary, dl1dx_times_y_var,
                           x_sample, subgradient_index, gradient_index, R,
                           x_equilibrium, Ry_lo, Ry_up, V_lambda):
        dtype = torch.float64

        def sample_y():
            found_y = False
            while not found_y:
                y_sample = 10 * torch.rand(2, dtype=dtype) - 5
                Ry = R @ y_sample
                if torch.all(Ry <= Ry_up) and torch.all(Ry >= Ry_lo):
                    found_y = True
            return y_sample

        # Now test subgradient.
        for idx in subgradient_index:
            for i in range(3):
                l1_binary[idx][0].lb = 0
                l1_binary[idx][0].ub = 0
                l1_binary[idx][1].lb = 0
                l1_binary[idx][1].ub = 1
                l1_binary[idx][2].lb = 0
                l1_binary[idx][2].ub = 0
        set_l1_binary_val(R, x_sample, x_equilibrium, l1_binary,
                          gradient_index)
        for _ in range(10):
            # Test with different y_sample
            y_sample = sample_y()
            for i in range(2):
                y[i].lb = y_sample[i].item()
                y[i].ub = y_sample[i].item()
            dl1dx_times_y_extreme = V_lambda * utils.l1_gradient(
                R @ (x_sample - x_equilibrium)) @ R @ y_sample
            # First maximize/minimize dl1dx_times_y_expected. It should
            # correspond to the left/right direvative.
            milp.gurobi_model.setObjective(dl1dx_times_y_var,
                                           gurobipy.GRB.MAXIMIZE)
            milp.gurobi_model.optimize()
            self.assertEqual(milp.gurobi_model.status,
                             gurobipy.GRB.Status.OPTIMAL)
            self.assertAlmostEqual(dl1dx_times_y_var.x,
                                   torch.max(dl1dx_times_y_extreme).item())
            milp.gurobi_model.setObjective(dl1dx_times_y_var,
                                           gurobipy.GRB.MINIMIZE)
            milp.gurobi_model.optimize()
            self.assertEqual(milp.gurobi_model.status,
                             gurobipy.GRB.Status.OPTIMAL)
            self.assertAlmostEqual(dl1dx_times_y_var.x,
                                   torch.min(dl1dx_times_y_extreme).item())
            # Set the gradient to a desired value, check if dl1dx_times_y_var
            # can obtain this desired value.
            milp.gurobi_model.setObjective(0)
            gradient = torch.zeros((1, 4), dtype=dtype)
            Rx = R @ (x_sample - x_equilibrium)
            for i in range(4):
                if Rx[i] > 0:
                    gradient[0, i] = 1
                elif Rx[i] < 0:
                    gradient[0, i] = -1
                else:
                    gradient[0, i] = torch.rand(1, dtype=dtype) * 2 - 1
            dl1dx_times_y_expected = V_lambda * (gradient @ R) @ y_sample
            dl1dx_times_y_var.lb = dl1dx_times_y_expected
            dl1dx_times_y_var.ub = dl1dx_times_y_expected
            milp.gurobi_model.optimize()
            self.assertEqual(milp.gurobi_model.status,
                             gurobipy.GRB.Status.OPTIMAL)
            dl1dx_times_y_var.lb = -gurobipy.GRB.INFINITY
            dl1dx_times_y_var.ub = gurobipy.GRB.INFINITY

    def test2(self):
        # test with search_subgradient=True
        dtype = torch.float64
        milp = gurobi_torch_mip.GurobiTorchMIP(dtype)
        R = torch.tensor([[1, -2], [0, 1], [-1, 1], [1, 0]], dtype=dtype)
        Ry_lo = torch.tensor([1, -3, -5, -3], dtype=dtype)
        Ry_up = torch.tensor([9, -1, 5, 5], dtype=dtype)

        def sample_y():
            found_y = False
            while not found_y:
                y_sample = 10 * torch.rand(2, dtype=dtype) - 5
                Ry = R @ y_sample
                if torch.all(Ry <= Ry_up) and torch.all(Ry >= Ry_lo):
                    found_y = True
            return y_sample

        V_lambda = 0.5
        l1_binary = [None] * 4
        for i in range(4):
            l1_binary[i] = milp.addVars(3, vtype=gurobipy.GRB.BINARY)
        y = milp.addVars(2, lb=-gurobipy.GRB.INFINITY)
        dl1dx_times_y_var = mut._compute_dl1dx_times_y(milp, l1_binary, y, R,
                                                       Ry_lo, Ry_up, V_lambda,
                                                       "dl1dx_times_y", True)
        # First test values of x that doesn't have subgradient.
        x_equilibrium = torch.tensor([0, 0], dtype=dtype)
        torch.manual_seed(0)
        x_samples = utils.uniform_sample_in_box(
            torch.tensor([-3, -3], dtype=dtype),
            torch.tensor([3, 3], dtype=dtype), 100)
        for i in range(x_samples.shape[0]):
            dl1dx = V_lambda * (
                utils.l1_gradient(R @ (x_samples[i] - x_equilibrium)) @ R)
            if dl1dx.shape[0] > 1:
                # We will test the non-unique gradient case separately.
                continue
            y_sample = sample_y()
            dl1dx_times_y = dl1dx @ y_sample
            for j in range(2):
                y[j].lb = y_sample[j].item()
                y[j].ub = y_sample[j].item()
            set_l1_binary_val(R, x_samples[i], x_equilibrium, l1_binary,
                              [0, 1, 2, 3])
            milp.gurobi_model.setParam(gurobipy.GRB.Param.OutputFlag, False)
            milp.gurobi_model.optimize()
            self.assertEqual(milp.gurobi_model.status,
                             gurobipy.GRB.Status.OPTIMAL)
            self.assertAlmostEqual(dl1dx_times_y_var.x, dl1dx_times_y.item())

        # Now test subgradient.
        # R @ (x_sample - x_equilibrium)= [0, 1, -1, 2]
        x_sample = torch.tensor([2, 1], dtype=dtype)
        self.subgradient_tester(milp, y, l1_binary, dl1dx_times_y_var,
                                x_sample, [0], [1, 2, 3], R, x_equilibrium,
                                Ry_lo, Ry_up, V_lambda)
        # R @ (x_sample - x_equilibrium) = [1, 0, -1, 1]
        x_sample = torch.tensor([1, 0], dtype=dtype)
        self.subgradient_tester(milp, y, l1_binary, dl1dx_times_y_var,
                                x_sample, [1], [0, 2, 3], R, x_equilibrium,
                                Ry_lo, Ry_up, V_lambda)
        # R @ (x_sample - x_equilibrium) = [-1, 1, 0, 1]
        x_sample = torch.tensor([1, 1], dtype=dtype)
        self.subgradient_tester(milp, y, l1_binary, dl1dx_times_y_var,
                                x_sample, [2], [0, 1, 3], R, x_equilibrium,
                                Ry_lo, Ry_up, V_lambda)

        # R @ (x_sample - x_equilibrium) = [-2, 1, 1, 0]
        x_sample = torch.tensor([0, 1], dtype=dtype)
        self.subgradient_tester(milp, y, l1_binary, dl1dx_times_y_var,
                                x_sample, [3], [0, 1, 2], R, x_equilibrium,
                                Ry_lo, Ry_up, V_lambda)

        # R @ (x_sample - x_equilibrium) = [0, 0, 0, 0]
        x_sample = torch.tensor([0, 0], dtype=dtype)
        self.subgradient_tester(milp, y, l1_binary, dl1dx_times_y_var,
                                x_sample, [0, 1, 2, 3], [], R, x_equilibrium,
                                Ry_lo, Ry_up, V_lambda)


class TestComputeDl1dxTimesYSampledGradient(unittest.TestCase):
    def setUp(self):
        self.dtype = torch.float64
        self.milp = gurobi_torch_mip.GurobiTorchMILP(self.dtype)
        self.R = torch.tensor([[1, -2], [0, 1], [-1, 1]], dtype=self.dtype)
        self.Ry_lo = torch.tensor([1, -3, -5], dtype=self.dtype)
        self.Ry_up = torch.tensor([9, -1, 5], dtype=self.dtype)
        self.V_lambda = 0.5
        self.l1_binary = [None] * 3
        self.subgradient_samples = np.array([-0.6, -0.4, 0, 0.3, 0.5])
        self.subgradient_binary = [None] * 3
        self.y = self.milp.addVars(2, lb=-gurobipy.GRB.INFINITY)
        for i in range(3):
            self.l1_binary[i] = self.milp.addVars(3, vtype=gurobipy.GRB.BINARY)
            self.subgradient_binary[i] = self.milp.addVars(
                5, vtype=gurobipy.GRB.BINARY)
            self.milp.addLConstr([
                torch.ones(5, dtype=self.dtype),
                torch.tensor([-1], dtype=self.dtype)
            ], [self.subgradient_binary[i], [self.l1_binary[i][1]]],
                                 sense=gurobipy.GRB.EQUAL,
                                 rhs=0.)
        self.dl1dx_times_y_var = \
            mut._compute_dl1dx_times_y_sampled_subgradient(
                self.milp, self.l1_binary, self.y, self.subgradient_binary,
                self.subgradient_samples, self.R, self.Ry_lo, self.Ry_up,
                self.V_lambda, "dl1dx_times_y")

    def sample_y(self):
        found_y = False
        while not found_y:
            y_sample = 10 * torch.rand(2, dtype=self.dtype) - 5
            Ry = self.R @ y_sample
            if torch.all(Ry <= self.Ry_up) and torch.all(Ry >= self.Ry_lo):
                found_y = True
                return y_sample

    def test1(self):
        # Test with no subgradient.

        x_samples = utils.uniform_sample_in_box(
            torch.tensor([-3, -4], dtype=self.dtype),
            torch.tensor([4, 5], dtype=self.dtype), 100)
        x_equilibrium = torch.tensor([0, 0], dtype=self.dtype)
        for i in range(x_samples.shape[0]):
            dl1dx = self.V_lambda * utils.l1_gradient(
                self.R @ (x_samples[i] - x_equilibrium)) @ self.R
            if dl1dx.shape[0] > 1:
                continue
            y_sample = self.sample_y()
            dl1dx_times_y = dl1dx @ y_sample
            for j in range(2):
                self.y[j].lb = y_sample[j].item()
                self.y[j].ub = y_sample[j].item()

            set_l1_binary_val(self.R, x_samples[i], x_equilibrium,
                              self.l1_binary, [0, 1, 2])
            self.milp.gurobi_model.setParam(gurobipy.GRB.Param.OutputFlag,
                                            False)
            self.milp.gurobi_model.optimize()
            self.assertEqual(self.milp.gurobi_model.status,
                             gurobipy.GRB.Status.OPTIMAL)
            self.assertAlmostEqual(self.dl1dx_times_y_var.x,
                                   dl1dx_times_y.item())

    def test2(self):
        # Test with subgradients.
        x_equilibrium = torch.tensor([0, 0], dtype=self.dtype)
        x_sample = torch.tensor([2, 1], dtype=self.dtype)
        # R @ (x - x*) = [0, 1, -1]
        # set the subgradient to 0.3
        dl1dx = self.V_lambda * torch.tensor(
            [self.subgradient_samples[3], 1, -1], dtype=self.dtype) @ self.R
        y_sample = self.sample_y()
        for i in range(2):
            self.y[i].lb = y_sample[i].item()
            self.y[i].ub = y_sample[i].item()
        set_l1_binary_val(self.R, x_sample, x_equilibrium, self.l1_binary,
                          [1, 2])
        self.l1_binary[0][0].lb = 0.
        self.l1_binary[0][0].ub = 0.
        self.l1_binary[0][1].lb = 1.
        self.l1_binary[0][1].ub = 1.
        self.l1_binary[0][2].lb = 0.
        self.l1_binary[0][2].ub = 0.
        self.subgradient_binary[0][3].lb = 1.
        self.subgradient_binary[0][3].ub = 1.
        self.milp.gurobi_model.setParam(gurobipy.GRB.Param.OutputFlag, False)
        self.milp.gurobi_model.optimize()
        self.assertEqual(self.milp.gurobi_model.status,
                         gurobipy.GRB.Status.OPTIMAL)
        self.assertAlmostEqual(self.dl1dx_times_y_var.x,
                               (dl1dx @ y_sample).item())

        # x_sample = [1, 1], R @ (x - x*) = [-1, 1, 0]
        # set the subgradient to -0.4
        x_sample = torch.tensor([1, 1], dtype=self.dtype)
        dl1dx = self.V_lambda * torch.tensor(
            [-1, 1, self.subgradient_samples[1]], dtype=self.dtype) @ self.R
        y_sample = self.sample_y()
        for i in range(2):
            self.y[i].lb = y_sample[i].item()
            self.y[i].ub = y_sample[i].item()
        set_l1_binary_val(self.R, x_sample, x_equilibrium, self.l1_binary,
                          [0, 1])
        self.l1_binary[2][0].lb = 0.
        self.l1_binary[2][0].ub = 0.
        self.l1_binary[2][1].lb = 1.
        self.l1_binary[2][1].ub = 1.
        self.l1_binary[2][2].lb = 0.
        self.l1_binary[2][2].ub = 0.
        # Reset the bounds on subgradient_binary.
        for i in range(3):
            for j in range(5):
                self.subgradient_binary[i][j].lb = 0.
                self.subgradient_binary[i][j].ub = 1.
        self.subgradient_binary[2][1].lb = 1.
        self.subgradient_binary[2][1].ub = 1.
        self.milp.gurobi_model.setParam(gurobipy.GRB.Param.OutputFlag, False)
        self.milp.gurobi_model.optimize()
        self.assertEqual(self.milp.gurobi_model.status,
                         gurobipy.GRB.Status.OPTIMAL)
        self.assertAlmostEqual(self.dl1dx_times_y_var.x,
                               (dl1dx @ y_sample).item())


if __name__ == "__main__":
    unittest.main()
