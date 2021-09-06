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
        dl1_dx = V_lambda * utils.l1_gradient(R @ (x - x_equilibrium),
                                              zero_tol=zero_tol) @ R
        v = dut.lyapunov_value(x, x_equilibrium, V_lambda, R=R)
        if subgradient_rule == "max":
            vdot_expected = -np.inf
        elif subgradient_rule == "min":
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
                if subgradient_rule == "max" and vdot_expected_ij > \
                        vdot_expected:
                    vdot_expected = vdot_expected_ij
                elif subgradient_rule == "min" and vdot_expected_ij < \
                        vdot_expected:
                    vdot_expected = vdot_expected_ij
                elif subgradient_rule == "all":
                    vdot_expected.append(vdot_expected_ij)
        if subgradient_rule in ("min", "max"):
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
        dut = mut.ControlLyapunov(self.linear_system, self.lyapunov_relu1)
        x_equilibrium = torch.tensor([0, -1], dtype=self.dtype)
        V_lambda = 0.5
        R = torch.tensor([[1, 0], [0, 2], [1, 3]], dtype=self.dtype)
        epsilon = 0.1

        for subgradient_rule in ("min", "max", "all"):
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
        dut = mut.ControlLyapunov(self.linear_system, self.lyapunov_relu1)
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
        dut = mut.ControlLyapunov(self.linear_system, self.lyapunov_relu1)
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
        Vdot_coeff = []
        Vdot_vars = []
        dl1dx_times_f_slack = dut._add_dl1dx_times_f(milp, x, l1_binary, f, R,
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
            # Check dl1dx_times_f_slack is correct.
            dl1dx_times_f_slack_sol = np.array(
                [v.x for v in dl1dx_times_f_slack])
            dl1dx_times_f_slack_expected = (
                l1_binary_sol * (R @ f_samples[i])).detach().numpy()
            np.testing.assert_allclose(dl1dx_times_f_slack_sol,
                                       dl1dx_times_f_slack_expected)
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

    def test_add_dVdx_times_G(self):
        dut = mut.ControlLyapunov(self.linear_system, self.lyapunov_relu1)
        x_equilibrium = torch.tensor([0.1, 0.2], dtype=dut.system.dtype)
        R = torch.tensor([[1, 2], [-1, 1], [0, 4]], dtype=dut.system.dtype)
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
        G_flat_lo, G_flat_up = dut.system.compute_G_range_ia()
        RG_lo, RG_up = dut._compute_RG_bounds_IA(R, G_flat_lo, G_flat_up)
        V_lambda = 0.5
        Vdot_coeff = []
        Vdot_vars = []
        dVdx_times_G_ret, dVdx_times_G_binary = \
            dut._add_dVdx_times_G(
                milp, x, l1_binary, relu_beta, Gt, R, G_flat_lo, G_flat_up,
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
            np.testing.assert_allclose(
                np.array([v.x for v in dVdx_times_G_ret.dVdx_times_G]),
                dVdx_times_G_expected.detach().numpy())

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
            subgradient_rule = "max"
        else:
            subgradient_rule = "min"
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
        dut = mut.ControlLyapunov(self.linear_system, self.lyapunov_relu1)
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
        dut = mut.ControlLyapunov(self.linear_system, self.lyapunov_relu1)
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
        dut = mut.ControlLyapunov(self.linear_system, self.lyapunov_relu1)
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

    def compute_dVdx_times_G_tester(self, dut, x_equilibrium, R, G_flat_lo,
                                    G_flat_up, RG_lo, RG_up, V_lambda):
        milp = gurobi_torch_mip.GurobiTorchMIP(self.dtype)
        x = milp.addVars(dut.system.x_dim, lb=-gurobipy.GRB.INFINITY)
        l1_slack, l1_binary = dut.add_state_error_l1_constraint(milp,
                                                                x_equilibrium,
                                                                x,
                                                                R=R)
        _, relu_beta, _, _, _ = dut.add_lyap_relu_output_constraint(milp, x)
        Gt = [None] * dut.system.u_dim
        for i in range(dut.system.u_dim):
            Gt[i] = milp.addVars(dut.system.x_dim, lb=-gurobipy.GRB.INFINITY)
        ret = dut._compute_dVdx_times_G(milp, x, relu_beta, l1_binary, Gt,
                                        G_flat_lo, G_flat_up, RG_lo, RG_up, R,
                                        V_lambda)
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
            G = dut.system.G(x_samples)
            dphidx = utils.relu_network_gradient(dut.lyapunov_relu,
                                                 x_samples[i])
            assert (dphidx.shape[0] == 1)
            dphidx_times_G_expected = dphidx[0][0] @ G
            dphidx_times_G_sol = np.array([v.x for v in ret.dphidx_times_G])
            np.testing.assert_allclose(
                dphidx_times_G_sol,
                dphidx_times_G_expected.detach().numpy())
            l1_binary_sol = torch.tensor([v.x for v in l1_binary],
                                         dtype=self.dtype)
            l1_binary_times_RG_sol = [None] * dut.system.u_dim
            RG = R @ G
            for j in range(dut.system.u_dim):
                l1_binary_times_RG_sol[j] = l1_binary_sol * RG[:, j]
                np.testing.assert_allclose(
                    np.array([v.x for v in ret.l1_binary_times_RG[j]]),
                    l1_binary_times_RG_sol[j].detach().numpy())
            dl1dx = utils.l1_gradient(R @ (x_samples[i] - x_equilibrium))
            assert (dl1dx.shape[0] == 1)
            dVdx = dphidx[0][0] + V_lambda * dl1dx[0] @ R
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

    def test_compute_dVdx_times_G(self):
        dut = mut.ControlLyapunov(self.linear_system, self.lyapunov_relu1)
        G_flat_lo, G_flat_up = dut.system.compute_G_range_ia()
        R = torch.tensor([[1., 2.], [0., 1.], [1., -2.]], dtype=self.dtype)
        RG_lo, RG_up = dut._compute_RG_bounds_IA(R, G_flat_lo, G_flat_up)
        x_equilibrium = torch.tensor([0.5, -1.], dtype=self.dtype)
        V_lambda = 0.5
        self.compute_dVdx_times_G_tester(dut, x_equilibrium, R, G_flat_lo,
                                         G_flat_up, RG_lo, RG_up, V_lambda)

    def test_lyapunov_derivative_loss_at_samples_and_next_states(self):
        # Test with the linear system.
        dut = mut.ControlLyapunov(self.linear_system, self.lyapunov_relu1)
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
                                    subgradient_rule="max",
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


if __name__ == "__main__":
    unittest.main()
