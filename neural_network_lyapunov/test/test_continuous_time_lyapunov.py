import gurobipy
import numpy as np
import unittest
import torch
import torch.nn as nn

import os

import neural_network_lyapunov.lyapunov as lyapunov
import neural_network_lyapunov.continuous_time_lyapunov as\
    continuous_time_lyapunov
import neural_network_lyapunov.relu_to_optimization as relu_to_optimization
import neural_network_lyapunov.gurobi_torch_mip as gurobi_torch_mip
import neural_network_lyapunov.utils as utils
import neural_network_lyapunov.test.test_hybrid_linear_system as\
    test_hybrid_linear_system
import neural_network_lyapunov.test.test_lyapunov as test_lyapunov


class TestLyapunovContinuousTimeHybridSystem(unittest.TestCase):
    def setUp(self):
        self.dtype = torch.float64
        self.x_equilibrium1 = torch.tensor([0, 0], dtype=self.dtype)
        self.system1 = test_hybrid_linear_system.\
            setup_johansson_continuous_time_system1()
        self.x_equilibrium2 = torch.tensor([0, 0], dtype=self.dtype)
        self.system2 = test_hybrid_linear_system.\
            setup_johansson_continuous_time_system2()
        self.x_equilibrium3 = torch.tensor([1., 2.], dtype=self.dtype)
        self.system3 = test_hybrid_linear_system.\
            setup_johansson_continuous_time_system3(self.x_equilibrium3)

    def test_lyapunov_as_milp(self):
        """
        Test some cases caught in the wild. Instead of using Gurobi's
        feasibility tolerance as the active_constraint_tolerance, we have to
        relax the active constraint tolerance to match the objective in
        Gurobi.
        """
        system = test_hybrid_linear_system.\
            setup_johansson_continuous_time_system4()
        x_equilibrium = torch.tensor([0, 0], dtype=torch.float64)
        data_dir_path = os.path.dirname(os.path.realpath(__file__)) + "/data/"
        V_lambda = 0.1
        positivity_epsilon = 0.01
        derivative_epsilon = 0.01
        R = None
        for relu_model_data_file in \
                ("negative_loss_relu_1.pt", "negative_loss_relu_2.pt"):
            relu_model_data = torch.load(data_dir_path+relu_model_data_file)
            relu = utils.setup_relu(
                relu_model_data["linear_layer_width"], params=None,
                negative_slope=relu_model_data["negative_slope"],
                bias=relu_model_data["bias"], dtype=torch.float64)
            relu.load_state_dict(relu_model_data["state_dict"])
            dut = continuous_time_lyapunov.LyapunovContinuousTimeHybridSystem(
                system, relu)

            lyapunov_positivity_mip_return = dut.lyapunov_positivity_as_milp(
                x_equilibrium, V_lambda, positivity_epsilon, R=R, fixed_R=True)
            lyapunov_positivity_mip_return[0].gurobi_model.setParam(
                gurobipy.GRB.Param.OutputFlag, False)
            lyapunov_positivity_mip_return[0].gurobi_model.optimize()
            positivity_objective = lyapunov_positivity_mip_return[0].\
                compute_objective_from_mip_data_and_solution()
            self.assertAlmostEqual(
                positivity_objective.item(),
                lyapunov_positivity_mip_return[0].gurobi_model.ObjVal,
                places=3)

            lyapunov_derivative_mip_return = dut.lyapunov_derivative_as_milp(
                x_equilibrium, V_lambda, derivative_epsilon,
                lyapunov.ConvergenceEps.ExpLower, R=R, fixed_R=True)
            lyapunov_derivative_mip_return[0].gurobi_model.setParam(
                gurobipy.GRB.Param.OutputFlag, False)
            lyapunov_derivative_mip_return[0].gurobi_model.optimize()
            derivative_objective = lyapunov_derivative_mip_return[0].\
                compute_objective_from_mip_data_and_solution()
            self.assertAlmostEqual(
                derivative_objective.item(),
                lyapunov_derivative_mip_return[0].gurobi_model.ObjVal,
                places=3)

    def test_lyapunov_derivative(self):
        linear1 = nn.Linear(2, 3)
        linear1.weight.data = torch.tensor(
            [[1, 2], [3, 4], [5, 6]], dtype=self.dtype)
        linear1.bias.data = torch.tensor(
            [-1, 13, 4], dtype=self.dtype)
        linear2 = nn.Linear(3, 3)
        linear2.weight.data = torch.tensor(
            [[3, -2, -1], [1, -4, 0], [0, 1, -2]], dtype=self.dtype)
        linear2.bias.data = torch.tensor(
            [-11, 13, 48], dtype=self.dtype)
        linear3 = nn.Linear(3, 1)
        linear3.weight.data = torch.tensor([[1, 2, 3]], dtype=self.dtype)
        linear3.bias.data = torch.tensor([1], dtype=self.dtype)
        relu = nn.Sequential(linear1, nn.ReLU(), linear2, nn.ReLU(), linear3)

        V_lambda = 2.
        epsilon = 0.1
        R = None
        dut1 = continuous_time_lyapunov.LyapunovContinuousTimeHybridSystem(
            self.system1, relu)
        x = torch.tensor([0.5, 0.2], dtype=self.system1.dtype)
        lyapunov_derivatives = dut1.lyapunov_derivative(
            x, self.x_equilibrium1, V_lambda, epsilon, R=R)
        self.assertEqual(len(lyapunov_derivatives), 1)
        x.requires_grad = True
        V = dut1.lyapunov_value(x, self.x_equilibrium1, V_lambda, R=R)
        V.backward()
        activation_pattern = relu_to_optimization.\
            ComputeReLUActivationPattern(relu, x)
        g, _, _, _ = relu_to_optimization.ReLUGivenActivationPattern(
            relu, self.system1.x_dim, activation_pattern, self.dtype)
        xdot = self.system1.step_forward(x)
        Vdot = x.grad @ xdot
        self.assertAlmostEqual(
            lyapunov_derivatives[0].item(),
            (Vdot + epsilon * dut1.lyapunov_value(
                x, self.x_equilibrium1, V_lambda, R=R)).item())

        # x has multiple activation patterns.
        x = torch.tensor([0.5, 0.25], dtype=self.dtype)
        lyapunov_derivatives = dut1.lyapunov_derivative(
            x, self.x_equilibrium1, V_lambda, epsilon, R=R)
        self.assertEqual(len(lyapunov_derivatives), 2)
        activation_patterns = relu_to_optimization.\
            compute_all_relu_activation_patterns(relu, x)
        dReLU_dx_all = [relu_to_optimization.ReLUGivenActivationPattern(
            relu, self.system1.x_dim, pattern, self.dtype)[0] for pattern in
            activation_patterns]
        dVdx_all = [
            dReLU_dx.squeeze() +
            V_lambda * torch.tensor([1, 1], dtype=self.dtype) for dReLU_dx in
            dReLU_dx_all]
        xdot = self.system1.step_forward(x)
        Vdot_all = [dV_dx @ xdot for dV_dx in dVdx_all]
        V = dut1.lyapunov_value(x, self.x_equilibrium1, V_lambda, R=R)
        self.assertAlmostEqual(
            (lyapunov_derivatives[0]).item(),
            (Vdot_all[0] + epsilon * V).item())
        self.assertAlmostEqual(
            (lyapunov_derivatives[1]).item(),
            (Vdot_all[1] + epsilon * V).item())

        # The gradient of |R *(x-x*)|₁ has multiple possible gradients.
        x = torch.tensor([0.25, 0], dtype=self.dtype)
        lyapunov_derivatives = dut1.lyapunov_derivative(
            x, self.x_equilibrium1, V_lambda, epsilon, R=R)
        self.assertEqual(len(lyapunov_derivatives), 2)
        activation_pattern = relu_to_optimization.\
            ComputeReLUActivationPattern(relu, x)
        g, _, _, _ = relu_to_optimization.ReLUGivenActivationPattern(
            relu, self.system1.x_dim, activation_pattern, self.dtype)
        xdot = self.system1.step_forward(x)
        Vdot0 = g.squeeze() @ xdot + V_lambda * torch.tensor(
            [1, 1], dtype=self.dtype) @ xdot
        Vdot1 = g.squeeze() @ xdot + V_lambda * torch.tensor(
            [1, -1], dtype=self.dtype) @ xdot
        self.assertAlmostEqual(
            lyapunov_derivatives[0].item(),
            (Vdot0 + epsilon * dut1.lyapunov_value(
                x, self.x_equilibrium1, V_lambda, R=R)).item())
        self.assertAlmostEqual(
            lyapunov_derivatives[1].item(),
            (Vdot1 + epsilon * dut1.lyapunov_value(
                x, self.x_equilibrium1, V_lambda, R=R)).item())

        # x is on the boundary of the hybrid modes, and the gradient of
        # |R *(x-x*)|₁ has multiple values.
        x = torch.tensor([0., 0.1], dtype=self.dtype)
        lyapunov_derivatives = dut1.lyapunov_derivative(
            x, self.x_equilibrium1, V_lambda, epsilon, R=R)
        self.assertEqual(len(lyapunov_derivatives), 4)
        activation_pattern = relu_to_optimization.\
            ComputeReLUActivationPattern(relu, x)
        g, _, _, _ = relu_to_optimization.ReLUGivenActivationPattern(
            relu, self.system1.x_dim, activation_pattern, self.dtype)
        xdot_all = self.system1.possible_dx(x)
        dVdx0 = g.squeeze() + V_lambda * torch.tensor([1, 1], dtype=self.dtype)
        dVdx1 = g.squeeze() + V_lambda * \
            torch.tensor([-1, 1], dtype=self.dtype)
        Vdot = [None] * 4
        Vdot[0] = dVdx0 @ xdot_all[0]
        Vdot[1] = dVdx0 @ xdot_all[1]
        Vdot[2] = dVdx1 @ xdot_all[0]
        Vdot[3] = dVdx1 @ xdot_all[1]
        V = dut1.lyapunov_value(x, self.x_equilibrium1, V_lambda, R=R)
        for i in range(4):
            self.assertAlmostEqual(
                lyapunov_derivatives[i].item(),
                (Vdot[i] + epsilon * V).item())

    def test_add_relu_gradient_times_dynamics(self):
        """
        test add_relu_gradient_times_Aisi() and
        add_relu_gradient_times_gigammai()
        """
        lyapunov_relu1 = test_lyapunov.setup_relu(self.dtype)
        lyapunov_relu2 = test_lyapunov.setup_leaky_relu(self.dtype)

        def test_fun(relu, system, x_val, Aisi_flag):
            """
            Setup a MILP with fixed x, if Aisi_flag = True, solve
            ∑ᵢ ∂ReLU(x)/∂x * Aᵢsᵢ
            if Aisi_flag=False, solve
            ∑ᵢ ∂ReLU(x)/∂x * gᵢγᵢ
            """
            dut = continuous_time_lyapunov.LyapunovContinuousTimeHybridSystem(
                system, relu)
            milp = gurobi_torch_mip.GurobiTorchMILP(system.dtype)
            x = milp.addVars(
                system.x_dim, lb=-gurobipy.GRB.INFINITY,
                vtype=gurobipy.GRB.CONTINUOUS)
            s, gamma = dut.add_system_constraint(milp, x, None)
            (_, beta, _, _) = dut.add_relu_output_constraint(milp, x)
            if Aisi_flag:
                (z, cost_z_coeff) = dut.add_relu_gradient_times_Aisi(
                    milp, s, beta)
            else:
                (z, cost_z_coeff) = dut.add_relu_gradient_times_gigammai(
                    milp, gamma, beta)
            for i in range(system.x_dim):
                milp.addLConstr(
                    [torch.tensor([1.], dtype=system.dtype)], [[x[i]]],
                    rhs=x_val[i], sense=gurobipy.GRB.EQUAL)
            milp.setObjective(
                cost_z_coeff, z, 0., gurobipy.GRB.MAXIMIZE)
            milp.gurobi_model.setParam(gurobipy.GRB.Param.OutputFlag, False)
            milp.gurobi_model.optimize()
            if (milp.gurobi_model.status != gurobipy.GRB.OPTIMAL):
                milp.gurobi_model.computeIIS()
                milp.gurobi_model.write("milp.ilp")
            self.assertEqual(milp.gurobi_model.status, gurobipy.GRB.OPTIMAL)
            mode_index = system.mode(x_val)
            activation_pattern = relu_to_optimization.\
                ComputeReLUActivationPattern(relu, x_val)
            (g, _, _, _) = relu_to_optimization.ReLUGivenActivationPattern(
                relu, system.x_dim, activation_pattern, system.dtype)
            if Aisi_flag:
                self.assertAlmostEqual(
                    (g.squeeze() @ (system.A[mode_index] @ x_val)).item(),
                    milp.gurobi_model.ObjVal)
            else:
                self.assertAlmostEqual(
                    (g.squeeze() @ (system.g[mode_index])).item(),
                    milp.gurobi_model.ObjVal)

        for relu in (lyapunov_relu1, lyapunov_relu2):
            for Aisi_flag in (True, False):
                for system in (self.system1, self.system2):
                    test_fun(
                        relu, system,
                        torch.tensor([0.5, 0.2], dtype=self.dtype), Aisi_flag)
                    test_fun(
                        relu, system,
                        torch.tensor([-0.5, 0.2], dtype=self.dtype), Aisi_flag)
                    test_fun(
                        relu, system,
                        torch.tensor([0.5, -0.2], dtype=self.dtype), Aisi_flag)
                    test_fun(
                        relu, system,
                        torch.tensor([-0.5, -0.2], dtype=self.dtype),
                        Aisi_flag)
                test_fun(
                    relu, self.system3,
                    torch.tensor([0.5, 0.3], dtype=self.dtype) +
                    self.x_equilibrium3, Aisi_flag)
                test_fun(
                    relu, self.system3,
                    torch.tensor([-1.5, 0.3], dtype=self.dtype) +
                    self.x_equilibrium3, Aisi_flag)
                test_fun(
                    relu, self.system3,
                    torch.tensor([0.5, -0.3], dtype=self.dtype) +
                    self.x_equilibrium3, Aisi_flag)
                test_fun(
                    relu, self.system3,
                    torch.tensor([-0.5, -0.3], dtype=self.dtype) +
                    self.x_equilibrium3, Aisi_flag)

    def test_add_relu_gradient_times_xdot(self):
        lyapunov_relu1 = test_lyapunov.setup_relu(self.dtype)
        lyapunov_relu2 = test_lyapunov.setup_leaky_relu(self.dtype)

        def test_fun(relu, system, x_val):
            """
            Setup a MILP with fixed x, solve
            ∂ReLU(x)/∂x * ẋ
            """
            dut = continuous_time_lyapunov.LyapunovContinuousTimeHybridSystem(
                system, relu)
            milp = gurobi_torch_mip.GurobiTorchMILP(system.dtype)
            x = milp.addVars(
                system.x_dim, lb=-gurobipy.GRB.INFINITY,
                vtype=gurobipy.GRB.CONTINUOUS)
            xdot = milp.addVars(
                system.x_dim, lb=-gurobipy.GRB.INFINITY,
                vtype=gurobipy.GRB.CONTINUOUS)
            s, gamma = dut.add_system_constraint(milp, x, xdot)
            (_, beta, _, _) = dut.add_relu_output_constraint(milp, x)
            (z, cost_z_coeff) = dut.add_relu_gradient_times_xdot(
                milp, xdot, beta,
                system.dx_lower, system.dx_upper)
            milp.addMConstrs(
                [torch.eye(system.x_dim, dtype=system.dtype)], [x],
                sense=gurobipy.GRB.EQUAL, b=x_val)
            milp.setObjective(
                [cost_z_coeff], [z], 0., gurobipy.GRB.MAXIMIZE)
            milp.gurobi_model.setParam(gurobipy.GRB.Param.OutputFlag, False)
            milp.gurobi_model.optimize()
            if (milp.gurobi_model.status != gurobipy.GRB.OPTIMAL):
                milp.gurobi_model.computeIIS()
                milp.gurobi_model.write("milp.ilp")
            self.assertEqual(milp.gurobi_model.status, gurobipy.GRB.OPTIMAL)
            mode_index = system.mode(x_val)
            activation_pattern = relu_to_optimization.\
                ComputeReLUActivationPattern(relu, x_val)
            (g, _, _, _) = relu_to_optimization.ReLUGivenActivationPattern(
                relu, system.x_dim, activation_pattern, system.dtype)
            self.assertAlmostEqual(
                (g.squeeze() @ system.step_forward(x_val, mode_index)).
                item(), milp.gurobi_model.ObjVal, places=5)

        for relu in (lyapunov_relu1, lyapunov_relu2):
            for system in (self.system1, self.system2):
                test_fun(
                    relu, system,
                    torch.tensor([0.5, 0.2], dtype=self.dtype))
                test_fun(
                    relu, system,
                    torch.tensor([-0.5, 0.2], dtype=self.dtype))
                test_fun(
                    relu, system,
                    torch.tensor([0.5, -0.2], dtype=self.dtype))
                test_fun(
                    relu, system,
                    torch.tensor([-0.5, -0.2], dtype=self.dtype))
            test_fun(
                relu, self.system3,
                torch.tensor([0.5, 0.3], dtype=self.dtype) +
                self.x_equilibrium3)
            test_fun(
                relu, self.system3,
                torch.tensor([-1.5, 0.3], dtype=self.dtype) +
                self.x_equilibrium3)
            test_fun(
                relu, self.system3,
                torch.tensor([0.5, -0.3], dtype=self.dtype) +
                self.x_equilibrium3)
            test_fun(
                relu, self.system3,
                torch.tensor([-0.5, -0.3], dtype=self.dtype) +
                self.x_equilibrium3)

    def test_add_sign_state_error_times_dynamics(self):
        """
        test add_sign_state_error_times_Aisi() and
        add_sign_state_error_times_gigammai()
        """
        relu = test_lyapunov.setup_leaky_relu(self.dtype)

        def test_fun(system, x_equilibrium, x_val, Aisi_flag):
            dut = continuous_time_lyapunov.LyapunovContinuousTimeHybridSystem(
                system, relu)
            milp = gurobi_torch_mip.GurobiTorchMILP(system.dtype)
            x = milp.addVars(
                system.x_dim, lb=-gurobipy.GRB.INFINITY,
                vtype=gurobipy.GRB.CONTINUOUS)
            s, gamma = dut.add_system_constraint(milp, x, None)
            (_, alpha) = dut.add_state_error_l1_constraint(
                milp, x_equilibrium, x)
            if Aisi_flag:
                (z, z_coeff, s_coeff) = dut.add_sign_state_error_times_Aisi(
                    milp, s, alpha)
            else:
                (z, z_coeff, gamma_coeff) = dut.\
                    add_sign_state_error_times_gigammai(milp, gamma, alpha)
            self.assertEqual(len(z), system.num_modes)
            for i in range(system.x_dim):
                milp.addLConstr(
                    [torch.tensor([1.], dtype=system.dtype)], [[x[i]]],
                    sense=gurobipy.GRB.EQUAL, rhs=x_val[i])
            milp.gurobi_model.setParam(gurobipy.GRB.Param.OutputFlag, 0)
            milp.gurobi_model.optimize()
            self.assertEqual(milp.gurobi_model.status, gurobipy.GRB.OPTIMAL)
            alpha_expected = torch.empty(system.x_dim, dtype=system.dtype)
            for i in range(system.x_dim):
                if x_val[i] >= x_equilibrium[i]:
                    alpha_expected[i] = 1
                else:
                    alpha_expected[i] = 0
            np.testing.assert_allclose(
                np.array([v.x for v in alpha]),
                alpha_expected.detach().numpy())
            z_sol = [None] * system.num_modes
            s_sol = torch.tensor([v.x for v in s], dtype=system.dtype)
            gamma_sol = torch.tensor([v.x for v in gamma], dtype=system.dtype)
            for i in range(system.num_modes):
                z_sol[i] = torch.tensor(
                    [v.x for v in z[i]], dtype=system.dtype)
                if Aisi_flag:
                    np.testing.assert_allclose(
                        z_sol[i].detach().numpy(),
                        (system.A[i] @ s_sol[
                            i*system.x_dim: (i+1) * system.x_dim])
                        * alpha_expected)
                else:
                    np.testing.assert_allclose(
                        z_sol[i].detach().numpy(),
                        (system.g[i] * gamma_sol[i]) * alpha_expected)
            mode_idx = system.mode(x_val)
            gamma_expected = torch.zeros(system.num_modes, dtype=system.dtype)
            gamma_expected[mode_idx] = 1
            np.testing.assert_allclose(
                gamma_sol.detach().numpy(), gamma_expected.detach().numpy())
            if Aisi_flag:
                self.assertAlmostEqual(
                    torch.sum(torch.sign(x_val - x_equilibrium) *
                              (system.A[mode_idx] @ x_val)).item(),
                    (torch.cat(z_coeff) @ torch.cat(z_sol) +
                     torch.cat(s_coeff) @ s_sol).item())
            else:
                self.assertAlmostEqual(
                    torch.sum(torch.sign(x_val - x_equilibrium) *
                              (system.g[mode_idx])).item(),
                    (torch.cat(z_coeff) @ torch.cat(z_sol) +
                     torch.cat(gamma_coeff) @ gamma_sol).item())

        for Aisi_flag in (True, False):
            test_fun(
                self.system1, self.x_equilibrium1,
                torch.tensor([0.2, -0.5], dtype=self.system1.dtype),
                Aisi_flag)
            test_fun(
                self.system1, self.x_equilibrium1,
                torch.tensor([-0.2, -0.5], dtype=self.system1.dtype),
                Aisi_flag)
            test_fun(
                self.system1, self.x_equilibrium1,
                torch.tensor([0.2, 0.5], dtype=self.system1.dtype), Aisi_flag)
            test_fun(
                self.system2, self.x_equilibrium2,
                torch.tensor([0.2, -0.5], dtype=self.system2.dtype),
                Aisi_flag)
            test_fun(
                self.system2, self.x_equilibrium2,
                torch.tensor([-0.2, -0.5], dtype=self.system2.dtype),
                Aisi_flag)
            test_fun(
                self.system2, self.x_equilibrium2,
                torch.tensor([0.2, 0.5], dtype=self.system2.dtype), Aisi_flag)
            test_fun(
                self.system3, self.x_equilibrium3,
                torch.tensor([-1.2, -0.5], dtype=self.system3.dtype) +
                self.x_equilibrium3, Aisi_flag)
            test_fun(
                self.system3, self.x_equilibrium3,
                torch.tensor([0.2, -0.5], dtype=self.system3.dtype) +
                self.x_equilibrium3, Aisi_flag)
            test_fun(
                self.system3, self.x_equilibrium3,
                torch.tensor([1.2, 0.5], dtype=self.system3.dtype) +
                self.x_equilibrium3, Aisi_flag)

    def test_add_sign_state_error_times_xdot(self):
        relu = test_lyapunov.setup_leaky_relu(self.dtype)

        def test_fun(system, x_equilibrium, x_val):
            dut = continuous_time_lyapunov.LyapunovContinuousTimeHybridSystem(
                system, relu)
            milp = gurobi_torch_mip.GurobiTorchMILP(system.dtype)
            x = milp.addVars(
                system.x_dim, lb=-gurobipy.GRB.INFINITY,
                vtype=gurobipy.GRB.CONTINUOUS)
            xdot = milp.addVars(
                system.x_dim, lb=-gurobipy.GRB.INFINITY,
                vtype=gurobipy.GRB.CONTINUOUS)
            s, gamma = dut.add_system_constraint(milp, x, xdot)
            (_, alpha) = dut.add_state_error_l1_constraint(
                milp, x_equilibrium, x)
            (z, z_coeff, xdot_coeff) = dut.add_sign_state_error_times_xdot(
                milp, xdot, alpha, system.dx_lower, system.dx_upper)
            milp.addMConstrs(
                [torch.eye(system.x_dim, dtype=system.dtype)], [x],
                sense=gurobipy.GRB.EQUAL, b=x_val)
            milp.gurobi_model.setParam(gurobipy.GRB.Param.OutputFlag, 0)
            milp.gurobi_model.optimize()
            self.assertEqual(milp.gurobi_model.status, gurobipy.GRB.OPTIMAL)
            alpha_expected = torch.empty(system.x_dim, dtype=system.dtype)
            for i in range(system.x_dim):
                if x_val[i] >= x_equilibrium[i]:
                    alpha_expected[i] = 1
                else:
                    alpha_expected[i] = 0
            np.testing.assert_allclose(
                np.array([v.x for v in alpha]),
                alpha_expected.detach().numpy())
            gamma_sol = torch.tensor([v.x for v in gamma], dtype=system.dtype)
            z_sol = torch.tensor([v.x for v in z], dtype=system.dtype)
            xdot_val = system.step_forward(x_val)
            np.testing.assert_allclose(
                z_sol.detach().numpy(),
                (xdot_val * alpha_expected).detach().numpy())
            mode_idx = system.mode(x_val)
            gamma_expected = torch.zeros(system.num_modes, dtype=system.dtype)
            gamma_expected[mode_idx] = 1
            np.testing.assert_allclose(
                gamma_sol.detach().numpy(), gamma_expected.detach().numpy())
            self.assertAlmostEqual(
                (torch.sign(x_val - x_equilibrium) @ xdot_val).item(),
                (z_coeff @ z_sol + xdot_coeff @ xdot_val).item())

        test_fun(
            self.system1, self.x_equilibrium1,
            torch.tensor([0.2, -0.5], dtype=self.system1.dtype))
        test_fun(
            self.system1, self.x_equilibrium1,
            torch.tensor([-0.2, -0.5], dtype=self.system1.dtype))
        test_fun(
            self.system1, self.x_equilibrium1,
            torch.tensor([0.2, 0.5], dtype=self.system1.dtype))
        test_fun(
            self.system2, self.x_equilibrium2,
            torch.tensor([0.2, -0.5], dtype=self.system2.dtype))
        test_fun(
            self.system2, self.x_equilibrium2,
            torch.tensor([-0.2, -0.5], dtype=self.system2.dtype))
        test_fun(
            self.system2, self.x_equilibrium2,
            torch.tensor([0.2, 0.5], dtype=self.system2.dtype))
        test_fun(
            self.system3, self.x_equilibrium3,
            torch.tensor([-1.2, -0.5], dtype=self.system3.dtype) +
            self.x_equilibrium3)
        test_fun(
            self.system3, self.x_equilibrium3,
            torch.tensor([0.2, -0.5], dtype=self.system3.dtype) +
            self.x_equilibrium3)
        test_fun(
            self.system3, self.x_equilibrium3,
            torch.tensor([1.2, 0.5], dtype=self.system3.dtype) +
            self.x_equilibrium3)

    def test_lyapunov_derivative_as_milp(self):
        """
        Test both lyapunov_derivative_as_milp and lyapunov_derivative_as_milp2
        """
        lyapunov_relu1 = test_lyapunov.setup_relu(self.dtype)
        lyapunov_relu2 = test_lyapunov.setup_leaky_relu(self.dtype)
        V_lambda = 2.
        epsilon = 0.1
        R = None

        def test_fun(relu, x_equilibrium, system, x_val, formulation):
            dut = continuous_time_lyapunov.LyapunovContinuousTimeHybridSystem(
                system, relu)
            if formulation == 1:
                (milp, x, beta, gamma) = dut.lyapunov_derivative_as_milp(
                    x_equilibrium, V_lambda, epsilon,
                    lyapunov.ConvergenceEps.ExpLower, R=R, fixed_R=True,
                    lyapunov_lower=None, lyapunov_upper=None)
            elif formulation == 2:
                (milp, x, beta, gamma) = dut.lyapunov_derivative_as_milp2(
                    x_equilibrium, V_lambda, epsilon,
                    lyapunov.ConvergenceEps.ExpLower, R=R, fixed_R=True,
                    lyapunov_lower=None, lyapunov_upper=None)
            for i in range(system.x_dim):
                milp.addLConstr(
                    [torch.tensor([1.], dtype=system.dtype)], [[x[i]]],
                    sense=gurobipy.GRB.EQUAL, rhs=x_val[i])
            milp.gurobi_model.setParam(gurobipy.GRB.Param.OutputFlag, False)
            milp.gurobi_model.setParam(gurobipy.GRB.Param.DualReductions, 0)
            milp.gurobi_model.optimize()
            if (milp.gurobi_model.status == gurobipy.GRB.Status.INFEASIBLE):
                milp.gurobi_model.computeIIS()
                milp.gurobi_model.write("milp.ilp")

            self.assertEqual(milp.gurobi_model.status,
                             gurobipy.GRB.Status.OPTIMAL)
            x_val.requires_grad = True
            V = dut.lyapunov_value(x_val, x_equilibrium, V_lambda, R=R)
            V.backward()
            Vdot = x_val.grad @ system.step_forward(x_val)
            self.assertAlmostEqual(
                milp.gurobi_model.ObjVal,
                (Vdot + epsilon * V).squeeze().item())

        for formulation in (1, 2):
            for relu in [lyapunov_relu1, lyapunov_relu2]:
                for (system, x_equilibrium) in zip(
                    (self.system1, self.system2),
                        (self.x_equilibrium1, self.x_equilibrium2)):
                    test_fun(
                        relu, x_equilibrium, system,
                        torch.tensor([0.2, 0.3], dtype=self.dtype),
                        formulation)
                    test_fun(
                        relu, x_equilibrium, system,
                        torch.tensor([0.2, -0.3], dtype=self.dtype),
                        formulation)
                    test_fun(
                        relu, x_equilibrium, system,
                        torch.tensor([-0.2, -0.3], dtype=self.dtype),
                        formulation)
                test_fun(
                    relu, self.x_equilibrium3, self.system3,
                    torch.tensor([1.2, 0.5], dtype=self.dtype) +
                    self.x_equilibrium3, formulation)
                test_fun(
                    relu, self.x_equilibrium3, self.system3,
                    torch.tensor([-.5, 0.5], dtype=self.dtype) +
                    self.x_equilibrium3, formulation)
                test_fun(
                    relu, self.x_equilibrium3, self.system3,
                    torch.tensor([-1.5, 0.5], dtype=self.dtype) +
                    self.x_equilibrium3, formulation)

    def test_lyapunov_derivative_as_milp_gradient(self):
        """
        Test if we can compute the gradient of maxₓ V̇+εV w.r.t the ReLU
        network weights/bias.
        """
        V_lambda = 2.
        epsilon = 0.1
        R = None

        def compute_milp_cost_given_relu(
                system, relu_index, x_equilibrium, params_val, formulation,
                requires_grad):
            if relu_index == 1:
                relu = test_lyapunov.setup_relu(system.dtype, params_val)
            elif relu_index == 2:
                relu = test_lyapunov.setup_leaky_relu(system.dtype, params_val)
            dut = continuous_time_lyapunov.LyapunovContinuousTimeHybridSystem(
                system, relu)
            if formulation == 1:
                milp = dut.lyapunov_derivative_as_milp(
                    x_equilibrium, V_lambda, epsilon,
                    lyapunov.ConvergenceEps.ExpLower, R=R, fixed_R=True,
                    lyapunov_lower=None, lyapunov_upper=None)[0]
            elif formulation == 2:
                milp = dut.lyapunov_derivative_as_milp2(
                    x_equilibrium, V_lambda, epsilon,
                    lyapunov.ConvergenceEps.ExpLower, R=R, fixed_R=True,
                    lyapunov_lower=None, lyapunov_upper=None)[0]
            milp.gurobi_model.setParam(gurobipy.GRB.Param.OutputFlag, False)
            milp.gurobi_model.optimize()
            if requires_grad:
                objective = milp.compute_objective_from_mip_data_and_solution(
                    penalty=0.)
                objective.backward()
                grad = np.concatenate(
                    [p.grad.detach().numpy().reshape((-1,)) for p in
                     relu.parameters()], axis=0)
                return grad
            else:
                return milp.gurobi_model.ObjVal

        def test_fun(
                system, relu_index, x_equilibrium, params_val, formulation):
            grad = compute_milp_cost_given_relu(
                system, relu_index, x_equilibrium, params_val, formulation,
                True)
            grad_numerical = utils.compute_numerical_gradient(
                lambda p: compute_milp_cost_given_relu(
                    system, relu_index, x_equilibrium, torch.from_numpy(p),
                    formulation, False), params_val, dx=1e-7)
            np.testing.assert_allclose(
                grad, grad_numerical, atol=1.2e-5, rtol=1e-6)

        relu_params = []
        relu_params.append(torch.tensor([
            0.5, 1.2, 0.3, 2.5, 4.2, -1.5, -3.4, -12.1, 20.5, 32.1, 0.5, -.85,
            0.33, 0.345, 0.54, 0.325, 0.8, 2.4, 1.5, 4.2, 1.4, 2.3, 4.1, -4.5,
            3.2, 0.5, 4.6, 12.4, -0.05, 0.2], dtype=self.dtype))
        relu_params.append(torch.tensor([
            2.5, 1.7, 4.1, 3.1, -5.7, 1.7, 3.8, 12.8, -23.5, -2.1, 2.9, 1.85,
            -.33, 2.645, 2.54, 3.295, 2.8, -.4, -.5, 2.2, 3.4, 1.3, 3.1, 1.5,
            3.4, 0.9, 2.9, 1.2, -0.45, 0.8], dtype=self.dtype))
        for formulation in (1, 2):
            for relu_param in relu_params:
                for relu_index in (1, 2):
                    for system, x_equilibrium in (
                        (self.system1, self.x_equilibrium1),
                        (self.system2, self.x_equilibrium2),
                            (self.system3, self.x_equilibrium3)):
                        test_fun(
                            system, relu_index, x_equilibrium, relu_param,
                            formulation)

    def compute_lyapunov_derivative_loss_single_sample(
        self, system, V_lambda, epsilon, x_equilibrium, R, relu, x_val,
            eps_type, margin):
        xdot = system.step_forward(x_val)
        activation_pattern = relu_to_optimization.\
            ComputeReLUActivationPattern(relu, x_val)
        (g, _, _, _) = relu_to_optimization.ReLUGivenActivationPattern(
            relu, system.x_dim, activation_pattern, system.dtype)
        dVdx = g.squeeze() + \
            V_lambda * torch.sign(x_val - x_equilibrium)
        Vdot = dVdx @ xdot
        dut = continuous_time_lyapunov.LyapunovContinuousTimeHybridSystem(
            system, relu)
        V = dut.lyapunov_value(
            x_val, x_equilibrium, V_lambda, R=R,
            relu_at_equilibrium=relu.forward(x_equilibrium))
        if eps_type == lyapunov.ConvergenceEps.ExpLower:
            loss_expected = torch.max(
                Vdot + epsilon * V + margin,
                torch.tensor(0., dtype=system.dtype))
        elif eps_type == lyapunov.ConvergenceEps.ExpUpper:
            loss_expected = torch.max(
                -(Vdot + epsilon * V) + margin,
                torch.tensor(0., dtype=system.dtype))
        elif eps_type == lyapunov.ConvergenceEps.Asymp:
            loss_expected = torch.max(
                Vdot + epsilon * torch.norm(x_val - x_equilibrium, p=1) +
                margin, torch.tensor(0., dtype=system.dtype))
        return loss_expected

    def test_lyapunov_derivative_loss_at_samples(self):
        V_lambda = 2.
        epsilon = 0.2
        R = None
        relu = test_lyapunov.setup_leaky_relu(self.dtype)

        lyapunov_relu1 = test_lyapunov.setup_relu(torch.float64)
        lyapunov_relu2 = test_lyapunov.setup_leaky_relu(torch.float64)
        for relu in (lyapunov_relu1, lyapunov_relu2):
            for system, x_equilibrium in (
                (self.system1, self.x_equilibrium1),
                    (self.system2, self.x_equilibrium2)):
                dut = continuous_time_lyapunov.\
                    LyapunovContinuousTimeHybridSystem(system, relu)
                margin = 0.1
                for eps_type in list(lyapunov.ConvergenceEps):
                    # First compute the loss for each single point.
                    for pt in ([0.1, 0.3], [-0.2, 0.4], [0.5, 0.6],
                               [0.1, -0.4]):
                        loss_expected = self.\
                            compute_lyapunov_derivative_loss_single_sample(
                                system, V_lambda, epsilon, x_equilibrium, R,
                                relu, torch.tensor(pt, dtype=torch.float64),
                                eps_type, margin=margin)
                        loss = dut.lyapunov_derivative_loss_at_samples(
                            V_lambda, epsilon,
                            torch.tensor([pt], dtype=torch.float64),
                            x_equilibrium, eps_type, R=R, margin=margin)
                        self.assertAlmostEqual(
                            loss_expected.item(), loss.item())

                    # Now compute the loss for a batch of points.
                    points = torch.tensor([
                        [0.1, 0.3], [-0.2, 0.4], [0.5, 0.6], [0.1, -0.4]],
                        dtype=system.dtype)
                    relu.zero_grad()
                    loss = dut.lyapunov_derivative_loss_at_samples(
                        V_lambda, epsilon, points, x_equilibrium, eps_type,
                        R=R, margin=margin)
                    loss.backward()
                    loss_grad = [p.grad.data.clone() for p in
                                 relu.parameters()]

                    relu.zero_grad()
                    loss_expected = 0
                    for i in range(points.shape[0]):
                        loss_expected += self.\
                            compute_lyapunov_derivative_loss_single_sample(
                                system, V_lambda, epsilon, x_equilibrium, R,
                                relu, points[i], eps_type, margin)
                    loss_expected = loss_expected / points.shape[0]
                    loss_expected.backward()
                    loss_grad_expected = [
                        p.grad.data.clone() for p in relu.parameters()]
                    self.assertAlmostEqual(loss.item(), loss_expected.item())
                    for i in range(len(loss_grad)):
                        np.testing.assert_allclose(
                            loss_grad[i].detach().numpy(),
                            loss_grad_expected[i].detach().numpy(), atol=1e-15)

    def test_lyapunov_derivative_loss_at_samples_gradient(self):
        # Check the gradient of lyapunov_derivative_loss_at_samples computed
        # from autodiff and numerical gradient
        V_lambda = 0.1
        epsilon = 0.01
        R = None

        def compute_loss_batch_sample(
                relu_params, points_test, eps_type, requires_grad):
            margin = 0.1
            if isinstance(relu_params, np.ndarray):
                relu_params_torch = torch.from_numpy(relu_params)
            else:
                relu_params_torch = relu_params
            relu_test = test_lyapunov.setup_leaky_relu(
                torch.float64, relu_params_torch)
            if requires_grad:
                dut = continuous_time_lyapunov.\
                    LyapunovContinuousTimeHybridSystem(self.system1, relu_test)
                loss = dut.lyapunov_derivative_loss_at_samples(
                    V_lambda, epsilon, points_test,
                    self.x_equilibrium1, eps_type, R=R, margin=margin)
                loss.backward()
                grad = utils.extract_relu_parameters_grad(relu_test)
                return grad
            else:
                with torch.no_grad():
                    loss = 0
                    for i in range(points_test.shape[0]):
                        loss += self.\
                            compute_lyapunov_derivative_loss_single_sample(
                                self.system1, V_lambda, epsilon,
                                self.x_equilibrium1, R, relu_test, points[i],
                                eps_type, margin)
                    loss = loss / points_test.shape[0]
                    return np.array([loss.item()])
        relu_params = []
        relu_params.append(torch.tensor([
            0.1, 0.2, -0.3, 0.15, 0.32, 1.5, 0.45, 2.35, 0.35, 4.09, -2.14,
            -4.51, 0.99, 1.5, 4.2, 3.1, 0.45, 0.09, 5.43, 2.35, 0.34, 0.44,
            5.42, -3.43, -4.51, -4.53, 2.09, 4.90, 0.55, 6.5],
            dtype=torch.float64, requires_grad=True))
        relu_params.append(torch.tensor([
            2.1, 2.2, -4.3, -0.45, 0.32, 1.9, 2.43, 2.45, 1.35, 4.09, -2.14,
            -14.51, 3.91, 1.9, 2.2, 3.1, 0.45, -0.9, -3.43, -2.35, 2.31, 2.41,
            -5.42, -13.43, -4.11, 3.53, -2.59, -4.9, 0.55, -3.5],
            dtype=torch.float64, requires_grad=True))
        points = torch.tensor([
            [0.1, 0.3], [-0.2, 0.4], [0.5, 0.6], [0.1, -0.4]],
            dtype=torch.float64)

        for relu_param in relu_params:
            for eps_type in list(lyapunov.ConvergenceEps):
                grad = compute_loss_batch_sample(
                    relu_param, points, eps_type, True)
                grad_numerical = utils.compute_numerical_gradient(
                    lambda p: compute_loss_batch_sample(
                        p, points, eps_type, False),
                    relu_param.detach().numpy())
                np.testing.assert_allclose(
                    grad, grad_numerical.squeeze(), atol=2e-7)


if __name__ == "__main__":
    unittest.main()
