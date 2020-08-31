import gurobipy
import numpy as np
import unittest
import torch
import torch.nn as nn

import os

import robust_value_approx.lyapunov as lyapunov
import robust_value_approx.relu_to_optimization as relu_to_optimization
import robust_value_approx.gurobi_torch_mip as gurobi_torch_mip
import robust_value_approx.hybrid_linear_system as hybrid_linear_system
import robust_value_approx.relu_system as relu_system
import robust_value_approx.utils as utils
import robust_value_approx.test.test_hybrid_linear_system as\
    test_hybrid_linear_system


def setup_relu(dtype, params=None):
    # Construct a simple ReLU model with 2 hidden layers
    # params is the value of weights/bias after concatenation.
    if params is not None:
        assert(isinstance(params, torch.Tensor))
        assert(params.shape == (30,))
    linear1 = nn.Linear(2, 3)
    if params is None:
        linear1.weight.data = torch.tensor([[1, 2], [3, 4], [5, 6]],
                                           dtype=dtype)
        linear1.bias.data = torch.tensor([-11, 10, 5], dtype=dtype)
    else:
        linear1.weight.data = params[:6].clone().reshape((3, 2))
        linear1.bias.data = params[6:9].clone()
    linear2 = nn.Linear(3, 4)
    if params is None:
        linear2.weight.data = torch.tensor(
                [[-1, -0.5, 1.5], [2, 5, 6], [-2, -3, -4], [1.5, 4, 6]],
                dtype=dtype)
        linear2.bias.data = torch.tensor([-3, 2, 0.7, 1.5], dtype=dtype)
    else:
        linear2.weight.data = params[9:21].clone().reshape((4, 3))
        linear2.bias.data = params[21:25].clone()
    linear3 = nn.Linear(4, 1)
    if params is None:
        linear3.weight.data = torch.tensor([[4, 5, 6, 7]], dtype=dtype)
        linear3.bias.data = torch.tensor([-9], dtype=dtype)
    else:
        linear3.weight.data = params[25:29].clone().reshape((1, 4))
        linear3.bias.data = params[29].clone().reshape((1))
    relu1 = nn.Sequential(
        linear1, nn.ReLU(), linear2, nn.ReLU(), linear3, nn.ReLU())
    assert(not relu1.forward(torch.tensor([0, 0], dtype=dtype)).item() == 0)
    return relu1


def setup_leaky_relu(dtype, params=None, bias=True):
    if params is not None and bias:
        assert(isinstance(params, torch.Tensor))
        assert(params.shape == (30,))
    linear1 = nn.Linear(2, 3, bias=bias)
    param_count = 0
    if params is None:
        linear1.weight.data = torch.tensor(
            [[-1.3405, -0.2602], [-0.9392, 0.9033], [-2.1063, 1.3141]],
            dtype=dtype)
        if bias:
            linear1.bias.data = torch.tensor(
                [0.913, 0.6429, 0.0011], dtype=dtype)
    else:
        linear1.weight.data = params[:6].clone().reshape((3, 2))
        param_count += 6
        if bias:
            linear1.bias.data = params[param_count: param_count + 3].clone()
            param_count += 3
    linear2 = nn.Linear(3, 4, bias=bias)
    if params is None:
        linear2.weight.data = torch.tensor(
            [[-0.4209, -1.1947, 1.4353], [1.7519, -1.3908, 2.6274],
             [-2.7574, 0.3764, -0.5544], [-0.3721, -1.0413, 0.52]],
            dtype=dtype)
        if bias:
            linear2.bias.data = torch.tensor(
                [-0.9802, 1.1129, 1.0941, 1.582], dtype=dtype)
    else:
        linear2.weight.data = params[param_count: param_count + 12].clone().\
            reshape((4, 3))
        param_count += 12
        if bias:
            linear2.bias.data = params[param_count: param_count + 4].clone()
            param_count += 4
    linear3 = nn.Linear(4, 1)
    if params is None:
        linear3.weight.data = torch.tensor(
            [[-1.1727, 0.2846, 1.2452, 0.8230]], dtype=dtype)
        linear3.bias.data = torch.tensor([0.4431], dtype=dtype)
    else:
        linear3.weight.data = params[param_count:param_count + 4].clone().\
            reshape((1, 4))
        param_count += 4
        if bias:
            linear3.bias.data = params[param_count].clone().reshape((1))
            param_count += 1
    relu = nn.Sequential(
        linear1, nn.LeakyReLU(0.1), linear2, nn.LeakyReLU(0.1), linear3)
    return relu


def setup_relu_dyn(dtype, params=None):
    # Construct a simple ReLU model with 2 hidden layers
    # params is the value of weights/bias after concatenation.
    # the network has the same number of outputs as inputs (2)
    if params is not None:
        assert(isinstance(params, torch.Tensor))
        assert(params.shape == (35,))
    linear1 = nn.Linear(2, 3)
    if params is None:
        linear1.weight.data = torch.tensor([[1, 2], [3, 4], [5, 6]],
                                           dtype=dtype)
        linear1.bias.data = torch.tensor([-11, 10, 5], dtype=dtype)
    else:
        linear1.weight.data = params[:6].clone().reshape((3, 2))
        linear1.bias.data = params[6:9].clone()
    linear2 = nn.Linear(3, 4)
    if params is None:
        linear2.weight.data = torch.tensor(
                [[-1, -0.5, 1.5], [2, 5, 6], [-2, -3, -4], [1.5, 4, 6]],
                dtype=dtype)
        linear2.bias.data = torch.tensor([-3, 2, 0.7, 1.5], dtype=dtype)
    else:
        linear2.weight.data = params[9:21].clone().reshape((4, 3))
        linear2.bias.data = params[21:25].clone()
    linear3 = nn.Linear(4, 2)
    if params is None:
        linear3.weight.data = torch.tensor([[4, 5, 6, 7], [8, 7, 5.5, 4.5]],
                                           dtype=dtype)
        linear3.bias.data = torch.tensor([-9, 3], dtype=dtype)
    else:
        linear3.weight.data = params[25:33].clone().reshape((2, 4))
        linear3.bias.data = params[33:35].clone().reshape((2))
    relu1 = nn.Sequential(
        linear1, nn.ReLU(), linear2, nn.ReLU(), linear3)
    assert(not relu1.forward(torch.tensor([0, 0], dtype=dtype))[0].item() == 0)
    assert(not relu1.forward(torch.tensor([0, 0], dtype=dtype))[1].item() == 0)
    return relu1


class TestLyapunovHybridSystem(unittest.TestCase):
    def setUp(self):
        self.dtype = torch.float64
        self.x_equilibrium1 = torch.tensor([0, 0], dtype=self.dtype)
        self.system1 = test_hybrid_linear_system.\
            setup_trecate_discrete_time_system()
        self.theta2 = np.pi / 5
        cos_theta2 = np.cos(self.theta2)
        sin_theta2 = np.sin(self.theta2)
        self.R2 = torch.tensor(
            [[cos_theta2, -sin_theta2], [sin_theta2, cos_theta2]],
            dtype=self.dtype)
        self.x_equilibrium2 = torch.tensor([0.4, 2.5], dtype=self.dtype)
        self.system2 = \
            test_hybrid_linear_system.setup_transformed_trecate_system(
                self.theta2, self.x_equilibrium2)
        self.system3 = test_hybrid_linear_system.\
            setup_johansson_continuous_time_system1()
        self.x_equilibrium3 = torch.tensor([0, 0], dtype=self.dtype)

    def test_add_hybrid_system_constraint(self):

        def test_fun(system, x_val, is_x_valid):
            milp = gurobi_torch_mip.GurobiTorchMILP(self.dtype)
            relu = setup_leaky_relu(self.dtype)
            dut = lyapunov.LyapunovHybridLinearSystem(system, relu)
            (x, s, gamma, Aeq_s, Aeq_gamma) = \
                dut.add_hybrid_system_constraint(milp)
            # Now fix x to x_val
            for i in range(system.x_dim):
                milp.addLConstr(
                    [torch.tensor([1.], dtype=self.dtype)], [[x[i]]],
                    sense=gurobipy.GRB.EQUAL, rhs=x_val[i])
            milp.gurobi_model.setParam(gurobipy.GRB.Param.OutputFlag, 0)
            milp.gurobi_model.setParam(gurobipy.GRB.Param.DualReductions, 0)
            milp.gurobi_model.optimize()
            if is_x_valid:
                self.assertEqual(
                    milp.gurobi_model.status, gurobipy.GRB.OPTIMAL)
                mode_index = system.mode(x_val)
                gamma_expected = np.zeros(system.num_modes)
                gamma_expected[mode_index] = 1.
                s_expected = np.zeros(system.num_modes * system.x_dim)
                s_expected[mode_index * system.x_dim:
                           (mode_index+1) * system.x_dim] =\
                    x_val.detach().numpy()
                s_sol = np.array([v.x for v in s])
                np.testing.assert_allclose(s_sol, s_expected)
                gamma_sol = np.array([v.x for v in gamma])
                np.testing.assert_allclose(gamma_sol, gamma_expected)
                np.testing.assert_allclose(
                    Aeq_s @ torch.from_numpy(s_sol) +
                    Aeq_gamma @ torch.from_numpy(gamma_sol),
                    system.A[mode_index] @ x_val + system.g[mode_index])
            else:
                self.assertEqual(
                    milp.gurobi_model.status, gurobipy.GRB.INFEASIBLE)

        test_fun(
            self.system1, torch.tensor([0.5, 0.2], dtype=self.dtype), True)
        test_fun(
            self.system1, torch.tensor([-0.5, 0.2], dtype=self.dtype), True)
        test_fun(
            self.system1, torch.tensor([-1.5, 0.2], dtype=self.dtype), False)
        test_fun(
            self.system2, self.R2 @ torch.tensor([-0.5, 0.2], dtype=self.dtype)
            + self.x_equilibrium2, True)
        test_fun(
            self.system2, self.R2 @ torch.tensor([-0.5, 1.2], dtype=self.dtype)
            + self.x_equilibrium2, False)
        test_fun(
            self.system3, torch.tensor([-0.5, 0.7], dtype=self.dtype), True)

    def test_add_relu_output_constraint(self):

        def test_fun(relu, system, x_val):
            relu_free_pattern = relu_to_optimization.ReLUFreePattern(
                relu, self.dtype)
            milp = gurobi_torch_mip.GurobiTorchMILP(self.dtype)
            x = milp.addVars(
                system.x_dim, lb=-gurobipy.GRB.INFINITY,
                vtype=gurobipy.GRB.CONTINUOUS)
            dut = lyapunov.LyapunovHybridLinearSystem(system, relu)
            (z, beta, a_out, b_out) = dut.add_relu_output_constraint(milp, x)
            for i in range(system.x_dim):
                milp.addLConstr(
                    [torch.tensor([1.], dtype=self.dtype)], [[x[i]]],
                    sense=gurobipy.GRB.EQUAL, rhs=x_val[i])
            milp.gurobi_model.setParam(gurobipy.GRB.Param.OutputFlag, 0)
            milp.gurobi_model.optimize()
            self.assertEqual(milp.gurobi_model.status, gurobipy.GRB.OPTIMAL)
            activation_pattern = \
                relu_to_optimization.ComputeReLUActivationPattern(relu, x_val)
            for i in range(len(relu_free_pattern.relu_unit_index)):
                for j in range(len(relu_free_pattern.relu_unit_index[i])):
                    self.assertAlmostEqual(
                        activation_pattern[i][j],
                        beta[relu_free_pattern.relu_unit_index[i][j]].x)
            self.assertAlmostEqual(
                (a_out @ torch.tensor([v.x for v in z], dtype=system.dtype) +
                 b_out).item(), relu.forward(x_val).item())

        relu1 = setup_relu(self.dtype)
        relu2 = setup_leaky_relu(self.dtype)
        test_fun(
            relu1, self.system1, torch.tensor([0.5, 0.2], dtype=self.dtype))
        test_fun(
            relu2, self.system1, torch.tensor([0.5, 0.2], dtype=self.dtype))
        test_fun(
            relu2, self.system1, torch.tensor([-0.5, 0.2], dtype=self.dtype))

    def test_add_state_error_l1_constraint(self):
        relu = setup_leaky_relu(self.dtype)

        def test_fun(system, x_equilibrium, x_val):
            milp = gurobi_torch_mip.GurobiTorchMILP(self.dtype)
            x = milp.addVars(
                system.x_dim, lb=-gurobipy.GRB.INFINITY,
                vtype=gurobipy.GRB.CONTINUOUS)
            dut = lyapunov.LyapunovHybridLinearSystem(system, relu)
            s, alpha = dut.add_state_error_l1_constraint(
                milp, x_equilibrium, x)
            self.assertEqual(len(s), system.x_dim)
            self.assertEqual(len(alpha), system.x_dim)
            for i in range(system.x_dim):
                milp.addLConstr(
                    [torch.tensor([1.], dtype=system.dtype)], [[x[i]]],
                    sense=gurobipy.GRB.EQUAL, rhs=x_val[i])
            milp.gurobi_model.setParam(gurobipy.GRB.Param.OutputFlag, 0)
            milp.gurobi_model.optimize()
            self.assertEqual(milp.gurobi_model.status, gurobipy.GRB.OPTIMAL)
            for i in range(system.x_dim):
                if x_val[i] >= x_equilibrium[i]:
                    self.assertAlmostEqual(alpha[i].x, 1)
                    self.assertAlmostEqual(
                        s[i].x, (x_val[i] - x_equilibrium[i]).item())
                else:
                    self.assertAlmostEqual(alpha[i].x, 0)
                    self.assertAlmostEqual(
                        s[i].x, -(x_val[i] - x_equilibrium[i]).item())
        test_fun(
            self.system1, self.x_equilibrium1,
            torch.tensor([0.5, -0.3], dtype=self.dtype))
        test_fun(
            self.system1, self.x_equilibrium1,
            torch.tensor([-0.5, -0.3], dtype=self.dtype))
        test_fun(
            self.system2, self.x_equilibrium2,
            self.R2 @ torch.tensor([-0.5, -0.3], dtype=self.dtype) +
            self.x_equilibrium2)
        test_fun(
            self.system2, self.x_equilibrium2,
            self.R2 @ torch.tensor([0.5, -0.3], dtype=self.dtype) +
            self.x_equilibrium2)
        # system 5 has some x_lo equal to x_equilibrium.
        system5 = test_hybrid_linear_system.\
            setup_johansson_continuous_time_system5(keep_positive_x=True)
        test_fun(
            system5, torch.tensor([0., 0], dtype=self.dtype),
            torch.tensor([0.5, 0.3], dtype=self.dtype))
        # system 6 has some x_up equal to x_equilibrium.
        system5_full = test_hybrid_linear_system.\
            setup_johansson_continuous_time_system5()
        system6 = hybrid_linear_system.AutonomousHybridLinearSystem(
            2, dtype=self.dtype)
        system6.add_mode(
            system5_full.A[1], system5_full.g[1], system5_full.P[1],
            system5_full.q[1])
        system6.add_mode(
            system5_full.A[2], system5_full.g[2], system5_full.P[2],
            system5_full.q[2])
        test_fun(
            system6, torch.tensor([0., 0], dtype=self.dtype),
            torch.tensor([-0.2, 0.3], dtype=self.dtype))

    def test_add_lyapunov_bounds_constraint(self):
        V_lambda = 0.5

        def test_fun(
            lyapunov_lower, lyapunov_upper, system, relu, x_equilibrium,
                x_val):
            """
            Set x = x_val, check if the MILP
            lyapunov_lower <= V(x) <= lyapunov_upper is feasible or not.
            """
            dut = lyapunov.LyapunovHybridLinearSystem(system, relu)
            milp = gurobi_torch_mip.GurobiTorchMILP(self.dtype)
            x = milp.addVars(
                system.x_dim, lb=-gurobipy.GRB.INFINITY,
                vtype=gurobipy.GRB.CONTINUOUS)
            (relu_z, _, a_relu, b_relu) = dut.add_relu_output_constraint(
                milp, x)
            (s, _) = dut.add_state_error_l1_constraint(milp, x_equilibrium, x)
            relu_x_equilibrium = relu.forward(x_equilibrium)
            dut.add_lyapunov_bounds_constraint(
                lyapunov_lower, lyapunov_upper, milp, a_relu, b_relu, V_lambda,
                relu_x_equilibrium, relu_z, s)
            for i in range(system.x_dim):
                milp.addLConstr(
                    [torch.tensor([1.], dtype=system.dtype)], [[x[i]]],
                    sense=gurobipy.GRB.EQUAL, rhs=x_val[i])
            milp.gurobi_model.setParam(gurobipy.GRB.Param.OutputFlag, 0)
            milp.gurobi_model.setParam(gurobipy.GRB.Param.DualReductions, 0)
            milp.gurobi_model.optimize()

            V_val = dut.lyapunov_value(
                x_val, x_equilibrium, V_lambda, relu_x_equilibrium)
            is_satisfied = True
            if lyapunov_lower is not None:
                is_satisfied = is_satisfied and V_val >= lyapunov_lower
            if lyapunov_upper is not None:
                is_satisfied = is_satisfied and V_val <= lyapunov_upper
            if is_satisfied:
                self.assertEqual(
                    milp.gurobi_model.status, gurobipy.GRB.Status.OPTIMAL)
            else:
                self.assertEqual(
                    milp.gurobi_model.status, gurobipy.GRB.Status.INFEASIBLE)

        relu1 = setup_relu(self.dtype)
        relu2 = setup_leaky_relu(self.dtype)
        for relu in (relu1, relu2):
            test_fun(
                None, None, self.system1, relu, self.x_equilibrium1,
                torch.tensor([-0.5, 0.3], dtype=self.dtype))
            test_fun(
                0.5, None, self.system1, relu, self.x_equilibrium1,
                torch.tensor([-0.5, 0.3], dtype=self.dtype))
            test_fun(
                None, 30., self.system1, relu, self.x_equilibrium1,
                torch.tensor([-0.5, 0.3], dtype=self.dtype))
            test_fun(
                -2., 30., self.system1, relu, self.x_equilibrium1,
                torch.tensor([-0.5, 0.3], dtype=self.dtype))
            test_fun(
                -2., 1., self.system1, relu, self.x_equilibrium1,
                torch.tensor([-0.1, 0.4], dtype=self.dtype))
            test_fun(
                1., 3., self.system1, relu, self.x_equilibrium1,
                torch.tensor([0.3, 0.4], dtype=self.dtype))

    def test_lyapunov_value(self):
        relu = setup_leaky_relu(self.system1.dtype)
        dut1 = lyapunov.LyapunovHybridLinearSystem(self.system1, relu)
        dut2 = lyapunov.LyapunovHybridLinearSystem(self.system2, relu)
        V_lambda = 0.1

        def test_fun(x):
            self.assertAlmostEqual(
                (relu.forward(x) - relu.forward(self.x_equilibrium1) +
                 V_lambda * torch.norm(x - self.x_equilibrium1, p=1)).item(),
                dut1.lyapunov_value(x, self.x_equilibrium1, V_lambda).item())
            self.assertAlmostEqual(
                (relu.forward(x) - relu.forward(self.x_equilibrium2) +
                 V_lambda * torch.norm(x - self.x_equilibrium2, p=1)).item(),
                dut2.lyapunov_value(x, self.x_equilibrium2, V_lambda).item())

        test_fun(torch.tensor([0., 0.], dtype=dut1.system.dtype))
        test_fun(torch.tensor([1., 0.], dtype=dut1.system.dtype))
        test_fun(torch.tensor([-0.2, 0.4], dtype=dut1.system.dtype))

        def test_batch_fun(dut, x_equilibrium, x):
            # Test a batch of x.
            expected = torch.empty((x.shape[0],), dtype=x.dtype)
            for i in range(x.shape[0]):
                expected[i] = dut.lyapunov_value(
                    x[i], x_equilibrium, V_lambda)
            val = dut.lyapunov_value(x, x_equilibrium, V_lambda)
            np.testing.assert_allclose(
                expected.detach().numpy(), val.detach().numpy())
        for dut, x_equilibrium in \
                ((dut1, self.x_equilibrium1), (dut2, self.x_equilibrium2)):
            test_batch_fun(
                dut, x_equilibrium, torch.tensor([
                    [0., 0.], [1., 0.], [0., 1.], [0.2, 0.4], [0.5, -0.8]],
                    dtype=dut.system.dtype))

    def test_lyapunov_positivity_loss_at_samples(self):
        # Construct a simple ReLU model with 2 hidden layers
        linear1 = nn.Linear(2, 3)
        linear1.weight.data = torch.tensor([[1, 2], [3, 4], [5, 6]],
                                           dtype=self.dtype)
        linear1.bias.data = torch.tensor([-11, 10, 5], dtype=self.dtype)
        linear2 = nn.Linear(3, 4)
        linear2.weight.data = torch.tensor(
                [[-1, -0.5, 1.5], [2, 5, 6], [-2, -3, -4], [1.5, 4, 6]],
                dtype=self.dtype)
        linear2.bias.data = torch.tensor([-3, 2, 0.7, 1.5], dtype=self.dtype)
        linear3 = nn.Linear(4, 1)
        linear3.weight.data = torch.tensor([[4, 5, 6, 7]], dtype=self.dtype)
        linear3.bias.data = torch.tensor([-9], dtype=self.dtype)
        relu1 = nn.Sequential(
            linear1, nn.ReLU(), linear2, nn.ReLU(), linear3, nn.ReLU())
        dut = lyapunov.LyapunovDiscreteTimeHybridSystem(self.system1, relu1)

        x_equilibrium = torch.tensor([0., 0.], dtype=self.dtype)
        relu_at_equilibrium = relu1.forward(x_equilibrium)

        V_lambda = 0.01
        margin = 2.
        epsilon = 2.

        def test_fun(x_samples):
            loss = 0
            for i in range(x_samples.shape[0]):
                relu_x = relu1.forward(x_samples[i])
                v = (relu_x - relu_at_equilibrium) + V_lambda * torch.norm(
                    x_samples[i] - x_equilibrium, p=1)
                v_minus_l1 = v - epsilon * torch.norm(
                    x_samples[i] - x_equilibrium, p=1)
                loss += 0 if v_minus_l1 > margin else margin - v_minus_l1
            loss = loss / x_samples.shape[0]
            self.assertAlmostEqual(
                loss.item(), dut.lyapunov_positivity_loss_at_samples(
                    relu_at_equilibrium, x_equilibrium, x_samples,
                    V_lambda, epsilon, margin=margin).item())

        test_fun(torch.tensor([[0, 0]], dtype=self.dtype))
        test_fun(torch.tensor([[0, 0], [0, 1]], dtype=self.dtype))
        test_fun(torch.tensor([
            [0, 0], [0, 1], [1, 0], [0.5, 0.4], [0.2, -0.1], [0.4, 0.3],
            [-0.2, 0.3]], dtype=self.dtype))


class TestLyapunovDiscreteTimeHybridSystem(unittest.TestCase):

    def setUp(self):
        """
        The piecewise affine system is from "Analysis of discrete-time
        piecewise affine and hybrid systems" by Giancarlo Ferrari-Trecate
        et.al.
        """
        self.dtype = torch.float64
        self.x_equilibrium1 = torch.tensor([0, 0], dtype=self.dtype)
        self.system1 = test_hybrid_linear_system.\
            setup_trecate_discrete_time_system()
        self.theta2 = np.pi / 5
        cos_theta2 = np.cos(self.theta2)
        sin_theta2 = np.sin(self.theta2)
        self.R2 = torch.tensor(
            [[cos_theta2, -sin_theta2], [sin_theta2, cos_theta2]],
            dtype=self.dtype)
        self.x_equilibrium2 = torch.tensor([0.4, 2.5], dtype=self.dtype)
        self.system2 = \
            test_hybrid_linear_system.setup_transformed_trecate_system(
                self.theta2, self.x_equilibrium2)

    def test_lyapunov_derivative(self):
        relu = setup_relu(torch.float64)
        V_lambda = 2.
        epsilon = 0.1

        def test_fun(system, x, x_equilibrium):
            x_next_possible = system.possible_dx(x)
            relu_at_equilibrium = relu.forward(x_equilibrium)
            dut = lyapunov.LyapunovDiscreteTimeHybridSystem(system, relu)
            V_next_possible = [dut.lyapunov_value(
                x_next, x_equilibrium, V_lambda, relu_at_equilibrium) for
                x_next in x_next_possible]
            V = dut.lyapunov_value(
                x, x_equilibrium, V_lambda, relu_at_equilibrium)
            lyapunov_derivative_expected = [
                V_next - V + epsilon * V for V_next in V_next_possible]
            lyapunov_derivative = dut.lyapunov_derivative(
                x, x_equilibrium, V_lambda, epsilon)
            self.assertEqual(
                len(lyapunov_derivative), len(lyapunov_derivative_expected))
            for i in range(len(lyapunov_derivative)):
                self.assertAlmostEqual(
                    lyapunov_derivative[i].item(),
                    lyapunov_derivative_expected[i].item())

        for x in ([0.2, 0.5], [0.1, -0.4], [0., 0.5], [-0.2, 0.]):
            test_fun(
                self.system1, torch.tensor(x, dtype=self.system1.dtype),
                self.x_equilibrium1)
            test_fun(
                self.system2,
                self.R2 @ torch.tensor(x, dtype=self.system2.dtype) +
                self.x_equilibrium2, self.x_equilibrium2)

    def test_lyapunov_positivity_as_milp(self):
        relu1 = setup_relu(self.dtype)
        relu2 = setup_leaky_relu(self.dtype, bias=False)
        V_epsilon = 0.01
        V_lambda = 0.1

        def test_fun(system, relu, x_equilibrium, relu_x_equilibrium, x_val):
            # Fix x to different values. Now check if the optimal cost is
            # ReLU(x) - ReLU(x*) + (ρ - epsilon) * |x - x*|₁
            dut = lyapunov.LyapunovDiscreteTimeHybridSystem(system, relu)
            (milp, x) = dut.lyapunov_positivity_as_milp(
                x_equilibrium, V_lambda, V_epsilon)
            for i in range(dut.system.x_dim):
                milp.addLConstr(
                    [torch.tensor([1.], dtype=self.dtype)], [[x[i]]],
                    rhs=x_val[i], sense=gurobipy.GRB.EQUAL)
            milp.gurobi_model.setParam(gurobipy.GRB.Param.OutputFlag, False)
            milp.gurobi_model.optimize()
            self.assertEqual(milp.gurobi_model.status,
                             gurobipy.GRB.Status.OPTIMAL)
            self.assertAlmostEqual(
                milp.gurobi_model.ObjVal, relu.forward(x_val).item() -
                relu_x_equilibrium.item() +
                (V_lambda - V_epsilon) *
                torch.norm(x_val - x_equilibrium, p=1).item())

        relu_x_equilibrium1 = relu1.forward(self.x_equilibrium1)
        relu_x_equilibrium2 = relu1.forward(self.x_equilibrium2)
        relu_x_equilibrium3 = relu2.forward(self.x_equilibrium1)

        test_fun(
            self.system1, relu1, self.x_equilibrium1, relu_x_equilibrium1,
            torch.tensor([0, 0], dtype=self.dtype))
        test_fun(
            self.system1, relu1, self.x_equilibrium1, relu_x_equilibrium1,
            torch.tensor([0, 0.5], dtype=self.dtype))
        test_fun(
            self.system1, relu1, self.x_equilibrium1, relu_x_equilibrium1,
            torch.tensor([0.1, 0.5], dtype=self.dtype))
        test_fun(
            self.system1, relu1, self.x_equilibrium1, relu_x_equilibrium1,
            torch.tensor([-0.3, 0.8], dtype=self.dtype))
        test_fun(
            self.system1, relu1, self.x_equilibrium1, relu_x_equilibrium1,
            torch.tensor([-0.3, -0.2], dtype=self.dtype))
        test_fun(
            self.system1, relu1, self.x_equilibrium1, relu_x_equilibrium1,
            torch.tensor([0.6, -0.2], dtype=self.dtype))
        test_fun(
            self.system2, relu1, self.x_equilibrium2, relu_x_equilibrium2,
            self.x_equilibrium2)
        test_fun(
            self.system2, relu1, self.x_equilibrium2, relu_x_equilibrium2,
            self.R2 @ torch.tensor([0, 0.5], dtype=self.dtype) +
            self.x_equilibrium2)
        test_fun(
            self.system2, relu1, self.x_equilibrium2, relu_x_equilibrium2,
            self.R2 @ torch.tensor([0.1, 0.5], dtype=self.dtype) +
            self.x_equilibrium2)
        test_fun(
            self.system2, relu1, self.x_equilibrium2, relu_x_equilibrium2,
            self.R2 @ torch.tensor([-0.3, 0.5], dtype=self.dtype) +
            self.x_equilibrium2)
        test_fun(
            self.system2, relu1, self.x_equilibrium2, relu_x_equilibrium2,
            self.R2 @ torch.tensor([0.5, -0.8], dtype=self.dtype) +
            self.x_equilibrium2)
        test_fun(
            self.system2, relu1, self.x_equilibrium2, relu_x_equilibrium2,
            self.R2 @ torch.tensor([-0.2, -0.8], dtype=self.dtype) +
            self.x_equilibrium2)
        test_fun(
            self.system1, relu2, self.x_equilibrium1, relu_x_equilibrium3,
            torch.tensor([-0.3, -0.2], dtype=self.dtype))
        test_fun(
            self.system1, relu2, self.x_equilibrium1, relu_x_equilibrium3,
            torch.tensor([0.5, -0.2], dtype=self.dtype))

    def test_lyapunov_derivative_as_milp(self):
        """
        Test lyapunov_derivative_as_milp without bounds on V(x[n])
        """

        relu1 = setup_leaky_relu(self.dtype)
        relu2 = setup_relu(self.dtype)
        V_lambda = 2.
        dV_epsilon = 0.1

        def test_milp(system, x_equilibrium, relu):
            dut = lyapunov.LyapunovDiscreteTimeHybridSystem(system, relu)
            (milp, x, beta, gamma, x_next, s, z, z_next, beta_next) =\
                dut.lyapunov_derivative_as_milp(
                    x_equilibrium, V_lambda, dV_epsilon)
            # First solve this MILP. The solution has to satisfy that
            # x_next = Ai * x + g_i where i is the active mode inferred from
            # gamma.
            milp.gurobi_model.setParam(gurobipy.GRB.Param.OutputFlag, 0)
            milp.gurobi_model.optimize()
            if (milp.gurobi_model.status == gurobipy.GRB.Status.INFEASIBLE):
                milp.gurobi_model.computeIIS()
                milp.gurobi_model.write("milp.ilp")
            self.assertEqual(
                milp.gurobi_model.status, gurobipy.GRB.Status.OPTIMAL)
            x_sol = np.array([var.x for var in x])
            x_next_sol = np.array([var.x for var in x_next])
            gamma_sol = np.array([var.x for var in gamma])
            self.assertAlmostEqual(np.sum(gamma_sol), 1)
            for mode in range(self.system1.num_modes):
                if np.abs(gamma_sol[mode] - 1) < 1E-4:
                    # This mode is active.
                    # Pi * x <= qi
                    np.testing.assert_array_less(
                        dut.system.P[mode].detach().numpy() @ x_sol,
                        dut.system.q[mode].detach().numpy() + 1E-5)
                    # x_next = Ai * x + gi
                    np.testing.assert_array_almost_equal(
                        dut.system.A[mode].detach().numpy() @ x_sol +
                        dut.system.g[mode].detach().numpy(),
                        x_next_sol, decimal=5)
            v_next = dut.lyapunov_value(
                torch.from_numpy(x_next_sol), x_equilibrium, V_lambda)
            v = dut.lyapunov_value(
                torch.from_numpy(x_sol), x_equilibrium, V_lambda)
            self.assertAlmostEqual(
                milp.gurobi_model.objVal,
                (v_next - v + dV_epsilon * v).item())

        test_milp(self.system1, self.x_equilibrium1, relu1)
        test_milp(self.system1, self.x_equilibrium1, relu2)
        test_milp(self.system2, self.x_equilibrium2, relu1)
        test_milp(self.system2, self.x_equilibrium2, relu2)

        # Now solve MILP to optimal for system1 and system2
        dut11 = lyapunov.LyapunovDiscreteTimeHybridSystem(self.system1, relu1)
        milp1 = dut11.lyapunov_derivative_as_milp(
                self.x_equilibrium1, V_lambda, dV_epsilon)[0]
        milp1.gurobi_model.setParam(gurobipy.GRB.Param.OutputFlag, 0)
        milp1.gurobi_model.optimize()
        milp_optimal_cost1 = milp1.gurobi_model.ObjVal
        dut21 = lyapunov.LyapunovDiscreteTimeHybridSystem(self.system2, relu1)
        milp2 = dut21.lyapunov_derivative_as_milp(
                self.x_equilibrium2, V_lambda, dV_epsilon)[0]
        milp2.gurobi_model.setParam(gurobipy.GRB.Param.OutputFlag, 0)
        milp2.gurobi_model.optimize()
        milp_optimal_cost2 = milp2.gurobi_model.ObjVal

        # Now test reformulating
        # ReLU(x[n+1]) + ρ|x[n+1]-x*|₁ - ReLU(x[n]) - ρ|x[n]-x*|₁ +
        # epsilon * (Relu(x[n]) - ReLU(x*) + ρ|x[n]-x*|₁) as a
        # mixed-integer linear program. We fix x[n] to some value, compute the
        # cost function of the MILP, and then check if it is the same as
        # evaluating the ReLU network on x[n] and x[n+1]
        def test_milp_cost(dut, mode, x_val, x_equilibrium, milp_optimal_cost):
            assert(torch.all(
                dut.system.P[mode] @ x_val <= dut.system.q[mode]))
            x_next_val = dut.system.A[mode] @ x_val + dut.system.g[mode]
            v_next = dut.lyapunov_value(
                x_next_val, x_equilibrium, V_lambda)
            v = dut.lyapunov_value(x_val, x_equilibrium, V_lambda)
            cost_expected = (v_next - v + dV_epsilon * v).item()
            (milp_test, x_test, _, _, _, _, _, _, _) =\
                dut.lyapunov_derivative_as_milp(
                    x_equilibrium, V_lambda, dV_epsilon)
            for i in range(dut.system.x_dim):
                milp_test.addLConstr(
                    [torch.tensor([1.], dtype=milp_test.dtype)], [[x_test[i]]],
                    rhs=x_val[i], sense=gurobipy.GRB.EQUAL)
            milp_test.gurobi_model.setParam(gurobipy.GRB.Param.OutputFlag, 0)
            milp_test.gurobi_model.optimize()
            self.assertEqual(milp_test.gurobi_model.status,
                             gurobipy.GRB.Status.OPTIMAL)
            self.assertAlmostEqual(cost_expected,
                                   milp_test.gurobi_model.objVal)
            # milp_test solves the problem with fixed x[n], so it should
            # achieve less optimal cost than milp1 or milp2.
            self.assertLessEqual(milp_test.gurobi_model.objVal,
                                 milp_optimal_cost)

        # Now test with random x[n]
        torch.manual_seed(0)
        np.random.seed(0)

        def sample_state(system):
            while True:
                x_val = torch.tensor(
                    [np.random.uniform(system.x_lo_all[i], system.x_up_all[i])
                     for i in range(self.system1.x_dim)]).type(system.dtype)
                if torch.all(system.P[i] @ x_val <= system.q[i]):
                    return x_val

        for i in range(self.system1.num_modes):
            for _ in range(20):
                x_val1 = sample_state(self.system1)
                test_milp_cost(
                    dut11, i, x_val1, self.x_equilibrium1, milp_optimal_cost1)
        for i in range(self.system2.num_modes):
            for _ in range(20):
                x_val2 = sample_state(self.system2)
                test_milp_cost(
                    dut21, i, x_val2, self.x_equilibrium2, milp_optimal_cost2)

    def test_lyapunov_derivative_as_milp_bounded(self):
        """
        Test lyapunov_derivative_as_milp function, but with a lower and upper
        bounds on V(x[n])
        """
        x_equilibrium = torch.tensor([0, 0], dtype=self.dtype)
        V_lambda = 0.01

        relu1 = setup_relu(self.dtype)
        dut = lyapunov.LyapunovDiscreteTimeHybridSystem(self.system1, relu1)
        # First find out what is the lower and upper bound of the ReLU network.
        milp_relu = gurobi_torch_mip.GurobiTorchMILP(dut.system.dtype)
        relu1_free_pattern = relu_to_optimization.ReLUFreePattern(
            relu1, dut.system.dtype)
        Ain_x, Ain_z, Ain_beta, rhs_in, Aeq_x, Aeq_z, Aeq_beta, rhs_eq, a_out,\
            b_out, _, _, _, _ = relu1_free_pattern.output_constraint(
                torch.tensor([-1.0, -1.0], dtype=dut.system.dtype),
                torch.tensor([1.0, 1.0], dtype=dut.system.dtype))
        x = milp_relu.addVars(
            2, lb=-gurobipy.GRB.INFINITY, vtype=gurobipy.GRB.CONTINUOUS)
        z = milp_relu.addVars(
            Ain_z.shape[1], lb=-gurobipy.GRB.INFINITY,
            vtype=gurobipy.GRB.CONTINUOUS)
        beta = milp_relu.addVars(Ain_beta.shape[1], vtype=gurobipy.GRB.BINARY)
        for i in range(Ain_x.shape[0]):
            milp_relu.addLConstr(
                [Ain_x[i], Ain_z[i], Ain_beta[i]], [x, z, beta],
                rhs=rhs_in[i], sense=gurobipy.GRB.LESS_EQUAL)
        for i in range(Aeq_x.shape[0]):
            milp_relu.addLConstr(
                [Aeq_x[i], Aeq_z[i], Aeq_beta[i]], [x, z, beta],
                rhs=rhs_eq[i], sense=gurobipy.GRB.EQUAL)
        # Add rho * |x - x*|₁. To do so, we introduce slack variable s_x_norm,
        # such that s_x_norm(i) = x(i) - x*(i).
        s_x_norm = milp_relu.addVars(
            self.system1.x_dim, lb=-gurobipy.GRB.INFINITY,
            vtype=gurobipy.GRB.CONTINUOUS)
        beta_x_norm = milp_relu.addVars(
            self.system1.x_dim, vtype=gurobipy.GRB.BINARY)
        for i in range(self.system1.x_dim):
            Ain_x_norm, Ain_s_x_norm, Ain_beta_x_norm, rhs_x_norm = \
                utils.replace_absolute_value_with_mixed_integer_constraint(
                    self.system1.x_lo_all[i] - x_equilibrium[i],
                    self.system1.x_up_all[i] - x_equilibrium[i],
                    self.system1.dtype)
            for j in range(Ain_x_norm.shape[0]):
                milp_relu.addLConstr(
                    [torch.tensor([Ain_x_norm[j], Ain_s_x_norm[j],
                                   Ain_beta_x_norm[j]],
                                  dtype=self.system1.dtype)],
                    [[x[i], s_x_norm[i], beta_x_norm[i]]],
                    rhs=rhs_x_norm[j]+Ain_x_norm[j]*x_equilibrium[i],
                    sense=gurobipy.GRB.LESS_EQUAL)
        relu_x_equilibrium = relu1.forward(x_equilibrium).item()
        milp_relu.setObjective(
            [a_out, V_lambda *
             torch.ones((self.system1.x_dim,), dtype=self.system1.dtype)],
            [z, s_x_norm], float(b_out) - relu_x_equilibrium,
            sense=gurobipy.GRB.MAXIMIZE)
        milp_relu.gurobi_model.setParam(gurobipy.GRB.Param.OutputFlag, False)
        milp_relu.gurobi_model.optimize()
        self.assertEqual(
            milp_relu.gurobi_model.status, gurobipy.GRB.Status.OPTIMAL)
        v_upper = milp_relu.gurobi_model.ObjVal
        x_sol = torch.tensor([v.x for v in x], dtype=dut.system.dtype)
        self.assertAlmostEqual(dut.lyapunov_value(
            x_sol, x_equilibrium, V_lambda, relu_x_equilibrium).item(),
            v_upper)
        milp_relu.setObjective(
            [a_out, V_lambda *
             torch.ones((self.system1.x_dim,), dtype=self.system1.dtype)],
            [z, s_x_norm], float(b_out) - relu_x_equilibrium,
            sense=gurobipy.GRB.MINIMIZE)
        milp_relu.gurobi_model.optimize()
        self.assertEqual(
            milp_relu.gurobi_model.status, gurobipy.GRB.Status.OPTIMAL)
        v_lower = milp_relu.gurobi_model.ObjVal
        x_sol = torch.tensor([v.x for v in x], dtype=dut.system.dtype)
        self.assertAlmostEqual(dut.lyapunov_value(
            x_sol, x_equilibrium, V_lambda, relu_x_equilibrium).item(),
            v_lower)

        # If we set lyapunov_lower to be v_upper + 1, the problem should be
        # infeasible.
        dV_epsilon = 0.01
        (milp, _, _, _, _, _, _, _, _) =\
            dut.lyapunov_derivative_as_milp(
                x_equilibrium, V_lambda, dV_epsilon, v_upper + 1,
                v_upper + 2)
        milp.gurobi_model.setParam(gurobipy.GRB.Param.OutputFlag, False)
        milp.gurobi_model.setParam(gurobipy.GRB.Param.DualReductions, 0)
        milp.gurobi_model.optimize()
        self.assertEqual(
            milp.gurobi_model.status, gurobipy.GRB.Status.INFEASIBLE)
        # If we set lyapunov_upper to be v_lower - 1, the problem should be
        # infeasible.
        (milp, _, _, _, _, _, _, _, _) =\
            dut.lyapunov_derivative_as_milp(
                x_equilibrium, V_lambda, dV_epsilon, v_lower - 2,
                v_lower - 1)
        milp.gurobi_model.setParam(gurobipy.GRB.Param.OutputFlag, False)
        milp.gurobi_model.setParam(gurobipy.GRB.Param.DualReductions, 0)
        milp.gurobi_model.optimize()
        self.assertEqual(
            milp.gurobi_model.status, gurobipy.GRB.Status.INFEASIBLE)

        # Now solve the MILP with valid lyapunov_lower and lyapunov_upper.
        # Then take many sample state. If lyapunov_lower <= V(x_sample) <=
        # lyapunov_upper, then the Lyapunov condition violation should be
        # smaller than milp optimal.
        lyapunov_lower = 0.9 * v_lower + 0.1 * v_upper
        lyapunov_upper = 0.1 * v_lower + 0.9 * v_upper
        (milp, _, _, _, _, _, _, _, _) =\
            dut.lyapunov_derivative_as_milp(
                x_equilibrium, V_lambda, dV_epsilon, lyapunov_lower,
                lyapunov_upper)
        milp.gurobi_model.setParam(gurobipy.GRB.Param.OutputFlag, False)
        milp.gurobi_model.optimize()
        self.assertEqual(milp.gurobi_model.status, gurobipy.GRB.Status.OPTIMAL)
        for _ in range(100):
            x_sample = torch.from_numpy(np.random.random((2,)) * 2 - 1)
            v = relu1.forward(x_sample) - relu_x_equilibrium +\
                V_lambda * torch.norm(x_sample - x_equilibrium, p=1)
            if v >= lyapunov_lower and v <= lyapunov_upper:
                x_next = self.system1.step_forward(x_sample)
                v_next = relu1.forward(x_next) - relu_x_equilibrium +\
                    V_lambda * torch.norm(x_sample - x_equilibrium, p=1)
                self.assertLessEqual(v_next - v + dV_epsilon * v,
                                     milp.gurobi_model.ObjVal)

    def test_lyapunov_derivative_as_milp_gradient(self):
        """
        Test the gradient of the MILP optimal cost w.r.t the ReLU network
        weights and bias. I can first compute the gradient through pytorch
        autograd, and then compare that against numerical gradient.
        """

        def compute_milp_cost_given_relu(weight_all, bias_all, requires_grad):
            # Construct a simple ReLU model with 2 hidden layers
            assert(isinstance(weight_all, np.ndarray))
            assert(isinstance(bias_all, np.ndarray))
            assert(weight_all.shape == (22,))
            assert(bias_all.shape == (8,))
            weight_tensor = torch.from_numpy(weight_all).type(self.dtype)
            weight_tensor.requires_grad = requires_grad
            bias_tensor = torch.from_numpy(bias_all).type(self.dtype)
            bias_tensor.requires_grad = requires_grad
            linear1 = nn.Linear(2, 3)
            linear1.weight.data = weight_tensor[:6].clone().reshape((3, 2))
            linear1.bias.data = bias_tensor[:3].clone()
            linear2 = nn.Linear(3, 4)
            linear2.weight.data = weight_tensor[6:18].clone().reshape((4, 3))
            linear2.bias.data = bias_tensor[3:7].clone()
            linear3 = nn.Linear(4, 1)
            linear3.weight.data = weight_tensor[18:].clone().reshape((1, 4))
            linear3.bias.data = bias_tensor[7:].clone()
            relu1 = nn.Sequential(
                linear1, nn.ReLU(), linear2, nn.LeakyReLU(0.1), linear3,
                nn.ReLU())
            dut = lyapunov.LyapunovDiscreteTimeHybridSystem(
                self.system1, relu1)

            V_lambda = 0.1
            dV_epsilon = 0.01
            milp_return = dut.lyapunov_derivative_as_milp(
                    torch.tensor([0, 0], dtype=self.system1.dtype),
                    V_lambda, dV_epsilon)
            milp = milp_return[0]

            milp.gurobi_model.setParam(gurobipy.GRB.Param.OutputFlag, False)
            milp.gurobi_model.optimize()
            objective = milp.compute_objective_from_mip_data_and_solution(
                penalty=0.)
            if requires_grad:
                objective.backward()
                weight_grad = np.concatenate(
                    (linear1.weight.grad.detach().numpy().reshape((-1,)),
                     linear2.weight.grad.detach().numpy().reshape((-1,)),
                     linear3.weight.grad.detach().numpy().reshape((-1,))),
                    axis=0)
                bias_grad = np.concatenate(
                    (linear1.bias.grad.detach().numpy().reshape((-1,)),
                     linear2.bias.grad.detach().numpy().reshape((-1,)),
                     linear3.bias.grad.detach().numpy().reshape((-1,))),
                    axis=0)
                return (weight_grad, bias_grad)
            else:
                self.assertAlmostEqual(
                    milp.gurobi_model.ObjVal, objective.item(), places=6)
                return milp.gurobi_model.ObjVal

        # Test arbitrary weight and bias.
        weight_all_list = []
        bias_all_list = []
        weight_all_list.append(
            np.array(
                [0.1, 0.5, -0.2, -2.5, 0.9, 4.5, -11, -2.4, 0.6, 12.5, 21.3,
                 0.32, -2.9, 4.98, -14.23, 16.8, 0.54, 0.42, 1.54, 13.22, 20.1,
                 -4.5]))
        bias_all_list.append(
            np.array([0.45, -2.3, -4.3, 0.58, 2.45, 12.1, 4.6, -3.2]))
        weight_all_list.append(
            np.array(
                [-0.3, 2.5, -3.2, -2.9, 4.9, 4.1, -1.1, -5.43, 0.9, 12.1, 29.3,
                 4.32, -2.98, 4.92, 12.13, -16.8, 0.94, -4.42, 1.54, -13.22,
                 29.1, -14.5]))
        bias_all_list.append(
            np.array([2.45, -12.3, -4.9, 3.58, -2.15, 10.1, -4.6, -3.8]))
        weight_all_list.append(np.array(
            [0.1, -0.2, 2, -0.4, 0.21, 3.2, 14.2, 47.1, 0.1, -2.5, 12.1, 0.3,
             0.5, -3.21, 0.75, 0.42, 3.45, 1.25, 2.41, 2.96, -3.22, -0.01]))
        bias_all_list.append(np.array(
            [0.25, 0.32, 0.34, -0.21, 0.46, 4.21, 12.4, -2.5]))
        weight_all_list.append(np.array(
            [3.1, -1.3, 2.4, -2.4, 3.01, -3.1, 1.2, -41.3, 4.1, -2.4, 14.8,
             1.5, 2.5, -1.81, 3.78, 2.32, -.45, 2.25, 1.4, -.96, -3.95,
             -2.01]))
        bias_all_list.append(np.array(
            [4.25, 2.37, 0.39, -0.24, 1.49, -4.31, 82.5, -12.5]))

        for weight_all, bias_all in zip(weight_all_list, bias_all_list):
            (weight_grad, bias_grad) = compute_milp_cost_given_relu(
                weight_all, bias_all, True)
            grad_numerical = utils.compute_numerical_gradient(
                lambda weight, bias: compute_milp_cost_given_relu(
                    weight, bias, False), weight_all, bias_all, dx=1e-6)
            np.testing.assert_allclose(
                    weight_grad, grad_numerical[0].squeeze(), atol=4e-6)
            np.testing.assert_allclose(
                    bias_grad, grad_numerical[1].squeeze(), atol=1e-6)

    def test_lyapunov_derivative_loss_at_samples(self):
        # Construct a simple ReLU model with 2 hidden layers
        relu1 = setup_relu(torch.float64)
        relu2 = setup_leaky_relu(torch.float64)

        x_samples = []
        x_samples.append(torch.tensor([-0.5, -0.2], dtype=self.dtype))
        x_samples.append(torch.tensor([-0.5, 0.25], dtype=self.dtype))
        x_samples.append(torch.tensor([-0.7, -0.55], dtype=self.dtype))
        x_samples.append(torch.tensor([0.1, 0.85], dtype=self.dtype))
        x_samples.append(torch.tensor([0.45, 0.35], dtype=self.dtype))
        x_samples.append(torch.tensor([0.45, -0.78], dtype=self.dtype))
        x_samples.append(torch.tensor([0.95, -0.23], dtype=self.dtype))
        margin = 0.1
        x_equilibrium = torch.tensor([0, 0], dtype=self.dtype)
        V_lambda = 0.1
        epsilon = 0.5
        for relu in (relu1, relu2):
            dut = lyapunov.LyapunovDiscreteTimeHybridSystem(
                self.system1, relu)
            loss_expected = [None] * len(x_samples)
            for i, x_sample in enumerate(x_samples):
                loss = dut.lyapunov_derivative_loss_at_samples(
                    V_lambda, epsilon, x_sample.unsqueeze(0),
                    x_equilibrium, margin)
                x_next = self.system1.step_forward(x_sample)
                V_x_sample = dut.lyapunov_value(
                    x_sample, x_equilibrium, V_lambda)
                V_x_next = dut.lyapunov_value(
                    x_next, x_equilibrium, V_lambda)
                V_diff = V_x_next - V_x_sample + epsilon * V_x_sample
                loss_expected[i] = torch.max(
                        V_diff + margin, torch.tensor(0., dtype=self.dtype))
                self.assertAlmostEqual(loss.item(), loss_expected[i].item())

            # Test for a batch of x.
            loss_batch = dut.lyapunov_derivative_loss_at_samples(
                V_lambda, epsilon, torch.stack(x_samples), x_equilibrium,
                margin)
            loss_batch_expected = torch.mean(torch.cat(loss_expected))

            self.assertAlmostEqual(
                loss_batch.item(), loss_batch_expected.item())
            relu.zero_grad()
            loss_batch.backward()
            grad = [p.grad.data.clone() for p in relu.parameters()]
            relu.zero_grad()
            loss_batch_expected.backward()
            grad_expected = [p.grad.data.clone() for p in relu.parameters()]
            for i in range(len(grad)):
                np.testing.assert_allclose(
                    grad[i].detach().numpy(),
                    grad_expected[i].detach().numpy(), atol=1e-15)


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
        for relu_model_data in \
                ("negative_loss_relu_1.pt", "negative_loss_relu_2.pt"):
            relu = torch.load(data_dir_path + relu_model_data)
            dut = lyapunov.LyapunovContinuousTimeHybridSystem(system, relu)

            lyapunov_positivity_mip_return = dut.lyapunov_positivity_as_milp(
                x_equilibrium, V_lambda, positivity_epsilon)
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
                x_equilibrium, V_lambda, derivative_epsilon)
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
        dut1 = lyapunov.LyapunovContinuousTimeHybridSystem(self.system1, relu)
        x = torch.tensor([0.5, 0.2], dtype=self.system1.dtype)
        lyapunov_derivatives = dut1.lyapunov_derivative(
            x, self.x_equilibrium1, V_lambda, epsilon)
        self.assertEqual(len(lyapunov_derivatives), 1)
        x.requires_grad = True
        V = dut1.lyapunov_value(x, self.x_equilibrium1, V_lambda)
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
                x, self.x_equilibrium1, V_lambda)).item())

        # x has multiple activation patterns.
        x = torch.tensor([0.5, 0.25], dtype=self.dtype)
        lyapunov_derivatives = dut1.lyapunov_derivative(
            x, self.x_equilibrium1, V_lambda, epsilon)
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
        V = dut1.lyapunov_value(x, self.x_equilibrium1, V_lambda)
        self.assertAlmostEqual(
            (lyapunov_derivatives[0]).item(),
            (Vdot_all[0] + epsilon * V).item())
        self.assertAlmostEqual(
            (lyapunov_derivatives[1]).item(),
            (Vdot_all[1] + epsilon * V).item())

        # The gradient of |x-x*|₁ has multiple possible gradients.
        x = torch.tensor([0.25, 0], dtype=self.dtype)
        lyapunov_derivatives = dut1.lyapunov_derivative(
            x, self.x_equilibrium1, V_lambda, epsilon)
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
                x, self.x_equilibrium1, V_lambda)).item())
        self.assertAlmostEqual(
            lyapunov_derivatives[1].item(),
            (Vdot1 + epsilon * dut1.lyapunov_value(
                x, self.x_equilibrium1, V_lambda)).item())

        # x is on the boundary of the hybrid modes, and the gradient of
        # |x-x*|₁ has multiple values.
        x = torch.tensor([0., 0.1], dtype=self.dtype)
        lyapunov_derivatives = dut1.lyapunov_derivative(
            x, self.x_equilibrium1, V_lambda, epsilon)
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
        V = dut1.lyapunov_value(x, self.x_equilibrium1, V_lambda)
        for i in range(4):
            self.assertAlmostEqual(
                lyapunov_derivatives[i].item(),
                (Vdot[i] + epsilon * V).item())

    def test_add_relu_gradient_times_dynamics(self):
        """
        test add_relu_gradient_times_Aisi() and
        add_relu_gradient_times_gigammai()
        """
        relu1 = setup_relu(self.dtype)
        relu2 = setup_leaky_relu(self.dtype)

        def test_fun(relu, system, x_val, Aisi_flag):
            """
            Setup a MILP with fixed x, if Aisi_flag = True, solve
            ∑ᵢ ∂ReLU(x)/∂x * Aᵢsᵢ
            if Aisi_flag=False, solve
            ∑ᵢ ∂ReLU(x)/∂x * gᵢγᵢ
            """
            dut = lyapunov.LyapunovContinuousTimeHybridSystem(system, relu)
            milp = gurobi_torch_mip.GurobiTorchMILP(system.dtype)
            (x, s, gamma, _, _) = dut.add_hybrid_system_constraint(milp)
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

        for relu in (relu1, relu2):
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
        relu1 = setup_relu(self.dtype)
        relu2 = setup_leaky_relu(self.dtype)

        def test_fun(relu, system, x_val):
            """
            Setup a MILP with fixed x, solve
            ∂ReLU(x)/∂x * ẋ
            """
            dut = lyapunov.LyapunovContinuousTimeHybridSystem(system, relu)
            milp = gurobi_torch_mip.GurobiTorchMILP(system.dtype)
            (x, s, gamma, Aeq_s, Aeq_gamma) = \
                dut.add_hybrid_system_constraint(milp)
            xdot = milp.addVars(
                system.x_dim, lb=-gurobipy.GRB.INFINITY,
                vtype=gurobipy.GRB.CONTINUOUS)
            milp.addMConstrs(
                [torch.eye(system.x_dim, dtype=system.dtype), -Aeq_s,
                 -Aeq_gamma], [xdot, s, gamma], sense=gurobipy.GRB.EQUAL,
                b=torch.zeros(system.x_dim, dtype=system.dtype))
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

        for relu in (relu1, relu2):
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
        relu = setup_leaky_relu(self.dtype)

        def test_fun(system, x_equilibrium, x_val, Aisi_flag):
            dut = lyapunov.LyapunovContinuousTimeHybridSystem(system, relu)
            milp = gurobi_torch_mip.GurobiTorchMILP(system.dtype)
            (x, s, gamma, _, _) = dut.add_hybrid_system_constraint(milp)
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
        relu = setup_leaky_relu(self.dtype)

        def test_fun(system, x_equilibrium, x_val):
            dut = lyapunov.LyapunovContinuousTimeHybridSystem(system, relu)
            milp = gurobi_torch_mip.GurobiTorchMILP(system.dtype)
            (x, s, gamma, Aeq_s, Aeq_gamma) = \
                dut.add_hybrid_system_constraint(milp)
            (_, alpha) = dut.add_state_error_l1_constraint(
                milp, x_equilibrium, x)
            xdot = milp.addVars(
                system.x_dim, lb=-gurobipy.GRB.INFINITY,
                vtype=gurobipy.GRB.CONTINUOUS)
            milp.addMConstrs(
                [torch.eye(system.x_dim, dtype=system.dtype), -Aeq_s,
                 -Aeq_gamma], [xdot, s, gamma], sense=gurobipy.GRB.EQUAL,
                b=torch.zeros(system.x_dim, dtype=system.dtype))
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
        relu1 = setup_relu(self.dtype)
        relu2 = setup_leaky_relu(self.dtype)
        V_lambda = 2.
        epsilon = 0.1

        def test_fun(relu, x_equilibrium, system, x_val, formulation):
            dut = lyapunov.LyapunovContinuousTimeHybridSystem(system, relu)
            if formulation == 1:
                (milp, x, beta, gamma) = dut.lyapunov_derivative_as_milp(
                    x_equilibrium, V_lambda, epsilon, None, None)
            elif formulation == 2:
                (milp, x, beta, gamma) = dut.lyapunov_derivative_as_milp2(
                    x_equilibrium, V_lambda, epsilon, None, None)
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
            V = dut.lyapunov_value(x_val, x_equilibrium, V_lambda)
            V.backward()
            Vdot = x_val.grad @ system.step_forward(x_val)
            self.assertAlmostEqual(
                milp.gurobi_model.ObjVal,
                (Vdot + epsilon * V).squeeze().item())

        for formulation in (1, 2):
            for relu in [relu1, relu2]:
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

        def compute_milp_cost_given_relu(
                system, relu_index, x_equilibrium, params_val, formulation,
                requires_grad):
            if relu_index == 1:
                relu = setup_relu(system.dtype, params_val)
            elif relu_index == 2:
                relu = setup_leaky_relu(system.dtype, params_val)
            dut = lyapunov.LyapunovContinuousTimeHybridSystem(system, relu)
            if formulation == 1:
                milp = dut.lyapunov_derivative_as_milp(
                    x_equilibrium, V_lambda, epsilon, None, None)[0]
            elif formulation == 2:
                milp = dut.lyapunov_derivative_as_milp2(
                    x_equilibrium, V_lambda, epsilon, None, None)[0]
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
            np.testing.assert_allclose(grad, grad_numerical, atol=1.2e-5)

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

    def test_lyapunov_derivative_loss_at_samples(self):
        V_lambda = 2.
        epsilon = 0.2
        relu = setup_leaky_relu(self.dtype)

        def compute_loss_single_sample(
                system, x_equilibrium, relu, x_val, margin):
            xdot = system.step_forward(x_val)
            activation_pattern = relu_to_optimization.\
                ComputeReLUActivationPattern(relu, x_val)
            (g, _, _, _) = relu_to_optimization.ReLUGivenActivationPattern(
                relu, system.x_dim, activation_pattern, system.dtype)
            dVdx = g.squeeze() + \
                V_lambda * torch.sign(x_val - x_equilibrium)
            Vdot = dVdx @ xdot
            dut = lyapunov.LyapunovContinuousTimeHybridSystem(system, relu)
            V = dut.lyapunov_value(
                x_val, x_equilibrium, V_lambda, relu.forward(x_equilibrium))
            loss_expected = torch.max(
                Vdot + epsilon * V + margin,
                torch.tensor(0., dtype=system.dtype))
            return loss_expected

        def test_fun(system, x_equilibrium, relu, x_val, margin):
            xdot = system.step_forward(x_val)
            activation_pattern = relu_to_optimization.\
                ComputeReLUActivationPattern(relu, x_val)
            (g, _, _, _) = relu_to_optimization.ReLUGivenActivationPattern(
                relu, system.x_dim, activation_pattern, system.dtype)
            dVdx = g.squeeze() + \
                V_lambda * torch.sign(x_val - x_equilibrium)
            Vdot = dVdx @ xdot
            dut = lyapunov.LyapunovContinuousTimeHybridSystem(system, relu)
            V = dut.lyapunov_value(
                x_val, x_equilibrium, V_lambda,
                relu.forward(x_equilibrium))
            loss_expected = torch.max(
                Vdot + epsilon * V + margin,
                torch.tensor(0., dtype=system.dtype))
            self.assertAlmostEqual(
                loss_expected.item(),
                dut.lyapunov_derivative_loss_at_samples(
                    V_lambda, epsilon, x_val, x_equilibrium, margin).item())

        relu1 = setup_relu(torch.float64)
        relu2 = setup_leaky_relu(torch.float64)
        for relu in (relu1, relu2):
            for system, x_equilibrium in (
                (self.system1, self.x_equilibrium1),
                    (self.system2, self.x_equilibrium2)):
                dut = lyapunov.LyapunovContinuousTimeHybridSystem(system, relu)
                margin = 0.1
                # First compute the loss for each single point.
                for pt in ([0.1, 0.3], [-0.2, 0.4], [0.5, 0.6],
                           [0.1, -0.4]):
                    loss_expected = compute_loss_single_sample(
                        system, x_equilibrium, relu,
                        torch.tensor(pt, dtype=torch.float64), margin)
                    loss = dut.lyapunov_derivative_loss_at_samples(
                        V_lambda, epsilon,
                        torch.tensor([pt], dtype=torch.float64), x_equilibrium,
                        margin)
                    self.assertAlmostEqual(loss_expected.item(), loss.item())

                # Now compute the loss for a batch of points.
                points = torch.tensor([
                    [0.1, 0.3], [-0.2, 0.4], [0.5, 0.6], [0.1, -0.4]],
                    dtype=system.dtype)
                relu.zero_grad()
                loss = dut.lyapunov_derivative_loss_at_samples(
                    V_lambda, epsilon, points, x_equilibrium, margin)
                loss.backward()
                loss_grad = [p.grad.data.clone() for p in relu.parameters()]

                relu.zero_grad()
                loss_expected = 0
                for i in range(points.shape[0]):
                    loss_expected += compute_loss_single_sample(
                        system, x_equilibrium, relu, points[i], margin)
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

        def compute_loss_single_sample(
                system, x_equilibrium, relu, x_val, margin):
            xdot = system.step_forward(x_val)
            activation_pattern = relu_to_optimization.\
                ComputeReLUActivationPattern(relu, x_val)
            (g, _, _, _) = relu_to_optimization.ReLUGivenActivationPattern(
                relu, system.x_dim, activation_pattern, system.dtype)
            dVdx = g.squeeze() + \
                V_lambda * torch.sign(x_val - x_equilibrium)
            Vdot = dVdx @ xdot
            dut = lyapunov.LyapunovContinuousTimeHybridSystem(system, relu)
            V = dut.lyapunov_value(
                x_val, x_equilibrium, V_lambda, relu.forward(x_equilibrium))
            loss_expected = torch.max(
                Vdot + epsilon * V + margin,
                torch.tensor(0., dtype=system.dtype))
            return loss_expected

        def compute_loss_batch_sample(relu_params, points_test, requires_grad):
            margin = 0.1
            if isinstance(relu_params, np.ndarray):
                relu_params_torch = torch.from_numpy(relu_params)
            else:
                relu_params_torch = relu_params
            relu_test = setup_leaky_relu(torch.float64, relu_params_torch)
            if requires_grad:
                dut = lyapunov.LyapunovContinuousTimeHybridSystem(
                    self.system1, relu_test)
                loss = dut.lyapunov_derivative_loss_at_samples(
                    V_lambda, epsilon, points_test,
                    self.x_equilibrium1, margin)
                loss.backward()
                grad = np.concatenate(
                    [p.grad.detach().numpy().reshape((-1,)) for p in
                     relu_test.parameters()], axis=0)
                return grad
            else:
                with torch.no_grad():
                    loss = 0
                    for i in range(points_test.shape[0]):
                        loss += compute_loss_single_sample(
                            self.system1, self.x_equilibrium1, relu_test,
                            points[i], margin)
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
            grad = compute_loss_batch_sample(relu_param, points, True)
            grad_numerical = utils.compute_numerical_gradient(
                lambda p: compute_loss_batch_sample(p, points, False),
                relu_param.detach().numpy())
            np.testing.assert_allclose(
                grad, grad_numerical.squeeze(), atol=1e-7)


class TestLyapunovDiscreteTimeAutonomousReLUSystem(unittest.TestCase):

    def setUp(self):
        self.dtype = torch.float64
        self.x_equilibrium1 = torch.tensor([0, 0], dtype=self.dtype)
        self.relu_dyn = setup_relu_dyn(self.dtype)
        self.x_lo = torch.tensor([-1e4, -1e4], dtype=self.dtype)
        self.x_up = torch.tensor([1e4, 1e4], dtype=self.dtype)
        self.system1 = relu_system.AutonomousReLUSystem(2, self.dtype,
                                                        self.x_lo, self.x_up,
                                                        self.relu_dyn)

    def test_lyapunov_derivative_as_milp(self):
        """
        Test lyapunov_derivative_as_milp without bounds on V(x[n])
        """
        dut1 = lyapunov.LyapunovDiscreteTimeAutonomousReLUSystem(self.system1)

        relu1 = setup_leaky_relu(self.dtype)
        relu2 = setup_relu(self.dtype)
        V_rho = 2.
        dV_epsilon = 0.1

        def test_milp(dut, x_equilibrium, relu):
            (milp, x, beta, gamma, x_next, s, z, z_next, beta_next) =\
                dut.lyapunov_derivative_as_milp(
                    relu, self.relu_dyn, x_equilibrium, V_rho, dV_epsilon)
            # First solve this MILP. The solution has to satisfy that
            # x_next = Ai * x + g_i where i is the active mode inferred from
            # gamma.
            # milp.gurobi_model.setParam(gurobipy.GRB.Param.DualReductions, 0)
            milp.gurobi_model.setParam(gurobipy.GRB.Param.OutputFlag, 0)
            milp.gurobi_model.optimize()
            if (milp.gurobi_model.status == gurobipy.GRB.Status.INFEASIBLE):
                milp.gurobi_model.computeIIS()
                milp.gurobi_model.write("milp.ilp")
            self.assertEqual(
                milp.gurobi_model.status, gurobipy.GRB.Status.OPTIMAL)
            x_sol = np.array([var.x for var in x])
            x_next_sol = np.array([var.x for var in x_next])
            np.testing.assert_array_almost_equal(
                self.relu_dyn(
                    torch.tensor(x_sol, dtype=self.dtype)).detach().numpy(),
                x_next_sol, decimal=5)
            v_next = dut.lyapunov_value(
                relu, torch.from_numpy(x_next_sol), x_equilibrium, V_rho)
            v = dut.lyapunov_value(
                relu, torch.from_numpy(x_sol), x_equilibrium, V_rho)
            self.assertAlmostEqual(
                milp.gurobi_model.objVal,
                (v_next - v + dV_epsilon * v).item())

        test_milp(dut1, self.x_equilibrium1, relu1)
        test_milp(dut1, self.x_equilibrium1, relu2)

        # Now solve MILP to optimal for system1 and system2
        milp1 = dut1.lyapunov_derivative_as_milp(
                relu1, self.relu_dyn,
                self.x_equilibrium1, V_rho, dV_epsilon)[0]
        milp1.gurobi_model.setParam(gurobipy.GRB.Param.OutputFlag, 0)
        milp1.gurobi_model.optimize()
        milp_optimal_cost1 = milp1.gurobi_model.ObjVal
        milp2 = dut1.lyapunov_derivative_as_milp(
                relu2, self.relu_dyn,
                self.x_equilibrium1, V_rho, dV_epsilon)[0]
        milp2.gurobi_model.setParam(gurobipy.GRB.Param.OutputFlag, 0)
        milp2.gurobi_model.optimize()
        milp_optimal_cost2 = milp2.gurobi_model.ObjVal

        # Now test reformulating
        # ReLU(x[n+1]) + ρ|x[n+1]-x*|₁ - ReLU(x[n]) - ρ|x[n]-x*|₁ +
        # epsilon * (Relu(x[n]) - ReLU(x*) + ρ|x[n]-x*|₁) as a
        # mixed-integer linear program. We fix x[n] to some value, compute the
        # cost function of the MILP, and then check if it is the same as
        # evaluating the ReLU network on x[n] and x[n+1]
        def test_milp_cost(dut, relu, x_val, x_equilibrium, milp_optimal_cost):
            x_next_val = self.relu_dyn(x_val)
            v_next = dut.lyapunov_value(
                relu, x_next_val, x_equilibrium, V_rho)
            v = dut.lyapunov_value(relu, x_val, x_equilibrium, V_rho)
            cost_expected = (v_next - v + dV_epsilon * v).item()
            (milp_test, x_test, _, _, _, _, _, _, _) =\
                dut.lyapunov_derivative_as_milp(
                    relu, self.relu_dyn, x_equilibrium, V_rho, dV_epsilon)
            for i in range(dut.system.x_dim):
                milp_test.addLConstr(
                    [torch.tensor([1.], dtype=milp_test.dtype)], [[x_test[i]]],
                    rhs=x_val[i], sense=gurobipy.GRB.EQUAL)
            milp_test.gurobi_model.setParam(gurobipy.GRB.Param.OutputFlag, 0)
            milp_test.gurobi_model.optimize()
            self.assertEqual(milp_test.gurobi_model.status,
                             gurobipy.GRB.Status.OPTIMAL)
            self.assertAlmostEqual(cost_expected,
                                   milp_test.gurobi_model.objVal)
            # milp_test solves the problem with fixed x[n], so it should
            # achieve less optimal cost than milp1 or milp2.
            self.assertLessEqual(milp_test.gurobi_model.objVal,
                                 milp_optimal_cost)

        # Now test with random x[n]
        torch.manual_seed(0)
        np.random.seed(0)

        def sample_state():
            while True:
                x_val = torch.rand(self.system1.x_dim, dtype=self.dtype) * (
                  self.system1.x_up - self.system1.x_lo) + self.system1.x_lo
                x_val_next = self.relu_dyn(x_val)
                if (torch.all(x_val_next <= self.system1.x_up) and torch.all(
                  x_val_next >= self.system1.x_lo)):
                    return x_val

        for _ in range(20):
            x_val = sample_state()
            test_milp_cost(dut1, relu1, x_val, self.x_equilibrium1,
                           milp_optimal_cost1)
            test_milp_cost(dut1, relu2, x_val, self.x_equilibrium1,
                           milp_optimal_cost2)


if __name__ == "__main__":
    unittest.main()
