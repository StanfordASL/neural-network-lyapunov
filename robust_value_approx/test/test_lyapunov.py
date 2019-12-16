import gurobipy
import numpy as np
import unittest
import torch
import torch.nn as nn

import robust_value_approx.lyapunov as lyapunov
import robust_value_approx.relu_to_optimization as relu_to_optimization
import robust_value_approx.gurobi_torch_mip as gurobi_torch_mip
import robust_value_approx.utils as utils
import robust_value_approx.test.test_hybrid_linear_system as\
    test_hybrid_linear_system


def setup_relu(dtype):
    # Construct a simple ReLU model with 2 hidden layers
    linear1 = nn.Linear(2, 3)
    linear1.weight.data = torch.tensor([[1, 2], [3, 4], [5, 6]],
                                       dtype=dtype)
    linear1.bias.data = torch.tensor([-11, 10, 5], dtype=dtype)
    linear2 = nn.Linear(3, 4)
    linear2.weight.data = torch.tensor(
            [[-1, -0.5, 1.5], [2, 5, 6], [-2, -3, -4], [1.5, 4, 6]],
            dtype=dtype)
    linear2.bias.data = torch.tensor([-3, 2, 0.7, 1.5], dtype=dtype)
    linear3 = nn.Linear(4, 1)
    linear3.weight.data = torch.tensor([[4, 5, 6, 7]], dtype=dtype)
    linear3.bias.data = torch.tensor([-9], dtype=dtype)
    relu1 = nn.Sequential(
        linear1, nn.ReLU(), linear2, nn.ReLU(), linear3, nn.ReLU())
    assert(not relu1.forward(torch.tensor([0, 0], dtype=dtype)).item() == 0)
    return relu1


class TestLyapunovDiscreteTimeHybridSystem(unittest.TestCase):

    def setUp(self):
        """
        The piecewise affine system is from "Analysis of discrete-time
        piecewise affine and hybrid systems" by Giancarlo Ferrari-Trecate
        et.al.
        """
        self.dtype = torch.float64
        self.system1 = test_hybrid_linear_system.\
            setup_trecate_discrete_time_system()

    def test_lyapunov_positivity_as_milp(self):
        dut = lyapunov.LyapunovDiscreteTimeHybridSystem(self.system1)

        relu1 = setup_relu(dut.system.dtype)
        x_equilibrium = torch.tensor([0, 0], dtype=dut.system.dtype)
        V_epsilon = 0.01
        relu_x_equilibrium = relu1.forward(x_equilibrium)

        def test_fun(x_val):
            # Fix x to different values. Now check if the optimal cost is
            # ReLU(x) - ReLU(x*) - epsilon * |x - x*|₁
            (milp, x) = dut.lyapunov_positivity_as_milp(
                relu1, x_equilibrium, V_epsilon)
            for i in range(dut.system.x_dim):
                milp.addLConstr(
                    [torch.tensor([1.], dtype=self.dtype)], [[x[i]]],
                    rhs=x_val[i], sense=gurobipy.GRB.EQUAL)
            milp.gurobi_model.setParam(gurobipy.GRB.Param.OutputFlag, False)
            milp.gurobi_model.optimize()
            self.assertEqual(milp.gurobi_model.status,
                             gurobipy.GRB.Status.OPTIMAL)
            self.assertAlmostEqual(
                milp.gurobi_model.ObjVal, relu1.forward(x_val).item() -
                relu_x_equilibrium.item() -
                V_epsilon * torch.norm(x_val - x_equilibrium, p=1).item())

        test_fun(torch.tensor([0, 0], dtype=self.dtype))
        test_fun(torch.tensor([0, 0.5], dtype=self.dtype))
        test_fun(torch.tensor([0.1, 0.5], dtype=self.dtype))
        test_fun(torch.tensor([-0.3, 0.8], dtype=self.dtype))
        test_fun(torch.tensor([-0.3, -0.2], dtype=self.dtype))
        test_fun(torch.tensor([0.6, -0.2], dtype=self.dtype))

    def test_lyapunov_gradient_as_milp(self):
        """
        Test lyapunov_gradient_as_milp without bounds on V(x[n])
        """
        dut = lyapunov.LyapunovDiscreteTimeHybridSystem(self.system1)

        relu1 = setup_relu(dut.system.dtype)
        x_equilibrium = torch.tensor([0, 0], dtype=dut.system.dtype)
        dV_epsilon = 0.01
        (milp, x, x_next, s, gamma, z, z_next, beta, beta_next) =\
            dut.lyapunov_gradient_as_milp(relu1, x_equilibrium, dV_epsilon)
        # First solve this MILP. The solution has to satisfy that
        # x_next = Ai * x + g_i where i is the active mode inferred from gamma.
        milp.gurobi_model.setParam(gurobipy.GRB.Param.OutputFlag, 0)
        milp.gurobi_model.optimize()
        if (milp.gurobi_model.status == gurobipy.GRB.Status.INFEASIBLE):
            milp.gurobi_model.computeIIS()
            milp.gurobi_model.write("milp.ilp")
        self.assertEqual(milp.gurobi_model.status, gurobipy.GRB.Status.OPTIMAL)
        x_sol = np.array([var.x for var in x])
        x_next_sol = np.array([var.x for var in x_next])
        gamma_sol = np.array([var.x for var in gamma])
        self.assertAlmostEqual(np.sum(gamma_sol), 1)
        for mode in range(self.system1.num_modes):
            if np.abs(gamma_sol[mode] - 1) < 1E-4:
                # This mode is active.
                # Pi * x <= qi
                np.testing.assert_array_less(
                    self.system1.P[mode].detach().numpy() @ x_sol,
                    self.system1.q[mode].detach().numpy() + 1E-5)
                # x_next = Ai * x + gi
                np.testing.assert_array_almost_equal(
                    self.system1.A[mode].detach().numpy() @ x_sol +
                    self.system1.g[mode].detach().numpy(),
                    x_next_sol, decimal=5)
        self.assertAlmostEqual(
            milp.gurobi_model.objVal,
            (relu1.forward(torch.from_numpy(x_next_sol)) -
             relu1.forward(torch.from_numpy(x_sol)) +
             dV_epsilon * (relu1.forward(torch.from_numpy(x_sol)) -
                           relu1.forward(x_equilibrium))).item())

        # Now test reformulating
        # ReLU(x[n+1]) - ReLU(x[n]) + epsilon * (Relu(x[n]) - ReLU(x*)) as a
        # mixed-integer linear program. We fix x[n] to some value, compute the
        # cost function of the MILP, and then check if it is the same as
        # evaluating the ReLU network on x[n] and x[n+1]
        def test_milp_cost(mode, x_val):
            assert(torch.all(
                self.system1.P[mode] @ x_val <= self.system1.q[mode]))
            x_next_val = self.system1.A[mode] @ x_val + self.system1.g[mode]
            cost_expected = \
                (relu1.forward(x_next_val) - relu1.forward(x_val) +
                 dV_epsilon * (relu1.forward(x_val) -
                               relu1.forward(x_equilibrium))).item()
            (milp_test, x_test, _, _, _, _, _, _, _) =\
                dut.lyapunov_gradient_as_milp(relu1, x_equilibrium, dV_epsilon)
            for i in range(self.system1.x_dim):
                milp_test.addLConstr(
                    [torch.tensor([1.], dtype=milp_test.dtype)], [[x_test[i]]],
                    rhs=x_val[i], sense=gurobipy.GRB.EQUAL)
            milp_test.gurobi_model.setParam(gurobipy.GRB.Param.OutputFlag, 0)
            milp_test.gurobi_model.optimize()
            self.assertEqual(milp_test.gurobi_model.status,
                             gurobipy.GRB.Status.OPTIMAL)
            self.assertAlmostEqual(cost_expected,
                                   milp_test.gurobi_model.objVal)
            # milp solves the problem without the bound on x[n], so it should
            # achieve the largest cost.
            self.assertLessEqual(milp_test.gurobi_model.objVal,
                                 milp.gurobi_model.objVal)

        # Now test with random x[n]
        torch.manual_seed(0)
        np.random.seed(0)
        for i in range(self.system1.num_modes):
            for _ in range(20):
                found_x = False
                while not found_x:
                    x_val = torch.tensor(
                        [np.random.uniform(
                            self.system1.x_lo_all[i], self.system1.x_up_all[i])
                         for i in range(self.system1.x_dim)]
                        ).type(self.system1.dtype)
                    if torch.all(
                            self.system1.P[i] @ x_val <= self.system1.q[i]):
                        found_x = True
                test_milp_cost(i, x_val)

    def test_lyapunov_gradient_as_milp_bounded(self):
        """
        Test lyapunov_gradient_as_milp function, but with a lower and upper
        bounds on V(x[n])
        """
        dut = lyapunov.LyapunovDiscreteTimeHybridSystem(self.system1)
        x_equilibrium = torch.tensor([0, 0], dtype=dut.system.dtype)

        relu1 = setup_relu(dut.system.dtype)
        # First find out what is the lower and upper bound of the ReLU network.
        milp_relu = gurobi_torch_mip.GurobiTorchMILP(dut.system.dtype)
        relu1_free_pattern = relu_to_optimization.ReLUFreePattern(
            relu1, dut.system.dtype)
        Ain_x, Ain_z, Ain_beta, rhs_in, Aeq_x, Aeq_z, Aeq_beta, rhs_eq, a_out,\
            b_out, _, _, _, _ = relu1_free_pattern.output_constraint(
                relu1, torch.tensor([-1.0, -1.0], dtype=dut.system.dtype),
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
        milp_relu.setObjective(
            [a_out], [z], float(b_out), sense=gurobipy.GRB.MAXIMIZE)
        milp_relu.gurobi_model.setParam(gurobipy.GRB.Param.OutputFlag, False)
        milp_relu.gurobi_model.optimize()
        self.assertEqual(
            milp_relu.gurobi_model.status, gurobipy.GRB.Status.OPTIMAL)
        relu_x_equilibrium = relu1.forward(x_equilibrium).item()
        v_upper = milp_relu.gurobi_model.ObjVal - relu_x_equilibrium
        self.assertAlmostEqual(
            relu1.forward(torch.tensor([v.x for v in x],
                          dtype=dut.system.dtype)).item(),
            v_upper + relu_x_equilibrium)
        milp_relu.setObjective(
            [a_out], [z], float(b_out), sense=gurobipy.GRB.MINIMIZE)
        milp_relu.gurobi_model.optimize()
        self.assertEqual(
            milp_relu.gurobi_model.status, gurobipy.GRB.Status.OPTIMAL)
        v_lower = milp_relu.gurobi_model.ObjVal - relu_x_equilibrium
        self.assertAlmostEqual(
            relu1.forward(torch.tensor([v.x for v in x],
                          dtype=dut.system.dtype)).item(),
            v_lower + relu_x_equilibrium)

        # If we set lyapunov_lower to be v_upper + 1, the problem should be
        # infeasible.
        dV_epsilon = 0.01
        (milp, _, _, _, _, _, _, _, _) =\
            dut.lyapunov_gradient_as_milp(
                relu1, x_equilibrium, dV_epsilon, v_upper + 1, v_upper + 2)
        milp.gurobi_model.setParam(gurobipy.GRB.Param.OutputFlag, False)
        milp.gurobi_model.setParam(gurobipy.GRB.Param.DualReductions, 0)
        milp.gurobi_model.optimize()
        self.assertEqual(
            milp.gurobi_model.status, gurobipy.GRB.Status.INFEASIBLE)
        # If we set lyapunov_upper to be v_lower - 1, the problem should be
        # infeasible.
        (milp, _, _, _, _, _, _, _, _) =\
            dut.lyapunov_gradient_as_milp(
                relu1, x_equilibrium, dV_epsilon, v_lower - 2, v_lower - 1)
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
            dut.lyapunov_gradient_as_milp(
                relu1, x_equilibrium, dV_epsilon, lyapunov_lower,
                lyapunov_upper)
        milp.gurobi_model.setParam(gurobipy.GRB.Param.OutputFlag, False)
        milp.gurobi_model.optimize()
        self.assertEqual(milp.gurobi_model.status, gurobipy.GRB.Status.OPTIMAL)
        for _ in range(100):
            x_sample = torch.from_numpy(np.random.random((2,)) * 2 - 1)
            v = relu1.forward(x_sample) - relu_x_equilibrium
            if v >= lyapunov_lower and v <= lyapunov_upper:
                x_next = self.system1.step_forward(x_sample)
                v_next = relu1.forward(x_next) - relu_x_equilibrium
                self.assertLessEqual(v_next - v + dV_epsilon * v,
                                     milp.gurobi_model.ObjVal)

    def test_lyapunov_gradient_as_milp_gradient(self):
        """
        Test the gradient of the MILP optimal cost w.r.t the ReLU network
        weights and bias. I can first compute the gradient through pytorch
        autograd, and then compare that against numerical gradient.
        """
        dut = lyapunov.LyapunovDiscreteTimeHybridSystem(self.system1)

        def compute_milp_cost_given_relu(weight_all, bias_all, requires_grad):
            # Construct a simple ReLU model with 2 hidden layers
            assert(isinstance(weight_all, np.ndarray))
            assert(isinstance(bias_all, np.ndarray))
            assert(weight_all.shape == (22,))
            assert(bias_all.shape == (8,))
            weight_tensor = torch.from_numpy(weight_all).type(self.dtype)
            weight_tensor.requires_grad = True
            bias_tensor = torch.from_numpy(bias_all).type(self.dtype)
            bias_tensor.requires_grad = True
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
                linear1, nn.ReLU(), linear2, nn.ReLU(), linear3, nn.ReLU())

            dV_epsilon = 0.01
            (milp, x, x_next, s, gamma, z, z_next, beta, beta_next) =\
                dut.lyapunov_gradient_as_milp(
                    relu1, torch.tensor([0, 0], dtype=self.system1.dtype),
                    dV_epsilon)

            milp.gurobi_model.setParam(gurobipy.GRB.Param.OutputFlag, False)
            milp.gurobi_model.optimize()
            objective = milp.compute_objective_from_mip_data_and_solution()
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
                return np.array([objective.item()])

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

        for weight_all, bias_all in zip(weight_all_list, bias_all_list):
            (weight_grad, bias_grad) = compute_milp_cost_given_relu(
                weight_all, bias_all, True)
            grad_numerical = utils.compute_numerical_gradient(
                lambda weight, bias: compute_milp_cost_given_relu(
                    weight, bias, False), weight_all, bias_all, dx=1e-6)
            np.testing.assert_allclose(
                weight_grad, grad_numerical[0].squeeze(), rtol=1e-2, atol=0.1)
            np.testing.assert_allclose(
                bias_grad, grad_numerical[1].squeeze(), rtol=1e-2, atol=0.2)

    def test_lyapunov_gradient_loss_at_sample(self):
        dut = lyapunov.LyapunovDiscreteTimeHybridSystem(self.system1)
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

        x_samples = []
        x_samples.append(torch.tensor([-0.5, -0.2], dtype=self.dtype))
        x_samples.append(torch.tensor([-0.5, 0.25], dtype=self.dtype))
        x_samples.append(torch.tensor([-0.7, -0.55], dtype=self.dtype))
        x_samples.append(torch.tensor([0.1, 0.85], dtype=self.dtype))
        x_samples.append(torch.tensor([0.45, 0.35], dtype=self.dtype))
        x_samples.append(torch.tensor([0.45, -0.78], dtype=self.dtype))
        x_samples.append(torch.tensor([0.95, -0.23], dtype=self.dtype))
        margin = 0.1
        for x_sample in x_samples:
            loss = dut.lyapunov_gradient_loss_at_sample(
                relu1, x_sample, margin)
            is_in_mode = False
            for i in range(self.system1.num_modes):
                if (torch.all(
                        self.system1.P[i] @ x_sample <= self.system1.q[i])):
                    x_next = self.system1.A[i] @ x_sample + self.system1.g[i]
                    is_in_mode = True
                    break
            assert(is_in_mode)
            V_x_sample = relu1.forward(x_sample)
            V_x_next = relu1.forward(x_next)
            V_diff = V_x_next - V_x_sample
            loss_expected = V_diff + margin if V_diff > margin else\
                torch.tensor(0., dtype=self.dtype)
            self.assertAlmostEqual(loss.item(), loss_expected.item())


if __name__ == "__main__":
    unittest.main()
