import gurobipy
import numpy as np
import unittest
import torch
import torch.nn as nn

import robust_value_approx.hybrid_linear_system as hybrid_linear_system
import robust_value_approx.lyapunov as lyapunov
import robust_value_approx.utils as utils


class TestLyapunovDiscreteTimeHybridSystem(unittest.TestCase):

    def setUp(self):
        """
        The piecewise affine system is from "Analysis of discrete-time
        piecewise affine and hybrid systems" by Giancarlo Ferrari-Trecate
        et.al.
        """
        self.dtype = torch.float64
        self.system1 = hybrid_linear_system.AutonomousHybridLinearSystem(
            2, self.dtype)
        self.system1.add_mode(
            torch.tensor([[-0.999, 0], [-0.139, 0.341]], dtype=self.dtype),
            torch.zeros((2,), dtype=self.dtype),
            torch.tensor([[1, 0], [-1, 0], [0, 1], [0, -1]], dtype=self.dtype),
            torch.tensor([1, 0, 0, 1], dtype=self.dtype))
        self.system1.add_mode(
            torch.tensor([[0.436, 0.323], [0.388, -0.049]], dtype=self.dtype),
            torch.zeros((2,), dtype=self.dtype),
            torch.tensor([[1, 0], [-1, 0], [0, 1], [0, -1]], dtype=self.dtype),
            torch.tensor([1, 0, 1, 0], dtype=self.dtype))
        self.system1.add_mode(
            torch.tensor([[-0.457, 0.215], [0.491, 0.49]], dtype=self.dtype),
            torch.zeros((2,), dtype=self.dtype),
            torch.tensor([[1, 0], [-1, 0], [0, 1], [0, -1]], dtype=self.dtype),
            torch.tensor([0, 1, 0, 1], dtype=self.dtype))
        self.system1.add_mode(
            torch.tensor([[-0.022, 0.344], [0.458, 0.271]], dtype=self.dtype),
            torch.zeros((2,), dtype=self.dtype),
            torch.tensor([[1, 0], [-1, 0], [0, 1], [0, -1]], dtype=self.dtype),
            torch.tensor([0, 1, 1, 0], dtype=self.dtype))

    def test_lyapunov_as_milp(self):
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

        (milp, x, x_next, s, gamma, z, z_next, beta, beta_next) =\
            dut.lyapunov_as_milp(relu1)
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
             relu1.forward(torch.from_numpy(x_sol))).item())

        # Now test reformulating ReLU(x[n+1]) - ReLU(x[n]) as a mixed-integer
        # linear program. We fix x[n] to some value, compute the cost function
        # of the MILP, and then check if it is the same as evaluating the
        # ReLU network on x[n] and x[n+1]
        def test_milp_cost(mode, x_val):
            assert(torch.all(
                self.system1.P[mode] @ x_val <= self.system1.q[mode]))
            x_next_val = self.system1.A[mode] @ x_val + self.system1.g[mode]
            cost_expected = (relu1.forward(x_next_val) -
                             relu1.forward(x_val)).item()
            (milp_test, x_test, _, _, _, _, _, _, _) =\
                dut.lyapunov_as_milp(relu1)
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

    def test_lyapunov_as_milp_gradient(self):
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

            (milp, x, x_next, s, gamma, z, z_next, beta, beta_next) =\
                dut.lyapunov_as_milp(relu1)

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
                weight_grad, grad_numerical[0].squeeze(), rtol=1e-3, atol=0.04)
            np.testing.assert_allclose(
                bias_grad, grad_numerical[1].squeeze(), rtol=1e-3, atol=0.02)

    def test_lyapunov_loss_at_sample(self):
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
            loss = dut.lyapunov_loss_at_sample(relu1, x_sample, margin)
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
