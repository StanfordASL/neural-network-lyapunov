import gurobipy
import numpy as np
import unittest
import torch
import torch.nn as nn

import robust_value_approx.hybrid_linear_system as hybrid_linear_system
import robust_value_approx.lyapunov as lyapunov


class TestLyapunovDiscreteTimeHybridSystem(unittest.TestCase):

    def setUp(self):
        """
        The piecewise affine system is from "Analysis of discrete-time
        piecewise affine and hybrid systems
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
        milp.setParam(gurobipy.GRB.Param.OutputFlag, 0)
        milp.optimize()
        if (milp.status == gurobipy.GRB.Status.INFEASIBLE):
            milp.computeIIS()
            milp.write("milp.ilp")
        self.assertEqual(milp.status, gurobipy.GRB.Status.OPTIMAL)
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
            milp.objVal,
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
                milp_test.addConstr(x_test[i] == x_val[i])
            milp_test.setParam(gurobipy.GRB.Param.OutputFlag, 0)
            milp_test.optimize()
            self.assertEqual(milp_test.status, gurobipy.GRB.Status.OPTIMAL)
            self.assertAlmostEqual(cost_expected, milp_test.objVal)
            # milp solves the problem without the bound on x[n], so it should
            # achieve the largest cost.
            self.assertLessEqual(milp_test.objVal, milp.objVal)

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


if __name__ == "__main__":
    unittest.main()
