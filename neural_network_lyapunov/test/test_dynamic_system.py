import neural_network_lyapunov.dynamic_system as mut
import unittest
import torch
import numpy as np
import gurobipy

import neural_network_lyapunov.test.test_hybrid_linear_system as\
    test_hybrid_linear_system
import neural_network_lyapunov.test.test_relu_system as test_relu_system
import neural_network_lyapunov.relu_system as relu_system
import neural_network_lyapunov.gurobi_torch_mip as gurobi_torch_mip
import neural_network_lyapunov.test.test_lyapunov as test_lyapunov


class TestAddSystemConstraint(unittest.TestCase):
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
        self.system4 = relu_system.AutonomousReLUSystemGivenEquilibrium(
            self.dtype, torch.tensor([-2, -3], dtype=self.dtype),
            torch.tensor([1, -2], dtype=self.dtype),
            test_relu_system.setup_relu_dyn(self.dtype),
            torch.tensor([-1, -2.5], dtype=self.dtype))

    def test(self):
        def test_fun(system, x_val, is_x_valid):
            milp = gurobi_torch_mip.GurobiTorchMILP(self.dtype)
            x = milp.addVars(system.x_dim,
                             lb=-gurobipy.GRB.INFINITY,
                             vtype=gurobipy.GRB.CONTINUOUS)
            mip_cnstr_return = system.mixed_integer_constraints()
            system_constraint_ret = mut._add_system_constraint(
                system, milp, x, None)
            s = system_constraint_ret.slack
            gamma = system_constraint_ret.binary
            # Now fix x to x_val
            for i in range(system.x_dim):
                milp.addLConstr([torch.tensor([1.], dtype=self.dtype)],
                                [[x[i]]],
                                sense=gurobipy.GRB.EQUAL,
                                rhs=x_val[i])
            milp.gurobi_model.setParam(gurobipy.GRB.Param.OutputFlag, 0)
            milp.gurobi_model.setParam(gurobipy.GRB.Param.DualReductions, 0)
            milp.gurobi_model.optimize()
            if is_x_valid:
                self.assertEqual(milp.gurobi_model.status,
                                 gurobipy.GRB.OPTIMAL)
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
                    mip_cnstr_return.Aout_slack @ torch.from_numpy(s_sol) +
                    mip_cnstr_return.Aout_binary @ torch.from_numpy(gamma_sol),
                    system.A[mode_index] @ x_val + system.g[mode_index])
            else:
                self.assertEqual(milp.gurobi_model.status,
                                 gurobipy.GRB.INFEASIBLE)

        test_fun(self.system1, torch.tensor([0.5, 0.2], dtype=self.dtype),
                 True)
        test_fun(self.system1, torch.tensor([-0.5, 0.2], dtype=self.dtype),
                 True)
        test_fun(self.system1, torch.tensor([-1.5, 0.2], dtype=self.dtype),
                 False)
        test_fun(
            self.system2,
            self.R2 @ torch.tensor([-0.5, 0.2], dtype=self.dtype) +
            self.x_equilibrium2, True)
        test_fun(
            self.system2,
            self.R2 @ torch.tensor([-0.5, 1.2], dtype=self.dtype) +
            self.x_equilibrium2, False)
        test_fun(self.system3, torch.tensor([-0.5, 0.7], dtype=self.dtype),
                 True)

    def test_add_system_constraint_binary_relax(self):
        # Test add_system_constraint with binary_var_type=BINARYRELAX
        dtype = torch.float64
        closed_loop_system, _ = \
            test_lyapunov.setup_relu_feedback_system_and_lyapunov(dtype)
        milp = gurobi_torch_mip.GurobiTorchMIP(dtype)
        x = milp.addVars(closed_loop_system.x_dim, lb=-gurobipy.GRB.INFINITY)
        x_next = milp.addVars(closed_loop_system.x_dim,
                              lb=-gurobipy.GRB.INFINITY)
        system_cnstr_return = mut._add_system_constraint(
            closed_loop_system,
            milp,
            x,
            x_next,
            binary_var_type=gurobi_torch_mip.BINARYRELAX)
        self.assertGreater(len(system_cnstr_return.binary), 0)
        for v in system_cnstr_return.binary:
            self.assertEqual(v.vtype, gurobipy.GRB.CONTINUOUS)


if __name__ == "__main__":
    unittest.main()
