import gurobipy
import torch
import robust_value_approx.gurobi_torch_mip as gurobi_torch_mip
import robust_value_approx.utils as utils
import unittest
import numpy as np


def setup_milp1():
    dtype = torch.float64
    dut = gurobi_torch_mip.GurobiTorchMILP(dtype)
    # The constraints are
    # x[i] >= 0
    # alpha[0] + alpha[1] = 1
    # x[0] + x[1] + x[2] = 1
    # x[0] + x[1] <= alpha[0]
    # x[1] + x[2] <= alpha[1]
    x = dut.addVars(3, lb=0, vtype=gurobipy.GRB.CONTINUOUS)
    alpha = dut.addVars(2, vtype=gurobipy.GRB.BINARY)
    dut.addLConstr(
        [torch.ones(2, dtype=dtype, requires_grad=True)], [alpha],
        sense=gurobipy.GRB.EQUAL,
        rhs=torch.tensor(1, dtype=dtype, requires_grad=True))
    dut.addLConstr([torch.ones(3, dtype=dtype, requires_grad=True)], [x],
                   sense=gurobipy.GRB.EQUAL,
                   rhs=torch.tensor(1, dtype=dtype, requires_grad=True))
    dut.addLConstr(
        [torch.ones(2, dtype=dtype, requires_grad=True),
         torch.tensor([-1], dtype=dtype, requires_grad=True)],
        [x[:2], [alpha[0]]], sense=gurobipy.GRB.LESS_EQUAL,
        rhs=torch.tensor(0, dtype=dtype, requires_grad=True))
    dut.addLConstr(
        [torch.ones(2, dtype=dtype, requires_grad=True),
         torch.tensor([-1], dtype=dtype, requires_grad=True)],
        [x[1:], [alpha[1]]], sense=gurobipy.GRB.LESS_EQUAL,
        rhs=torch.tensor(0, dtype=dtype, requires_grad=True))
    return (dut, x, alpha)


class TestGurobiTorchMIP(unittest.TestCase):
    def test_add_vars(self):
        dut = gurobi_torch_mip.GurobiTorchMIP(torch.float64)
        # Add continuous variables with no bounds
        x = dut.addVars(
            2, lb=-gurobipy.GRB.INFINITY, vtype=gurobipy.GRB.CONTINUOUS)
        self.assertEqual(
            dut.gurobi_model.getAttr(gurobipy.GRB.Attr.NumVars), 2)
        self.assertEqual(dut.r, [x[0], x[1]])
        self.assertEqual(len(dut.zeta), 0)
        self.assertEqual(len(dut.Ain_r_row), 0)
        self.assertEqual(len(dut.Ain_r_col), 0)
        self.assertEqual(len(dut.Ain_r_val), 0)
        self.assertEqual(len(dut.rhs_in), 0)
        self.assertEqual(dut.r_indices, {x[0]: 0, x[1]: 1})
        self.assertEqual(len(dut.zeta_indices), 0)

        # Add continuous variables with bounds
        y = dut.addVars(3, lb=1, ub=2, vtype=gurobipy.GRB.CONTINUOUS)
        self.assertEqual(
            dut.gurobi_model.getAttr(gurobipy.GRB.Attr.NumVars), 5)
        self.assertEqual(dut.r, [x[0], x[1], y[0], y[1], y[2]])
        self.assertEqual(len(dut.zeta), 0)
        self.assertEqual(dut.Ain_r_row, [0, 1, 2, 3, 4, 5])
        self.assertEqual(dut.Ain_r_col, [2, 3, 4, 2, 3, 4])
        self.assertEqual(dut.Ain_r_val,
                         [torch.tensor(-1, dtype=torch.float64),
                          torch.tensor(-1, dtype=torch.float64),
                          torch.tensor(-1, dtype=torch.float64),
                          torch.tensor(1, dtype=torch.float64),
                          torch.tensor(1, dtype=torch.float64),
                          torch.tensor(1, dtype=torch.float64)])
        self.assertEqual(dut.rhs_in,
                         [torch.tensor(-1, dtype=torch.float64),
                          torch.tensor(-1, dtype=torch.float64),
                          torch.tensor(-1, dtype=torch.float64),
                          torch.tensor(2, dtype=torch.float64),
                          torch.tensor(2, dtype=torch.float64),
                          torch.tensor(2, dtype=torch.float64)])
        self.assertEqual(
            dut.r_indices, {x[0]: 0, x[1]: 1, y[0]: 2, y[1]: 3, y[2]: 4})
        self.assertEqual(len(dut.zeta_indices), 0)

        # Add binary variables
        alpha = dut.addVars(2, vtype=gurobipy.GRB.BINARY)
        self.assertEqual(
            dut.gurobi_model.getAttr(gurobipy.GRB.Attr.NumVars), 7)
        self.assertEqual(
            dut.gurobi_model.getAttr(gurobipy.GRB.Attr.NumBinVars), 2)
        self.assertEqual(dut.zeta, [alpha[0], alpha[1]])
        self.assertEqual(len(dut.Ain_zeta_row), 0)
        self.assertEqual(len(dut.Aeq_zeta_row), 0)
        self.assertEqual(
            dut.r_indices, {x[0]: 0, x[1]: 1, y[0]: 2, y[1]: 3, y[2]: 4})
        self.assertEqual(dut.zeta_indices, {alpha[0]: 0, alpha[1]: 1})

    def test_addLConstr(self):
        dut = gurobi_torch_mip.GurobiTorchMIP(torch.float64)
        x = dut.addVars(2, lb=0, vtype=gurobipy.GRB.CONTINUOUS)
        alpha = dut.addVars(2, vtype=gurobipy.GRB.BINARY)
        y = dut.addVars(2, lb=-gurobipy.GRB.INFINITY,
                        vtype=gurobipy.GRB.CONTINUOUS)
        beta = dut.addVars(2, vtype=gurobipy.GRB.BINARY)
        self.assertEqual(len(dut.Ain_r_row), 2)
        # Add an equality constraint on continuous variables.
        _ = dut.addLConstr(
            [torch.tensor([1, 2], dtype=torch.float64)], [x],
            sense=gurobipy.GRB.EQUAL, rhs=torch.tensor(2, dtype=torch.float64))
        dut.gurobi_model.update()
        self.assertEqual(
            dut.gurobi_model.getAttr(gurobipy.GRB.Attr.NumConstrs), 1)
        self.assertEqual(len(dut.Ain_r_row), 2)
        self.assertEqual(len(dut.Ain_r_col), 2)
        self.assertEqual(len(dut.Ain_r_val), 2)
        self.assertEqual(len(dut.Ain_zeta_row), 0)
        self.assertEqual(len(dut.Ain_zeta_col), 0)
        self.assertEqual(len(dut.Ain_zeta_val), 0)
        self.assertEqual(len(dut.rhs_in), 2)
        self.assertEqual(dut.Aeq_r_row, [0, 0])
        self.assertEqual(dut.Aeq_r_col, [0, 1])
        self.assertEqual(dut.Aeq_r_val,
                         [torch.tensor(1, dtype=torch.float64),
                          torch.tensor(2, dtype=torch.float64)])
        self.assertEqual(len(dut.Aeq_zeta_row), 0)
        self.assertEqual(len(dut.Aeq_zeta_col), 0)
        self.assertEqual(len(dut.Aeq_zeta_val), 0)
        self.assertEqual(dut.rhs_eq, [torch.tensor(2, dtype=torch.float64)])

        # Add an equality constraint on binary variables.
        _ = dut.addLConstr(
            [torch.tensor([1, 2], dtype=torch.float64),
             torch.tensor([3, 4], dtype=torch.float64)],
            [beta, alpha], rhs=torch.tensor(3, dtype=torch.float64),
            sense=gurobipy.GRB.EQUAL)
        dut.gurobi_model.update()
        self.assertEqual(
            dut.gurobi_model.getAttr(gurobipy.GRB.Attr.NumConstrs), 2)
        self.assertEqual(dut.Aeq_zeta_row, [1, 1, 1, 1])
        self.assertEqual(dut.Aeq_zeta_col, [2, 3, 0, 1])
        self.assertEqual(
            dut.Aeq_zeta_val,
            [torch.tensor(1, dtype=torch.float64),
             torch.tensor(2, dtype=torch.float64),
             torch.tensor(3, dtype=torch.float64),
             torch.tensor(4, dtype=torch.float64)])
        self.assertEqual(
            dut.rhs_eq, [torch.tensor(2, dtype=torch.float64),
                         torch.tensor(3, dtype=torch.float64)])

        # Add <= constraint on both continuous and binary variables.
        _ = dut.addLConstr(
            [torch.tensor([5, 6], dtype=torch.float64),
             torch.tensor([-1, -2], dtype=torch.float64)], [y, alpha],
            sense=gurobipy.GRB.LESS_EQUAL,
            rhs=torch.tensor(4, dtype=torch.float64))
        dut.gurobi_model.update()
        self.assertEqual(
            dut.gurobi_model.getAttr(gurobipy.GRB.Attr.NumConstrs), 3)
        self.assertEqual(dut.Ain_r_row, [0, 1, 2, 2])
        self.assertEqual(dut.Ain_r_col, [0, 1, 2, 3])
        self.assertEqual(
            dut.Ain_r_val,
            [torch.tensor(-1, dtype=torch.float64),
             torch.tensor(-1, dtype=torch.float64),
             torch.tensor(5, dtype=torch.float64),
             torch.tensor(6, dtype=torch.float64)])
        self.assertEqual(dut.Ain_zeta_row, [2, 2])
        self.assertEqual(dut.Ain_zeta_col, [0, 1])
        self.assertEqual(
            dut.Ain_zeta_val,
            [torch.tensor(-1, dtype=torch.float64),
             torch.tensor(-2, dtype=torch.float64)])
        self.assertEqual(
            dut.rhs_in, [torch.tensor(0, dtype=torch.float64),
                         torch.tensor(0, dtype=torch.float64),
                         torch.tensor(4, dtype=torch.float64)])
        self.assertEqual(
            dut.rhs_eq, [torch.tensor(2, dtype=torch.float64),
                         torch.tensor(3, dtype=torch.float64)])

        # Add >= constraint on both continuous and binary variables.
        _ = dut.addLConstr(
            [torch.tensor([7, 8], dtype=torch.float64),
             torch.tensor([-3, -4], dtype=torch.float64)], [x, beta],
            sense=gurobipy.GRB.GREATER_EQUAL,
            rhs=torch.tensor(5, dtype=torch.float64))
        dut.gurobi_model.update()
        self.assertEqual(
            dut.gurobi_model.getAttr(gurobipy.GRB.Attr.NumConstrs), 4)
        self.assertEqual(dut.Ain_r_row, [0, 1, 2, 2, 3, 3])
        self.assertEqual(dut.Ain_r_col, [0, 1, 2, 3, 0, 1])
        self.assertEqual(
            dut.Ain_r_val,
            [torch.tensor(-1, dtype=torch.float64),
             torch.tensor(-1, dtype=torch.float64),
             torch.tensor(5, dtype=torch.float64),
             torch.tensor(6, dtype=torch.float64),
             torch.tensor(-7, dtype=torch.float64),
             torch.tensor(-8, dtype=torch.float64)])
        self.assertEqual(dut.Ain_zeta_row, [2, 2, 3, 3])
        self.assertEqual(dut.Ain_zeta_col, [0, 1, 2, 3])
        self.assertEqual(
            dut.Ain_zeta_val,
            [torch.tensor(-1, dtype=torch.float64),
             torch.tensor(-2, dtype=torch.float64),
             torch.tensor(3, dtype=torch.float64),
             torch.tensor(4, dtype=torch.float64)])
        self.assertEqual(
            dut.rhs_in, [torch.tensor(0, dtype=torch.float64),
                         torch.tensor(0, dtype=torch.float64),
                         torch.tensor(4, dtype=torch.float64),
                         torch.tensor(-5, dtype=torch.float64)])
        self.assertEqual(
            dut.rhs_eq, [torch.tensor(2, dtype=torch.float64),
                         torch.tensor(3, dtype=torch.float64)])

    def test_get_active_constraints(self):
        dtype = torch.float64
        dut, x, alpha = setup_milp1()

        (A_act, b_act) = dut.get_active_constraints(
            {2, 3}, torch.tensor([1, 0], dtype=dtype))
        self.assertTrue(
            torch.all(
                A_act == torch.tensor([[0, 0, 0], [1, 1, 1], [0, 0, -1],
                                       [1, 1, 0]], dtype=dtype,
                                      requires_grad=True)))
        self.assertTrue(torch.all(b_act == torch.tensor(
            [0, 1, 0, 1], dtype=dtype, requires_grad=True)))

        (A_act, b_act) = dut.get_active_constraints(
            {0, 1, 4}, torch.tensor([0, 1], dtype=dtype))
        self.assertTrue(
            torch.all(
                A_act == torch.tensor([[0, 0, 0], [1, 1, 1], [-1, 0, 0],
                                       [0, -1, 0], [0, 1, 1]], dtype=dtype,
                                      requires_grad=True)))
        self.assertTrue(torch.all(b_act == torch.tensor(
            [0, 1, 0, 0, 1], dtype=dtype, requires_grad=True)))

    def test_get_inequality_constraints(self):
        dtype = torch.float64
        dut = gurobi_torch_mip.GurobiTorchMIP(dtype)
        x = dut.addVars(3, lb=-1., vtype=gurobipy.GRB.CONTINUOUS)
        alpha = dut.addVars(2, vtype=gurobipy.GRB.BINARY)
        y = dut.addVars(2, lb=-gurobipy.GRB.INFINITY,
                        vtype=gurobipy.GRB.CONTINUOUS)
        beta = dut.addVars(3, vtype=gurobipy.GRB.BINARY)
        dut.addLConstr(
            [torch.tensor([1, 2, 3], dtype=dtype),
             torch.tensor([0.5, 1.5], dtype=dtype)], [x, alpha],
            sense=gurobipy.GRB.EQUAL, rhs=2.)
        dut.addLConstr([torch.tensor([2.5, 0.3, 2], dtype=dtype),
                        torch.tensor([0.1, 4.2], dtype=dtype)],
                       [beta, y], sense=gurobipy.GRB.LESS_EQUAL, rhs=3.)
        dut.addLConstr([torch.tensor([0.3, 0.5, 0.1], dtype=dtype),
                        torch.tensor([0.2, 1.5, 3], dtype=dtype),
                        torch.tensor([2.1, 0.5], dtype=dtype)], [x, beta, y],
                       sense=gurobipy.GRB.GREATER_EQUAL, rhs=0.5)
        Ain_r, Ain_zeta, rhs_in = dut.get_inequality_constraints()
        self.assertTrue(
            torch.all(Ain_r == torch.tensor(
                [[-1, 0, 0, 0, 0], [0, -1, 0, 0, 0], [0, 0, -1, 0, 0],
                 [0, 0, 0, 0.1, 4.2], [-0.3, -0.5, -0.1, -2.1, -0.5]],
                dtype=dtype)))
        self.assertTrue(
            torch.all(Ain_zeta == torch.tensor(
                [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0],
                 [0, 0, 2.5, 0.3, 2], [0, 0, -0.2, -1.5, -3]], dtype=dtype)))
        self.assertTrue(
            torch.all(rhs_in == torch.tensor([1, 1, 1, 3, -0.5], dtype=dtype)))


class TestGurobiTorchMILP(unittest.TestCase):
    def test_setObjective(self):
        dtype = torch.float64
        dut = gurobi_torch_mip.GurobiTorchMILP(dtype)
        x = dut.addVars(2, lb=0, vtype=gurobipy.GRB.CONTINUOUS)
        alpha = dut.addVars(3, vtype=gurobipy.GRB.BINARY)
        y = dut.addVars(4, vtype=gurobipy.GRB.CONTINUOUS)
        beta = dut.addVars(1, vtype=gurobipy.GRB.BINARY)
        for sense in (gurobipy.GRB.MINIMIZE, gurobipy.GRB.MAXIMIZE):
            dut.setObjective([torch.tensor([1, 2], dtype=dtype),
                              torch.tensor([2, 0.5], dtype=dtype),
                              torch.tensor([0.5], dtype=dtype),
                              torch.tensor([2.5], dtype=dtype)],
                             [x, [alpha[0], alpha[2]], beta, [y[2]]],
                             constant=3., sense=sense)
            self.assertTrue(
                torch.all(dut.c_r == torch.tensor([1, 2, 0, 0, 2.5, 0],
                                                  dtype=dtype)))
            self.assertTrue(
                torch.all(dut.c_zeta == torch.tensor([2, 0, 0.5, 0.5],
                                                     dtype=dtype)))
            self.assertTrue(dut.c_constant == torch.tensor(3, dtype=dtype))
            self.assertEqual(dut.sense, sense)

    def test_compute_objective_given_active_constraints(self):
        dtype = torch.float64
        dut, x, alpha = setup_milp1()
        # If the objective is max x[0] + 1, then the active constraints are
        # x[1] >= 0, x[2] >= 0, x[0] + x[1] <= alpha[0],
        # x[1] + x[2] <= alpha[1], the binary variable solution is
        # alpha = [1, 0]
        dut.setObjective([torch.tensor([1], dtype=dtype)], [[x[0]]], 1.,
                         gurobipy.GRB.MAXIMIZE)
        self.assertAlmostEqual(
            dut.compute_objective_from_mip_data(
                {1, 2, 3, 4}, torch.tensor([1, 0], dtype=dtype)).item(), 2.)
        # First solve the MILP default pool search mode.
        dut.gurobi_model.setParam(gurobipy.GRB.Param.OutputFlag, 0)
        dut.gurobi_model.optimize()
        self.assertAlmostEqual(
            dut.compute_objective_from_mip_data_and_solution().item(), 2.)
        # Now solve MILP with pool search mode = 2
        dut.gurobi_model.setParam(gurobipy.GRB.Param.PoolSearchMode, 2)
        dut.gurobi_model.setParam(gurobipy.GRB.Param.PoolSolutions, 2)
        dut.gurobi_model.optimize()
        self.assertAlmostEqual(
            dut.compute_objective_from_mip_data_and_solution().item(), 2.)
        self.assertAlmostEqual(
            dut.compute_objective_from_mip_data_and_solution(0).item(), 2.)
        self.assertAlmostEqual(
            dut.compute_objective_from_mip_data_and_solution(1).item(), 1.)

        # If the objective is min x[0] + 2x[1], then the active constraints
        # are x[0] >= 0, x[1] >= 0, x[0] + x[1] <= alpha[0],
        # x[1] + x[2] <= alpha[1], the binary variable solution is
        # alpha = [0, 1]
        dut.setObjective([torch.tensor([1, 2], dtype=dtype)], [x[:2]], 0.,
                         gurobipy.GRB.MINIMIZE)
        self.assertAlmostEqual(
            dut.compute_objective_from_mip_data(
                {0, 1, 3, 4}, torch.tensor([0, 1], dtype=dtype)).item(), 0.)
        # First solve the MILP default pool search mode.
        dut.gurobi_model.optimize()
        self.assertAlmostEqual(
            dut.compute_objective_from_mip_data_and_solution().item(), 0.)
        # Now solve MILP with pool search mode = 2
        dut.gurobi_model.setParam(gurobipy.GRB.Param.PoolSearchMode, 2)
        dut.gurobi_model.setParam(gurobipy.GRB.Param.PoolSolutions, 2)
        dut.gurobi_model.optimize()
        self.assertAlmostEqual(
            dut.compute_objective_from_mip_data_and_solution().item(), 0.)
        self.assertAlmostEqual(
            dut.compute_objective_from_mip_data_and_solution(0).item(), 0.)
        self.assertAlmostEqual(
            dut.compute_objective_from_mip_data_and_solution(1).item(), 1.)

    def test_objective_gradient(self):
        """
        Test if we can compute the gradient of the MIP objective w.r.t the
        constraint/cost data, by using pytorch autograd.
        """
        dtype = torch.float64

        def compute_milp_example_cost(a_numpy, autograd_flag):
            a = torch.from_numpy(a_numpy).type(dtype)
            if autograd_flag:
                a.requires_grad = True
            dut = gurobi_torch_mip.GurobiTorchMILP(dtype)
            x = dut.addVars(3, lb=0., vtype=gurobipy.GRB.CONTINUOUS)
            alpha = dut.addVars(2, vtype=gurobipy.GRB.BINARY)
            dut.addLConstr(
                [torch.stack((a[0] * a[1], a[1] + a[2], 3 * a[2]))],
                [x], sense=gurobipy.GRB.LESS_EQUAL, rhs=a[0] + 2 * a[2] * a[1])
            dut.addLConstr(
                [torch.stack((torch.tensor(2., dtype=dtype), a[1]**2,
                              torch.tensor(0.5, dtype=dtype))),
                 torch.tensor([1., 1.], dtype=dtype)], [x, alpha],
                sense=gurobipy.GRB.EQUAL, rhs=2 * a[0] + 1)
            dut.setObjective(
                [torch.stack((a[0] + a[1], a[0], a[2])),
                 torch.stack((a[1], torch.tensor(3, dtype=dtype)))],
                [x, alpha], a[0] * a[1],
                sense=gurobipy.GRB.MAXIMIZE)
            dut.gurobi_model.setParam(gurobipy.GRB.Param.OutputFlag, False)
            dut.gurobi_model.optimize()
            objective = dut.compute_objective_from_mip_data_and_solution(0)
            if autograd_flag:
                objective.backward()
                return (objective, a.grad)
            else:
                return objective.item()

        (_, grad) = compute_milp_example_cost(np.array([1., 2., 3.]), True)
        grad_numerical = utils.compute_numerical_gradient(
            lambda a: compute_milp_example_cost(a, False),
            np.array([1., 2., 3.]))
        np.testing.assert_array_almost_equal(
            grad.detach().numpy(), grad_numerical)


class TestGurobiTorchMIQP(unittest.TestCase):
    def test_setObjective(self):
        dtype = torch.float64
        dut = gurobi_torch_mip.GurobiTorchMIQP(dtype)
        x = dut.addVars(2, lb=0, vtype=gurobipy.GRB.CONTINUOUS, name="x")
        alpha = dut.addVars(3, vtype=gurobipy.GRB.BINARY, name="alpha")
        y = dut.addVars(4, vtype=gurobipy.GRB.CONTINUOUS, name="y")
        beta = dut.addVars(1, vtype=gurobipy.GRB.BINARY, name="beta")
        for sense in (gurobipy.GRB.MINIMIZE, gurobipy.GRB.MAXIMIZE):
            dut.setObjective([torch.tensor([[1, 2], [3, 4]], dtype=dtype),
                              torch.tensor([[5]], dtype=dtype),
                              torch.tensor([[6], [7]], dtype=dtype)],
                             [(x, x), ([alpha[0]], [alpha[1]]), (x, beta)],
                             [torch.tensor([1, 2], dtype=dtype),
                              torch.tensor([2, 0.5], dtype=dtype),
                              torch.tensor([0.5], dtype=dtype),
                              torch.tensor([2.5], dtype=dtype)],
                             [x, [alpha[0], alpha[2]], beta, [y[2]]],
                             constant=3., sense=sense)
            self.assertTrue(
                torch.all(dut.c_r == torch.tensor([1, 2, 0, 0, 2.5, 0],
                                                  dtype=dtype)))
            self.assertTrue(
                torch.all(dut.c_zeta == torch.tensor([2, 0, 0.5, 0.5],
                                                     dtype=dtype)))
            self.assertTrue(dut.c_constant == torch.tensor(3, dtype=dtype))
            self.assertEqual(dut.sense, sense)
            self.assertTrue(torch.all(dut.Q_r ==
                            torch.tensor([[1, 2, 0, 0, 0, 0],
                                          [3, 4, 0, 0, 0, 0],
                                          [0, 0, 0, 0, 0, 0],
                                          [0, 0, 0, 0, 0, 0],
                                          [0, 0, 0, 0, 0, 0],
                                          [0, 0, 0, 0, 0, 0]], dtype=dtype)))
            self.assertTrue(torch.all(dut.Q_zeta ==
                            torch.tensor([[0, 5, 0, 0],
                                          [0, 0, 0, 0],
                                          [0, 0, 0, 0],
                                          [0, 0, 0, 0]], dtype=dtype)))
            self.assertTrue(torch.all(dut.Q_rzeta ==
                            torch.tensor([[0, 0, 0, 6],
                                          [0, 0, 0, 7],
                                          [0, 0, 0, 0],
                                          [0, 0, 0, 0],
                                          [0, 0, 0, 0],
                                          [0, 0, 0, 0]], dtype=dtype)))


if __name__ == "__main__":
    unittest.main()
