import gurobipy
import torch
import robust_value_approx.gurobi_torch_mip as gurobi_torch_mip
import unittest


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


if __name__ == "__main__":
    unittest.main()
