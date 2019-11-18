import gurobipy
import torch
import robust_value_approx.gurobi_torch_mip as gurobi_torch_mip
import unittest

class GurobiTorchMIP(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
