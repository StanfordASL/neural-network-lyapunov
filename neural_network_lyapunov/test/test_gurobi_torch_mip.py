import gurobipy
import torch
import neural_network_lyapunov.gurobi_torch_mip as gurobi_torch_mip
import neural_network_lyapunov.utils as utils
import unittest
import numpy as np


class TestMixedIntegerConstraintsReturn(unittest.TestCase):
    def test_constructor(self):
        dut = gurobi_torch_mip.MixedIntegerConstraintsReturn()
        self.assertEqual(dut.num_out(), 0)
        self.assertEqual(dut.num_eq(), 0)
        self.assertEqual(dut.num_ineq(), 0)
        self.assertEqual(dut.num_input(), 0)
        self.assertEqual(dut.num_slack(), 0)
        self.assertEqual(dut.num_binary(), 0)
        dtype = torch.float64

        dut.Ain_input = torch.tensor([[2], [3]], dtype=dtype)
        dut.rhs_in = torch.tensor([2, 5], dtype=dtype)
        self.assertEqual(dut.num_out(), 0)
        self.assertEqual(dut.num_ineq(), 2)
        self.assertEqual(dut.num_eq(), 0)
        self.assertEqual(dut.num_input(), 1)
        self.assertEqual(dut.num_slack(), 0)
        self.assertEqual(dut.num_binary(), 0)

        dut.Aeq_slack = torch.tensor([[1, 2], [2, 3], [3, 4]], dtype=dtype)
        dut.rhs_eq = torch.tensor([1, 2, 3], dtype=dtype)
        self.assertEqual(dut.num_out(), 0)
        self.assertEqual(dut.num_ineq(), 2)
        self.assertEqual(dut.num_eq(), 3)
        self.assertEqual(dut.num_input(), 1)
        self.assertEqual(dut.num_slack(), 2)
        self.assertEqual(dut.num_binary(), 0)

        dut.Aout_binary = torch.tensor([[2, 1, 3], [3, 2, 1]], dtype=dtype)
        self.assertEqual(dut.num_out(), 2)
        self.assertEqual(dut.num_ineq(), 2)
        self.assertEqual(dut.num_eq(), 3)
        self.assertEqual(dut.num_input(), 1)
        self.assertEqual(dut.num_slack(), 2)
        self.assertEqual(dut.num_binary(), 3)

        dut.Aout_binary = None
        dut.Cout = torch.tensor([2, 1, 3, 4], dtype=dtype)
        self.assertEqual(dut.num_out(), 4)
        self.assertEqual(dut.num_ineq(), 2)
        self.assertEqual(dut.num_eq(), 3)
        self.assertEqual(dut.num_input(), 1)
        self.assertEqual(dut.num_slack(), 2)
        self.assertEqual(dut.num_binary(), 0)

    def test_clone(self):
        dtype = torch.float64
        dut = gurobi_torch_mip.MixedIntegerConstraintsReturn()
        dut.Ain_input = torch.tensor([[2], [3]], dtype=dtype)
        dut.rhs_in = torch.tensor([2, 5], dtype=dtype)
        other = dut.clone()
        np.testing.assert_allclose(other.Ain_input.detach().numpy(),
                                   dut.Ain_input.detach().numpy())
        np.testing.assert_allclose(other.rhs_in.detach().numpy(),
                                   dut.rhs_in.detach().numpy())
        for item in dut.__dict__.keys():
            if item not in ("Ain_input", "rhs_in"):
                self.assertIsNone(other.__dict__[item])

        # Changing other won't affect dut, and vice versa.
        dut.Ain_input = torch.tensor([[3], [4]], dtype=dtype)
        np.testing.assert_allclose(other.Ain_input.detach().numpy(),
                                   np.array([[2], [3]]))
        other.rhs_in = torch.tensor([1, 4], dtype=dtype)
        np.testing.assert_allclose(dut.rhs_in.detach().numpy(),
                                   np.array([2., 5]))

    def transform_input_tester(self, dut, x_eq):
        dtype = torch.float64
        other = dut.clone()
        A = torch.tensor([[1, 0, 2], [2, 1, 2], [3, 1, 2]], dtype=dtype)
        b = torch.tensor([1, 3, 2], dtype=dtype)
        other.transform_input(A, b)
        np.testing.assert_array_less((dut.Ain_input @ x_eq).detach().numpy(),
                                     dut.rhs_in.detach().numpy())
        x_other = A.inverse() @ (x_eq - b)
        np.testing.assert_array_less(
            (other.Ain_input @ x_other).detach().numpy(),
            other.rhs_in.detach().numpy())
        np.testing.assert_allclose(
            (other.Aeq_input @ x_other).detach().numpy(),
            other.rhs_eq.detach().numpy())
        x_samples = utils.uniform_sample_in_box(
            -10 * torch.ones(3, dtype=dtype), 10 * torch.ones(3, dtype=dtype),
            100)
        for i in range(x_samples.shape[0]):
            self.assertEqual(
                torch.all(
                    dut.Ain_input @ (A @ x_samples[i] + b) <= dut.rhs_in),
                torch.all(other.Ain_input @ x_samples[i] <= other.rhs_in))
            out = dut.Aout_input @ (A @ x_samples[i] + b)
            if dut.Cout is not None:
                out += dut.Cout
            np.testing.assert_allclose(out.detach().numpy(),
                                       (other.Aout_input @ x_samples[i] +
                                        other.Cout).detach().numpy())

    def test_transform_input(self):
        dut = gurobi_torch_mip.MixedIntegerConstraintsReturn()
        dtype = torch.float64
        dut.Ain_input = torch.tensor([[1, 2, 3], [3, 4, 5]], dtype=dtype)
        dut.rhs_in = torch.tensor([1, 5], dtype=dtype)
        dut.Aeq_input = torch.tensor([[1, 2, 5], [7, 8, -1]], dtype=dtype)
        dut.Aout_input = torch.tensor([[1, 3, 2]], dtype=dtype)
        x_eq = torch.tensor([-2, -3, 1], dtype=dtype)
        dut.rhs_eq = dut.Aeq_input @ x_eq

        # First test dut.Cout = None
        dut.Cout = None
        self.transform_input_tester(dut, x_eq)
        # Now set dut.Cout
        dut.Cout = torch.tensor([2], dtype=dtype)
        self.transform_input_tester(dut, x_eq)


class TestConcatenateMixedIntegerConstraints(unittest.TestCase):
    def concatenate_tester(self, cnstr1, cnstr2, same_slack, same_binary,
                           stack_output):
        dtype = torch.float64
        ret = gurobi_torch_mip.concatenate_mixed_integer_constraints(
            cnstr1, cnstr2, same_slack, same_binary, stack_output)
        if stack_output:
            self.assertEqual(ret.num_out(),
                             cnstr1.num_out() + cnstr2.num_out())
        self.assertEqual(ret.num_ineq(), cnstr1.num_ineq() + cnstr2.num_ineq())
        self.assertEqual(ret.num_eq(), cnstr1.num_eq() + cnstr2.num_eq())
        self.assertEqual(ret.num_input(), cnstr1.num_input())
        if same_slack:
            self.assertEqual(ret.num_slack(), cnstr1.num_slack())
        else:
            self.assertEqual(ret.num_slack(),
                             cnstr1.num_slack() + cnstr2.num_slack())
        if same_binary:
            self.assertEqual(ret.num_binary(), cnstr1.num_binary())
        else:
            self.assertEqual(ret.num_binary(),
                             cnstr1.num_binary() + cnstr2.num_binary())

        def get_matrix(mat, mat_size):
            if mat is not None:
                return mat
            else:
                return torch.zeros(mat_size, dtype=dtype)

        def check_cat(mat1, mat2, mat_cat, mat1_size, mat2_size, same_var):
            if mat1 is None and mat2 is None:
                self.assertIsNone(mat_cat)
            else:
                mat1_tensor = get_matrix(mat1, mat1_size)
                mat2_tensor = get_matrix(mat2, mat2_size)
                if same_var:
                    np.testing.assert_allclose(
                        mat_cat.detach().numpy(),
                        torch.cat((mat1_tensor, mat2_tensor),
                                  dim=0).detach().numpy())
                else:
                    np.testing.assert_allclose(
                        mat_cat.detach().numpy(),
                        torch.block_diag(mat1_tensor,
                                         mat2_tensor).detach().numpy())

        if stack_output:
            check_cat(cnstr1.Aout_input, cnstr2.Aout_input, ret.Aout_input,
                      (cnstr1.num_out(), cnstr1.num_input()),
                      (cnstr2.num_out(), cnstr2.num_input()), True)
            check_cat(cnstr1.Aout_slack, cnstr2.Aout_slack, ret.Aout_slack,
                      (cnstr1.num_out(), cnstr1.num_slack()),
                      (cnstr2.num_out(), cnstr2.num_slack()), same_slack)
            check_cat(cnstr1.Aout_binary, cnstr2.Aout_binary, ret.Aout_binary,
                      (cnstr1.num_out(), cnstr1.num_binary()),
                      (cnstr2.num_out(), cnstr2.num_binary()), same_binary)
            check_cat(cnstr1.Cout, cnstr2.Cout, ret.Cout, (cnstr1.num_out(), ),
                      (cnstr2.num_out(), ), True)

        check_cat(cnstr1.Ain_input, cnstr2.Ain_input, ret.Ain_input,
                  (cnstr1.num_ineq(), cnstr1.num_input()),
                  (cnstr2.num_ineq(), cnstr2.num_input()), True)
        check_cat(cnstr1.Ain_slack, cnstr2.Ain_slack, ret.Ain_slack,
                  (cnstr1.num_ineq(), cnstr1.num_slack()),
                  (cnstr2.num_ineq(), cnstr2.num_slack()), same_slack)
        check_cat(cnstr1.Ain_binary, cnstr2.Ain_binary, ret.Ain_binary,
                  (cnstr1.num_ineq(), cnstr1.num_binary()),
                  (cnstr2.num_ineq(), cnstr2.num_binary()), same_binary)
        check_cat(cnstr1.rhs_in, cnstr2.rhs_in, ret.rhs_in,
                  (cnstr1.num_ineq(), ), (cnstr2.num_ineq(), ), True)

        check_cat(cnstr1.Aeq_input, cnstr2.Aeq_input, ret.Aeq_input,
                  (cnstr1.num_eq(), cnstr1.num_input()),
                  (cnstr2.num_eq(), cnstr2.num_input()), True)
        check_cat(cnstr1.Aeq_slack, cnstr2.Aeq_slack, ret.Aeq_slack,
                  (cnstr1.num_eq(), cnstr1.num_slack()),
                  (cnstr2.num_eq(), cnstr2.num_slack()), same_slack)
        check_cat(cnstr1.Aeq_binary, cnstr2.Aeq_binary, ret.Aeq_binary,
                  (cnstr1.num_eq(), cnstr1.num_binary()),
                  (cnstr2.num_eq(), cnstr2.num_binary()), same_binary)
        check_cat(cnstr1.rhs_eq, cnstr2.rhs_eq, ret.rhs_eq,
                  (cnstr1.num_eq(), ), (cnstr2.num_eq(), ), True)

        def check_bnd(bnd1, bnd2, bnd_cat, num_bnd1, num_bnd2, same_var,
                      upper_bound):
            if bnd1 is None and bnd2 is None:
                self.assertIsNone(bnd_cat)
            else:
                bnd_default_val = np.inf if upper_bound else -np.inf
                if bnd1 is None:
                    bnd1_tensor = torch.full((num_bnd1, ),
                                             bnd_default_val,
                                             dtype=dtype)
                else:
                    bnd1_tensor = bnd1
                if bnd2 is None:
                    bnd2_tensor = torch.full((num_bnd2, ),
                                             bnd_default_val,
                                             dtype=dtype)
                else:
                    bnd2_tensor = bnd2
                if same_var:
                    if upper_bound:
                        np.testing.assert_allclose(
                            bnd_cat.detach().numpy(),
                            torch.minimum(bnd1_tensor,
                                          bnd2_tensor).detach().numpy())
                    else:
                        np.testing.assert_allclose(
                            bnd_cat.detach().numpy(),
                            torch.maximum(bnd1_tensor,
                                          bnd2_tensor).detach().numpy())
                else:
                    np.testing.assert_allclose(
                        bnd_cat.detach().numpy(),
                        torch.cat((bnd1_tensor, bnd2_tensor)).detach().numpy())

        check_bnd(cnstr1.input_lo, cnstr2.input_lo, ret.input_lo,
                  cnstr1.num_input(), cnstr2.num_input(), True, False)
        check_bnd(cnstr1.input_up, cnstr2.input_up, ret.input_up,
                  cnstr1.num_input(), cnstr2.num_input(), True, True)
        check_bnd(cnstr1.slack_lo, cnstr2.slack_lo, ret.slack_lo,
                  cnstr1.num_slack(), cnstr2.num_slack(), same_slack, False)
        check_bnd(cnstr1.slack_up, cnstr2.slack_up, ret.slack_up,
                  cnstr1.num_slack(), cnstr2.num_slack(), same_slack, True)
        check_bnd(cnstr1.binary_lo, cnstr2.binary_lo, ret.binary_lo,
                  cnstr1.num_binary(), cnstr2.num_binary(), same_binary, False)
        check_bnd(cnstr1.binary_up, cnstr2.binary_up, ret.binary_up,
                  cnstr1.num_binary(), cnstr2.num_binary(), same_binary, True)

    def test1(self):
        cnstr1 = gurobi_torch_mip.MixedIntegerConstraintsReturn()
        cnstr2 = gurobi_torch_mip.MixedIntegerConstraintsReturn()
        ret = gurobi_torch_mip.concatenate_mixed_integer_constraints(
            cnstr1,
            cnstr2,
            same_slack=True,
            same_binary=True,
            stack_output=True)
        self.assertEqual(ret.num_out(), 0)
        self.assertEqual(ret.num_ineq(), 0)
        self.assertEqual(ret.num_eq(), 0)
        self.assertEqual(ret.num_input(), 0)
        self.assertEqual(ret.num_slack(), 0)
        self.assertEqual(ret.num_binary(), 0)

    def test2(self):
        # Test the input
        dtype = torch.float64
        cnstr1 = gurobi_torch_mip.MixedIntegerConstraintsReturn()
        cnstr2 = gurobi_torch_mip.MixedIntegerConstraintsReturn()
        cnstr1.Ain_input = torch.tensor([[2, 3, 4], [1, 2, 4]], dtype=dtype)
        cnstr1.rhs_in = torch.tensor([2, 3], dtype=dtype)
        cnstr2.Aout_input = torch.tensor([[1, 2, 3]], dtype=dtype)
        self.concatenate_tester(cnstr1,
                                cnstr2,
                                same_slack=False,
                                same_binary=False,
                                stack_output=False)
        self.concatenate_tester(cnstr1,
                                cnstr2,
                                same_slack=False,
                                same_binary=True,
                                stack_output=True)

        cnstr1.input_lo = torch.tensor([-2, -3, -1], dtype=dtype)
        cnstr2.input_lo = torch.tensor([-1, -4, 2], dtype=dtype)
        cnstr1.input_up = None
        cnstr2.input_up = torch.tensor([3, 4, 5], dtype=dtype)
        self.concatenate_tester(cnstr1,
                                cnstr2,
                                same_slack=False,
                                same_binary=True,
                                stack_output=True)

    def test3(self):
        # test the slack.
        dtype = torch.float64
        cnstr1 = gurobi_torch_mip.MixedIntegerConstraintsReturn()
        cnstr2 = gurobi_torch_mip.MixedIntegerConstraintsReturn()
        cnstr1.Ain_input = torch.tensor([[2, 3, 4], [1, 2, 4]], dtype=dtype)
        cnstr1.rhs_in = torch.tensor([2, 3], dtype=dtype)
        cnstr1.Ain_slack = torch.tensor([[1, 3], [2, 5]], dtype=dtype)
        cnstr2.Aout_input = torch.tensor([[1, 2, 3]], dtype=dtype)
        cnstr2.Ain_input = torch.tensor([[2, 1, 0]], dtype=dtype)
        cnstr2.rhs_in = torch.tensor([1], dtype=dtype)
        self.concatenate_tester(cnstr1,
                                cnstr2,
                                same_slack=False,
                                same_binary=False,
                                stack_output=True)

        cnstr2.Aeq_slack = torch.tensor([[1, 3], [2, 4]], dtype=dtype)
        cnstr2.rhs_eq = torch.tensor([1, 3], dtype=dtype)
        self.concatenate_tester(cnstr1,
                                cnstr2,
                                same_slack=False,
                                same_binary=False,
                                stack_output=True)

        cnstr1.Aeq_slack = torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=dtype)
        cnstr1.rhs_eq = torch.tensor([1, 2, 3], dtype=dtype)
        cnstr1.slack_lo = torch.tensor([1, 3], dtype=dtype)
        cnstr1.slack_up = torch.tensor([2, 5], dtype=dtype)
        cnstr2.slack_lo = None
        cnstr2.slack_up = torch.tensor([1, 6], dtype=dtype)
        self.concatenate_tester(cnstr1,
                                cnstr2,
                                same_slack=False,
                                same_binary=False,
                                stack_output=True)
        self.concatenate_tester(cnstr1,
                                cnstr2,
                                same_slack=True,
                                same_binary=False,
                                stack_output=True)

    def test4(self):
        dtype = torch.float64
        cnstr1 = gurobi_torch_mip.MixedIntegerConstraintsReturn()
        cnstr2 = gurobi_torch_mip.MixedIntegerConstraintsReturn()
        cnstr1.Ain_input = torch.tensor([[2, 3, 4], [1, 2, 4]], dtype=dtype)
        cnstr1.rhs_in = torch.tensor([2, 3], dtype=dtype)
        cnstr1.Aout_binary = torch.tensor([[1, 3]], dtype=dtype)
        cnstr2.Aout_input = torch.tensor([[1, 2, 3]], dtype=dtype)
        cnstr2.Ain_binary = torch.tensor([[2, 3], [1, 4], [5, 6]], dtype=dtype)
        cnstr2.rhs_in = torch.tensor([1, 3, 4], dtype=dtype)
        self.concatenate_tester(cnstr1,
                                cnstr2,
                                same_slack=True,
                                same_binary=True,
                                stack_output=False)
        cnstr2.binary_lo = torch.tensor([0, 1], dtype=dtype)
        cnstr1.binary_up = torch.tensor([1, 0], dtype=dtype)
        self.concatenate_tester(cnstr1,
                                cnstr2,
                                same_slack=True,
                                same_binary=True,
                                stack_output=False)
        self.concatenate_tester(cnstr1,
                                cnstr2,
                                same_slack=True,
                                same_binary=False,
                                stack_output=True)


def setup_mip1(dut):
    dtype = torch.float64
    # The constraints are
    # x[i] >= 0
    # alpha[0] + alpha[1] = 1
    # x[0] + x[1] + x[2] = 1
    # x[0] + x[1] <= alpha[0]
    # x[1] + x[2] <= alpha[1]
    x = dut.addVars(3, lb=0, vtype=gurobipy.GRB.CONTINUOUS)
    alpha = dut.addVars(2, vtype=gurobipy.GRB.BINARY)
    dut.addLConstr([torch.ones(2, dtype=dtype, requires_grad=True)], [alpha],
                   sense=gurobipy.GRB.EQUAL,
                   rhs=torch.tensor(1, dtype=dtype, requires_grad=True))
    dut.addLConstr([torch.ones(3, dtype=dtype, requires_grad=True)], [x],
                   sense=gurobipy.GRB.EQUAL,
                   rhs=torch.tensor(1, dtype=dtype, requires_grad=True))
    dut.addLConstr([
        torch.ones(2, dtype=dtype, requires_grad=True),
        torch.tensor([-1], dtype=dtype, requires_grad=True)
    ], [x[:2], [alpha[0]]],
                   sense=gurobipy.GRB.LESS_EQUAL,
                   rhs=torch.tensor(0, dtype=dtype, requires_grad=True))
    dut.addLConstr([
        torch.ones(2, dtype=dtype, requires_grad=True),
        torch.tensor([-1], dtype=dtype, requires_grad=True)
    ], [x[1:], [alpha[1]]],
                   sense=gurobipy.GRB.LESS_EQUAL,
                   rhs=torch.tensor(0, dtype=dtype, requires_grad=True))
    return (x, alpha)


class TestGurobiTorchMIP(unittest.TestCase):
    def test_add_vars1(self):
        dut = gurobi_torch_mip.GurobiTorchMIP(torch.float64)
        # Add continuous variables with no bounds
        x = dut.addVars(2,
                        lb=-gurobipy.GRB.INFINITY,
                        vtype=gurobipy.GRB.CONTINUOUS)
        self.assertEqual(dut.gurobi_model.getAttr(gurobipy.GRB.Attr.NumVars),
                         2)
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
        for i in range(3):
            self.assertEqual(y[i].lb, 1)
            self.assertEqual(y[i].ub, 2)
        self.assertEqual(dut.gurobi_model.getAttr(gurobipy.GRB.Attr.NumVars),
                         5)
        self.assertEqual(dut.r, [x[0], x[1], y[0], y[1], y[2]])
        self.assertEqual(len(dut.zeta), 0)
        self.assertEqual(dut.Ain_r_row, [0, 1, 2, 3, 4, 5])
        self.assertEqual(dut.Ain_r_col, [2, 3, 4, 2, 3, 4])
        self.assertEqual(dut.Ain_r_val, [
            torch.tensor(-1, dtype=torch.float64),
            torch.tensor(-1, dtype=torch.float64),
            torch.tensor(-1, dtype=torch.float64),
            torch.tensor(1, dtype=torch.float64),
            torch.tensor(1, dtype=torch.float64),
            torch.tensor(1, dtype=torch.float64)
        ])
        self.assertEqual(dut.rhs_in, [
            torch.tensor(-1, dtype=torch.float64),
            torch.tensor(-1, dtype=torch.float64),
            torch.tensor(-1, dtype=torch.float64),
            torch.tensor(2, dtype=torch.float64),
            torch.tensor(2, dtype=torch.float64),
            torch.tensor(2, dtype=torch.float64)
        ])
        self.assertEqual(dut.r_indices, {
            x[0]: 0,
            x[1]: 1,
            y[0]: 2,
            y[1]: 3,
            y[2]: 4
        })
        self.assertEqual(len(dut.zeta_indices), 0)

        # Add binary variables
        alpha = dut.addVars(2, vtype=gurobipy.GRB.BINARY)
        self.assertEqual(dut.gurobi_model.getAttr(gurobipy.GRB.Attr.NumVars),
                         7)
        self.assertEqual(
            dut.gurobi_model.getAttr(gurobipy.GRB.Attr.NumBinVars), 2)
        self.assertEqual(dut.zeta, [alpha[0], alpha[1]])
        self.assertEqual(len(dut.Ain_zeta_row), 0)
        self.assertEqual(len(dut.Aeq_zeta_row), 0)
        self.assertEqual(dut.r_indices, {
            x[0]: 0,
            x[1]: 1,
            y[0]: 2,
            y[1]: 3,
            y[2]: 4
        })
        self.assertEqual(dut.zeta_indices, {alpha[0]: 0, alpha[1]: 1})

    def test_addVars2(self):
        # addVars for continuous variable with a tensor type of lb and(or) ub.
        dtype = torch.float64
        dut = gurobi_torch_mip.GurobiTorchMIP(dtype)
        lb = torch.tensor([-2, -np.inf, 4., -np.inf, 5], dtype=dtype)
        ub = torch.tensor([4., 3, np.inf, np.inf, 5], dtype=dtype)
        x = dut.addVars(5, lb=lb, ub=ub, vtype=gurobipy.GRB.CONTINUOUS)
        self.assertEqual(len(x), 5)
        for i in range(5):
            self.assertEqual(x[i].lb, lb[i].item())
            self.assertEqual(x[i].ub, ub[i].item())
            self.assertEqual(x[i].vtype, gurobipy.GRB.CONTINUOUS)
        self.assertListEqual(dut.Ain_r_row, [0, 1, 2, 3])
        self.assertListEqual(dut.Ain_r_col, [0, 2, 0, 1])
        self.assertEqual(dut.Ain_r_val, [
            torch.tensor(-1, dtype=dtype),
            torch.tensor(-1, dtype=dtype),
            torch.tensor(1, dtype=dtype),
            torch.tensor(1, dtype=dtype)
        ])
        self.assertEqual(dut.rhs_in, [
            torch.tensor(2, dtype=dtype),
            torch.tensor(-4, dtype=dtype),
            torch.tensor(4, dtype=dtype),
            torch.tensor(3, dtype=dtype)
        ])
        self.assertEqual(len(dut.Ain_zeta_row), 0)
        self.assertEqual(len(dut.Ain_zeta_col), 0)
        self.assertEqual(len(dut.Ain_zeta_val), 0)

        self.assertEqual(dut.Aeq_r_row, [0])
        self.assertEqual(dut.Aeq_r_col, [4])
        self.assertEqual(dut.Aeq_r_val, [torch.tensor(1, dtype=dtype)])
        self.assertEqual(dut.rhs_eq, [torch.tensor(5, dtype=dtype)])

        self.assertEqual(len(dut.Aeq_zeta_row), 0)
        self.assertEqual(len(dut.Aeq_zeta_col), 0)
        self.assertEqual(len(dut.Aeq_zeta_val), 0)

    def test_addVars3(self):
        # addVars for binary variable with a tensor type of lb and(or) ub.
        dtype = torch.float64
        dut = gurobi_torch_mip.GurobiTorchMIP(dtype)
        lb = torch.tensor([0., -1., 1., -np.inf, 0], dtype=dtype)
        ub = torch.tensor([1., 0., 2., np.inf, 0], dtype=dtype)
        b = dut.addVars(5, lb=lb, ub=ub, vtype=gurobipy.GRB.BINARY)
        for i in range(5):
            self.assertEqual(b[i].lb, torch.clamp(lb[i], 0, 1).item())
            self.assertEqual(b[i].ub, torch.clamp(ub[i], 0, 1).item())
            self.assertEqual(b[i].vtype, gurobipy.GRB.BINARY)
        self.assertEqual(len(dut.Ain_r_row), 0)
        self.assertEqual(len(dut.Ain_r_col), 0)
        self.assertEqual(len(dut.Ain_r_val), 0)
        self.assertEqual(len(dut.Aeq_r_row), 0)
        self.assertEqual(len(dut.Aeq_r_col), 0)
        self.assertEqual(len(dut.Aeq_r_val), 0)
        self.assertEqual(len(dut.Ain_zeta_row), 0)
        self.assertEqual(len(dut.Ain_zeta_col), 0)
        self.assertEqual(len(dut.Ain_zeta_val), 0)
        self.assertEqual(len(dut.Aeq_zeta_row), 0)
        self.assertEqual(len(dut.Aeq_zeta_col), 0)
        self.assertEqual(len(dut.Aeq_zeta_val), 0)

    def test_addVars4(self):
        # Test addVars with vtype = BINARYRELAX
        dtype = torch.float64
        dut = gurobi_torch_mip.GurobiTorchMIP(dtype)
        x = dut.addVars(2,
                        lb=-gurobipy.GRB.INFINITY,
                        vtype=gurobi_torch_mip.BINARYRELAX,
                        name="x")
        self.assertEqual(len(x), 2)
        self.assertEqual(
            dut.gurobi_model.getAttr(gurobipy.GRB.Attr.NumBinVars), 0)
        self.assertEqual(dut.gurobi_model.getAttr(gurobipy.GRB.Attr.NumVars),
                         2)
        for i in range(2):
            self.assertEqual(x[i].lb, 0.)
            self.assertEqual(x[i].ub, 1.)
            self.assertEqual(x[i].vtype, gurobipy.GRB.CONTINUOUS)
        # Now check if x is registered in zeta
        self.assertEqual(len(dut.zeta), 2)
        self.assertEqual(len(dut.r), 0)

        # Now add binary_relax variable with specified bounds.
        y = dut.addVars(3, lb=0.5, ub=0.6, vtype=gurobi_torch_mip.BINARYRELAX)
        self.assertEqual(dut.gurobi_model.getAttr(gurobipy.GRB.Attr.NumVars),
                         5)
        self.assertEqual(
            dut.gurobi_model.getAttr(gurobipy.GRB.Attr.NumBinVars), 0)
        for i in range(3):
            self.assertEqual(y[i].lb, 0.5)
            self.assertEqual(y[i].ub, 0.6)
            self.assertEqual(y[i].vtype, gurobipy.GRB.CONTINUOUS)
        # Now check if y is registered in zeta
        self.assertEqual(len(dut.zeta), 5)
        self.assertEqual(len(dut.r), 0)

        # Now add a constraint on x and y. I should expect to see coefficients
        # on zeta.
        dut.addLConstr(
            [torch.ones((2, ), dtype=dtype),
             torch.ones((3, ), dtype=dtype)], [x, y],
            rhs=1.,
            sense=gurobipy.GRB.EQUAL)
        self.assertEqual(len(dut.Aeq_r_row), 0)
        self.assertEqual(len(dut.Aeq_r_col), 0)
        self.assertEqual(len(dut.Aeq_r_val), 0)
        self.assertEqual(len(dut.Aeq_zeta_row), 5)
        self.assertEqual(len(dut.Aeq_zeta_col), 5)
        self.assertEqual(len(dut.Aeq_zeta_val), 5)

    def test_addLConstr(self):
        dut = gurobi_torch_mip.GurobiTorchMIP(torch.float64)
        x = dut.addVars(2, lb=0, vtype=gurobipy.GRB.CONTINUOUS)
        alpha = dut.addVars(2, vtype=gurobipy.GRB.BINARY)
        y = dut.addVars(2,
                        lb=-gurobipy.GRB.INFINITY,
                        vtype=gurobipy.GRB.CONTINUOUS)
        beta = dut.addVars(2, vtype=gurobipy.GRB.BINARY)
        self.assertEqual(len(dut.Ain_r_row), 2)
        # Add an equality constraint on continuous variables.
        _ = dut.addLConstr([torch.tensor([1, 2], dtype=torch.float64)], [x],
                           sense=gurobipy.GRB.EQUAL,
                           rhs=torch.tensor(2, dtype=torch.float64))
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
        self.assertEqual(dut.Aeq_r_val, [
            torch.tensor(1, dtype=torch.float64),
            torch.tensor(2, dtype=torch.float64)
        ])
        self.assertEqual(len(dut.Aeq_zeta_row), 0)
        self.assertEqual(len(dut.Aeq_zeta_col), 0)
        self.assertEqual(len(dut.Aeq_zeta_val), 0)
        self.assertEqual(dut.rhs_eq, [torch.tensor(2, dtype=torch.float64)])

        # Add an equality constraint on binary variables.
        _ = dut.addLConstr([
            torch.tensor([1, 2], dtype=torch.float64),
            torch.tensor([3, 4], dtype=torch.float64)
        ], [beta, alpha],
                           rhs=torch.tensor(3, dtype=torch.float64),
                           sense=gurobipy.GRB.EQUAL)
        dut.gurobi_model.update()
        self.assertEqual(
            dut.gurobi_model.getAttr(gurobipy.GRB.Attr.NumConstrs), 2)
        self.assertEqual(dut.Aeq_zeta_row, [1, 1, 1, 1])
        self.assertEqual(dut.Aeq_zeta_col, [2, 3, 0, 1])
        self.assertEqual(dut.Aeq_zeta_val, [
            torch.tensor(1, dtype=torch.float64),
            torch.tensor(2, dtype=torch.float64),
            torch.tensor(3, dtype=torch.float64),
            torch.tensor(4, dtype=torch.float64)
        ])
        self.assertEqual(dut.rhs_eq, [
            torch.tensor(2, dtype=torch.float64),
            torch.tensor(3, dtype=torch.float64)
        ])

        # Add <= constraint on both continuous and binary variables.
        _ = dut.addLConstr([
            torch.tensor([5, 6], dtype=torch.float64),
            torch.tensor([-1, -2], dtype=torch.float64)
        ], [y, alpha],
                           sense=gurobipy.GRB.LESS_EQUAL,
                           rhs=torch.tensor(4, dtype=torch.float64))
        dut.gurobi_model.update()
        self.assertEqual(
            dut.gurobi_model.getAttr(gurobipy.GRB.Attr.NumConstrs), 3)
        self.assertEqual(dut.Ain_r_row, [0, 1, 2, 2])
        self.assertEqual(dut.Ain_r_col, [0, 1, 2, 3])
        self.assertEqual(dut.Ain_r_val, [
            torch.tensor(-1, dtype=torch.float64),
            torch.tensor(-1, dtype=torch.float64),
            torch.tensor(5, dtype=torch.float64),
            torch.tensor(6, dtype=torch.float64)
        ])
        self.assertEqual(dut.Ain_zeta_row, [2, 2])
        self.assertEqual(dut.Ain_zeta_col, [0, 1])
        self.assertEqual(dut.Ain_zeta_val, [
            torch.tensor(-1, dtype=torch.float64),
            torch.tensor(-2, dtype=torch.float64)
        ])
        self.assertEqual(dut.rhs_in, [
            torch.tensor(0, dtype=torch.float64),
            torch.tensor(0, dtype=torch.float64),
            torch.tensor(4, dtype=torch.float64)
        ])
        self.assertEqual(dut.rhs_eq, [
            torch.tensor(2, dtype=torch.float64),
            torch.tensor(3, dtype=torch.float64)
        ])

        # Add >= constraint on both continuous and binary variables.
        _ = dut.addLConstr([
            torch.tensor([7, 8], dtype=torch.float64),
            torch.tensor([-3, -4], dtype=torch.float64)
        ], [x, beta],
                           sense=gurobipy.GRB.GREATER_EQUAL,
                           rhs=torch.tensor(5, dtype=torch.float64))
        dut.gurobi_model.update()
        self.assertEqual(
            dut.gurobi_model.getAttr(gurobipy.GRB.Attr.NumConstrs), 4)
        self.assertEqual(dut.Ain_r_row, [0, 1, 2, 2, 3, 3])
        self.assertEqual(dut.Ain_r_col, [0, 1, 2, 3, 0, 1])
        self.assertEqual(dut.Ain_r_val, [
            torch.tensor(-1, dtype=torch.float64),
            torch.tensor(-1, dtype=torch.float64),
            torch.tensor(5, dtype=torch.float64),
            torch.tensor(6, dtype=torch.float64),
            torch.tensor(-7, dtype=torch.float64),
            torch.tensor(-8, dtype=torch.float64)
        ])
        self.assertEqual(dut.Ain_zeta_row, [2, 2, 3, 3])
        self.assertEqual(dut.Ain_zeta_col, [0, 1, 2, 3])
        self.assertEqual(dut.Ain_zeta_val, [
            torch.tensor(-1, dtype=torch.float64),
            torch.tensor(-2, dtype=torch.float64),
            torch.tensor(3, dtype=torch.float64),
            torch.tensor(4, dtype=torch.float64)
        ])
        self.assertEqual(dut.rhs_in, [
            torch.tensor(0, dtype=torch.float64),
            torch.tensor(0, dtype=torch.float64),
            torch.tensor(4, dtype=torch.float64),
            torch.tensor(-5, dtype=torch.float64)
        ])
        self.assertEqual(dut.rhs_eq, [
            torch.tensor(2, dtype=torch.float64),
            torch.tensor(3, dtype=torch.float64)
        ])

    def test_addMConstrs(self):
        dtype = torch.float64
        dut = gurobi_torch_mip.GurobiTorchMIP(dtype)
        x = dut.addVars(2, lb=0, vtype=gurobipy.GRB.CONTINUOUS)
        alpha = dut.addVars(2, vtype=gurobipy.GRB.BINARY)
        y = dut.addVars(2,
                        lb=-gurobipy.GRB.INFINITY,
                        vtype=gurobipy.GRB.CONTINUOUS)
        beta = dut.addVars(2, vtype=gurobipy.GRB.BINARY)
        # Add constraint [[1, 2], [3, 4]] * x + [[2, 3], [4, 5]] * beta ==
        # [5, 6]
        A1 = [
            torch.tensor([[1., 2.], [3., 4.]], dtype=dtype),
            torch.tensor([[2, 3], [4, 5]], dtype=dtype)
        ]
        _ = dut.addMConstrs(A1, [x, beta],
                            b=torch.tensor([3, 7], dtype=dtype),
                            sense=gurobipy.GRB.EQUAL)
        dut.gurobi_model.update()
        self.assertEqual(
            dut.gurobi_model.getAttr(gurobipy.GRB.Attr.NumConstrs), 2)
        dut.gurobi_model.setParam(gurobipy.GRB.Param.OutputFlag, False)
        dut.gurobi_model.optimize()
        np.testing.assert_allclose(
            A1[0].detach().numpy() @ np.array([x[0].x, x[1].x]) +
            A1[1].detach().numpy() @ np.array([beta[0].x, beta[1].x]),
            np.array([3., 7.]))

        self.assertEqual(
            dut.rhs_eq,
            [torch.tensor(3, dtype=dtype),
             torch.tensor(7, dtype=dtype)])
        self.assertEqual(dut.Aeq_r_row, [0, 0, 1, 1])
        self.assertEqual(dut.Aeq_r_col, [0, 1, 0, 1])
        self.assertEqual(dut.Aeq_r_val,
                         [A1[0][0, 0], A1[0][0, 1], A1[0][1, 0], A1[0][1, 1]])
        self.assertEqual(dut.Aeq_zeta_row, [0, 0, 1, 1])
        self.assertEqual(dut.Aeq_zeta_col, [2, 3, 2, 3])
        self.assertEqual(dut.Aeq_zeta_val,
                         [A1[1][0, 0], A1[1][0, 1], A1[1][1, 0], A1[1][1, 1]])
        # The inequality constraint are x >= 0
        self.assertEqual(dut.Ain_r_row, [0, 1])
        self.assertEqual(dut.Ain_r_col, [0, 1])
        self.assertEqual(
            dut.Ain_r_val,
            [torch.tensor(-1, dtype=dtype),
             torch.tensor(-1, dtype=dtype)])
        self.assertEqual(
            dut.rhs_in,
            [torch.tensor(0, dtype=dtype),
             torch.tensor(0, dtype=dtype)])
        self.assertEqual(len(dut.Ain_zeta_row), 0)
        self.assertEqual(len(dut.Ain_zeta_col), 0)
        self.assertEqual(len(dut.Ain_zeta_val), 0)

        # Now add <= inequality constraints
        A2 = [
            torch.tensor([[2., 3.], [1., 2.]], dtype=dtype),
            torch.tensor([[3.], [1.]], dtype=dtype)
        ]
        dut.addMConstrs(A2, [y, [alpha[1]]],
                        sense=gurobipy.GRB.LESS_EQUAL,
                        b=torch.tensor([2., 5.], dtype=dtype))
        dut.gurobi_model.optimize()
        np.testing.assert_array_less(
            A2[0].detach().numpy() @ np.array([y[0].x, y[1].x]) +
            A2[1].detach().numpy().squeeze() * alpha[1].x,
            np.array([2., 5.]) + 1e-6)
        self.assertEqual(dut.Ain_r_row, [0, 1, 2, 2, 3, 3])
        self.assertEqual(dut.Ain_r_col, [0, 1, 2, 3, 2, 3])
        self.assertEqual(dut.Ain_r_val, [
            torch.tensor(-1, dtype=dtype),
            torch.tensor(-1, dtype=dtype), A2[0][0, 0], A2[0][0, 1],
            A2[0][1, 0], A2[0][1, 1]
        ])
        self.assertEqual(dut.rhs_in, [
            torch.tensor(0, dtype=dtype),
            torch.tensor(0, dtype=dtype),
            torch.tensor(2, dtype=dtype),
            torch.tensor(5, dtype=dtype)
        ])
        self.assertEqual(dut.Ain_zeta_row, [2, 3])
        self.assertEqual(dut.Ain_zeta_col, [1, 1])
        self.assertEqual(dut.Ain_zeta_val, [A2[1][0, 0], A2[1][1, 0]])
        self.assertEqual(len(dut.Aeq_r_row), 4)
        self.assertEqual(len(dut.Aeq_r_col), 4)
        self.assertEqual(len(dut.Aeq_r_val), 4)
        self.assertEqual(len(dut.Aeq_zeta_row), 4)
        self.assertEqual(len(dut.Aeq_zeta_col), 4)
        self.assertEqual(len(dut.Aeq_zeta_val), 4)
        self.assertEqual(len(dut.rhs_eq), 2)

        # Now add >= inequality constraint.
        A3 = [
            torch.tensor([[2], [1]], dtype=dtype),
            torch.tensor([[2, 3], [1, 2]], dtype=dtype)
        ]
        dut.addMConstrs(A3, [[x[1]], beta],
                        b=torch.tensor([-2, -4], dtype=dtype),
                        sense=gurobipy.GRB.GREATER_EQUAL)
        dut.gurobi_model.optimize()
        np.testing.assert_array_less(
            np.array([-2., -4.]) - 1e-6,
            A3[0].squeeze().detach().numpy() * x[1].x +
            A3[1].detach().numpy() @ np.array([beta[0].x, beta[1].x]))
        self.assertEqual(dut.Ain_r_row, [0, 1, 2, 2, 3, 3, 4, 5])
        self.assertEqual(dut.Ain_r_col, [0, 1, 2, 3, 2, 3, 1, 1])
        self.assertEqual(dut.Ain_r_val, [
            torch.tensor(-1, dtype=dtype),
            torch.tensor(-1, dtype=dtype), A2[0][0, 0], A2[0][0, 1],
            A2[0][1, 0], A2[0][1, 1], -A3[0][0, 0], -A3[0][1, 0]
        ])
        self.assertEqual(dut.rhs_in, [
            torch.tensor(0, dtype=dtype),
            torch.tensor(0, dtype=dtype),
            torch.tensor(2, dtype=dtype),
            torch.tensor(5, dtype=dtype),
            torch.tensor(2, dtype=dtype),
            torch.tensor(4, dtype=dtype)
        ])
        self.assertEqual(dut.Ain_zeta_row, [2, 3, 4, 4, 5, 5])
        self.assertEqual(dut.Ain_zeta_col, [1, 1, 2, 3, 2, 3])
        self.assertEqual(dut.Ain_zeta_val, [
            A2[1][0, 0], A2[1][1, 0], -A3[1][0, 0], -A3[1][0, 1], -A3[1][1, 0],
            -A3[1][1, 1]
        ])
        self.assertEqual(len(dut.Aeq_r_row), 4)
        self.assertEqual(len(dut.Aeq_r_col), 4)
        self.assertEqual(len(dut.Aeq_r_val), 4)
        self.assertEqual(len(dut.Aeq_zeta_row), 4)
        self.assertEqual(len(dut.Aeq_zeta_col), 4)
        self.assertEqual(len(dut.Aeq_zeta_val), 4)
        self.assertEqual(len(dut.rhs_eq), 2)

    def test_get_active_constraints1(self):
        dtype = torch.float64
        dut = gurobi_torch_mip.GurobiTorchMILP(dtype)
        x, alpha = setup_mip1(dut)

        (A_act, b_act) = dut.get_active_constraints({2, 3},
                                                    torch.tensor([1, 0],
                                                                 dtype=dtype))
        self.assertTrue(
            torch.all(A_act == torch.tensor(
                [[0, 0, 0], [1, 1, 1], [0, 0, -1], [1, 1, 0]],
                dtype=dtype,
                requires_grad=True)))
        self.assertTrue(
            torch.all(b_act == torch.tensor(
                [0, 1, 0, 1], dtype=dtype, requires_grad=True)))

        (A_act, b_act) = dut.get_active_constraints({0, 1, 4},
                                                    torch.tensor([0, 1],
                                                                 dtype=dtype))
        self.assertTrue(
            torch.all(A_act == torch.tensor(
                [[0, 0, 0], [1, 1, 1], [-1, 0, 0], [0, -1, 0], [0, 1, 1]],
                dtype=dtype,
                requires_grad=True)))
        self.assertTrue(
            torch.all(b_act == torch.tensor(
                [0, 1, 0, 0, 1], dtype=dtype, requires_grad=True)))

    def test_get_active_constraints2(self):
        """
        Test with no equality constraint
        """
        dtype = torch.float64
        dut = gurobi_torch_mip.GurobiTorchMILP(dtype)
        x = dut.addVars(1, lb=0, vtype=gurobipy.GRB.CONTINUOUS)
        zeta = dut.addVars(1, vtype=gurobipy.GRB.BINARY)
        dut.addLConstr([torch.tensor([1, 1], dtype=dtype)], [[x[0], zeta[0]]],
                       sense=gurobipy.GRB.LESS_EQUAL,
                       rhs=1.5)
        dut.setObjective([torch.tensor([1], dtype=dtype)], [x], 0.,
                         gurobipy.GRB.MAXIMIZE)
        dut.gurobi_model.setParam(gurobipy.GRB.Param.OutputFlag, False)
        dut.gurobi_model.optimize()
        # The active constraint is x[0] + zeta[0] <= 1.5
        active_ineq_row_indices, zeta_sol = \
            dut.get_active_constraint_indices_and_binary_val()
        self.assertEqual(active_ineq_row_indices, {1})
        np.testing.assert_allclose(zeta_sol, np.array([0.]), atol=1e-12)
        A_act, b_act = dut.get_active_constraints(active_ineq_row_indices,
                                                  zeta_sol)
        np.testing.assert_allclose(A_act, np.array([[1.]]))
        np.testing.assert_allclose(b_act, np.array([1.5]))

    def test_get_active_constraints3(self):
        """
        MILP with inequality constraints, but the inequality constraints are
        not active.
        """
        dtype = torch.float64
        dut = gurobi_torch_mip.GurobiTorchMILP(dtype)
        x = dut.addVars(1, lb=0, vtype=gurobipy.GRB.CONTINUOUS)
        zeta = dut.addVars(1, vtype=gurobipy.GRB.BINARY)
        dut.addLConstr([torch.tensor([1, 1], dtype=dtype)], [[x[0], zeta[0]]],
                       rhs=2.,
                       sense=gurobipy.GRB.EQUAL)
        dut.setObjective([torch.tensor([1.], dtype=dtype)], [x],
                         0.,
                         sense=gurobipy.GRB.MINIMIZE)
        dut.gurobi_model.setParam(gurobipy.GRB.Param.OutputFlag, False)
        dut.gurobi_model.optimize()
        active_ineq_row_indices, zeta_sol = \
            dut.get_active_constraint_indices_and_binary_val()
        self.assertEqual(len(active_ineq_row_indices), 0)
        np.testing.assert_allclose(zeta_sol, np.array([1.]), atol=1e-12)
        A_act, b_act = dut.get_active_constraints(active_ineq_row_indices,
                                                  zeta_sol)
        np.testing.assert_allclose(A_act, np.array([[1.]]))
        np.testing.assert_allclose(b_act, np.array([1.]))

    def test_get_inequality_constraints(self):
        dtype = torch.float64
        dut = gurobi_torch_mip.GurobiTorchMIP(dtype)
        x = dut.addVars(3, lb=-1., vtype=gurobipy.GRB.CONTINUOUS)
        alpha = dut.addVars(2, vtype=gurobipy.GRB.BINARY)
        y = dut.addVars(2,
                        lb=-gurobipy.GRB.INFINITY,
                        vtype=gurobipy.GRB.CONTINUOUS)
        beta = dut.addVars(3, vtype=gurobipy.GRB.BINARY)
        dut.addLConstr([
            torch.tensor([1, 2, 3], dtype=dtype),
            torch.tensor([0.5, 1.5], dtype=dtype)
        ], [x, alpha],
                       sense=gurobipy.GRB.EQUAL,
                       rhs=2.)
        dut.addLConstr([
            torch.tensor([2.5, 0.3, 2], dtype=dtype),
            torch.tensor([0.1, 4.2], dtype=dtype)
        ], [beta, y],
                       sense=gurobipy.GRB.LESS_EQUAL,
                       rhs=3.)
        dut.addLConstr([
            torch.tensor([0.3, 0.5, 0.1], dtype=dtype),
            torch.tensor([0.2, 1.5, 3], dtype=dtype),
            torch.tensor([2.1, 0.5], dtype=dtype)
        ], [x, beta, y],
                       sense=gurobipy.GRB.GREATER_EQUAL,
                       rhs=0.5)
        Ain_r, Ain_zeta, rhs_in = dut.get_inequality_constraints()
        self.assertTrue(
            torch.all(Ain_r == torch.tensor(
                [[-1, 0, 0, 0, 0], [0, -1, 0, 0, 0], [0, 0, -1, 0, 0],
                 [0, 0, 0, 0.1, 4.2], [-0.3, -0.5, -0.1, -2.1, -0.5]],
                dtype=dtype)))
        self.assertTrue(
            torch.all(Ain_zeta == torch.tensor(
                [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0],
                 [0, 0, 2.5, 0.3, 2], [0, 0, -0.2, -1.5, -3]],
                dtype=dtype)))
        self.assertTrue(
            torch.all(rhs_in == torch.tensor([1, 1, 1, 3, -0.5], dtype=dtype)))

    def test_get_active_constraint_indices_and_binary_val(self):
        dtype = torch.float64
        dut = gurobi_torch_mip.GurobiTorchMILP(dtype)
        x, alpha = setup_mip1(dut)
        # If the objective is max x[0] + 1, then the active constraints are
        # x[1] >= 0, x[2] >= 0, x[0] + x[1] <= alpha[0],
        # x[1] + x[2] <= alpha[1], the binary variable solution is
        # alpha = [1, 0]
        dut.setObjective([torch.tensor([1], dtype=dtype)], [[x[0]]], 1.,
                         gurobipy.GRB.MAXIMIZE)
        dut.gurobi_model.setParam(gurobipy.GRB.Param.OutputFlag, False)
        dut.gurobi_model.optimize()
        active_ineq_row_indices, zeta_sol = \
            dut.get_active_constraint_indices_and_binary_val()
        self.assertEqual(active_ineq_row_indices, {1, 2, 3, 4})
        np.testing.assert_allclose(zeta_sol, np.array([1, 0]), atol=1e-12)

    def add_mixed_integer_linear_constraints_tester(
            self, dut, Ain_r_expected, Ain_zeta_expected, rhs_in_expected,
            Aeq_r_expected, Aeq_zeta_expected, rhs_eq_expected):
        Ain_r = torch.zeros(len(dut.rhs_in), len(dut.r), dtype=dut.dtype)
        for i in range(len(dut.Ain_r_val)):
            Ain_r[dut.Ain_r_row[i], dut.Ain_r_col[i]] = dut.Ain_r_val[i]
        np.testing.assert_allclose(Ain_r_expected.detach().numpy(),
                                   Ain_r.detach().numpy())
        Ain_zeta = torch.zeros(len(dut.rhs_in), len(dut.zeta), dtype=dut.dtype)
        for i in range(len(dut.Ain_zeta_val)):
            Ain_zeta[dut.Ain_zeta_row[i], dut.Ain_zeta_col[i]] = \
                dut.Ain_zeta_val[i]
        np.testing.assert_allclose(Ain_zeta_expected.detach().numpy(),
                                   Ain_zeta.detach().numpy())
        np.testing.assert_allclose(rhs_in_expected.detach().numpy(),
                                   np.array(dut.rhs_in).reshape((-1, 1)))
        Aeq_r = torch.zeros(len(dut.rhs_eq), len(dut.r), dtype=dut.dtype)
        for i in range(len(dut.Aeq_r_val)):
            Aeq_r[dut.Aeq_r_row[i], dut.Aeq_r_col[i]] = dut.Aeq_r_val[i]
        np.testing.assert_allclose(Aeq_r_expected.detach().numpy(),
                                   Aeq_r.detach().numpy())
        Aeq_zeta = torch.zeros(len(dut.rhs_eq), len(dut.zeta), dtype=dut.dtype)
        for i in range(len(dut.Aeq_zeta_val)):
            Aeq_zeta[dut.Aeq_zeta_row[i], dut.Aeq_zeta_col[i]] =\
                dut.Aeq_zeta_val[i]
        np.testing.assert_allclose(Aeq_zeta_expected.detach().numpy(),
                                   Aeq_zeta.detach().numpy())
        np.testing.assert_allclose(rhs_eq_expected.detach().numpy(),
                                   np.array(dut.rhs_eq).reshape((-1, 1)))

    def setup_mixed_integer_constraints_return(self):
        mip_constr_return = gurobi_torch_mip.MixedIntegerConstraintsReturn()
        dtype = torch.float64
        mip_constr_return.Aout_input = torch.tensor([[1, 3]], dtype=dtype)
        mip_constr_return.Aout_slack = torch.tensor([[0, 2, 3]], dtype=dtype)
        mip_constr_return.Aout_binary = torch.tensor([[1, -1]], dtype=dtype)
        mip_constr_return.Cout = torch.tensor([[2]], dtype=dtype)
        mip_constr_return.Ain_input = torch.tensor([[1, 2], [0, 1]],
                                                   dtype=dtype)
        mip_constr_return.Ain_slack = torch.tensor([[0, 2, 3], [1, 2, 4]],
                                                   dtype=dtype)
        mip_constr_return.Ain_binary = torch.tensor([[1, -1], [0, 1]],
                                                    dtype=dtype)
        mip_constr_return.rhs_in = torch.tensor([[1], [3]], dtype=dtype)
        mip_constr_return.Aeq_input = torch.tensor([[1, 3]], dtype=dtype)
        mip_constr_return.Aeq_slack = torch.tensor([[1, 4, 5]], dtype=dtype)
        mip_constr_return.Aeq_binary = torch.tensor([[0, 2]], dtype=dtype)
        mip_constr_return.rhs_eq = torch.tensor([[4]], dtype=dtype)
        return mip_constr_return, dtype

    def test_add_mixed_integer_linear_constraints1(self):
        """
        Test with MixedIntegerConstraintsReturn that doesn't contain any None
        items (except binary_lo and binary_up), it also adds the output
        constraint.
        """
        mip_constr_return, dtype = \
            self.setup_mixed_integer_constraints_return()
        dut = gurobi_torch_mip.GurobiTorchMILP(dtype=dtype)
        x = dut.addVars(2,
                        lb=-gurobipy.GRB.INFINITY,
                        vtype=gurobipy.GRB.CONTINUOUS,
                        name="x")
        y = dut.addVars(1,
                        lb=-gurobipy.GRB.INFINITY,
                        vtype=gurobipy.GRB.CONTINUOUS,
                        name="y")
        (slack, binary) = dut.add_mixed_integer_linear_constraints(
            mip_constr_return, x, y, "s", "gamma", "ineq_constr", "eq_constr",
            "out_constr")
        self.assertEqual(len(slack), 3)
        self.assertEqual(len(binary), 2)
        self.assertEqual(len(dut.zeta), 2)
        self.assertEqual(len(dut.r), 6)
        Ain_r_expected = torch.cat(
            (mip_constr_return.Ain_input, torch.zeros(
                2, len(y), dtype=dtype), mip_constr_return.Ain_slack),
            dim=1)
        Ain_zeta_expected = mip_constr_return.Ain_binary
        rhs_in_expected = mip_constr_return.rhs_in
        Aeq_r_expected = torch.cat((torch.cat(
            (mip_constr_return.Aeq_input, torch.zeros(
                (1, len(y)), dtype=dtype), mip_constr_return.Aeq_slack),
            dim=1),
                                    torch.cat((mip_constr_return.Aout_input,
                                               -torch.eye(1, dtype=dtype),
                                               mip_constr_return.Aout_slack),
                                              dim=1)),
                                   dim=0)
        Aeq_zeta_expected = torch.cat(
            (mip_constr_return.Aeq_binary, mip_constr_return.Aout_binary),
            dim=0)
        rhs_eq_expected = torch.cat(
            (mip_constr_return.rhs_eq, -mip_constr_return.Cout), dim=0)
        self.add_mixed_integer_linear_constraints_tester(
            dut, Ain_r_expected, Ain_zeta_expected, rhs_in_expected,
            Aeq_r_expected, Aeq_zeta_expected, rhs_eq_expected)

    def test_add_mixed_integer_linear_constraints2(self):
        """
        Test with MixedIntegerConstraintsReturn that doesn't contain any None
        items, it does NOT adds the output constraint.
        """
        mip_constr_return, dtype = \
            self.setup_mixed_integer_constraints_return()
        dut = gurobi_torch_mip.GurobiTorchMILP(dtype=dtype)
        x = dut.addVars(2,
                        lb=-gurobipy.GRB.INFINITY,
                        vtype=gurobipy.GRB.CONTINUOUS,
                        name="x")
        y = dut.addVars(1,
                        lb=-gurobipy.GRB.INFINITY,
                        vtype=gurobipy.GRB.CONTINUOUS,
                        name="y")
        (slack, binary) = dut.add_mixed_integer_linear_constraints(
            mip_constr_return, x, None, "s", "gamma", "ineq_constr",
            "eq_constr", "out_constr")
        self.assertEqual(len(slack), 3)
        self.assertEqual(len(binary), 2)
        self.assertEqual(len(dut.zeta), 2)
        self.assertEqual(len(dut.r), 6)
        Ain_r_expected = torch.cat(
            (mip_constr_return.Ain_input, torch.zeros(
                2, len(y), dtype=dtype), mip_constr_return.Ain_slack),
            dim=1)
        Ain_zeta_expected = mip_constr_return.Ain_binary
        rhs_in_expected = mip_constr_return.rhs_in
        Aeq_r_expected = torch.cat(
            (mip_constr_return.Aeq_input, torch.zeros(
                (1, len(y)), dtype=dtype), mip_constr_return.Aeq_slack),
            dim=1)
        Aeq_zeta_expected = mip_constr_return.Aeq_binary
        rhs_eq_expected = mip_constr_return.rhs_eq
        self.add_mixed_integer_linear_constraints_tester(
            dut, Ain_r_expected, Ain_zeta_expected, rhs_in_expected,
            Aeq_r_expected, Aeq_zeta_expected, rhs_eq_expected)

    def test_add_mixed_integer_linear_constraints3(self):
        # Test add_mixed_integer_linear_constraints with None items in both
        # inequality and equality constraints.
        mip_constr_return, dtype = \
            self.setup_mixed_integer_constraints_return()
        mip_constr_return.Ain_input = None
        mip_constr_return.Aeq_binary = None
        dut = gurobi_torch_mip.GurobiTorchMILP(dtype=dtype)
        x = dut.addVars(2,
                        lb=-gurobipy.GRB.INFINITY,
                        vtype=gurobipy.GRB.CONTINUOUS,
                        name="x")
        y = dut.addVars(1,
                        lb=-gurobipy.GRB.INFINITY,
                        vtype=gurobipy.GRB.CONTINUOUS,
                        name="y")
        (slack, binary) = dut.add_mixed_integer_linear_constraints(
            mip_constr_return, x, None, "s", "gamma", "ineq_constr",
            "eq_constr", "out_constr")
        self.assertEqual(len(slack), 3)
        self.assertEqual(len(binary), 2)
        self.assertEqual(len(dut.zeta), 2)
        self.assertEqual(len(dut.r), 6)
        Ain_r_expected = torch.cat(
            (torch.zeros(2, len(x), dtype=dtype),
             torch.zeros(2, len(y), dtype=dtype), mip_constr_return.Ain_slack),
            dim=1)
        Ain_zeta_expected = mip_constr_return.Ain_binary
        rhs_in_expected = mip_constr_return.rhs_in
        Aeq_r_expected = torch.cat(
            (mip_constr_return.Aeq_input, torch.zeros(
                (1, len(y)), dtype=dtype), mip_constr_return.Aeq_slack),
            dim=1)
        Aeq_zeta_expected = torch.zeros(
            (mip_constr_return.rhs_eq.numel(), len(binary)), dtype=dtype)
        rhs_eq_expected = mip_constr_return.rhs_eq
        self.add_mixed_integer_linear_constraints_tester(
            dut, Ain_r_expected, Ain_zeta_expected, rhs_in_expected,
            Aeq_r_expected, Aeq_zeta_expected, rhs_eq_expected)

    def test_add_mixed_integer_linear_constraints4(self):
        # Test adding bounds on the binary variables.
        dtype = torch.float64

        def check_binary_bounds(binary_lo, binary_up, lo_expected,
                                up_expected):
            mip_cnstr_return = gurobi_torch_mip.MixedIntegerConstraintsReturn()
            mip_cnstr_return.binary_lo = binary_lo
            mip_cnstr_return.binary_up = binary_up
            mip = gurobi_torch_mip.GurobiTorchMIP(dtype)
            slack, binary = mip.add_mixed_integer_linear_constraints(
                mip_cnstr_return, [], None, None, "binary", "ineq", "eq",
                "out")
            self.assertEqual(len(binary), 2)
            self.assertEqual(binary[0].vtype, gurobipy.GRB.BINARY)
            self.assertEqual(binary[1].vtype, gurobipy.GRB.BINARY)
            for i in range(2):
                self.assertEqual(binary[i].lb, lo_expected[i])
                self.assertEqual(binary[i].ub, up_expected[i])
            self.assertEqual(len(mip.Ain_r_row), 0)
            self.assertEqual(len(mip.Ain_r_col), 0)
            self.assertEqual(len(mip.Ain_r_val), 0)
            self.assertEqual(len(mip.Aeq_r_row), 0)
            self.assertEqual(len(mip.Aeq_r_col), 0)
            self.assertEqual(len(mip.Aeq_r_val), 0)
            self.assertEqual(len(mip.Ain_zeta_row), 0)
            self.assertEqual(len(mip.Ain_zeta_col), 0)
            self.assertEqual(len(mip.Ain_zeta_val), 0)
            self.assertEqual(len(mip.Aeq_zeta_row), 0)
            self.assertEqual(len(mip.Aeq_zeta_col), 0)
            self.assertEqual(len(mip.Aeq_zeta_val), 0)

        check_binary_bounds(None, torch.tensor([0, 1], dtype=dtype), [0, 0],
                            [0, 1])
        check_binary_bounds(torch.tensor([0, 1], dtype=dtype), None, [0, 1],
                            [1, 1])
        check_binary_bounds(torch.tensor([0, 1], dtype=dtype),
                            torch.tensor([0, 1], dtype=dtype), [0, 1], [0, 1])

    def test_add_mixed_integer_linear_constraints5(self):
        # Test adding bounds on the input variables.
        dtype = torch.float64

        def check_input_bounds(input_lo, input_up, lo_expected, up_expected):
            mip_cnstr_return = gurobi_torch_mip.MixedIntegerConstraintsReturn()
            mip_cnstr_return.input_lo = input_lo
            mip_cnstr_return.input_up = input_up
            mip = gurobi_torch_mip.GurobiTorchMIP(dtype)
            x = mip.addVars(len(lo_expected), lb=-2, ub=3)
            self.assertEqual(len(mip.Ain_r_row), 4)
            self.assertEqual(len(mip.Ain_r_col), 4)
            self.assertEqual(len(mip.Ain_r_val), 4)
            self.assertEqual(len(mip.rhs_in), 4)
            slack, binary = mip.add_mixed_integer_linear_constraints(
                mip_cnstr_return, x, None, None, "binary", "ineq", "eq", "out")
            self.assertEqual(len(slack), 0)
            self.assertEqual(len(binary), 0)
            for i in range(len(x)):
                self.assertEqual(x[i].lb, lo_expected[i])
                self.assertEqual(x[i].ub, up_expected[i])
            self.assertEqual(len(mip.Ain_r_row), 4)
            self.assertEqual(len(mip.Ain_r_col), 4)
            self.assertEqual(len(mip.Ain_r_val), 4)
            self.assertEqual(len(mip.rhs_in), 4)
            self.assertEqual(len(mip.Aeq_r_row), 0)
            self.assertEqual(len(mip.Aeq_r_col), 0)
            self.assertEqual(len(mip.Aeq_r_val), 0)
            self.assertEqual(len(mip.Ain_zeta_row), 0)
            self.assertEqual(len(mip.Ain_zeta_col), 0)
            self.assertEqual(len(mip.Ain_zeta_val), 0)
            self.assertEqual(len(mip.Aeq_zeta_row), 0)
            self.assertEqual(len(mip.Aeq_zeta_col), 0)
            self.assertEqual(len(mip.Aeq_zeta_val), 0)

        check_input_bounds(None, torch.tensor([0, 5], dtype=dtype), [-2, -2],
                           [0, 3])
        check_input_bounds(torch.tensor([-4, 1], dtype=dtype), None, [-2, 1],
                           [3, 3])
        check_input_bounds(torch.tensor([-4, -1], dtype=dtype),
                           torch.tensor([1, 6], dtype=dtype), [-2, -1], [1, 3])

    def test_add_mixed_integer_linear_constraints6(self):
        # Test adding bounds on the slack variables.
        dtype = torch.float64

        def check_slack_bounds(slack_lo, slack_up, lo_expected, up_expected):
            mip_cnstr_return = gurobi_torch_mip.MixedIntegerConstraintsReturn()
            mip_cnstr_return.slack_lo = slack_lo
            mip_cnstr_return.slack_up = slack_up
            mip = gurobi_torch_mip.GurobiTorchMIP(dtype)
            slack, binary = mip.add_mixed_integer_linear_constraints(
                mip_cnstr_return, [], None, "slack", "binary", "ineq", "eq",
                "out")
            self.assertEqual(len(slack), len(lo_expected))
            self.assertEqual(len(binary), 0)
            for i in range(len(slack)):
                self.assertEqual(slack[i].lb, lo_expected[i])
                self.assertEqual(slack[i].ub, up_expected[i])
            self.assertEqual(len(mip.Ain_r_row), 0)
            self.assertEqual(len(mip.Ain_r_col), 0)
            self.assertEqual(len(mip.Ain_r_val), 0)
            self.assertEqual(len(mip.rhs_in), 0)
            self.assertEqual(len(mip.Aeq_r_row), 0)
            self.assertEqual(len(mip.Aeq_r_col), 0)
            self.assertEqual(len(mip.Aeq_r_val), 0)
            self.assertEqual(len(mip.Ain_zeta_row), 0)
            self.assertEqual(len(mip.Ain_zeta_col), 0)
            self.assertEqual(len(mip.Ain_zeta_val), 0)
            self.assertEqual(len(mip.Aeq_zeta_row), 0)
            self.assertEqual(len(mip.Aeq_zeta_col), 0)
            self.assertEqual(len(mip.Aeq_zeta_val), 0)

        check_slack_bounds(None, torch.tensor([0, 5], dtype=dtype),
                           [-np.inf, -np.inf], [0, 5])
        check_slack_bounds(torch.tensor([-4, 1], dtype=dtype), None, [-4, 1],
                           [np.inf, np.inf])
        check_slack_bounds(torch.tensor([-4, -1], dtype=dtype),
                           torch.tensor([1, 6], dtype=dtype), [-4, -1], [1, 6])

    def test_add_mixed_integer_linear_constraints7(self):
        # Test binary_var_type=CONTINUOUS
        mip_constr_return, dtype = \
            self.setup_mixed_integer_constraints_return()
        mip = gurobi_torch_mip.GurobiTorchMIP(dtype)
        input_vars = mip.addVars(2, lb=-gurobipy.GRB.INFINITY)
        output_vars = mip.addVars(1, lb=-gurobipy.GRB.INFINITY)
        slack, binary_relax = mip.add_mixed_integer_linear_constraints(
            mip_constr_return,
            input_vars,
            output_vars,
            "slack",
            "binary_relax",
            "ineq",
            "eq",
            "out",
            binary_var_type=gurobipy.GRB.CONTINUOUS)
        self.assertEqual(len(binary_relax), 2)
        for v in binary_relax:
            self.assertEqual(v.vtype, gurobipy.GRB.CONTINUOUS)
            self.assertEqual(v.lb, 0)
            self.assertEqual(v.ub, 1)
        self.assertEqual(len(mip.r), 2 + 1 + 3 + 2)
        self.assertEqual(len(mip.zeta), 0)
        self.assertEqual(len(mip.Ain_zeta_row), 0)
        self.assertEqual(len(mip.Ain_zeta_col), 0)
        self.assertEqual(len(mip.Ain_zeta_val), 0)
        self.assertEqual(len(mip.Aeq_zeta_row), 0)
        self.assertEqual(len(mip.Aeq_zeta_col), 0)
        self.assertEqual(len(mip.Aeq_zeta_val), 0)
        # First add the constraint 0 <= binary_slack <= 1.
        self.assertEqual(mip.Ain_r_row[:4], [0, 1, 2, 3])
        binary_relax_indices = [
            mip.r_indices[binary_relax[0]], mip.r_indices[binary_relax[1]]
        ]
        self.assertEqual(mip.Ain_r_col[:4],
                         binary_relax_indices + binary_relax_indices)

    def test_add_mixed_integer_linear_constraints8(self):
        # Test binary_var_type=BINARYRELAX
        mip_constr_return, dtype = \
            self.setup_mixed_integer_constraints_return()
        mip = gurobi_torch_mip.GurobiTorchMIP(dtype)
        input_vars = mip.addVars(2, lb=-gurobipy.GRB.INFINITY)
        output_vars = mip.addVars(1, lb=-gurobipy.GRB.INFINITY)
        slack, binary_relax = mip.add_mixed_integer_linear_constraints(
            mip_constr_return,
            input_vars,
            output_vars,
            "slack",
            "binary_relax",
            "ineq",
            "eq",
            "out",
            binary_var_type=gurobi_torch_mip.BINARYRELAX)
        self.assertEqual(len(binary_relax), 2)
        for v in binary_relax:
            self.assertEqual(v.vtype, gurobipy.GRB.CONTINUOUS)
            self.assertEqual(v.lb, 0)
            self.assertEqual(v.ub, 1)
        self.assertEqual(len(mip.r), 2 + 1 + 3)
        self.assertEqual(len(mip.zeta), 2)
        self.assertEqual(len(mip.Ain_zeta_row), 4)
        self.assertEqual(len(mip.Ain_zeta_col), 4)
        self.assertEqual(len(mip.Ain_zeta_val), 4)
        # Include both the equality constraint in mip_cnstr_return, and also
        # the equality constraint for output = Aout_input * input +
        # Aout_binary * binary + Aout_slack * slack
        self.assertEqual(len(mip.Aeq_zeta_row), 4)
        self.assertEqual(len(mip.Aeq_zeta_col), 4)
        self.assertEqual(len(mip.Aeq_zeta_val), 4)

    def test_add_mixed_integer_linear_constraints9(self):
        # Test with binary_var_name equals to a list of binary variables.
        mip_constr_return, dtype = \
            self.setup_mixed_integer_constraints_return()
        mip = gurobi_torch_mip.GurobiTorchMIP(dtype)
        input_vars = mip.addVars(2, lb=-gurobipy.GRB.INFINITY)
        output_vars = mip.addVars(1, lb=-gurobipy.GRB.INFINITY)
        binary_var = mip.addVars(2, vtype=gurobipy.GRB.BINARY)
        slack, binary_var_return = mip.add_mixed_integer_linear_constraints(
            mip_constr_return,
            input_vars,
            output_vars,
            "slack",
            binary_var,
            "ineq",
            "eq",
            "out",
            binary_var_type=gurobipy.GRB.BINARY)
        self.assertIs(binary_var, binary_var_return)
        self.assertEqual(len(mip.zeta), 2)

    def test_remove_binary_relaxation(self):
        dtype = torch.float64
        dut = gurobi_torch_mip.GurobiTorchMIP(dtype)
        x = dut.addVars(2, vtype=gurobi_torch_mip.BINARYRELAX)
        dut.addVars(3, vtype=gurobipy.GRB.CONTINUOUS)
        self.assertEqual(
            dut.gurobi_model.getAttr(gurobipy.GRB.Attr.NumBinVars), 0)
        for i in range(2):
            self.assertEqual(x[i].vtype, gurobipy.GRB.CONTINUOUS)
        dut.remove_binary_relaxation()
        dut.gurobi_model.update()
        self.assertEqual(
            dut.gurobi_model.getAttr(gurobipy.GRB.Attr.NumBinVars), 2)
        for i in range(2):
            self.assertEqual(x[i].vtype, gurobipy.GRB.BINARY)


class TestGurobiTorchMILP(unittest.TestCase):
    def test_setObjective(self):
        dtype = torch.float64
        dut = gurobi_torch_mip.GurobiTorchMILP(dtype)
        x = dut.addVars(2, lb=0, vtype=gurobipy.GRB.CONTINUOUS)
        alpha = dut.addVars(3, vtype=gurobipy.GRB.BINARY)
        y = dut.addVars(4, vtype=gurobipy.GRB.CONTINUOUS)
        beta = dut.addVars(1, vtype=gurobipy.GRB.BINARY)
        for sense in (gurobipy.GRB.MINIMIZE, gurobipy.GRB.MAXIMIZE):
            dut.setObjective([
                torch.tensor([1, 2], dtype=dtype),
                torch.tensor([2, 0.5], dtype=dtype),
                torch.tensor([0.5], dtype=dtype),
                torch.tensor([2.5], dtype=dtype)
            ], [x, [alpha[0], alpha[2]], beta, [y[2]]],
                             constant=3.,
                             sense=sense)
            self.assertTrue(
                torch.all(dut.c_r == torch.tensor([1, 2, 0, 0, 2.5, 0],
                                                  dtype=dtype)))
            self.assertTrue(
                torch.all(
                    dut.c_zeta == torch.tensor([2, 0, 0.5, 0.5], dtype=dtype)))
            self.assertTrue(dut.c_constant == torch.tensor(3, dtype=dtype))
            self.assertEqual(dut.sense, sense)

    def test_compute_objective_given_active_constraints(self):
        dtype = torch.float64
        dut = gurobi_torch_mip.GurobiTorchMILP(dtype)
        x, alpha = setup_mip1(dut)
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
            dut.addLConstr([torch.stack((a[0] * a[1], a[1] + a[2], 3 * a[2]))],
                           [x],
                           sense=gurobipy.GRB.LESS_EQUAL,
                           rhs=a[0] + 2 * a[2] * a[1])
            dut.addLConstr([
                torch.stack((torch.tensor(2., dtype=dtype), a[1] ** 2,
                             torch.tensor(0.5, dtype=dtype))),
                torch.tensor([1., 1.], dtype=dtype)
            ], [x, alpha],
                           sense=gurobipy.GRB.EQUAL,
                           rhs=2 * a[0] + 1)
            dut.setObjective([
                torch.stack((a[0] + a[1], a[0], a[2])),
                torch.stack((a[1], torch.tensor(3, dtype=dtype)))
            ], [x, alpha],
                             a[0] * a[1],
                             sense=gurobipy.GRB.MAXIMIZE)
            dut.gurobi_model.setParam(gurobipy.GRB.Param.OutputFlag, False)
            dut.gurobi_model.optimize()
            objective = dut.compute_objective_from_mip_data_and_solution()
            if autograd_flag:
                objective.backward()
                return (objective, a.grad)
            else:
                return objective.item()

        def compare_gradient(a_val):
            (_, grad) = compute_milp_example_cost(np.array(a_val), True)
            grad_numerical = utils.compute_numerical_gradient(
                lambda a: compute_milp_example_cost(a, False), np.array(a_val))
            np.testing.assert_array_almost_equal(grad.detach().numpy(),
                                                 grad_numerical)

        compare_gradient([1., 2., 3.])
        compare_gradient([2., -2., 1.])
        compare_gradient([0.5, 1.4, 0.3])
        compare_gradient([0.2, 1.5, 0.3])


class TestGurobiTorchMIQP(unittest.TestCase):
    def test_setObjective(self):
        dtype = torch.float64
        dut = gurobi_torch_mip.GurobiTorchMIQP(dtype)
        x = dut.addVars(2, lb=0, vtype=gurobipy.GRB.CONTINUOUS, name="x")
        alpha = dut.addVars(3, vtype=gurobipy.GRB.BINARY, name="alpha")
        y = dut.addVars(4, vtype=gurobipy.GRB.CONTINUOUS, name="y")
        beta = dut.addVars(1, vtype=gurobipy.GRB.BINARY, name="beta")
        for sense in (gurobipy.GRB.MINIMIZE, gurobipy.GRB.MAXIMIZE):
            dut.setObjective([
                torch.tensor([[1, 2], [3, 4]], dtype=dtype),
                torch.tensor([[5]], dtype=dtype),
                torch.tensor([[6], [7]], dtype=dtype)
            ], [(x, x), ([alpha[0]], [alpha[1]]), (x, beta)], [
                torch.tensor([1, 2], dtype=dtype),
                torch.tensor([2, 0.5], dtype=dtype),
                torch.tensor([0.5], dtype=dtype),
                torch.tensor([2.5], dtype=dtype)
            ], [x, [alpha[0], alpha[2]], beta, [y[2]]],
                             constant=3.,
                             sense=sense)
            self.assertTrue(
                torch.all(dut.c_r == torch.tensor([1, 2, 0, 0, 2.5, 0],
                                                  dtype=dtype)))
            self.assertTrue(
                torch.all(
                    dut.c_zeta == torch.tensor([2, 0, 0.5, 0.5], dtype=dtype)))
            self.assertTrue(dut.c_constant == torch.tensor(3, dtype=dtype))
            self.assertEqual(dut.sense, sense)
            self.assertTrue(
                torch.all(dut.Q_r == torch.tensor(
                    [[1, 2, 0, 0, 0, 0], [3, 4, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]],
                    dtype=dtype)))
            self.assertTrue(
                torch.all(dut.Q_zeta == torch.tensor(
                    [[0, 5, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                    dtype=dtype)))
            self.assertTrue(
                torch.all(dut.Q_rzeta == torch.tensor(
                    [[0, 0, 0, 6], [0, 0, 0, 7], [0, 0, 0, 0], [0, 0, 0, 0],
                     [0, 0, 0, 0], [0, 0, 0, 0]],
                    dtype=dtype)))

    def test_compute_objective_from_mip_data(self):
        dut = gurobi_torch_mip.GurobiTorchMIQP(torch.float64)
        x, alpha = setup_mip1(dut)
        # The objective is min x[0]² + x[1]² + 4 * alpha[0]² + 4 * alpha[1]²
        # + 3 * x[0]*alpha[0] + 3*x[1]*alpha[1] + 2*x[1] + 3*x[2] +
        # 4 * alpha[0] + 6 * alpha[1] + 1
        dut.setObjective([
            torch.eye(
                2, dtype=torch.float64), 4 * torch.eye(2, dtype=torch.float64),
            3 * torch.eye(2, dtype=torch.float64)
        ], [[x[:2], x[:2]], [alpha, alpha], [x[:2], alpha]], [
            torch.tensor([2, 3], dtype=torch.float64),
            torch.tensor([4, 6], dtype=torch.float64)
        ], [x[1:], alpha],
                         1.,
                         sense=gurobipy.GRB.MINIMIZE)
        self.assertAlmostEqual(dut.compute_objective_from_mip_data(
            {1, 2, 3, 4}, torch.tensor([1., 0.], dtype=torch.float64)).item(),
                               13.,
                               places=6)
        self.assertAlmostEqual(dut.compute_objective_from_mip_data(
            {0, 1, 3, 4}, torch.tensor([0., 1.], dtype=torch.float64)).item(),
                               14,
                               places=6)

        dut.gurobi_model.setParam(gurobipy.GRB.Param.OutputFlag, False)
        dut.gurobi_model.setParam(gurobipy.GRB.Param.PoolSolutions, 2)
        dut.gurobi_model.setParam(gurobipy.GRB.Param.PoolSearchMode, 2)
        dut.gurobi_model.optimize()
        self.assertAlmostEqual(
            dut.compute_objective_from_mip_data_and_solution(
                0, penalty=1e-8).item(),
            13,
            places=6)
        self.assertAlmostEqual(
            dut.compute_objective_from_mip_data_and_solution(
                1, penalty=1e-8).item(),
            14,
            places=6)


if __name__ == "__main__":
    unittest.main()
