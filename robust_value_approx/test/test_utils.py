import robust_value_approx.utils as utils
import torch
import unittest
import numpy as np
import gurobipy


class test_replace_binary_continuous_product(unittest.TestCase):
    def setUp(self):
        pass

    def test(self):
        def test_fun(x_lo, x_up):
            (A_x, A_s, A_alpha, rhs) = utils.replace_binary_continuous_product(
                x_lo, x_up, torch.float64)
            # Now test if the four vertices satisfy the constraints
            points = torch.tensor([[x_up, x_up, 1],
                                   [x_lo, x_lo, 1],
                                   [x_up, 0, 0],
                                   [x_lo, 0, 0],
                                   [x_up, x_up, 0],
                                   [x_lo, x_lo, 0],
                                   [x_up, 0, 1],
                                   [x_lo, 0, 1],
                                   [x_up, x_lo - 0.1, 0],
                                   [x_lo, x_up + 0.1, 1]],
                                  dtype=torch.float64).T
            for i in range(points.shape[1]):
                lhs = A_x * points[0, i] + A_s * \
                    points[1, i] + A_alpha * points[2, i]
                satisfied = (
                    torch.abs(points[2, i] * points[0, i] - points[1, i])
                    < 1E-10)
                self.assertEqual(torch.all(lhs <= rhs + 1E-12), satisfied)

        test_fun(0, 1)
        test_fun(1, 2)
        test_fun(-1, 0)
        test_fun(-2, -1)
        test_fun(-2, 1)


class test_replace_relu_with_mixed_integer_constraint(unittest.TestCase):
    def setUp(self):
        pass

    def test(self):
        def test_fun(x_lo, x_up):
            (A_x, A_y, A_beta, rhs) = utils.\
                    replace_relu_with_mixed_integer_constraint(x_lo, x_up)
            self.assertEqual(A_x.shape, (4, 1))
            self.assertEqual(A_y.shape, (4, 1))
            self.assertEqual(A_beta.shape, (4, 1))
            self.assertEqual(rhs.shape, (4, 1))
            # Now check at the vertices of the polytope, if there are 3
            # inequality constraints being active, and one inactive.
            vertices = torch.tensor([[x_lo, 0, 0],
                                     [0, 0, 0],
                                     [0, 0, 1],
                                     [x_up, x_up, 1]], dtype=torch.float64)

            for i in range(4):
                lhs = A_x * vertices[i, 0] + A_y * vertices[i, 1] +\
                        A_beta * vertices[i, 2]
                self.assertTrue(torch.all(lhs <= rhs + 1E-12))
                self.assertEqual((torch.abs(lhs - rhs) < 1E-12).
                                 squeeze().sum().item(), 3)
            # Given x, there is only one and only y satisfying y = max(0, x)
            for x in np.linspace(x_lo, x_up, 5):
                y = np.maximum(x, 0)
                beta = 1 if x > 0 else 0
                lhs = A_x * x + A_y * y + A_beta * beta
                self.assertTrue(torch.all(lhs <= rhs + 1E-12))
                # Now take many y that are not equal to max(0, x), such y
                # should not satisfy the constraint
                for y_not_satisfied in np.linspace(x_lo, x_up, 100):
                    if (y_not_satisfied != y):
                        lhs1 = A_x * x + A_y * y_not_satisfied + A_beta * 0
                        self.assertFalse(torch.all(lhs1 <= rhs + 1E-12))
                        lhs2 = A_x * x + A_y * y_not_satisfied + A_beta * 1
                        self.assertFalse(torch.all(lhs2 <= rhs + 1E-12))

        test_fun(-1, 1)
        test_fun(-2, 10)
        test_fun(-100, 1)


class test_compute_numerical_gradient(unittest.TestCase):

    def test_linear_fun(self):
        A = np.array([[1., 2.], [3., 4.]])
        b = np.array([[3.], [4.]])
        def linear_fun(x): return A.dot(x) + b

        grad = utils.compute_numerical_gradient(linear_fun, np.array([[4.],
                                                                      [10.]]))
        self.assertTrue(utils.compare_numpy_matrices(grad, A, 1e-7, 1e-7))

    def test_quadratic_fun(self):
        Q = np.array([[1., 3.], [2., 4.]])
        p = np.array([[2.], [3.]])
        def quadratic_fun(x): return 0.5 * x.T.dot(Q.dot(x)) + p.T.dot(x) + 1
        x = np.array([[10.], [-2.]])
        grad = utils.compute_numerical_gradient(quadratic_fun, x)
        self.assertTrue(
                utils.compare_numpy_matrices(grad,
                                             (0.5 * (Q + Q.T).dot(x) + p).T,
                                             1e-7, 1e-7))

    def test_vector_fun(self):
        def fun(x): return np.array([np.sin(x[0]), np.cos(x[1])])
        x = np.array([10., 5.])
        grad = utils.compute_numerical_gradient(fun, x)
        grad_expected = np.array([[np.cos(x[0]), 0], [0, -np.sin(x[1])]])
        self.assertTrue(utils.compare_numpy_matrices(grad, grad_expected,
                                                     1e-7, 1e-7))


class TestIsPolyhedronBounded(unittest.TestCase):
    def test1(self):
        self.assertTrue(utils.is_polyhedron_bounded(
            torch.tensor([[1, 0], [-1, 0], [0, 1.], [0, -1.]])))
        self.assertTrue(utils.is_polyhedron_bounded(
            torch.tensor([[1, 0], [0, 1], [-1, -1.]])))
        self.assertTrue(utils.is_polyhedron_bounded(
            torch.tensor([[1., 1], [0, 1], [-1, -1], [2, -1]])))
        self.assertFalse(utils.is_polyhedron_bounded(
            torch.tensor([[1, 0], [0, 1]])))
        self.assertFalse(utils.is_polyhedron_bounded(
            torch.tensor([[1, 0], [-1, 0]])))
        self.assertFalse(utils.is_polyhedron_bounded(
            torch.tensor([[1, 0], [-1, 0], [0, 1.]])))
        self.assertTrue(utils.is_polyhedron_bounded(
            torch.tensor([[1, 0, 0.], [0, 1, 0], [0, 0, 1], [-1, -1, -1]])))
        self.assertFalse(utils.is_polyhedron_bounded(
            torch.tensor([[1, 0, 0.], [0, 1, 0], [0, 0, 1]])))


class TestComputeBoundsFromPolytope(unittest.TestCase):
    def test1(self):
        P = np.array([[1., 1.], [0, -1], [-1, 1]])
        q = np.array([2, 3., 1.5])

        self.assertEqual(utils.compute_bounds_from_polytope(P, q, 0),
                         (-4.5, 5))
        self.assertEqual(utils.compute_bounds_from_polytope(P, q, 1),
                         (-3, 1.75))

    def test2(self):
        P = np.array([[1., 1.], [0, -1]])
        q = np.array([2, 3.])

        self.assertEqual(utils.compute_bounds_from_polytope(P, q, 0),
                         (-np.inf, 5))
        self.assertEqual(utils.compute_bounds_from_polytope(P, q, 1),
                         (-3, np.inf))


class TestLinearProgramCost(unittest.TestCase):
    def test(self):
        def test_fun(c, d, A_in, b_in, A_eq, b_eq):
            cost = utils.linear_program_cost(c, d, A_in, b_in, A_eq, b_eq)
            x_dim = A_in.shape[1]
            num_in = A_in.shape[0]
            num_eq = A_eq.shape[0]
            model = gurobipy.Model()
            x_vars = model.addVars(x_dim, lb=-np.inf,
                                   vtype=gurobipy.GRB.CONTINUOUS)
            x = [x_vars[i] for i in range(x_dim)]

            for i in range(num_in):
                model.addLConstr(
                    gurobipy.LinExpr(A_in[i].tolist(), x),
                    sense=gurobipy.GRB.LESS_EQUAL, rhs=b_in[i])
            for i in range(num_eq):
                model.addLConstr(
                    gurobipy.LinExpr(A_eq[i].tolist(), x),
                    sense=gurobipy.GRB.EQUAL, rhs=b_eq[i])
            model.setObjective(gurobipy.LinExpr(c, x) + d,
                               gurobipy.GRB.MAXIMIZE)
            model.setParam(gurobipy.GRB.Param.OutputFlag, 0)
            model.optimize()
            if (model.status != gurobipy.GRB.Status.OPTIMAL):
                self.assertIsNone(cost)
            else:
                self.assertAlmostEqual(cost, model.objVal)

        dtype = torch.float64
        test_fun(
            torch.tensor([1, 2], dtype=dtype),
            torch.tensor(2, dtype=dtype),
            -torch.eye(2, dtype=dtype),
            torch.tensor([0, 0], dtype=dtype),
            torch.tensor([[1, 1]], dtype=dtype),
            torch.tensor([1], dtype=dtype))
        test_fun(
            torch.tensor([1, 2], dtype=dtype),
            torch.tensor(2, dtype=dtype),
            -torch.eye(2, dtype=dtype),
            torch.tensor([0, 0], dtype=dtype),
            torch.tensor([[1, 1]], dtype=dtype),
            torch.tensor([-1], dtype=dtype))
        test_fun(
            torch.tensor([1, 2, 3], dtype=dtype),
            torch.tensor(2, dtype=dtype),
            -torch.tensor([[1, 3, 5], [2, 1, -2]], dtype=dtype),
            torch.tensor([5, 2], dtype=dtype),
            torch.tensor([[1, 2, 5]], dtype=dtype),
            torch.tensor([1], dtype=dtype))


def generate_milp1():
    milp = gurobipy.Model()
    x = milp.addVars(2, lb=0)
    b = milp.addVars(1, lb=0, vtype=gurobipy.GRB.BINARY)
    milp.addLConstr(x[0] + x[1] + b[0] == 2)
    milp.setObjective(x[0] + 2 * x[1])
    milp.update()
    return milp

class TestGurobiMILPToStandardForm(unittest.TestCase):
    def test(self):
        def test_fun(milp):
            utils.gurobi_milp_to_standard_form(milp)

        test_fun(generate_milp1())


if __name__ == "__main__":
    unittest.main()
