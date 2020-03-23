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
                torch.tensor(x_lo, dtype=torch.float64),
                torch.tensor(x_up, dtype=torch.float64), torch.float64)
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


class TestLeakyReLUGradientTimesX(unittest.TestCase):
    def test(self):

        def test_fun(x_lo, x_up, negative_slope, x, y, alpha, satisfied):
            A_x, A_y, A_alpha, rhs = utils.leaky_relu_gradient_times_x(
                torch.tensor(x_lo, dtype=torch.float64),
                torch.tensor(x_up, dtype=torch.float64), negative_slope)
            self.assertEqual(
                torch.all(A_x * x + A_y * y + A_alpha * alpha <= rhs),
                satisfied)

        test_fun(-0.5, 1.5, 0.1, 0., 0., 1., True)
        test_fun(-0.5, 1.5, 0.1, 0., 0., 0., True)
        test_fun(-0.5, 1.5, 0.1, 0.5, 0.5, 1, True)
        test_fun(-0.5, 1.5, 0.1, -0.25, -0.25, 1, True)
        test_fun(-0.5, 1.5, 0.1, 0.5, 0.05, 0, True)
        test_fun(-0.5, 1.5, 0.1, 0.5, 0.5, 0, False)
        test_fun(-0.5, 1.5, 0.1, -0.25, -0.025, 0, True)
        test_fun(-0.5, 1.5, 0.1, -0.25, -0.25, 0, False)
        test_fun(-0.5, 1.5, 0.1, -1.25, -1.25, 1, False)
        test_fun(-0.5, 1.5, 0.1, 1.55, 1.55, 1, False)
        test_fun(-0.5, 1.5, 0.1, 1.25, 0.125, 0, True)
        test_fun(0.5, 1.5, 0.1, 1.25, 0.125, 0, True)
        test_fun(0.5, 1.5, 0.1, 1.25, 1.25, 1, True)
        test_fun(0.5, 1.5, 0.1, 1.25, 0.25, 1, False)
        test_fun(0.5, 1.5, 0.1, 1.25, 0.25, 0, False)


class test_replace_absolute_value_with_mixed_integer_constraint(
        unittest.TestCase):
    def test(self):
        Ain_x, Ain_s, Ain_alpha, rhs_in =\
            utils.replace_absolute_value_with_mixed_integer_constraint(
                torch.tensor(-2, dtype=torch.float64),
                torch.tensor(3, dtype=torch.float64))

        def test_fun(x, s, alpha, satisfied):
            self.assertEqual(
                torch.all(Ain_x * x + Ain_s * s + Ain_alpha * alpha <= rhs_in),
                satisfied)

        test_fun(0., 0., 0., True)
        test_fun(0., 0., 1., True)
        test_fun(1., 1., 1., True)
        test_fun(2., 2., 1., True)
        test_fun(3., 3., 1., True)
        test_fun(2., 2.01, 1., False)
        test_fun(1., 0.99, 1., False)
        test_fun(1., 0.99, 0., False)
        test_fun(1., 1., 0., False)
        test_fun(4., 4., 1., False)
        test_fun(-1., 1., 0., True)
        test_fun(-2., 2., 0., True)
        test_fun(-3., 3., 0., False)
        test_fun(-1., 0.5, 0., False)
        test_fun(-1., -0.5, 0., False)
        test_fun(-1., 1., 1., False)


class test_replace_relu_with_mixed_integer_constraint(unittest.TestCase):
    def test(self):
        def test_fun(x_lo, x_up):
            (A_x, A_y, A_beta, rhs) = utils.\
                    replace_relu_with_mixed_integer_constraint(x_lo, x_up)
            self.assertEqual(A_x.shape, (4,))
            self.assertEqual(A_y.shape, (4,))
            self.assertEqual(A_beta.shape, (4,))
            self.assertEqual(rhs.shape, (4,))
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


class TestReplaceLeakyReluWithMixedIntegerConstraint(unittest.TestCase):
    def test(self):
        x_lo = -2
        x_up = 5

        def test_fun(negative_slope, x, y, beta):
            A_x, A_y, A_beta, rhs = utils.\
                replace_leaky_relu_mixed_integer_constraint(
                    negative_slope, x_lo, x_up)
            satisfied_flag = torch.all(A_x * x + A_y * y + A_beta * beta <=
                                       rhs)
            satisfied_expected = (
                x >= 0 and beta == 1 and y == x and x <= x_up) or \
                (x <= 0 and beta == 0 and y == negative_slope * x and
                 x >= x_lo)
            self.assertEqual(satisfied_flag, satisfied_expected)

        test_fun(0.5, 0., 0., 0)
        test_fun(0.5, 0., 0., 1)
        test_fun(0.5, 1., 1., 1)
        test_fun(0.5, 2., 2., 1)
        test_fun(0.5, 2., 2., 0)
        test_fun(0.5, 5., 5., 1)
        test_fun(0.5, 5., 5., 0)
        test_fun(0.5, 6., 6., 1)
        test_fun(0.5, 1., 2., 1)
        test_fun(0.5, 2., -2., 1)
        test_fun(0.5, -2., -1., 0)
        test_fun(0.5, -1., -0.5, 0)
        test_fun(0.5, -1., -0.6, 0)
        test_fun(0.5, -1., -0.9, 0)
        test_fun(0.5, -1., -1.9, 0)
        test_fun(0.5, -1., -1.9, 1)
        test_fun(0.5, -1., -0.5, 1)
        test_fun(2, -1., -2., 0)
        test_fun(2, -2., -4., 0)
        test_fun(2, -2., -3., 0)
        test_fun(2, -2., -5., 0)
        test_fun(2, -1.5, -3., 0)
        test_fun(2, -1.5, -3., 1)
        test_fun(2, 1.5, 1.5, 1)
        test_fun(2, 1.5, 3, 0)
        test_fun(2, 2.5, 5, 0)


class test_compute_numerical_gradient(unittest.TestCase):

    def test_linear_fun(self):
        A = np.array([[1., 2.], [3., 4.]])
        b = np.array([3., 4.])
        def linear_fun(x): return A.dot(x) + b

        grad = utils.compute_numerical_gradient(linear_fun, np.array([4.,
                                                                      10.]))
        self.assertTrue(utils.compare_numpy_matrices(grad, A, 1e-7, 1e-7))

    def test_quadratic_fun(self):
        Q = np.array([[1., 3.], [2., 4.]])
        p = np.array([2., 3.])
        def quadratic_fun(x): return 0.5 * x.T.dot(Q.dot(x)) + p.T.dot(x) + 1
        x = np.array([10., -2.])
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


class TestLeakyReLUInterval(unittest.TestCase):
    def test(self):
        self.assertEqual(utils.leaky_relu_interval(0.1, 1., 2.), (1., 2.))
        self.assertEqual(utils.leaky_relu_interval(
            0.1, torch.tensor(1.), torch.tensor(2.)),
            (torch.tensor(1.), torch.tensor(2.)))
        self.assertEqual(
            utils.leaky_relu_interval(0.1, -2., -1.), (-0.2, -0.1))
        self.assertEqual(
            utils.leaky_relu_interval(
                0.1, torch.tensor(-2.), torch.tensor(-1.)),
            (torch.tensor(-0.2), torch.tensor(-0.1)))
        self.assertEqual(
            utils.leaky_relu_interval(0.1, -2., 3.), (-0.2, 3.))
        self.assertEqual(
            utils.leaky_relu_interval(
                0.1, torch.tensor(-2.), torch.tensor(3.)),
            (torch.tensor(-0.2), torch.tensor(3.)))
        self.assertEqual(
            utils.leaky_relu_interval(-0.1, 1., 2.), (1., 2.))
        self.assertEqual(
            utils.leaky_relu_interval(
                -0.1, torch.tensor(1.), torch.tensor(2.)),
            (torch.tensor(1.), torch.tensor(2.)))
        self.assertEqual(
            utils.leaky_relu_interval(-0.1, -2., -1.), (0.1, 0.2))
        self.assertEqual(
            utils.leaky_relu_interval(
                -0.1, torch.tensor(-2.), torch.tensor(-1.)),
            (torch.tensor(0.1), torch.tensor(0.2)))
        self.assertEqual(
            utils.leaky_relu_interval(-0.1, -2., 1.), (0., 1.))
        self.assertEqual(
            utils.leaky_relu_interval(
                -0.1, torch.tensor(-2.), torch.tensor(1.)),
            (0., torch.tensor(1.)))
        self.assertEqual(
            utils.leaky_relu_interval(-0.1, -20., 1.), (0, 2.))
        self.assertEqual(
            utils.leaky_relu_interval(
                -0.1, torch.tensor(-20.), torch.tensor(1.)),
            (0., torch.tensor(2.)))


class TestProjectToPolyhedron(unittest.TestCase):
    def test(self):
        dtype = torch.float64
        A = torch.cat(
            [torch.eye(2).type(dtype), -torch.eye(2).type(dtype)], dim=0)
        b = torch.tensor([1, 1, 0, 0], dtype=dtype)
        np.testing.assert_almost_equal(utils.project_to_polyhedron(
            A, b, torch.tensor([0.5, 0.6], dtype=dtype)).detach().numpy(),
            np.array([0.5, 0.6]))
        np.testing.assert_almost_equal(utils.project_to_polyhedron(
            A, b, torch.tensor([1.2, 1.4], dtype=dtype)).detach().numpy(),
            np.array([1., 1.]))


if __name__ == "__main__":
    unittest.main()
