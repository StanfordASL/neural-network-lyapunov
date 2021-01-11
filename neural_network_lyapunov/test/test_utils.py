import neural_network_lyapunov.utils as utils
import neural_network_lyapunov.gurobi_torch_mip as gurobi_torch_mip
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
            points = torch.tensor(
                [[x_up, x_up, 1], [x_lo, x_lo, 1], [x_up, 0, 0], [x_lo, 0, 0],
                 [x_up, x_up, 0], [x_lo, x_lo, 0], [x_up, 0, 1], [x_lo, 0, 1],
                 [x_up, x_lo - 0.1, 0], [x_lo, x_up + 0.1, 1]],
                dtype=torch.float64).T
            for i in range(points.shape[1]):
                lhs = A_x * points[0, i] + A_s * \
                    points[1, i] + A_alpha * points[2, i]
                satisfied = (torch.abs(points[2, i] * points[0, i] -
                                       points[1, i]) < 1E-10)
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
            self.assertEqual(A_x.shape, (4, ))
            self.assertEqual(A_y.shape, (4, ))
            self.assertEqual(A_beta.shape, (4, ))
            self.assertEqual(rhs.shape, (4, ))
            # Now check at the vertices of the polytope, if there are 3
            # inequality constraints being active, and one inactive.
            vertices = torch.tensor(
                [[x_lo, 0, 0], [0, 0, 0], [0, 0, 1], [x_up, x_up, 1]],
                dtype=torch.float64)

            for i in range(4):
                lhs = A_x * vertices[i, 0] + A_y * vertices[i, 1] +\
                        A_beta * vertices[i, 2]
                self.assertTrue(torch.all(lhs <= rhs + 1E-12))
                self.assertEqual(
                    (torch.abs(lhs - rhs) < 1E-12).squeeze().sum().item(), 3)
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
            satisfied_flag = torch.all(
                A_x * x + A_y * y + A_beta * beta <= rhs)
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

        def linear_fun(x):
            return A.dot(x) + b

        grad = utils.compute_numerical_gradient(linear_fun, np.array([4.,
                                                                      10.]))
        self.assertTrue(utils.compare_numpy_matrices(grad, A, 1e-7, 1e-7))

    def test_quadratic_fun(self):
        Q = np.array([[1., 3.], [2., 4.]])
        p = np.array([2., 3.])

        def quadratic_fun(x):
            return 0.5 * x.T.dot(Q.dot(x)) + p.T.dot(x) + 1

        x = np.array([10., -2.])
        grad = utils.compute_numerical_gradient(quadratic_fun, x)
        self.assertTrue(
            utils.compare_numpy_matrices(grad, (0.5 * (Q + Q.T).dot(x) + p).T,
                                         1e-7, 1e-7))

    def test_vector_fun(self):
        def fun(x):
            return np.array([np.sin(x[0]), np.cos(x[1])])

        x = np.array([10., 5.])
        grad = utils.compute_numerical_gradient(fun, x)
        grad_expected = np.array([[np.cos(x[0]), 0], [0, -np.sin(x[1])]])
        self.assertTrue(
            utils.compare_numpy_matrices(grad, grad_expected, 1e-7, 1e-7))


class TestIsPolyhedronBounded(unittest.TestCase):
    def test1(self):
        self.assertTrue(
            utils.is_polyhedron_bounded(
                torch.tensor([[1, 0], [-1, 0], [0, 1.], [0, -1.]])))
        self.assertTrue(
            utils.is_polyhedron_bounded(
                torch.tensor([[1, 0], [0, 1], [-1, -1.]])))
        self.assertTrue(
            utils.is_polyhedron_bounded(
                torch.tensor([[1., 1], [0, 1], [-1, -1], [2, -1]])))
        self.assertFalse(
            utils.is_polyhedron_bounded(torch.tensor([[1, 0], [0, 1]])))
        self.assertFalse(
            utils.is_polyhedron_bounded(torch.tensor([[1, 0], [-1, 0]])))
        self.assertFalse(
            utils.is_polyhedron_bounded(
                torch.tensor([[1, 0], [-1, 0], [0, 1.]])))
        self.assertTrue(
            utils.is_polyhedron_bounded(
                torch.tensor([[1, 0, 0.], [0, 1, 0], [0, 0, 1], [-1, -1,
                                                                 -1]])))
        self.assertFalse(
            utils.is_polyhedron_bounded(
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
            x_vars = model.addVars(x_dim,
                                   lb=-np.inf,
                                   vtype=gurobipy.GRB.CONTINUOUS)
            x = [x_vars[i] for i in range(x_dim)]

            for i in range(num_in):
                model.addLConstr(gurobipy.LinExpr(A_in[i].tolist(), x),
                                 sense=gurobipy.GRB.LESS_EQUAL,
                                 rhs=b_in[i])
            for i in range(num_eq):
                model.addLConstr(gurobipy.LinExpr(A_eq[i].tolist(), x),
                                 sense=gurobipy.GRB.EQUAL,
                                 rhs=b_eq[i])
            model.setObjective(
                gurobipy.LinExpr(c, x) + d, gurobipy.GRB.MAXIMIZE)
            model.setParam(gurobipy.GRB.Param.OutputFlag, 0)
            model.optimize()
            if (model.status != gurobipy.GRB.Status.OPTIMAL):
                self.assertIsNone(cost)
            else:
                self.assertAlmostEqual(cost, model.objVal)

        dtype = torch.float64
        test_fun(torch.tensor([1, 2], dtype=dtype), torch.tensor(2,
                                                                 dtype=dtype),
                 -torch.eye(2, dtype=dtype), torch.tensor([0, 0], dtype=dtype),
                 torch.tensor([[1, 1]], dtype=dtype),
                 torch.tensor([1], dtype=dtype))
        test_fun(torch.tensor([1, 2], dtype=dtype), torch.tensor(2,
                                                                 dtype=dtype),
                 -torch.eye(2, dtype=dtype), torch.tensor([0, 0], dtype=dtype),
                 torch.tensor([[1, 1]], dtype=dtype),
                 torch.tensor([-1], dtype=dtype))
        test_fun(torch.tensor([1, 2, 3], dtype=dtype),
                 torch.tensor(2, dtype=dtype),
                 -torch.tensor([[1, 3, 5], [2, 1, -2]], dtype=dtype),
                 torch.tensor([5, 2], dtype=dtype),
                 torch.tensor([[1, 2, 5]], dtype=dtype),
                 torch.tensor([1], dtype=dtype))


class TestLeakyReLUInterval(unittest.TestCase):
    def test(self):
        self.assertEqual(utils.leaky_relu_interval(0.1, 1., 2.), (1., 2.))
        self.assertEqual(
            utils.leaky_relu_interval(0.1, torch.tensor(1.), torch.tensor(2.)),
            (torch.tensor(1.), torch.tensor(2.)))
        self.assertEqual(utils.leaky_relu_interval(0.1, -2., -1.),
                         (-0.2, -0.1))
        self.assertEqual(
            utils.leaky_relu_interval(0.1, torch.tensor(-2.),
                                      torch.tensor(-1.)),
            (torch.tensor(-0.2), torch.tensor(-0.1)))
        self.assertEqual(utils.leaky_relu_interval(0.1, -2., 3.), (-0.2, 3.))
        self.assertEqual(
            utils.leaky_relu_interval(0.1, torch.tensor(-2.),
                                      torch.tensor(3.)),
            (torch.tensor(-0.2), torch.tensor(3.)))
        self.assertEqual(utils.leaky_relu_interval(-0.1, 1., 2.), (1., 2.))
        self.assertEqual(
            utils.leaky_relu_interval(-0.1, torch.tensor(1.),
                                      torch.tensor(2.)),
            (torch.tensor(1.), torch.tensor(2.)))
        self.assertEqual(utils.leaky_relu_interval(-0.1, -2., -1.), (0.1, 0.2))
        self.assertEqual(
            utils.leaky_relu_interval(-0.1, torch.tensor(-2.),
                                      torch.tensor(-1.)),
            (torch.tensor(0.1), torch.tensor(0.2)))
        self.assertEqual(utils.leaky_relu_interval(-0.1, -2., 1.), (0., 1.))
        self.assertEqual(
            utils.leaky_relu_interval(-0.1, torch.tensor(-2.),
                                      torch.tensor(1.)),
            (0., torch.tensor(1.)))
        self.assertEqual(utils.leaky_relu_interval(-0.1, -20., 1.), (0, 2.))
        self.assertEqual(
            utils.leaky_relu_interval(-0.1, torch.tensor(-20.),
                                      torch.tensor(1.)),
            (0., torch.tensor(2.)))


class TestProjectToPolyhedron(unittest.TestCase):
    def test(self):
        dtype = torch.float64
        A = torch.cat([torch.eye(2).type(dtype), -torch.eye(2).type(dtype)],
                      dim=0)
        b = torch.tensor([1, 1, 0, 0], dtype=dtype)
        np.testing.assert_almost_equal(
            utils.project_to_polyhedron(
                A, b, torch.tensor([0.5, 0.6], dtype=dtype)).detach().numpy(),
            np.array([0.5, 0.6]))
        np.testing.assert_almost_equal(
            utils.project_to_polyhedron(
                A, b, torch.tensor([1.2, 1.4], dtype=dtype)).detach().numpy(),
            np.array([1., 1.]))


class TestSetupReLU(unittest.TestCase):
    """
    test both setup_relu and extract_relu_parameters
    """
    def test1(self):
        # Test with params=None and bias=True
        dut = utils.setup_relu((2, 4, 3),
                               None,
                               negative_slope=0.1,
                               bias=True,
                               dtype=torch.float64)
        self.assertEqual(len(dut), 3)
        self.assertIsInstance(dut[0], torch.nn.Linear)
        self.assertIsNotNone(dut[0].bias)
        self.assertEqual(dut[0].in_features, 2)
        self.assertEqual(dut[0].out_features, 4)
        self.assertIsInstance(dut[1], torch.nn.LeakyReLU)
        self.assertEqual(dut[1].negative_slope, 0.1)
        self.assertIsInstance(dut[2], torch.nn.Linear)
        self.assertIsNotNone(dut[2].bias)
        self.assertEqual(dut[2].in_features, 4)
        self.assertEqual(dut[2].out_features, 3)
        relu_params = utils.extract_relu_parameters(dut)
        self.assertEqual(relu_params.shape, (27, ))

    def test2(self):
        # Test with params=None and bias=False
        dut = utils.setup_relu((2, 4, 3),
                               None,
                               negative_slope=0.1,
                               bias=False,
                               dtype=torch.float64)
        self.assertEqual(len(dut), 3)
        self.assertIsInstance(dut[0], torch.nn.Linear)
        self.assertIsNone(dut[0].bias)
        self.assertEqual(dut[0].in_features, 2)
        self.assertEqual(dut[0].out_features, 4)
        self.assertIsInstance(dut[1], torch.nn.LeakyReLU)
        self.assertEqual(dut[1].negative_slope, 0.1)
        self.assertIsInstance(dut[2], torch.nn.Linear)
        self.assertIsNone(dut[2].bias)
        self.assertEqual(dut[2].in_features, 4)
        self.assertEqual(dut[2].out_features, 3)
        relu_params = utils.extract_relu_parameters(dut)
        self.assertEqual(relu_params.shape, (20, ))

    def test3(self):
        # Test with params and with_bias=True
        params = torch.tensor(list(range(27)), dtype=torch.float64)
        dut = utils.setup_relu((2, 4, 3),
                               params,
                               negative_slope=0.1,
                               bias=True,
                               dtype=torch.float64)
        self.assertEqual(len(dut), 3)
        np.testing.assert_allclose(dut[0].weight.detach().numpy(),
                                   params[:8].reshape((4, 2)).detach().numpy())
        np.testing.assert_allclose(dut[0].bias.detach().numpy(),
                                   params[8:12].detach().numpy())
        np.testing.assert_allclose(
            dut[2].weight.detach().numpy(), params[12:24].reshape(
                (3, 4)).detach().numpy())
        np.testing.assert_allclose(dut[2].bias.detach().numpy(),
                                   params[24:].detach().numpy())

        relu_params = utils.extract_relu_parameters(dut)
        np.testing.assert_allclose(relu_params.detach().numpy(),
                                   params.detach().numpy())

    def test4(self):
        # Test with params and with_bias=False
        params = torch.tensor(list(range(20)), dtype=torch.float64)
        dut = utils.setup_relu((2, 4, 3),
                               params,
                               negative_slope=0.1,
                               bias=False,
                               dtype=torch.float64)
        self.assertEqual(len(dut), 3)
        np.testing.assert_allclose(dut[0].weight.detach().numpy(),
                                   params[:8].reshape((4, 2)).detach().numpy())
        np.testing.assert_allclose(
            dut[2].weight.detach().numpy(), params[8:20].reshape(
                (3, 4)).detach().numpy())

        relu_params = utils.extract_relu_parameters(dut)
        np.testing.assert_allclose(relu_params.detach().numpy(),
                                   params.detach().numpy())


class TestAddSaturationAsMixedIntegerConstraint(unittest.TestCase):
    def saturation_tester(self, lower_limit, upper_limit, input_lower_bound,
                          input_upper_bound, x_val, y_val):
        mip = gurobi_torch_mip.GurobiTorchMILP(torch.float64)
        x = mip.addVars(1,
                        lb=-gurobipy.GRB.INFINITY,
                        vtype=gurobipy.GRB.CONTINUOUS)[0]
        y = mip.addVars(1,
                        lb=-gurobipy.GRB.INFINITY,
                        vtype=gurobipy.GRB.CONTINUOUS)[0]
        beta = utils.add_saturation_as_mixed_integer_constraint(
            mip, x, y, lower_limit, upper_limit, input_lower_bound,
            input_upper_bound)
        # Set input value
        mip.addLConstr([torch.tensor([1], dtype=torch.float64)], [[x]],
                       sense=gurobipy.GRB.EQUAL,
                       rhs=x_val)
        mip.gurobi_model.setParam(gurobipy.GRB.Param.OutputFlag, False)
        mip.gurobi_model.optimize()
        self.assertEqual(mip.gurobi_model.status, gurobipy.GRB.Status.OPTIMAL)
        self.assertAlmostEqual(y.x, y_val)

        return mip, beta

    def test_always_saturate_upper(self):
        # Test that the upper limit is always saturated.
        lower_limit = 1.
        upper_limit = 2.
        input_lower_bound = 3.
        input_upper_bound = 4.

        for x in np.linspace(input_lower_bound, input_upper_bound, 10):
            mip, beta = self.saturation_tester(lower_limit,
                                               upper_limit,
                                               input_lower_bound,
                                               input_upper_bound,
                                               x,
                                               y_val=upper_limit)
            self.assertEqual(len(beta), 0)
            self.assertEqual(len(mip.r), 2)
            self.assertEqual(len(mip.zeta), 0)

    def test_always_saturate_lower(self):
        # Test that the lower limit is always saturated.
        lower_limit = 1.
        upper_limit = 2.
        input_lower_bound = -2.
        input_upper_bound = 1.

        for x in np.linspace(input_lower_bound, input_upper_bound, 10):
            mip, beta = self.saturation_tester(lower_limit,
                                               upper_limit,
                                               input_lower_bound,
                                               input_upper_bound,
                                               x,
                                               y_val=lower_limit)
            self.assertEqual(len(beta), 0)
            self.assertEqual(len(mip.r), 2)
            self.assertEqual(len(mip.zeta), 0)

    def test_no_saturation(self):
        # Test when saturation can never happen.
        lower_limit = 1.
        upper_limit = 2.
        input_lower_bound = 1.
        input_upper_bound = 2.

        for x in np.linspace(input_lower_bound, input_upper_bound, 10):
            mip, beta = self.saturation_tester(lower_limit,
                                               upper_limit,
                                               input_lower_bound,
                                               input_upper_bound,
                                               x,
                                               y_val=x)
            self.assertEqual(len(beta), 0)
            self.assertEqual(len(mip.r), 2)
            self.assertEqual(len(mip.zeta), 0)

        input_lower_bound = 1.1
        input_upper_bound = 1.9
        for x in np.linspace(input_lower_bound, input_upper_bound, 10):
            mip, beta = self.saturation_tester(lower_limit,
                                               upper_limit,
                                               input_lower_bound,
                                               input_upper_bound,
                                               x,
                                               y_val=x)
            self.assertEqual(len(beta), 0)
            self.assertEqual(len(mip.r), 2)
            self.assertEqual(len(mip.zeta), 0)

    def test_maybe_saturate_lower(self):
        # The input bounds include the lower limit, but it cannot saturate the
        # upper limit.
        lower_limit = 1.
        upper_limit = 2.
        input_lower_bound = -1.
        for input_upper_bound in (2., 1.9):
            for x in np.linspace(input_lower_bound, input_upper_bound, 10):
                y_val = lower_limit if x <= lower_limit else x
                mip, beta = self.saturation_tester(lower_limit,
                                                   upper_limit,
                                                   input_lower_bound,
                                                   input_upper_bound,
                                                   x,
                                                   y_val=y_val)
                self.assertEqual(len(beta), 1)
                self.assertEqual(len(mip.r), 2)
                self.assertEqual(len(mip.zeta), 1)
                if x < lower_limit:
                    self.assertAlmostEqual(beta[0].x, 1)
                elif x > lower_limit:
                    self.assertAlmostEqual(beta[0].x, 0)

    def test_maybe_saturate_upper(self):
        # The input bounds include the upper limit, but it cannot saturate the
        # lower limit.
        lower_limit = 1.
        upper_limit = 2.
        input_upper_bound = 4.
        for input_lower_bound in (1., 1.1):
            for x in np.linspace(input_lower_bound, input_upper_bound, 10):
                y_val = upper_limit if x >= upper_limit else x
                mip, beta = self.saturation_tester(lower_limit,
                                                   upper_limit,
                                                   input_lower_bound,
                                                   input_upper_bound,
                                                   x,
                                                   y_val=y_val)
                self.assertEqual(len(beta), 1)
                self.assertEqual(len(mip.r), 2)
                self.assertEqual(len(mip.zeta), 1)
                if x < upper_limit:
                    self.assertAlmostEqual(beta[0].x, 0)
                elif x > upper_limit:
                    self.assertAlmostEqual(beta[0].x, 1)

    def test_maybe_saturate_both_limits(self):
        # The input bounds include both the lower limit and the upper limit. It
        # could saturate on both sides.
        lower_limit = 1.
        upper_limit = 2.
        input_lower_bound = -1.
        input_upper_bound = 3.

        for x in np.linspace(input_lower_bound, input_upper_bound, 10):
            beta_val = [None, None]
            if x > upper_limit:
                y_val = upper_limit
                beta_val = [0, 1]
            elif x < lower_limit:
                y_val = lower_limit
                beta_val = [1, 0]
            else:
                y_val = x
                if x > lower_limit:
                    beta_val[0] = 0
                if x < upper_limit:
                    beta_val[1] = 0
            mip, beta = self.saturation_tester(lower_limit,
                                               upper_limit,
                                               input_lower_bound,
                                               input_upper_bound,
                                               x,
                                               y_val=y_val)
            self.assertEqual(len(beta), 2)
            self.assertEqual(len(mip.r), 3)
            self.assertEqual(len(mip.zeta), 2)
            if beta_val[0] is not None:
                self.assertAlmostEqual(beta[0].x, beta_val[0])
            if beta_val[1] is not None:
                self.assertAlmostEqual(beta[1].x, beta_val[1])
            slack_val = lower_limit if x < lower_limit else x
            self.assertAlmostEqual(mip.r[2].x, slack_val)


class TestExtractReLUStructure(unittest.TestCase):
    def test(self):
        relu = utils.setup_relu((2, 4, 5),
                                params=None,
                                negative_slope=0.1,
                                bias=True)
        linear_layer_width, negative_slope, bias = \
            utils.extract_relu_structure(relu)
        self.assertEqual(linear_layer_width, (2, 4, 5))
        self.assertEqual(negative_slope, 0.1)
        self.assertTrue(bias)

        # negative_slope = 0
        relu = utils.setup_relu((2, 4, 5),
                                params=None,
                                negative_slope=0.,
                                bias=True)
        linear_layer_width, negative_slope, bias =\
            utils.extract_relu_structure(relu)
        self.assertEqual(linear_layer_width, (2, 4, 5))
        self.assertEqual(negative_slope, 0.)
        self.assertTrue(bias)

        # bias = False
        relu = utils.setup_relu((2, 4, 5),
                                params=None,
                                negative_slope=0.,
                                bias=False)
        linear_layer_width, negative_slope, bias = \
            utils.extract_relu_structure(relu)
        self.assertEqual(linear_layer_width, (2, 4, 5))
        self.assertEqual(negative_slope, 0.)
        self.assertFalse(bias)


class TestUpdateReLUParams(unittest.TestCase):
    def test(self):
        # no bias.
        network = utils.setup_relu((2, 4, 3, 2),
                                   params=None,
                                   negative_slope=0.01,
                                   bias=False,
                                   dtype=torch.float64)
        params = torch.linspace(1, 26, 26, dtype=torch.float64)
        utils.update_relu_params(network, params)
        params_extracted = utils.extract_relu_parameters(network)
        np.testing.assert_allclose(params_extracted.detach().numpy(),
                                   np.linspace(1, 26, 26))

        # with bias.
        network = utils.setup_relu((2, 4, 3, 2),
                                   params=None,
                                   negative_slope=0.01,
                                   bias=True,
                                   dtype=torch.float64)
        params = torch.linspace(1, 34, 34, dtype=torch.float64)
        utils.update_relu_params(network, params)
        params_extracted = utils.extract_relu_parameters(network)
        np.testing.assert_allclose(params_extracted.detach().numpy(),
                                   np.linspace(1, 34, 34))


class TestGetMeshgridSamples(unittest.TestCase):
    def test(self):
        lower = np.array([-1, -2])
        upper = np.array([1, 2])
        mesh_size = (11, 21)
        samples = utils.get_meshgrid_samples(lower, upper, mesh_size,
                                             torch.float64)
        self.assertIsInstance(samples, torch.Tensor)
        self.assertEqual(samples.shape, (11 * 21, 2))

        x_samples = torch.linspace(lower[0],
                                   upper[0],
                                   mesh_size[0],
                                   dtype=torch.float64)
        y_samples = torch.linspace(lower[1],
                                   upper[1],
                                   mesh_size[1],
                                   dtype=torch.float64)
        for i in range(mesh_size[0]):
            for j in range(mesh_size[1]):
                np.testing.assert_allclose(
                    samples[i * mesh_size[1] + j, :].detach().numpy(),
                    np.array([x_samples[i], y_samples[j]]))


class TestNetworkZeroGrad(unittest.TestCase):
    def test(self):
        # Test without bias
        network = utils.setup_relu((2, 4, 1),
                                   params=None,
                                   negative_slope=0.1,
                                   bias=False,
                                   dtype=torch.float64)
        data = (torch.tensor([[1., 2.], [0., 2.], [3., 1.]],
                             dtype=torch.float64),
                torch.tensor([[1.], [2.], [3.]], dtype=torch.float64))
        output = network(data[0])

        loss = torch.nn.MSELoss()(output, data[1])
        loss.backward()

        utils.network_zero_grad(network)
        for layer in (0, 2):
            np.testing.assert_allclose(
                network[layer].weight.grad.detach().numpy(),
                np.zeros_like(network[layer].weight.grad.detach().numpy()))
            self.assertIsNone(network[layer].bias)

        # Test with bias
        network = utils.setup_relu((2, 4, 1),
                                   params=None,
                                   negative_slope=0.1,
                                   bias=True,
                                   dtype=torch.float64)
        output = network(data[0])

        loss = torch.nn.MSELoss()(output, data[1])
        loss.backward()

        utils.network_zero_grad(network)
        for layer in (0, 2):
            np.testing.assert_allclose(
                network[layer].weight.grad.detach().numpy(),
                np.zeros_like(network[layer].weight.grad.detach().numpy()))
            np.testing.assert_allclose(
                network[layer].bias.grad.detach().numpy(),
                np.zeros_like(network[layer].bias.grad.detach().numpy()))


class TestSigmoidAnneal(unittest.TestCase):
    def test(self):
        dtype = torch.float64
        sa = utils.SigmoidAnneal(dtype, 0, 1, 10, 1)
        sig = torch.nn.Sigmoid()
        self.assertEqual(sa(10 + 2), sig(torch.tensor(2, dtype=dtype)))
        sa = utils.SigmoidAnneal(dtype, 1e-3, 100, 30, 7)
        self.assertEqual(sa(30), .5 * (100 - 1e-3) + 1e-3)
        self.assertAlmostEqual(sa(-100).item(), 1e-3, places=5)


class TestUniformSampleInBox(unittest.TestCase):
    def test(self):
        dtype = torch.float64
        lo = torch.tensor([1, 3], dtype=dtype)
        hi = torch.tensor([2, 4], dtype=dtype)
        samples = utils.uniform_sample_in_box(lo, hi, 10)
        self.assertEqual(samples.shape, (10, 2))
        for i in range(2):
            np.testing.assert_array_less(samples[:, i], hi[i])
            np.testing.assert_array_less(lo[i], samples[:, i])

        samples = utils.uniform_sample_in_box(lo, hi, 2)
        self.assertEqual(samples.shape, (2, 2))


if __name__ == "__main__":
    unittest.main()
