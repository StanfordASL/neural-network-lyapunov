import neural_network_lyapunov.mip_utils as mip_utils

import unittest

import torch
import numpy as np
from itertools import chain, combinations
import itertools
import gurobipy

import neural_network_lyapunov.utils as utils


class TestStrengthenReLUMipWIndices(unittest.TestCase):
    def special_cases_tester(self, c):
        dtype = torch.float64
        w = torch.tensor([3., -2.], dtype=dtype)
        b = torch.tensor(2., dtype=dtype)
        lo = torch.tensor([-2., -3.], dtype=dtype)
        up = torch.tensor([2., 4.], dtype=dtype)
        # Test the case when the index set is empty.
        # This should corresponds to
        # y <= c*(wx + b) + (1-c) * max(w*x+b) * beta
        x_coeff, binary_coeff, constant = \
            mip_utils.strengthen_relu_mip_w_indices(
                c, w, b, lo, up, set())
        self.assertEqual(constant.item(), c * b.item())
        np.testing.assert_allclose(x_coeff.detach().numpy(),
                                   c * w.detach().numpy())
        relu_input_lo, relu_input_up = mip_utils.compute_range_by_IA(
            w.reshape((1, -1)), b.reshape((-1, )), lo, up)
        self.assertAlmostEqual(binary_coeff.item(),
                               (1 - c) * relu_input_up[0].item())

        # Test with index set equals to the whole set. This should corresponds
        # to y <= w*x+b + (1-c) * relu_input_lo * beta + (c-1)*relu_input_lo
        x_coeff, binary_coeff, constant = \
            mip_utils.strengthen_relu_mip_w_indices(
                c, w, b, lo, up, {0, 1})
        self.assertEqual(constant.item(), (b + (c - 1) * relu_input_lo).item())
        np.testing.assert_allclose(x_coeff.detach().numpy(),
                                   w.detach().numpy())
        np.testing.assert_allclose(binary_coeff.item(),
                                   ((1 - c) * relu_input_lo).item())

    def test_special_cases(self):
        # Test ReLU
        self.special_cases_tester(0.)
        # Test leaky ReLU
        self.special_cases_tester(0.1)

    def general_case_tester(self, c):
        dtype = torch.float64
        w = torch.tensor([3., -2.], dtype=dtype)
        b = torch.tensor(2., dtype=dtype)
        lo = torch.tensor([-2., -3.], dtype=dtype)
        up = torch.tensor([2., 4.], dtype=dtype)
        # Test index set equals to {0}
        x_coeff, binary_coeff, constant = \
            mip_utils.strengthen_relu_mip_w_indices(
                c, w, b, lo, up, {0})
        self.assertAlmostEqual(constant.item(),
                               (b * c - (1 - c) * w[0] * lo[0]).item())
        self.assertAlmostEqual(binary_coeff.item(),
                               (b * (1 - c) + (1 - c) * w[0] * lo[0] +
                                (1 - c) * w[1] * lo[1]).item())
        np.testing.assert_allclose(x_coeff.detach().numpy(),
                                   np.array([w[0], c * w[1]]))

        # Test index set equals to {1}
        x_coeff, binary_coeff, constant = \
            mip_utils.strengthen_relu_mip_w_indices(
                c, w, b, lo, up, {1})
        self.assertAlmostEqual(constant.item(),
                               (b * c - (1 - c) * w[1] * up[1]).item())
        self.assertAlmostEqual(binary_coeff.item(),
                               (b * (1 - c) + (1 - c) * w[1] * up[1] +
                                (1 - c) * w[0] * up[0]).item())
        np.testing.assert_allclose(x_coeff.detach().numpy(),
                                   np.array([c * w[0], w[1]]))

    def test_general_case(self):
        self.general_case_tester(0.)
        self.general_case_tester(0.1)


class TestFindIndexSetToStrengthen(unittest.TestCase):
    def maximal_separation_tester(self, c, w, b, lo, up, xhat, beta_hat):
        # The maximal separated plane has the maximal violation of
        # y <= bc + b(1-c)β + ∑ i∈ℑ (wᵢxᵢ−(1−c)(1−β)wᵢL̅ᵢ)
        #      + ∑i∉ℑ(cwᵢxᵢ+(1−c)βwᵢU̅ᵢ)
        # Namely the right-hand side should be minimized
        indices = mip_utils.find_index_set_to_strengthen(
            w, lo, up, xhat, beta_hat)
        min_rhs = np.inf
        max_separation_set = None
        # Loop through all subsets of {0, 1, ..., nx-1}, make sure `indices`
        # returns the maximal separation plane.
        nx = w.shape[0]
        for candidate_index in chain.from_iterable(
                combinations(list(range(nx)), r) for r in range(nx + 1)):
            x_coeff, binary_coeff, constant = \
                mip_utils.strengthen_relu_mip_w_indices(
                    c, w, b, lo, up, set(candidate_index))
            rhs = x_coeff @ xhat + binary_coeff * beta_hat + constant
            if rhs < min_rhs:
                min_rhs = rhs
                max_separation_set = set(candidate_index)
        self.assertSetEqual(indices, max_separation_set)

    def test1(self):
        dtype = torch.float64
        self.maximal_separation_tester(c=0.1,
                                       w=torch.tensor([2., -3.], dtype=dtype),
                                       b=torch.tensor(-2., dtype=dtype),
                                       lo=torch.tensor([-2., -4.],
                                                       dtype=dtype),
                                       up=torch.tensor([3., 1.], dtype=dtype),
                                       xhat=torch.tensor([-2., 1.],
                                                         dtype=dtype),
                                       beta_hat=torch.tensor(0.3, dtype=dtype))

    def test2(self):
        dtype = torch.float64
        self.maximal_separation_tester(c=0.1,
                                       w=torch.tensor([2., -3.], dtype=dtype),
                                       b=torch.tensor(-2., dtype=dtype),
                                       lo=torch.tensor([-2., -4.],
                                                       dtype=dtype),
                                       up=torch.tensor([3., 1.], dtype=dtype),
                                       xhat=torch.tensor([-2., -4.],
                                                         dtype=dtype),
                                       beta_hat=torch.tensor(0.8, dtype=dtype))

    def test3(self):
        dtype = torch.float64
        self.maximal_separation_tester(c=0.1,
                                       w=torch.tensor([2., -3.], dtype=dtype),
                                       b=torch.tensor(2., dtype=dtype),
                                       lo=torch.tensor([-2., -4.],
                                                       dtype=dtype),
                                       up=torch.tensor([3., 1.], dtype=dtype),
                                       xhat=torch.tensor([3., -4.],
                                                         dtype=dtype),
                                       beta_hat=torch.tensor(0.8, dtype=dtype))

    def test4(self):
        dtype = torch.float64
        self.maximal_separation_tester(c=0.1,
                                       w=torch.tensor([2., -3.], dtype=dtype),
                                       b=torch.tensor(2., dtype=dtype),
                                       lo=torch.tensor([-2., -4.],
                                                       dtype=dtype),
                                       up=torch.tensor([3., 1.], dtype=dtype),
                                       xhat=torch.tensor([3., 1.],
                                                         dtype=dtype),
                                       beta_hat=torch.tensor(0.8, dtype=dtype))


class TestComputeBetaRange(unittest.TestCase):
    def empty_constraint_tester(self, c):
        # Test when x_coeffs = beta_coeffs = constants = None
        dtype = torch.float64
        w = torch.tensor([2., -3.], dtype=dtype)
        b = torch.tensor([-.5], dtype=dtype)
        beta_range = mip_utils._compute_beta_range(
            c, w, b, [], [], [], torch.tensor([-2, 3], dtype=dtype))
        self.assertEqual(beta_range[0].item(), 0.)
        self.assertEqual(beta_range[1].item(), 1.)

    def test_empty_constraint(self):
        self.empty_constraint_tester(0.)
        self.empty_constraint_tester(0.1)

    def nonempty_constraint_tester(self, c, w, b, lo, up, index_sets, xhat):
        relu_input_lo, relu_input_up = mip_utils.compute_range_by_IA(
            w.reshape((1, -1)), b.reshape((-1, )), lo, up)
        assert (relu_input_lo[0].item() < 0)
        assert (relu_input_up[0].item() > 0)
        x_coeffs = []
        beta_coeffs = []
        constants = []
        for index_set in index_sets:
            x_coeff, beta_coeff, constant =\
                mip_utils.strengthen_relu_mip_w_indices(
                    c, w, b, lo, up, index_set)
            x_coeffs.append(x_coeff)
            beta_coeffs.append(beta_coeff)
            constants.append(constant)

        beta_lo, beta_up = mip_utils._compute_beta_range(
            c, w, b, x_coeffs, beta_coeffs, constants, xhat)
        self.assertGreaterEqual(beta_lo.item(), 0)
        self.assertLessEqual(beta_up.item(), 1)
        self.assertGreaterEqual(beta_up.item(), beta_lo.item())
        x_coeffs_torch = torch.cat([v.reshape((1, -1)) for v in x_coeffs],
                                   dim=0)
        beta_coeffs_torch = torch.stack(beta_coeffs)
        constants_torch = torch.stack(constants)
        lhs = torch.max(c * (w @ xhat + b), w @ xhat + b)
        np.testing.assert_array_less(lhs.item() - 1E-6,
                                     (x_coeffs_torch @ xhat +
                                      beta_coeffs_torch * beta_up +
                                      constants_torch).detach().numpy())
        np.testing.assert_array_less(lhs.item() - 1E-6,
                                     (x_coeffs_torch @ xhat +
                                      beta_coeffs_torch * beta_lo +
                                      constants_torch).detach().numpy())
        np.testing.assert_array_less(
            lhs.item() - 1E-6,
            (x_coeffs_torch @ xhat + beta_coeffs_torch *
             (beta_lo + beta_up) / 2 + constants_torch).detach().numpy())
        # Make sure that if beta > beta_up, then it violates the constraints.
        if beta_up.item() < 1:
            self.assertFalse(
                torch.all(x_coeffs_torch @ xhat + beta_coeffs_torch *
                          (beta_up + 1E-5) + constants_torch >= lhs.item()))
        if beta_lo.item() > 0:
            self.assertFalse(
                torch.all(x_coeffs_torch @ xhat + beta_coeffs_torch *
                          (beta_lo - 1E-5) + constants_torch >= lhs.item()))

    def test_nonempty_constraint1(self):
        dtype = torch.float64
        w = torch.tensor([2., -3.], dtype=dtype)
        b = torch.tensor(-0.5, dtype=dtype)
        lo = torch.tensor([-1., -2.], dtype=dtype)
        up = torch.tensor([-0.5, 1.], dtype=dtype)
        for c in (0., 0.1):
            self.nonempty_constraint_tester(c, w, b, lo, up, [set()], lo)
            self.nonempty_constraint_tester(c, w, b, lo, up, [set()], up)
            self.nonempty_constraint_tester(
                c, w, b, lo, up, [set(), {0}],
                torch.tensor([lo[0], up[1]], dtype=dtype))
            self.nonempty_constraint_tester(
                c, w, b, lo, up, [set(), {0, 1}],
                torch.tensor([lo[0], up[1]], dtype=dtype))

    def test_nonempty_constraint2(self):
        dtype = torch.float64
        w = torch.tensor([2., -3., 1.5], dtype=dtype)
        b = torch.tensor(-0.5, dtype=dtype)
        lo = torch.tensor([-1., -2., 0.2], dtype=dtype)
        up = torch.tensor([-0.5, 1., 0.5], dtype=dtype)
        for c in (0., 0.1):
            self.nonempty_constraint_tester(c, w, b, lo, up, [set()], lo)
            self.nonempty_constraint_tester(c, w, b, lo, up, [set()], up)
            self.nonempty_constraint_tester(
                c, w, b, lo, up, [set(), {0}],
                torch.tensor([lo[0], up[1], lo[2]], dtype=dtype))
            self.nonempty_constraint_tester(
                c, w, b, lo, up, [set(), {0, 1}],
                torch.tensor([lo[0], up[1], up[2]], dtype=dtype))


class TestGetLinearInputVertices(unittest.TestCase):
    def test1(self):
        # Test with relu_input_lo, relu_input_up computed from interval
        # arithmetics.
        dtype = torch.float64
        lo = torch.tensor([-1, -3], dtype=dtype)
        up = torch.tensor([-0.5, 2], dtype=dtype)
        w = torch.tensor([-2, 3.], dtype=dtype)
        b = torch.tensor(0.4, dtype=dtype)
        relu_input_lo_ia, relu_input_up_ia = mip_utils.compute_range_by_IA(
            w.reshape((1, -1)), b.reshape((-1, )), lo, up)
        vertices = mip_utils._get_linear_input_vertices(
            lo, up, w, b, relu_input_lo_ia[0], relu_input_up_ia[0])
        box_vertices = itertools.product(*[(lo[i], up[i]) for i in range(2)])
        self.assertEqual(len(vertices), 4)
        i = 0
        for box_vertex in box_vertices:
            vertex_expect = torch.stack(box_vertex)
            np.testing.assert_allclose(vertices[i].detach().numpy(),
                                       vertex_expect.detach().numpy())
            i += 1

    def test2(self):
        # Test with relu_input_lo, relu_input_up tighter than that computed
        # from interval arithmetics.
        dtype = torch.float64
        lo = torch.tensor([-1, -3], dtype=dtype)
        up = torch.tensor([-0.5, 2], dtype=dtype)
        w = torch.tensor([-2, 3.], dtype=dtype)
        b = torch.tensor(0.4, dtype=dtype)
        relu_input_lo_ia, relu_input_up_ia = mip_utils.compute_range_by_IA(
            w.reshape((1, -1)), b.reshape((-1, )), lo, up)
        vertices = mip_utils._get_linear_input_vertices(
            lo, up, w, b, relu_input_lo_ia[0], relu_input_lo_ia[0] + 1E-5)
        self.assertEqual(len(vertices), 1)
        np.testing.assert_allclose(vertices[0], np.array([-0.5, -3]))


class TestMaxYGivenLinearInput(unittest.TestCase):
    def y_range_tester(self, c, w, b, x_coeffs, beta_coeffs, constants, x_hat):
        y_max, beta_val = mip_utils._max_y_given_linear_input(
            c, w, b, x_coeffs, beta_coeffs, constants, x_hat)
        # First make sure that this y_max and beta_val satisfy the constraint
        lhs = torch.max(c * (w @ x_hat + b), w @ x_hat + b).item()
        self.assertGreaterEqual(y_max, lhs - 1E-6)
        self.assertGreaterEqual(beta_val, 0)
        self.assertLessEqual(beta_val, 1)
        for x_coeff, beta_coeff, constant in zip(x_coeffs, beta_coeffs,
                                                 constants):
            self.assertLessEqual(
                y_max,
                (x_coeff @ x_hat + beta_coeff * beta_val + constant).item() +
                1E-6)
        x_coeffs_stack = torch.cat([v.reshape((1, -1)) for v in x_coeffs],
                                   dim=0)
        beta_coeffs_stack = torch.stack(beta_coeffs)
        constants_stack = torch.stack(constants)
        # Now sample many beta in the range [0, 1]. Either they violate the
        # constraint, or the corresponding y is smaller than y_max.
        for beta in np.linspace(0, 1, 51).tolist():
            rhs = x_coeffs_stack @ x_hat + beta_coeffs_stack * beta +\
                constants_stack
            self.assertTrue(
                torch.any(rhs < lhs) or torch.min(rhs).item() <= y_max)

    def test1(self):
        # Test when relu_input_lo, relu_input_up is computed from interval
        # arithmetic of linear_input_lo and linear_input_up
        dtype = torch.float64
        c = 0.
        w = torch.tensor([1, 1], dtype=dtype)
        b = torch.tensor(-1.5, dtype=dtype)
        lo = torch.tensor([0, 0], dtype=dtype)
        up = torch.tensor([1, 1], dtype=dtype)
        relu_input_lo, relu_input_up = mip_utils.compute_range_by_IA(
            w.reshape((1, -1)), b.reshape((-1, )), lo, up)
        x_coeffs = []
        beta_coeffs = []
        constants = []
        for index_set in (set(), {0, 1}):
            x_coeff, beta_coeff, constant = \
                mip_utils.strengthen_relu_mip_w_indices(
                    c, w, b, lo, up, index_set)
            x_coeffs.append(x_coeff)
            beta_coeffs.append(beta_coeff)
            constants.append(constant)

        for x_hat in (torch.tensor([0, 0], dtype=dtype),
                      torch.tensor([0, 1], dtype=dtype),
                      torch.tensor([1, 0], dtype=dtype),
                      torch.tensor([1, 1], dtype=dtype)):
            self.y_range_tester(c, w, b, x_coeffs, beta_coeffs, constants,
                                x_hat)
        x_coeff, beta_coeff, constant = \
            mip_utils.strengthen_relu_mip_w_indices(
                c, w, b, lo, up, {0})
        x_coeffs.append(x_coeff)
        beta_coeffs.append(beta_coeff)
        constants.append(constant)
        for x_hat in (torch.tensor([0, 0], dtype=dtype),
                      torch.tensor([0, 1], dtype=dtype),
                      torch.tensor([1, 0], dtype=dtype),
                      torch.tensor([1, 1], dtype=dtype)):
            self.y_range_tester(c, w, b, x_coeffs, beta_coeffs, constants,
                                x_hat)

    def test2(self):
        # Test when relu_input_lo, relu_input_up is tighter than those
        # computed from interval arithmetic of linear_input_lo and
        # linear_input_up
        dtype = torch.float64
        c = 0.1
        w = torch.tensor([2, -1], dtype=dtype)
        b = torch.tensor(-1.5, dtype=dtype)
        lo = torch.tensor([-1, 0.5], dtype=dtype)
        up = torch.tensor([1.5, 1], dtype=dtype)
        relu_input_lo_ia, relu_input_up_ia = mip_utils.compute_range_by_IA(
            w.reshape((1, -1)), b.reshape((-1, )), lo, up)
        relu_input_lo = 0.95 * relu_input_lo_ia[0] + 0.05 * relu_input_up_ia[0]
        relu_input_up = 0.1 * relu_input_lo_ia[0] + 0.9 * relu_input_up_ia[0]
        x_coeffs = [c * w, w]
        beta_coeffs = [(1 - c) * relu_input_up, (1 - c) * relu_input_lo]
        constants = [c * b, b - (1 - c) * relu_input_lo]

        for x_hat in mip_utils._get_linear_input_vertices(
                lo, up, w, b, relu_input_lo, relu_input_up):
            if relu_input_lo <= w @ x_hat + b <= relu_input_up:
                self.y_range_tester(c, w, b, x_coeffs, beta_coeffs, constants,
                                    x_hat)
        x_coeff, beta_coeff, constant = \
            mip_utils.strengthen_relu_mip_w_indices(
                c, w, b, lo, up, {0})
        x_coeffs.append(x_coeff)
        beta_coeffs.append(beta_coeff)
        constants.append(constant)
        for x_hat in mip_utils._get_linear_input_vertices(
                lo, up, w, b, relu_input_lo, relu_input_up):
            if relu_input_lo <= w @ x_hat + b <= relu_input_up:
                self.y_range_tester(c, w, b, x_coeffs, beta_coeffs, constants,
                                    x_hat)


class TestStrengthenReLUMip(unittest.TestCase):
    def construct_strengthened_mip(self, c, w, b, lo, up, relu_input_lo,
                                   relu_input_up, use_binary, selective):
        x_coeffs, binary_coeffs, constants = mip_utils.strengthen_relu_mip(
            c, w, b, lo, up, relu_input_lo, relu_input_up, selective)
        # Now setup an optimization problem with these constraints
        prog = gurobipy.Model()
        x = []
        for i in range(w.shape[0]):
            x.append(
                prog.addVar(lb=lo[i].item(), ub=up[i].item(), name=f"x_{i}"))
        beta_type = gurobipy.GRB.BINARY if use_binary else\
            gurobipy.GRB.CONTINUOUS
        beta = prog.addVar(lb=0., ub=1., name="b", vtype=beta_type)
        y = prog.addVar(lb=-gurobipy.GRB.INFINITY, name="y")
        # Add the lower bound constraint y >= c*(w'x+b), y >= w'x+b
        prog.addLConstr(gurobipy.LinExpr([1] + (-w).detach().numpy().tolist(),
                                         [y] + x),
                        rhs=b.item(),
                        sense=gurobipy.GRB.GREATER_EQUAL)
        prog.addLConstr(gurobipy.LinExpr(
            [1] + (-c * w).detach().numpy().tolist(), [y] + x),
                        rhs=c * b.item(),
                        sense=gurobipy.GRB.GREATER_EQUAL)
        # Now add the constraint
        # y <= c(wᵀx+b)+(1−c)m⁺β
        # y <= wᵀx+b−(1−c)m⁻(1−β)
        prog.addLConstr(gurobipy.LinExpr(
            [1] + (-c * w).detach().numpy().tolist() +
            [-(1 - c) * relu_input_up.item()], [y] + x + [beta]),
                        sense=gurobipy.GRB.LESS_EQUAL,
                        rhs=c * b.item())
        prog.addLConstr(gurobipy.LinExpr([1] + (-w).detach().numpy().tolist() +
                                         [-(1 - c) * relu_input_lo.item()],
                                         [y] + x + [beta]),
                        sense=gurobipy.GRB.LESS_EQUAL,
                        rhs=(b - (1 - c) * relu_input_lo).item())
        # Now add the strengthened constraints.
        # y <= x_coeffs * x + beta_coeffs * beta + constants
        for i in range(x_coeffs.shape[0]):
            prog.addLConstr(gurobipy.LinExpr(
                x_coeffs[i].detach().numpy().tolist() +
                [binary_coeffs[i].item()] + [-1], x + [beta] + [y]),
                            sense=gurobipy.GRB.GREATER_EQUAL,
                            rhs=-constants[i].item())
        return prog, x, y, beta

    def strengthened_constraint_tester(self, c, w, b, lo, up, relu_input_lo,
                                       relu_input_up, ideal, selective):
        # Make sure that after strengthening the constraint, the condition
        # y = relu(w*x+b) is satisfied. If we expect to get an ideal
        # formualtion, then also check that the vertices of the constrained
        # polytopes are integral solutions.

        # Now sample many different vector obj_coeff, and maximize
        # obj_coeff * [x;y]. We check if the optimal solution is integral.
        prog, x, y, beta = self.construct_strengthened_mip(c,
                                                           w,
                                                           b,
                                                           lo,
                                                           up,
                                                           relu_input_lo,
                                                           relu_input_up,
                                                           use_binary=False,
                                                           selective=selective)
        if ideal:
            for _ in range(40):
                obj_coeff = np.random.uniform(low=-1.,
                                              high=1.,
                                              size=(w.shape[0] + 1, ))
                prog.setObjective(gurobipy.LinExpr(obj_coeff.tolist(),
                                                   x + [y]),
                                  sense=gurobipy.GRB.MAXIMIZE)
                prog.setParam(gurobipy.GRB.Param.OutputFlag, False)
                prog.optimize()
                self.assertEqual(prog.status, gurobipy.GRB.Status.OPTIMAL)
                self.assertTrue(
                    np.abs(beta.x) < 1E-6 or np.abs(beta.x - 1) < 1E-6)

        # Now sample many points satisfying
        # lo <= x <= up
        # relu_input_lo <= w'*x+b <= relu_input_up
        # And check if (x, y) satisfies the strengthened constraint.
        prog, x, y, beta = self.construct_strengthened_mip(c,
                                                           w,
                                                           b,
                                                           lo,
                                                           up,
                                                           relu_input_lo,
                                                           relu_input_up,
                                                           use_binary=False,
                                                           selective=selective)
        x_samples = utils.uniform_sample_in_box(lo, up, 1000)
        for i in range(x_samples.shape[0]):
            if relu_input_lo <= w @ x_samples[i] + b <= relu_input_up:
                for j in range(w.shape[0]):
                    x[j].lb = x_samples[i, j].item()
                    prog.update()
                    x[j].ub = x_samples[i, j].item()
                    prog.update()
                y_val = torch.max(c * (w @ x_samples[i] + b),
                                  w @ x_samples[i] + b)
                y.lb = y_val.item()
                y.ub = y_val.item()
                prog.setParam(gurobipy.GRB.Param.OutputFlag, False)
                prog.optimize()
                self.assertEqual(prog.status, gurobipy.GRB.Status.OPTIMAL)

        # Now sample many points satisfying
        # lo <= x <= up
        # relu_input_lo <= w'*x+b <= relu_input_up
        # Make sure that if beta is a binary variable, then the solution
        # (y, beta) satisfies the relu condition.
        prog, x, y, beta = self.construct_strengthened_mip(c,
                                                           w,
                                                           b,
                                                           lo,
                                                           up,
                                                           relu_input_lo,
                                                           relu_input_up,
                                                           use_binary=True,
                                                           selective=selective)
        x_samples = utils.uniform_sample_in_box(lo, up, 1000)
        for i in range(x_samples.shape[0]):
            if relu_input_lo <= w @ x_samples[i] + b <= relu_input_up:
                for j in range(w.shape[0]):
                    x[j].lb = x_samples[i, j].item()
                    prog.update()
                    x[j].ub = x_samples[i, j].item()
                    prog.update()
                prog.setParam(gurobipy.GRB.Param.OutputFlag, False)
                prog.optimize()
                self.assertEqual(prog.status, gurobipy.GRB.Status.OPTIMAL)
                self.assertAlmostEqual(y.x,
                                       torch.max(c * (w @ x_samples[i] + b),
                                                 w @ x_samples[i] + b).item(),
                                       places=5)

    def test_2d_case1(self):
        # The example given in Strong mixed-integer programming formulations
        # for trained neural networks.
        c = 0.
        dtype = torch.float64
        w = torch.tensor([1, 1], dtype=dtype)
        b = torch.tensor(-1.5, dtype=dtype)
        lo = torch.tensor([0, 0], dtype=dtype)
        up = torch.tensor([1, 1], dtype=dtype)
        relu_input_lo_ia, relu_input_up_ia = mip_utils.compute_range_by_IA(
            w.reshape((1, -1)), b.reshape((-1, )), lo, up)
        relu_input_lo = relu_input_lo_ia[0]
        relu_input_up = relu_input_up_ia[0]
        for selective in (True, False):
            x_coeffs, binary_coeffs, constants = mip_utils.strengthen_relu_mip(
                c, w, b, lo, up, relu_input_lo, relu_input_up, selective)
            self.assertEqual(x_coeffs.shape, (2, 2))
            self.strengthened_constraint_tester(c,
                                                w,
                                                b,
                                                lo,
                                                up,
                                                relu_input_lo,
                                                relu_input_up,
                                                ideal=True,
                                                selective=selective)

        # relu_input_lo and relu_input_up are tighter than those computed from
        # interval arithematics.
        relu_input_lo = torch.tensor(-0.6, dtype=dtype)
        relu_input_up = relu_input_up_ia[0]
        for selective in (True, False):
            x_coeffs, binary_coeffs, constants = mip_utils.strengthen_relu_mip(
                c, w, b, lo, up, relu_input_lo, relu_input_up, selective)
            self.assertEqual(x_coeffs.shape, (2, 2))
            self.strengthened_constraint_tester(c,
                                                w,
                                                b,
                                                lo,
                                                up,
                                                relu_input_lo,
                                                relu_input_up,
                                                ideal=False,
                                                selective=selective)

        # Now tighten relu_input_lo further, such that the vertices
        # [x, y] = [0, 1, 0], [1, 0, 0] are not in the admissible region.
        relu_input_lo = torch.tensor(-0.45, dtype=dtype)
        for (selective, size) in ((True, 0), (False, 2)):
            x_coeffs, binary_coeffs, constants = mip_utils.strengthen_relu_mip(
                c, w, b, lo, up, relu_input_lo, relu_input_up, selective)
            self.assertEqual(x_coeffs.shape, (size, 2))
            self.assertEqual(binary_coeffs.shape, (size, ))
            self.assertEqual(constants.shape, (size, ))
            self.strengthened_constraint_tester(c,
                                                w,
                                                b,
                                                lo,
                                                up,
                                                relu_input_lo,
                                                relu_input_up,
                                                ideal=True,
                                                selective=selective)

    def test_2d_case2(self):
        # Test with leaky ReLU.
        c = 0.1
        dtype = torch.float64
        w = torch.tensor([2, -1], dtype=dtype)
        b = torch.tensor(0.5, dtype=dtype)
        lo = torch.tensor([-1, -1.5], dtype=dtype)
        up = torch.tensor([-0.5, 2], dtype=dtype)
        relu_input_lo_ia, relu_input_up_ia = mip_utils.compute_range_by_IA(
            w.reshape((1, -1)), b.reshape((-1, )), lo, up)
        assert (relu_input_lo_ia[0].item() < 0)
        assert (relu_input_up_ia[0].item() > 0)
        relu_input_lo = relu_input_lo_ia[0]
        relu_input_up = relu_input_up_ia[0]
        self.strengthened_constraint_tester(c,
                                            w,
                                            b,
                                            lo,
                                            up,
                                            relu_input_lo,
                                            relu_input_up,
                                            ideal=True,
                                            selective=False)
        self.strengthened_constraint_tester(c,
                                            w,
                                            b,
                                            lo,
                                            up,
                                            relu_input_lo,
                                            relu_input_up,
                                            ideal=False,
                                            selective=True)

    def test_3d_case(self):
        c = 0.1
        dtype = torch.float64
        w = torch.tensor([2, -1, 0.5], dtype=dtype)
        b = torch.tensor(-1.5, dtype=dtype)
        lo = torch.tensor([-1, 0.5, -1.5], dtype=dtype)
        up = torch.tensor([2, 1.5, 1], dtype=dtype)
        relu_input_lo_ia, relu_input_up_ia = mip_utils.compute_range_by_IA(
            w.reshape((1, -1)), b.reshape((-1, )), lo, up)
        assert (relu_input_lo_ia[0].item() < 0)
        assert (relu_input_up_ia[0].item() > 0)
        relu_input_lo = relu_input_lo_ia[0]
        relu_input_up = relu_input_up_ia[0]
        self.strengthened_constraint_tester(c,
                                            w,
                                            b,
                                            lo,
                                            up,
                                            relu_input_lo,
                                            relu_input_up,
                                            ideal=True,
                                            selective=False)
        self.strengthened_constraint_tester(c,
                                            w,
                                            b,
                                            lo,
                                            up,
                                            relu_input_lo,
                                            relu_input_up,
                                            ideal=False,
                                            selective=True)

        # Test a tighter relu input bound.
        relu_input_lo = 0.8 * relu_input_lo_ia[0] + 0.2 * relu_input_up_ia[0]
        relu_input_up = 0.1 * relu_input_lo_ia[0] + 0.9 * relu_input_up_ia[0]
        self.strengthened_constraint_tester(c,
                                            w,
                                            b,
                                            lo,
                                            up,
                                            relu_input_lo,
                                            relu_input_up,
                                            ideal=True,
                                            selective=False)
        self.strengthened_constraint_tester(c,
                                            w,
                                            b,
                                            lo,
                                            up,
                                            relu_input_lo,
                                            relu_input_up,
                                            ideal=False,
                                            selective=True)


class TestComputeRangeByLP(unittest.TestCase):
    def test_x_bounds(self):
        # Test with only bounds on x.
        x_lb = np.array([1., 2.])
        x_ub = np.array([3., 4.])
        A = np.array([[0., 1.], [1., 1.], [2., -1.]])
        b = np.array([0., 1., 3.])
        y_lb, y_ub = mip_utils.compute_range_by_lp(A, b, x_lb, x_ub, None,
                                                   None)
        y_lb_expected = np.array([2., 4, 1.])
        y_ub_expected = np.array([4, 8., 7.])
        np.testing.assert_allclose(y_lb, y_lb_expected, atol=1E-7)
        np.testing.assert_allclose(y_ub, y_ub_expected, atol=1E-7)

        # Some of the bounds should be infinity.
        x_lb = np.array([1., -np.inf])
        y_lb, y_ub = mip_utils.compute_range_by_lp(A, b, x_lb, x_ub, None,
                                                   None)
        y_lb_expected = np.array([-np.inf, -np.inf, 1.])
        y_ub_expected = np.array([4, 8., np.inf])
        np.testing.assert_allclose(y_lb, y_lb_expected, atol=1E-7)
        np.testing.assert_allclose(y_ub, y_ub_expected, atol=1E-7)

        # Now the problem is infeasible as x_lb > x_ub
        x_lb = np.array([4., 2])
        y_lb, y_ub = mip_utils.compute_range_by_lp(A, b, x_lb, x_ub, None,
                                                   None)
        np.testing.assert_allclose(y_lb, np.full((3, ), np.inf))
        np.testing.assert_allclose(y_ub, np.full((3, ), -np.inf))

    def test_ineq_bounds(self):
        # x has both bounds x_lb <= x <= x_ub and inequality bounds C * x <= d
        x_lb = np.array([-1., -2.])
        x_ub = np.array([3., 4.])
        C = np.array([[1., 1.], [1., -1], [-1, 1], [-1, -1.]])
        d = np.array([2, 2, 2, 2])
        A = np.array([[0., 1.], [1., 1.], [2., -1.]])
        b = np.array([0., 1., 3.])
        y_lb, y_ub = mip_utils.compute_range_by_lp(A, b, x_lb, x_ub, C, d)
        y_lb_expected = np.array([-2, -1, 0.])
        y_ub_expected = np.array([2., 3, 7.])
        np.testing.assert_allclose(y_lb, y_lb_expected, atol=1E-7)
        np.testing.assert_allclose(y_ub, y_ub_expected, atol=1E-7)


class TestComputeRangeByIA(unittest.TestCase):
    def test(self):
        dtype = torch.float64
        A = torch.tensor([[1., 2., -3.], [2., -1., -4.]], dtype=dtype)
        b = torch.tensor([2., 3.], dtype=dtype)
        x_lb = torch.tensor([-2., 3., -4.], dtype=dtype)
        x_ub = torch.tensor([1., 5., 7.], dtype=dtype)
        output_lb, output_ub = mip_utils.compute_range_by_IA(A, b, x_lb, x_ub)
        np.testing.assert_allclose(output_lb.detach().numpy(),
                                   np.array([-15, -34]))
        np.testing.assert_allclose(output_ub.detach().numpy(),
                                   np.array([25, 18]))

    def test_gradient(self):
        def test_fun(A_np, b_np, x_lb_np, x_ub_np):
            output_lb, output_ub = mip_utils.compute_range_by_IA(
                torch.from_numpy(A_np.reshape((2, 3))), torch.from_numpy(b_np),
                torch.from_numpy(x_lb_np), torch.from_numpy(x_ub_np))
            return torch.sum(output_lb * output_ub).item()

        dtype = torch.float64
        A = torch.tensor([[1., 2., -3.], [2., -1., -4.]],
                         dtype=dtype,
                         requires_grad=True)
        b = torch.tensor([2., 3.], dtype=dtype, requires_grad=True)
        x_lb = torch.tensor([-2., 3., -4.], dtype=dtype, requires_grad=True)
        x_ub = torch.tensor([1., 5., 7.], dtype=dtype, requires_grad=True)
        output_lb, output_ub = mip_utils.compute_range_by_IA(A, b, x_lb, x_ub)
        torch.sum(output_lb * output_ub).backward()
        A_grad = A.grad.detach().numpy()
        b_grad = b.grad.detach().numpy()
        x_lb_grad = x_lb.grad.detach().numpy()
        x_ub_grad = x_ub.grad.detach().numpy()

        A_grad_numerical, b_grad_numerical, x_lb_grad_numerical,\
            x_ub_grad_numerical = utils.compute_numerical_gradient(
                test_fun, A.detach().numpy().reshape((-1)), b.detach().numpy(),
                x_lb.detach().numpy(), x_ub.detach().numpy())
        np.testing.assert_allclose(A_grad,
                                   A_grad_numerical.reshape((2, 3)),
                                   atol=1E-6)
        np.testing.assert_allclose(b_grad, b_grad_numerical, atol=1E-6)
        np.testing.assert_allclose(x_lb_grad, x_lb_grad_numerical, atol=1E-6)
        np.testing.assert_allclose(x_ub_grad, x_ub_grad_numerical, atol=1E-6)


class TestPropagateBounds(unittest.TestCase):
    def test_relu(self):
        layer = torch.nn.ReLU()
        input_lo = torch.tensor([-2., 2., -3.], dtype=torch.float64)
        input_up = torch.tensor([3., 6., -1.], dtype=torch.float64)
        output_lo, output_up = mip_utils.propagate_bounds(
            layer, input_lo, input_up)
        np.testing.assert_allclose(output_lo, np.array([0, 2., 0.]))
        np.testing.assert_allclose(output_up, np.array([3., 6., 0.]))

    def test_leaky_relu1(self):
        # negative_slope > 0
        layer = torch.nn.LeakyReLU(0.1)
        input_lo = torch.tensor([-2., 2., -3.], dtype=torch.float64)
        input_up = torch.tensor([3., 6., -1.], dtype=torch.float64)
        output_lo, output_up = mip_utils.propagate_bounds(
            layer, input_lo, input_up)
        np.testing.assert_allclose(output_lo, np.array([-0.2, 2., -0.3]))
        np.testing.assert_allclose(output_up, np.array([3., 6., -0.1]))

    def test_leaky_relu2(self):
        # negative_slope < 0
        layer = torch.nn.LeakyReLU(-0.1)
        input_lo = torch.tensor([-10, 2, -8, -1], dtype=torch.float64)
        input_up = torch.tensor([0.5, 8, -2, 4], dtype=torch.float64)
        output_lo, output_up = mip_utils.propagate_bounds(
            layer, input_lo, input_up)
        np.testing.assert_allclose(output_lo, np.array([0, 2, 0.2, 0]))
        np.testing.assert_allclose(output_up, np.array([1., 8., 0.8, 4]))

    def test_linear_layer_no_bias(self):
        layer = torch.nn.Linear(3, 2, bias=False)
        layer.weight.data = torch.tensor([[-1, 0, 2], [3, -2, -1]],
                                         dtype=torch.float64)
        input_lo = torch.tensor([-2, 2, -3], dtype=torch.float64)
        input_up = torch.tensor([3, 6, -1], dtype=torch.float64)
        output_lo, output_up = mip_utils.propagate_bounds(
            layer, input_lo, input_up)
        np.testing.assert_allclose(output_lo.detach().numpy(),
                                   np.array([-9., -17]))
        np.testing.assert_allclose(output_up.detach().numpy(),
                                   np.array([0., 8.]))
        # Now check we can take the gradient.
        loss = output_lo.sum() + output_up.sum()
        loss.backward()

        for s in np.linspace(0, 1, 11):
            output = layer(s * input_lo + (1 - s) * input_up)
            np.testing.assert_array_less(output.detach().numpy(),
                                         output_up.detach().numpy() + 1E-10)
            np.testing.assert_array_less(output_lo.detach().numpy() - 1E-10,
                                         output.detach().numpy())

    def test_linear_layer_bias(self):
        layer = torch.nn.Linear(3, 2, bias=True)
        layer.weight.data = torch.tensor([[-1, 0, 2], [3, -2, -1]],
                                         dtype=torch.float64)
        layer.bias.data = torch.tensor([2., -1.], dtype=torch.float64)
        input_lo = torch.tensor([-2, 2, -3], dtype=torch.float64)
        input_up = torch.tensor([3, 6, -1], dtype=torch.float64)
        output_lo, output_up = mip_utils.propagate_bounds(
            layer, input_lo, input_up)
        np.testing.assert_allclose(output_lo.detach().numpy(),
                                   np.array([-7., -18.]))
        np.testing.assert_allclose(output_up.detach().numpy(),
                                   np.array([2., 7.]))
        # Now check we can take the gradient.
        loss = output_lo.sum() + output_up.sum()
        loss.backward()

        for s in np.linspace(0, 1, 11):
            output = layer(s * input_lo + (1 - s) * input_up)
            np.testing.assert_array_less(output.detach().numpy(),
                                         output_up.detach().numpy() + 1E-10)
            np.testing.assert_array_less(output_lo.detach().numpy() - 1E-10,
                                         output.detach().numpy())


if __name__ == "__main__":
    unittest.main()
