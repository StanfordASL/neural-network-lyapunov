import neural_network_lyapunov.relu_to_optimization as relu_to_optimization
import neural_network_lyapunov.gurobi_torch_mip as gurobi_torch_mip
import neural_network_lyapunov.utils as utils
import neural_network_lyapunov.mip_utils as mip_utils
import unittest
import numpy as np
import torch
import torch.nn as nn
import cvxpy as cp
import gurobipy


class TestReLU(unittest.TestCase):
    def setUp(self):
        self.dtype = torch.float64
        self.linear1 = nn.Linear(2, 3)
        self.linear1.weight.data = torch.tensor([[1, 2], [3, 4], [5, 6]],
                                                dtype=self.dtype)
        self.linear1.bias.data = torch.tensor([-11, 13, 4], dtype=self.dtype)
        self.linear1_no_bias = nn.Linear(2, 3, bias=False)
        self.linear1_no_bias.weight.data = self.linear1.weight.data.clone()

        self.linear2 = nn.Linear(3, 4)
        self.linear2.weight.data = torch.tensor(
            [[-1, 0.5, 1.5], [2, 5, 6], [-2, -3, -4], [1, 4, 6]],
            dtype=self.dtype)
        self.linear2.bias.data = torch.tensor([4, -1, -2, -20],
                                              dtype=self.dtype)
        self.linear2_no_bias = nn.Linear(3, 4, bias=False)
        self.linear2_no_bias.weight.data = self.linear2.weight.data.clone()

        self.linear3 = nn.Linear(4, 1)
        self.linear3.weight.data = torch.tensor([[4, 5, 6, 7]],
                                                dtype=self.dtype)
        self.linear3.bias.data = torch.tensor([-10], dtype=self.dtype)
        self.linear3_no_bias = nn.Linear(4, 1, bias=False)
        self.linear3_no_bias.weight.data = self.linear3.weight.data.clone()
        # Model with ReLU units
        self.model2 = nn.Sequential(self.linear1, nn.ReLU(), self.linear2,
                                    nn.ReLU(), self.linear3)

        # Model with leaky ReLU units.
        self.leaky_relus =\
            [nn.LeakyReLU(0.1), nn.LeakyReLU(0.2)]
        # Model with leaky ReLU with no ReLU unit in the output layer.
        self.model4 = nn.Sequential(self.linear1, self.leaky_relus[0],
                                    self.linear2, self.leaky_relus[1],
                                    self.linear3)

        # Model with leaky ReLU but no bias term
        self.model5 = nn.Sequential(self.linear1_no_bias, self.leaky_relus[0],
                                    self.linear2_no_bias, self.leaky_relus[1],
                                    self.linear3_no_bias)

        # Model with multiple outputs, normal relu
        self.model6 = nn.Sequential(self.linear1, nn.ReLU(), self.linear2)
        # Model with multiple outputs, leaky relu
        self.model7 = nn.Sequential(self.linear1, self.leaky_relus[0],
                                    self.linear2)

    def test_compute_relu_activation_pattern2(self):
        x = torch.tensor([-6, 4], dtype=self.dtype)
        activation_pattern = relu_to_optimization.ComputeReLUActivationPattern(
            self.model2, x)
        self.assertEqual(len(activation_pattern), 2)
        self.assertEqual(len(activation_pattern[0]), 3)
        self.assertEqual(len(activation_pattern[1]), 4)
        x_linear1 = self.linear1.forward(x)
        x_relu1 = nn.ReLU().forward(x_linear1)
        for i in range(3):
            self.assertEqual(x_linear1[i] >= 0, activation_pattern[0][i])
        x_linear2 = self.linear2.forward(x_relu1)
        for i in range(4):
            self.assertEqual(x_linear2[i] >= 0, activation_pattern[1][i])

    def test_compute_all_relu_activation_patterns(self):
        linear1 = nn.Linear(2, 3)
        linear1.weight.data = torch.tensor([[1, 2], [3, 4], [5, 6]],
                                           dtype=self.dtype)
        linear1.bias.data = torch.tensor([-11, 13, 4], dtype=self.dtype)
        linear2 = nn.Linear(3, 3)
        linear2.weight.data = torch.tensor(
            [[3, -2, -1], [1, -4, 0], [0, 1, -2]], dtype=self.dtype)
        linear2.bias.data = torch.tensor([-11, 13, 48], dtype=self.dtype)
        relu = nn.Sequential(linear1, nn.ReLU(), linear2, nn.ReLU())

        # For this input, all ReLU unit inputs are either > 0 or < 0
        patterns = relu_to_optimization.compute_all_relu_activation_patterns(
            relu, torch.tensor([1, 2], dtype=self.dtype))
        self.assertEqual(len(patterns), 1)
        self.assertEqual(patterns[0],
                         [[False, True, True], [False, False, True]])

        # For this input, the input to first layer relu are [0, 36, 39], and
        # the input to the second layer relu are [-122, -131, 6]. Note that
        # one of the input is 0.
        patterns = relu_to_optimization.compute_all_relu_activation_patterns(
            relu, torch.tensor([1, 5], dtype=self.dtype))
        self.assertEqual(len(patterns), 2)
        self.assertEqual(patterns[0],
                         [[True, True, True], [False, False, True]])
        self.assertEqual(patterns[1],
                         [[False, True, True], [False, False, True]])

        # For this input, the input to first layer relu are [0, 0, -33], and
        # the input to the second layer relu are [-11, 13, 38]. Note that
        # two of the inputs are 0.
        patterns = relu_to_optimization.compute_all_relu_activation_patterns(
            relu, torch.tensor([-35, 23], dtype=self.dtype))
        self.assertEqual(len(patterns), 4)
        self.assertEqual(patterns[0],
                         [[True, True, False], [False, True, True]])
        self.assertEqual(patterns[1],
                         [[True, False, False], [False, True, True]])
        self.assertEqual(patterns[2],
                         [[False, True, False], [False, True, True]])
        self.assertEqual(patterns[3],
                         [[False, False, False], [False, True, True]])

        # For this input, the input to first layer relu are [0, 38, 43], and
        # the input to the second layer relu are [-130, -139, 0]. Note that
        # two of the inputs are 0.
        patterns = relu_to_optimization.compute_all_relu_activation_patterns(
            relu, torch.tensor([3, 4], dtype=self.dtype))
        self.assertEqual(len(patterns), 4)
        self.assertEqual(patterns[0],
                         [[True, True, True], [False, False, True]])
        self.assertEqual(patterns[1],
                         [[True, True, True], [False, False, False]])
        self.assertEqual(patterns[2],
                         [[False, True, True], [False, False, True]])
        self.assertEqual(patterns[3],
                         [[False, True, True], [False, False, False]])

    def test_relu_activation_binary_to_pattern(self):
        for model in (self.model2, self.model4):
            activation_pattern = \
                relu_to_optimization.relu_activation_binary_to_pattern(
                    model, np.array([1, 1, 0, 0, 1, 0, 0]))
            self.assertEqual(
                activation_pattern,
                [[True, True, False], [False, True, False, False]])

    def test_relu_given_activation_pattern(self):
        def test_relu_given_activation_pattern_util(self, model, x):
            with torch.no_grad():
                activation_pattern = relu_to_optimization.\
                    ComputeReLUActivationPattern(model, x)
                (g, h, P, q) = relu_to_optimization.ReLUGivenActivationPattern(
                    model, 2, activation_pattern, self.dtype)
            x.requires_grad = True
            output_expected = model.forward(x)
            output_expected.backward()
            np.testing.assert_allclose(x.grad.detach().numpy(),
                                       g.squeeze().detach().numpy())
            with torch.no_grad():
                output = (g.T @ x.reshape((2, 1)) + h).item()
                self.assertAlmostEqual(output, output_expected.item(), 10)
                self.assertTrue(
                    torch.all(torch.le(P @ (x.reshape((-1, 1))), q)))
                # Randomly take 100 sample of inputs. If the sample shares the
                # same activation path as x, then it should satisfy P * x <= q
                # constraint. Otherwise it should violate the constraint.
                for _ in range(100):
                    x_sample = torch.tensor([
                        np.random.uniform(-10, 10),
                        np.random.uniform(-10, 10)
                    ],
                                            dtype=self.dtype)
                    activation_pattern_sample =\
                        relu_to_optimization.ComputeReLUActivationPattern(
                            model, x_sample)
                    output_sample_expected = model.forward(x_sample)
                    if (activation_pattern_sample == activation_pattern):
                        output_sample = g.T @ x_sample.reshape((2, 1)) + h
                        self.assertAlmostEqual(output_sample.item(),
                                               output_sample_expected.item(),
                                               10)
                        self.assertTrue(
                            torch.all(
                                torch.le(P @ (x_sample.reshape((-1, 1))), q)))
                    else:
                        self.assertFalse(
                            torch.all(
                                torch.le(P @ (x_sample.reshape((-1, 1))), q)))

        for model in (self.model2, self.model4, self.model5):
            test_relu_given_activation_pattern_util(
                self, model, torch.tensor([-6, 4], dtype=self.dtype))
            test_relu_given_activation_pattern_util(
                self, model, torch.tensor([-10, 4], dtype=self.dtype))
            test_relu_given_activation_pattern_util(
                self, model, torch.tensor([3, -4], dtype=self.dtype))
            test_relu_given_activation_pattern_util(
                self, model, torch.tensor([-3, -4], dtype=self.dtype))
            test_relu_given_activation_pattern_util(
                self, model, torch.tensor([-10, -20], dtype=self.dtype))

    def test_relu_free_pattern_constructor2(self):
        relu_free_pattern = relu_to_optimization.ReLUFreePattern(
            self.model2, self.dtype)
        self.assertEqual(len(relu_free_pattern.relu_unit_index), 2)
        self.assertListEqual(relu_free_pattern.relu_unit_index[0], [0, 1, 2])
        self.assertListEqual(relu_free_pattern.relu_unit_index[1],
                             [3, 4, 5, 6])
        self.assertEqual(relu_free_pattern.num_relu_units, 7)

    def test_relu_free_pattern_output_constraint(self):
        def test_model(model, method):
            relu_free_pattern = relu_to_optimization.ReLUFreePattern(
                model, self.dtype)
            x_lo = torch.tensor([-1, -2], dtype=self.dtype)
            x_up = torch.tensor([2, 3], dtype=self.dtype)
            (mip_constr_return,
                z_pre_relu_lo, z_pre_relu_up, z_post_relu_lo, z_post_relu_up,
                output_lo, output_up)\
                = relu_free_pattern.output_constraint(x_lo, x_up, method)
            self.assertIsNone(mip_constr_return.Aout_input)
            self.assertIsNone(mip_constr_return.Aout_binary)
            # print("z_pre_relu_lo:{}\nz_pre_relu_up:{}".format(
            #    z_pre_relu_lo, z_pre_relu_up))
            num_z_pre_relu_lo_positive = np.sum(
                [z_pre_relu_lo_i >= 0 for z_pre_relu_lo_i in z_pre_relu_lo])
            num_z_pre_relu_up_negative = np.sum(
                [z_pre_relu_up_i <= 0 for z_pre_relu_up_i in z_pre_relu_up])
            num_ineq = (relu_free_pattern.num_relu_units -
                        num_z_pre_relu_lo_positive - num_z_pre_relu_up_negative
                        ) * 4 + 4 + (num_z_pre_relu_lo_positive +
                                     num_z_pre_relu_up_negative) * 2
            num_eq = (num_z_pre_relu_lo_positive + num_z_pre_relu_up_negative)\
                * 2
            self.assertEqual(mip_constr_return.Ain_input.shape, (num_ineq, 2))
            self.assertEqual(mip_constr_return.Ain_slack.shape,
                             (num_ineq, relu_free_pattern.num_relu_units))
            self.assertEqual(mip_constr_return.Ain_binary.shape,
                             (num_ineq, relu_free_pattern.num_relu_units))
            self.assertEqual(mip_constr_return.rhs_in.shape, (num_ineq, ))
            self.assertEqual(mip_constr_return.Aeq_input.shape, (num_eq, 2))
            self.assertEqual(mip_constr_return.Aeq_slack.shape,
                             (num_eq, relu_free_pattern.num_relu_units))
            self.assertEqual(mip_constr_return.Aeq_binary.shape,
                             (num_eq, relu_free_pattern.num_relu_units))
            self.assertEqual(mip_constr_return.rhs_eq.shape, (num_eq))

            def test_input_output(x):
                (z, beta, output) = \
                    relu_free_pattern.compute_relu_unit_outputs_and_activation(
                    x)
                if torch.all(x <= x_up) and torch.all(x >= x_lo):
                    output_torch = relu_free_pattern.model(x)
                    np.testing.assert_array_less(
                        output_torch.detach().numpy(),
                        output_up.detach().numpy() + 1E-6)
                    np.testing.assert_array_less(
                        output_lo.detach().numpy() - 1E-6,
                        output_torch.detach().numpy())
                # Now formulate an optimization problem, with fixed input,
                # search for z and beta. There should be only one solution.
                z_var = cp.Variable(relu_free_pattern.num_relu_units)
                beta_var = cp.Variable(relu_free_pattern.num_relu_units,
                                       boolean=True)
                x_np = x.detach().numpy()
                con = []
                if mip_constr_return.rhs_in.shape[0] != 0:
                    con.append(
                        mip_constr_return.Ain_input.detach().numpy() @ x_np +
                        mip_constr_return.Ain_slack.detach().numpy() @ z_var +
                        mip_constr_return.Ain_binary.detach().numpy() @
                        beta_var
                        <= mip_constr_return.rhs_in.squeeze().detach().numpy())
                if mip_constr_return.rhs_eq.shape[0] != 0:
                    con.append(
                        mip_constr_return.Aeq_input.detach().numpy() @ x_np +
                        mip_constr_return.Aeq_slack.detach().numpy() @ z_var +
                        mip_constr_return.Aeq_binary.detach().numpy() @
                        beta_var
                        == mip_constr_return.rhs_eq.squeeze().detach().numpy())
                objective = cp.Minimize(0.)
                prob = cp.Problem(objective, con)
                prob.solve(solver=cp.GUROBI)
                if torch.all(x <= x_up) and torch.all(x >= x_lo):
                    self.assertEqual(prob.status, "optimal")
                    z_opt_var = z_var.value
                    beta_opt_var = beta_var.value
                    np.testing.assert_array_almost_equal(
                        z_opt_var,
                        z.squeeze().detach().numpy())
                    np.testing.assert_array_almost_equal(
                        beta_opt_var,
                        beta.squeeze().detach().numpy())
                    if len(mip_constr_return.Aout_slack.shape) > 1:
                        out_opt = mip_constr_return.Aout_slack @ z.squeeze()\
                            + mip_constr_return.Cout
                        output = model.forward(x)
                        for k in range(len(output)):
                            self.assertAlmostEqual(output[k].item(),
                                                   out_opt[k].item())
                    else:
                        self.assertAlmostEqual(
                            (mip_constr_return.Aout_slack @ z.squeeze() +
                             mip_constr_return.Cout).item(),
                            model.forward(x).item())
                else:
                    self.assertEqual(prob.status, "infeasible")

            def test_input_x(x):
                # For an arbitrary input x, compute its activation pattern and
                # output of each ReLU unit, check if they satisfy the
                # constraint
                # Ain1*x+Ain2*z+Ain3*β <= rhs_in
                # Aeq1*x+Aeq2*z+Aeq3*β <= rhs_eq
                assert (torch.all(torch.ge(x, x_lo.squeeze())))
                assert (torch.all(torch.le(x, x_up.squeeze())))
                (z, beta, output) = \
                    relu_free_pattern.compute_relu_unit_outputs_and_activation(
                    x)
                z_post_relu_up_numpy =\
                    np.array([zi.detach().numpy() for zi in z_post_relu_up])
                z_post_relu_lo_numpy =\
                    np.array([zi.detach().numpy() for zi in z_post_relu_lo])
                np.testing.assert_array_less(z.squeeze().detach().numpy(),
                                             z_post_relu_up_numpy + 1E-10)
                np.testing.assert_array_less(z_post_relu_lo_numpy - 1E-10,
                                             z.squeeze().detach().numpy())
                # Check the output
                if isinstance(output, torch.Tensor):
                    out_opt = mip_constr_return.Aout_slack @ z.squeeze() +\
                        mip_constr_return.Cout
                    for k in range(len(output)):
                        self.assertAlmostEqual(output[k], out_opt[k], 3)
                else:
                    self.assertAlmostEqual(output,
                                           (mip_constr_return.Aout_slack @ z +
                                            mip_constr_return.Cout).item(), 3)
                x_vec = x.reshape((-1, 1))
                lhs_in = mip_constr_return.Ain_input @ x_vec +\
                    mip_constr_return.Ain_slack @ z +\
                    mip_constr_return.Ain_binary @ beta
                lhs_eq = mip_constr_return.Aeq_input @ x_vec +\
                    mip_constr_return.Aeq_slack @ z +\
                    mip_constr_return.Aeq_binary @ beta
                precision = 1E-10
                np.testing.assert_array_less(
                    lhs_in.squeeze().detach().numpy(),
                    mip_constr_return.rhs_in.squeeze().detach().numpy() +
                    precision)
                np.testing.assert_allclose(
                    lhs_eq.squeeze().detach().numpy(),
                    mip_constr_return.rhs_eq.squeeze().detach().numpy())
                # Now perturb beta by changing some entry from 1 to 0, and vice
                # versa. Now it should not satisfy the constraint.
                perturbed_beta_entry = np.random.randint(0, beta.numel())
                beta_perturbed = beta.clone()
                beta_perturbed[perturbed_beta_entry] =\
                    1 - beta[perturbed_beta_entry]
                lhs_in_perturbed = mip_constr_return.Ain_input @ x_vec +\
                    mip_constr_return.Ain_slack @ z +\
                    mip_constr_return.Ain_slack @ beta_perturbed
                lhs_eq_perturbed = mip_constr_return.Aeq_input @ x_vec +\
                    mip_constr_return.Aeq_slack @ z +\
                    mip_constr_return.Aeq_binary @ beta_perturbed
                self.assertFalse(
                    torch.all(
                        torch.le(
                            lhs_in_perturbed.squeeze(),
                            mip_constr_return.rhs_in.squeeze() +
                            torch.tensor(precision)))
                    and torch.all(
                        torch.le(
                            torch.abs(lhs_eq_perturbed -
                                      mip_constr_return.rhs_eq), precision)))
                test_input_output(x)

            # Test with different input x.
            test_input_x(torch.tensor([0.7, 0.2], dtype=self.dtype))
            test_input_x(torch.tensor([-0.3, 0.2], dtype=self.dtype))
            test_input_x(torch.tensor([-0.15, -0.2], dtype=self.dtype))
            test_input_x(torch.tensor([1.1, -0.22], dtype=self.dtype))
            test_input_x(torch.tensor([1.5, -0.8], dtype=self.dtype))
            # The next two input x are outside of [x_lo, x_up]. The constraints
            # should be infeasible.
            test_input_output(torch.tensor([-2., 10.], dtype=self.dtype))
            test_input_output(torch.tensor([-2., 4.], dtype=self.dtype))
            # randomized test
            torch.manual_seed(0)
            np.random.seed(0)
            for _ in range(30):
                found_x = False
                while (not found_x):
                    x_random = torch.tensor(
                        [np.random.normal(0, 1), np.random.normal(0, 1)]).\
                        type(self.dtype)
                    if (torch.all(x_random >= x_lo)
                            and torch.all(x_random <= x_up)):
                        found_x = True
                test_input_x(x_random)

        for method in list(mip_utils.PropagateBoundsMethod):
            test_model(self.model2, method)
            test_model(self.model4, method)
            test_model(self.model5, method)
            test_model(self.model6, method)
            test_model(self.model7, method)

    def relu_free_pattern_output_constraint_gradient_tester(self, model):
        # This is the utility function for
        # test_relu_free_pattern_output_constraint_gradient()
        # Test the gradient of the returned argument in
        # ReLUFreePattern.output_constraint. Later we will use this gradient
        # extensively to compute the gradient of the loss w.r.t the network
        # weights/biases, so it is important to make sure that this gradient
        # is correct.
        def compute_loss(*network_parameters):
            param_index = 0
            for layer in model:
                if (isinstance(layer, torch.nn.Linear)):
                    layer.weight = network_parameters[param_index]
                    param_index += 1
                    if (layer.bias is not None):
                        layer.bias = network_parameters[param_index]
                        param_index += 1
            relu_free_pattern = relu_to_optimization.ReLUFreePattern(
                model, self.dtype)
            x_lo = torch.tensor([-1, -2], dtype=self.dtype)
            x_up = torch.tensor([2, 3], dtype=self.dtype)
            mip_constr_return, _, _, _, _, _, _ = \
                relu_free_pattern.output_constraint(
                    x_lo, x_up, mip_utils.PropagateBoundsMethod.IA)
            # This function compute the sum of all the return terms. If any of
            # the term has a wrong gradient, the gradient of the sum will also
            # be wrong.
            objective1 = mip_constr_return.Ain_input.sum() +\
                mip_constr_return.Ain_slack.sum() +\
                mip_constr_return.Ain_binary.sum() + \
                mip_constr_return.rhs_in.sum() +\
                mip_constr_return.Aeq_input.sum() +\
                mip_constr_return.Aeq_slack.sum() +\
                mip_constr_return.Aeq_binary.sum() + \
                mip_constr_return.rhs_eq.sum() +\
                mip_constr_return.Aout_slack.sum()
            if isinstance(mip_constr_return.Cout, torch.Tensor):
                objective1 += mip_constr_return.Cout.sum()
            return objective1
            # end of compute_loss

        # Now extract all the parameters in the model
        params_list = []
        for layer in model:
            if isinstance(layer, torch.nn.Linear):
                params_list.append(layer.weight)
                if layer.bias is not None:
                    params_list.append(layer.bias)
        torch.autograd.gradcheck(compute_loss, params_list, atol=1e-6)

    def test_relu_free_pattern_output_constraint_gradient2(self):
        self.relu_free_pattern_output_constraint_gradient_tester(self.model2)

    def test_relu_free_pattern_output_constraint_gradient4(self):
        self.relu_free_pattern_output_constraint_gradient_tester(self.model4)

    def test_relu_free_pattern_output_constraint_gradient5(self):
        self.relu_free_pattern_output_constraint_gradient_tester(self.model5)

    def test_relu_free_pattern_output_constraint_gradient6(self):
        self.relu_free_pattern_output_constraint_gradient_tester(self.model6)

    def test_relu_free_pattern_output_constraint_gradient7(self):
        self.relu_free_pattern_output_constraint_gradient_tester(self.model7)

    def test_compute_alpha_index2(self):
        relu_free_pattern = relu_to_optimization.\
            ReLUFreePattern(self.model2, self.dtype)
        self.assertEqual(relu_free_pattern.compute_alpha_index((0, 0)), 0)
        self.assertEqual(relu_free_pattern.compute_alpha_index((0, 1)), 1)
        self.assertEqual(relu_free_pattern.compute_alpha_index((0, 2)), 2)
        self.assertEqual(relu_free_pattern.compute_alpha_index((0, 3)), 3)
        self.assertEqual(relu_free_pattern.compute_alpha_index((1, 0)), 4)
        self.assertEqual(relu_free_pattern.compute_alpha_index((1, 1)), 5)
        self.assertEqual(relu_free_pattern.compute_alpha_index((1, 2)), 6)
        self.assertEqual(relu_free_pattern.compute_alpha_index((1, 3)), 7)
        self.assertEqual(relu_free_pattern.compute_alpha_index((2, 0)), 8)
        self.assertEqual(relu_free_pattern.compute_alpha_index((2, 1)), 9)
        self.assertEqual(relu_free_pattern.compute_alpha_index((2, 2)), 10)
        self.assertEqual(relu_free_pattern.compute_alpha_index((2, 3)), 11)

    def test_output_gradient(self):
        def test_model(model):
            relu_free_pattern = relu_to_optimization.\
                ReLUFreePattern(model, self.dtype)
            (M, B1, B2, d) = relu_free_pattern.output_gradient()
            num_alpha = 12
            self.assertListEqual(list(M.shape), [num_alpha, 2])

            # Enumerate all the possible activation path, with only one ReLU
            # unit active at each layer. Compute the gradient of the ReLU
            # network network for each activation path through
            # ReLUGivenActivationPattern(), and compare the result aginst M.
            activation_pattern = [[False, False, False],
                                  [False, False, False, False]]
            precision = 1E-10
            for i0 in range(3):
                activation_pattern[0] = [False, False, False]
                activation_pattern[0][i0] = True
                for i1 in range(4):
                    activation_pattern[1] = [False, False, False, False]
                    activation_pattern[1][i1] = True
                    (g, _, _, _) =\
                        relu_to_optimization.ReLUGivenActivationPattern(
                        model, 2, activation_pattern, self.dtype)
                    alpha_index = relu_free_pattern.compute_alpha_index(
                        (i0, i1))
                    self.assertTrue(
                        torch.all(
                            torch.abs(M[alpha_index] -
                                      g.reshape((1, -1))) < precision))
                    alpha_value = torch.zeros((num_alpha, 1), dtype=self.dtype)
                    alpha_value[alpha_index][0] = 1.
                    beta_value = torch.zeros(
                        (relu_free_pattern.num_relu_units, 1),
                        dtype=self.dtype)
                    beta_value[relu_free_pattern.relu_unit_index[0][i0]][0] =\
                        1.
                    beta_value[relu_free_pattern.relu_unit_index[1][i1]][0] =\
                        1.
                    self.assertTrue(
                        torch.all(B1 @ alpha_value + B2 @ beta_value -
                                  d < precision))
                    # Now perturb alpha value a bit, by negating a value from 1
                    # to 0 or vice versa, the perturbed alpha and beta should
                    # violate the constraint.
                    perturbed_alpha_entry = np.random.randint(
                        0, alpha_value.numel())
                    alpha_value[perturbed_alpha_entry] = 1. - \
                        alpha_value[perturbed_alpha_entry]
                    self.assertFalse(
                        torch.all(B1 @ alpha_value + B2 @ beta_value -
                                  d < precision))

        test_model(self.model2)

    def test_output_gradient_times_vector(self):
        def test_model(model, x, y, y_lo, y_up):
            assert (x.shape == (2, ))
            assert (y.shape == (2, ))
            assert (y_lo.shape == (2, ))
            assert (y_up.shape == (2, ))
            assert (torch.all(y <= y_up) and torch.all(y >= y_lo))
            activation_pattern =\
                relu_to_optimization.ComputeReLUActivationPattern(model, x)
            relu_free_pattern = relu_to_optimization.ReLUFreePattern(
                model, self.dtype)
            beta = torch.empty((relu_free_pattern.num_relu_units, ),
                               dtype=self.dtype)
            for layer in range(len(relu_free_pattern.relu_unit_index)):
                for index, beta_index in enumerate(
                        relu_free_pattern.relu_unit_index[layer]):
                    beta[beta_index] = 1. if activation_pattern[layer][index]\
                        else 0.
            (g, _, _, _) = relu_to_optimization.ReLUGivenActivationPattern(
                model, 2, activation_pattern, self.dtype)
            output_expected = g.squeeze() @ y
            (a_out, A_y, A_z, A_beta, rhs, z_lo, z_up) =\
                relu_free_pattern.output_gradient_times_vector(
                    y_lo, y_up)

            # Now compute z manually
            z_expected = torch.empty((relu_free_pattern.num_relu_units),
                                     dtype=self.dtype)
            z_pre = y
            layer_count = 0
            for layer in model:
                if (isinstance(layer, nn.Linear)):
                    z_cur = layer.weight.data @ z_pre
                elif isinstance(layer, nn.ReLU) or \
                        isinstance(layer, nn.LeakyReLU):
                    if isinstance(layer, nn.ReLU):
                        z_cur = beta[relu_free_pattern.
                                     relu_unit_index[layer_count]] * z_cur
                    else:
                        for i in range(
                                len(relu_free_pattern.
                                    relu_unit_index[layer_count])):
                            if activation_pattern[layer_count][i]:
                                z_cur[i] = z_cur[i]
                            else:
                                z_cur[i] = layer.negative_slope * z_cur[i]
                    z_expected[relu_free_pattern.relu_unit_index[layer_count]]\
                        = z_cur
                    z_pre = z_cur
                    layer_count += 1

            # Now check that z_expected is within the bound.
            np.testing.assert_array_less(z_expected.detach().numpy(),
                                         z_up.detach().numpy() + 1E-10)
            np.testing.assert_array_less(z_lo.detach().numpy() - 1E-10,
                                         z_expected.detach().numpy())
            # Check that the output equals to a_out.dot(z_expected)
            self.assertAlmostEqual((a_out @ z_expected).item(),
                                   output_expected.item())
            # Check that y, z, beta satisfies the constraint
            lhs = A_y @ y + A_z @ z_expected + A_beta @ beta
            np.testing.assert_array_less(lhs.detach().numpy(),
                                         rhs.detach().numpy() + 1E-10)

            # Now solve an optimization problem satisfying the constraint, and
            # fix y and beta. The only z that satisfies the constraint should
            # be z_expected.
            z_var = cp.Variable(relu_free_pattern.num_relu_units)
            objective = cp.Minimize(0.)
            con = [
                A_z.detach().numpy() @ z_var <=
                (rhs - A_y @ y - A_beta @ beta).detach().numpy()
            ]
            prob = cp.Problem(objective, con)
            prob.solve(solver=cp.GUROBI)
            np.testing.assert_array_almost_equal(z_var.value,
                                                 z_expected.detach().numpy())

        # Check for different models and inputs.
        for model in (self.model2, self.model4, self.model5):
            test_model(model, torch.tensor([1.5, 2.], dtype=self.dtype),
                       torch.tensor([0., 0.], dtype=self.dtype),
                       torch.tensor([-1., -2.], dtype=self.dtype),
                       torch.tensor([20., 3.], dtype=self.dtype))
            test_model(model, torch.tensor([2., 3.], dtype=self.dtype),
                       torch.tensor([1., 2.], dtype=self.dtype),
                       torch.tensor([-1., 0.], dtype=self.dtype),
                       torch.tensor([2., 3.], dtype=self.dtype))
            test_model(model, torch.tensor([2., -1.], dtype=self.dtype),
                       torch.tensor([1., 2.], dtype=self.dtype),
                       torch.tensor([-1., 0.], dtype=self.dtype),
                       torch.tensor([2., 3.], dtype=self.dtype))
            test_model(model, torch.tensor([-2., -1.], dtype=self.dtype),
                       torch.tensor([1., 2.], dtype=self.dtype),
                       torch.tensor([-1., 1.], dtype=self.dtype),
                       torch.tensor([2., 3.], dtype=self.dtype))
            test_model(model, torch.tensor([-4., -2.], dtype=self.dtype),
                       torch.tensor([1.5, -2.], dtype=self.dtype),
                       torch.tensor([-1., -4.], dtype=self.dtype),
                       torch.tensor([2., 3.], dtype=self.dtype))
            test_model(model, torch.tensor([-4., 2.5], dtype=self.dtype),
                       torch.tensor([2.5, -2.], dtype=self.dtype),
                       torch.tensor([-1., -4.], dtype=self.dtype),
                       torch.tensor([4., 3.], dtype=self.dtype))
        # randomized test.
        torch.manual_seed(0)
        np.random.seed(0)
        for _ in range(30):
            found_y_bound = False
            while (not found_y_bound):
                y_lo = torch.tensor(
                    [-1 + np.random.normal(0, 1), 1. + np.random.normal(0, 1)],
                    dtype=self.dtype)
                y_up = torch.tensor(
                    [2 + np.random.normal(0, 1), 3. + np.random.normal(0, 1)],
                    dtype=self.dtype)
                if torch.all(y_up > y_lo):
                    found_y_bound = True
            y = torch.tensor([
                np.random.uniform(y_lo[0], y_up[0]),
                np.random.uniform(y_lo[1], y_up[1])
            ],
                             dtype=self.dtype)
            x = torch.from_numpy(np.random.normal(0, 1,
                                                  (2, ))).type(self.dtype)
            test_model(self.model2, x, y, y_lo, y_up)

    def test_set_activation_warmstart(self):
        x = torch.tensor([1, 2], dtype=self.dtype)
        for model in [self.model2]:
            pattern = relu_to_optimization.ComputeReLUActivationPattern(
                model, x)
            pattern_flat = []
            for a in pattern:
                for b in a:
                    pattern_flat.append(b.item())
            milp = gurobi_torch_mip.GurobiTorchMILP(self.dtype)
            z = milp.addVars(len(pattern_flat),
                             lb=-gurobipy.GRB.INFINITY,
                             vtype=gurobipy.GRB.BINARY,
                             name="z")
            relu_to_optimization.set_activation_warmstart(model, z, x)
            milp.gurobi_model.update()
            for i in range(len(z)):
                self.assertEqual(bool(z[i].start), pattern_flat[i])

    def test_strengthen_mip_at_point1(self):
        # The un-strengthened LP relaxation has an optimal integral solution,
        # so calling strengthen function should return None
        dut = relu_to_optimization.ReLUFreePattern(self.model2, self.dtype)
        nn_input_lo = torch.tensor([-1, -2], dtype=self.dtype)
        nn_input_up = torch.tensor([1, 1.5], dtype=self.dtype)
        mip_cnstr_return, _, _, z_post_relu_lo, z_post_relu_up, _, _ = \
            dut.output_constraint(
                nn_input_lo, nn_input_up, mip_utils.PropagateBoundsMethod.IA)
        # Now set-up an LP to find a solution, that satisfies the linear
        # relaxation of the MIP.
        prog = gurobi_torch_mip.GurobiTorchMILP(self.dtype)
        x = prog.addVars(self.model2[0].in_features, lb=-gurobipy.GRB.INFINITY)
        nn_out = prog.addVars(self.model2[-1].out_features,
                              lb=-gurobipy.GRB.INFINITY)
        slack, activation = prog.add_mixed_integer_linear_constraints(
            mip_cnstr_return,
            x,
            nn_out,
            "s",
            "beta",
            "",
            "",
            "",
            lp_relaxation=True)
        # Optimize an arbitrary cost.
        prog.setObjective(
            [torch.ones(
                (self.model2[-1].out_features, ), dtype=self.dtype)], [nn_out],
            constant=0.,
            sense=gurobipy.GRB.MAXIMIZE)
        prog.gurobi_model.setParam(gurobipy.GRB.Param.OutputFlag, False)
        prog.gurobi_model.optimize()
        self.assertEqual(prog.gurobi_model.status, gurobipy.GRB.Status.OPTIMAL)
        x_sol = torch.tensor([v.x for v in x], dtype=self.dtype)
        slack_sol = torch.tensor([v.x for v in slack], dtype=self.dtype)
        activation_sol = torch.tensor([v.x for v in activation],
                                      dtype=self.dtype)
        # Make sure that the activation variable has integral value.
        # If not, we will abort this test.
        for v in activation:
            assert (np.abs(v.x) < 1E-6 or np.abs(v.x - 1) < 1E-6)
        point = (torch.cat((x_sol, slack_sol)), activation_sol)
        linear_inputs_lo = torch.cat((nn_input_lo, z_post_relu_lo))
        linear_inputs_up = torch.cat((nn_input_up, z_post_relu_up))
        Ain_input, Ain_slack, Ain_binary, rhs_in = \
            dut._strengthen_mip_at_point(
                point, linear_inputs_lo, linear_inputs_up)
        self.assertIsNone(Ain_input)
        self.assertIsNone(Ain_slack)
        self.assertIsNone(Ain_binary)
        self.assertIsNone(rhs_in)

    def strengthen_mip_at_point_tester(self, model, nn_input_lo, nn_input_up,
                                       method: mip_utils.PropagateBoundsMethod,
                                       cost_coeff_output):
        """
        1. First formulate an LP relaxation of the MIP for the network.
        2. Solve this LP relaxation. Make sure the solution is non-integral.
        3. Strengthen the big-M formulation at the solution in step 2.
        4. Solve this strengthened LP again. Make sure the objective is better.
        5. Sample many feasible solution of the network, make sure they still
           satisfies the strengthened formulation.
        """
        dut = relu_to_optimization.ReLUFreePattern(model, self.dtype)
        mip_cnstr_return, _, _, z_post_relu_lo, z_post_relu_up, _, _ = \
            dut.output_constraint(nn_input_lo, nn_input_up, method)
        # Now set-up an LP to find a solution, that satisfies the linear
        # relaxation of the MIP.
        prog = gurobi_torch_mip.GurobiTorchMILP(self.dtype)
        x = prog.addVars(model[0].in_features, lb=-gurobipy.GRB.INFINITY)
        nn_out = prog.addVars(model[-1].out_features,
                              lb=-gurobipy.GRB.INFINITY)
        slack, activation = prog.add_mixed_integer_linear_constraints(
            mip_cnstr_return,
            x,
            nn_out,
            "s",
            "beta",
            "",
            "",
            "",
            lp_relaxation=True)
        # Optimize an arbitrary cost.
        prog.setObjective([cost_coeff_output], [nn_out],
                          constant=0.,
                          sense=gurobipy.GRB.MAXIMIZE)
        prog.gurobi_model.setParam(gurobipy.GRB.Param.OutputFlag, False)
        prog.gurobi_model.optimize()
        self.assertEqual(prog.gurobi_model.status, gurobipy.GRB.Status.OPTIMAL)
        lp_obj = prog.gurobi_model.ObjVal
        x_sol = torch.tensor([v.x for v in x], dtype=self.dtype)
        slack_sol = torch.tensor([v.x for v in slack], dtype=self.dtype)
        activation_sol = torch.tensor([v.x for v in activation],
                                      dtype=self.dtype)
        # This LP should have non-integral optimal solutions. Abort this test
        # if the solution is integral.
        assert (torch.any(torch.abs(activation_sol) > 1E-6)
                or torch.any(torch.abs(activation_sol - 1) > 1E-6))
        pt = (torch.cat((x_sol, slack_sol)), activation_sol)
        linear_inputs_lo = torch.cat((nn_input_lo, z_post_relu_lo))
        linear_inputs_up = torch.cat((nn_input_up, z_post_relu_up))
        Ain_input, Ain_slack, Ain_binary, rhs_in = \
            dut._strengthen_mip_at_point(
                pt, linear_inputs_lo, linear_inputs_up)
        self.assertIsNotNone(Ain_input)
        self.assertIsNotNone(Ain_slack)
        self.assertIsNotNone(Ain_binary)
        self.assertIsNotNone(rhs_in)
        # I "think" the number of strengthend constraints equal to the number
        # of non-integral activations at the LP solution, this should be true
        # at least for LP constructed with bounds from interval arithmetics.
        if (method == mip_utils.PropagateBoundsMethod.IA):
            self.assertEqual(
                Ain_input.shape[0],
                torch.sum(
                    torch.logical_and(
                        torch.abs(activation_sol) > 1E-6,
                        torch.abs(activation_sol - 1) > 1E-6)))
        prog.addMConstrs([Ain_input, Ain_slack, Ain_binary],
                         [x, slack, activation],
                         sense=gurobipy.GRB.LESS_EQUAL,
                         b=rhs_in)
        prog.gurobi_model.optimize()
        self.assertEqual(prog.gurobi_model.status, gurobipy.GRB.Status.OPTIMAL)
        strengthen_lp_obj = prog.gurobi_model.ObjVal
        self.assertLess(strengthen_lp_obj, lp_obj)

        # Now sample many neural network inputs. Make sure the slack variables
        # and the binary variables computed from these inputs all satisfy the
        # newly strengthened constraints.
        num_samples = 1000
        nn_input_samples = utils.uniform_sample_in_box(nn_input_lo,
                                                       nn_input_up,
                                                       num_samples)
        slack_samples = []
        activation_samples = []
        linear_layer_input = nn_input_samples
        with torch.no_grad():
            for layer_count in range(len(dut.relu_unit_index)):
                linear_layer_output = model[2 *
                                            layer_count](linear_layer_input)
                relu_layer_activation = (linear_layer_output >= 0).double()
                relu_layer_output = model[2 * layer_count +
                                          1](linear_layer_output)
                slack_samples.append(relu_layer_output.clone())
                activation_samples.append(relu_layer_activation.clone())
                linear_layer_input = relu_layer_output
        slack_samples = torch.cat(slack_samples, dim=1)
        activation_samples = torch.cat(activation_samples, dim=1)
        np.testing.assert_array_less(
            (Ain_input @ (nn_input_samples.T) + Ain_slack @ (slack_samples.T) +
             Ain_binary @ (activation_samples.T)).detach().numpy(),
            rhs_in.reshape((-1, 1)).repeat(
                (1, num_samples)).detach().numpy() + 1E-6)

    def test_strengthen_mip_at_point2(self):
        # Test a leaky ReLU model
        model = utils.setup_relu((2, 4, 5, 3),
                                 params=None,
                                 negative_slope=0.1,
                                 bias=True,
                                 dtype=self.dtype)
        model[0].weight.data = torch.tensor(
            [[1, -1], [0, 2], [2, -1], [-2, -3]], dtype=self.dtype)
        model[0].bias.data = torch.tensor([0.5, 1., -2., -1.],
                                          dtype=self.dtype)
        model[2].weight.data = torch.tensor(
            [[0.5, 1.5, -0.5, -1], [-1, 2., 3., 0.5], [-1.5, 2.5, 0.5, -2],
             [0.5, -0.5, -1, 2.], [1.5, 2., 2.5, -1]],
            dtype=self.dtype)
        model[2].bias.data = torch.tensor([0.5, -0.5, -1., 2.5, -1],
                                          dtype=self.dtype)
        model[4].weight.data = torch.tensor(
            [[1., -2., -3., 0.5, -1.], [0.5, -1., 0.5, 1.5, -1],
             [1.5, -0.5, -1., -2., 0.5]],
            dtype=self.dtype)
        model[4].bias.data = torch.tensor([-1., -2., 1.5], dtype=self.dtype)

        nn_input_lo = torch.tensor([-1., -3.], dtype=self.dtype)
        nn_input_up = torch.tensor([2., -1.], dtype=self.dtype)

        cost_coeff_output = torch.ones((model[-1].out_features, ),
                                       dtype=self.dtype)
        for method in mip_utils.PropagateBoundsMethod:
            self.strengthen_mip_at_point_tester(
                model, nn_input_lo, nn_input_up, method, cost_coeff_output)

    def test_strengthen_mip_at_point3(self):
        # Test another leaky ReLU model
        model = utils.setup_relu((2, 4, 5, 3),
                                 params=None,
                                 negative_slope=0.1,
                                 bias=True,
                                 dtype=self.dtype)
        model[0].weight.data = torch.tensor(
            [[1, -3], [1, 2], [2, -4], [-2, -3]], dtype=self.dtype)
        model[0].bias.data = torch.tensor([1.5, 1., -2., -1.],
                                          dtype=self.dtype)
        model[2].weight.data = torch.tensor(
            [[0.5, 1.5, -0.5, -1], [-1, 2.5, 3., 0.5], [-1.5, 2.5, 0.5, -2],
             [0.5, -0.5, -1, 2.], [1.5, 2.5, 2.5, -1]],
            dtype=self.dtype)
        model[2].bias.data = torch.tensor([1.5, -2.5, -1., 2.5, -1],
                                          dtype=self.dtype)
        model[4].weight.data = torch.tensor(
            [[1., -2., -3., 1.5, -1.], [1.5, -1., 0.5, 1.5, -1],
             [1.5, -2.5, -1., -2., 0.5]],
            dtype=self.dtype)
        model[4].bias.data = torch.tensor([-1.5, -2., 1.5], dtype=self.dtype)

        nn_input_lo = torch.tensor([1., -3.], dtype=self.dtype)
        nn_input_up = torch.tensor([3., 1.], dtype=self.dtype)

        cost_coeff_output = -torch.ones((model[-1].out_features, ),
                                        dtype=self.dtype)

        for method in mip_utils.PropagateBoundsMethod:
            self.strengthen_mip_at_point_tester(
                model, nn_input_lo, nn_input_up, method, cost_coeff_output)


class TestReLUFreePatternOutputConstraintGradient(unittest.TestCase):
    """
    This class tests that the gradient of each entry in
    ReLUFreePattern.output_constraint() function is correct, by comparing the
    automatic differentiation result versus numerical gradient.
    """
    def entry_gradient_tester(self, network, network_param, x_lo, x_up,
                              mip_entry_name, mip_entry_size, atol, rtol):
        def get_entry(params, entry_index):
            if isinstance(params, np.ndarray):
                params_torch = torch.from_numpy(params)
            elif isinstance(params, torch.Tensor):
                params_torch = params
            utils.update_relu_params(network, params_torch)
            utils.network_zero_grad(network)
            dut = relu_to_optimization.ReLUFreePattern(network,
                                                       params_torch.dtype)
            mip_return, _, _, _, _, _, _ = dut.output_constraint(
                x_lo, x_up, mip_utils.PropagateBoundsMethod.IA)
            entry = getattr(mip_return, mip_entry_name)

            if entry is None:
                return None
            else:
                entry_flat = entry.reshape((-1))
                return entry_flat[entry_index]

        for entry_index in range(mip_entry_size):
            # First compute the gradient of that entry w.r.t network
            # weights/biases using autodiff.
            entry_i = get_entry(network_param, entry_index)
            if entry_i is None:
                continue
            if not entry_i.requires_grad:
                grad = torch.zeros_like(network_param)
            else:
                entry_i.backward()
                grad = utils.extract_relu_parameters_grad(network)

            grad_numerical = utils.compute_numerical_gradient(
                lambda param: get_entry(param, entry_index).item(),
                network_param.detach().numpy())
            np.testing.assert_allclose(grad.detach().numpy(),
                                       grad_numerical,
                                       atol=atol,
                                       rtol=rtol)

    def mip_return_gradient_tester(self, network, network_param, x_lo, x_up,
                                   atol, rtol):
        utils.update_relu_params(network, network_param)
        dut = relu_to_optimization.ReLUFreePattern(network,
                                                   network_param.dtype)
        mip_return, _, _, _, _, _, _ = dut.output_constraint(
            x_lo, x_up, mip_utils.PropagateBoundsMethod.IA)

        def test_entry(entry_name):
            entry = getattr(mip_return, entry_name)
            if entry is None:
                return
            self.entry_gradient_tester(network, network_param, x_lo, x_up,
                                       entry_name, entry.numel(), atol, rtol)

        test_entry("Aout_input")
        test_entry("Aout_slack")
        test_entry("Aout_binary")
        test_entry("Cout")
        test_entry("Ain_input")
        test_entry("Ain_slack")
        test_entry("Ain_binary")
        test_entry("rhs_in")
        test_entry("Aeq_input")
        test_entry("Aeq_slack")
        test_entry("Aeq_binary")
        test_entry("rhs_eq")

    def test_network1(self):
        """
        This is a network I found in the while when synthesizing controller for
        the pendulum.
        """
        network = utils.setup_relu((2, 2, 1),
                                   params=None,
                                   negative_slope=0.1,
                                   bias=True,
                                   dtype=torch.float64)
        network[0].weight.data = torch.tensor(
            [[1.6396, 2.7298], [-6.5492, -5.0761]], dtype=torch.float64)
        network[0].bias.data = torch.tensor([0.1370, -0.3323],
                                            dtype=torch.float64)
        network[2].weight.data = torch.tensor([-2.9836, 5.6627],
                                              dtype=torch.float64)
        network[2].bias.data = torch.tensor([-0.0927], dtype=torch.float64)

        network_param = utils.extract_relu_parameters(network)

        x_lo = torch.tensor([np.pi - 0.1, -0.2], dtype=torch.float64)
        x_up = torch.tensor([np.pi + 0.1, 0.2], dtype=torch.float64)

        self.mip_return_gradient_tester(network,
                                        network_param,
                                        x_lo,
                                        x_up,
                                        atol=1E-5,
                                        rtol=1E-5)


class TestComputeLinearOutputBoundByOptimization(unittest.TestCase):
    def setUp(self):
        self.dtype = torch.float64
        self.relu_with_bias = utils.setup_relu((2, 5, 4, 3),
                                               params=None,
                                               negative_slope=0.1,
                                               bias=True,
                                               dtype=self.dtype)
        self.relu_no_bias = utils.setup_relu((2, 5, 4, 3),
                                             params=None,
                                             negative_slope=0.1,
                                             bias=False,
                                             dtype=self.dtype)
        self.relu_with_bias[0].weight.data = torch.tensor(
            [[0.5, 0.4], [1.2, -0.3], [0.4, 2.1], [-1.5, -2.1], [-0.1, 1.3]],
            dtype=self.dtype)
        self.relu_with_bias[0].bias.data = torch.tensor(
            [-.5, 1.1, 0.3, -0.2, -2.1], dtype=self.dtype)
        self.relu_with_bias[2].weight.data = torch.tensor(
            [[0.1, 0.5, -0.2, 0.3, 0.2], [-0.3, -1.2, 2.1, -0.5, 3.2],
             [0.3, -0.4, 1.1, -0.2, 0.5], [2.1, 4.5, -0.4, -3.2, 0.4]],
            dtype=self.dtype)
        self.relu_with_bias[2].bias.data = torch.tensor([0.4, 0.3, 1.3, 2.5],
                                                        dtype=self.dtype)
        self.relu_with_bias[4].weight.data = torch.tensor(
            [[0.3, 0.2, -0.1, 0.4], [1.2, -0.5, 2.1, -0.2],
             [0.2, 0.5, 0.1, -1.5]],
            dtype=self.dtype)
        self.relu_with_bias[4].bias.data = torch.tensor([0.4, -2.1, -3.6],
                                                        dtype=self.dtype)
        for i in range(3):
            self.relu_no_bias[2 * i].weight.data = self.relu_with_bias[
                2 * i].weight.data.clone()

    def linear_output_tester(self, dut, layer_index, linear_output_row_index,
                             previous_neuron_input_lo,
                             previous_neuron_input_up, network_input_lo,
                             network_input_up, create_prog_callback,
                             input_checker, lp_relaxation):
        linear_output_lo, linear_output_up, lo_input_val, up_input_val = \
            dut._compute_linear_output_bound_by_optimization(
                layer_index, linear_output_row_index, previous_neuron_input_lo,
                previous_neuron_input_up, network_input_lo, network_input_up,
                create_prog_callback, lp_relaxation)
        truncated_network = torch.nn.Sequential(
            *[dut.model[i] for i in range(2 * layer_index + 1)])
        if not lp_relaxation:
            if lo_input_val is not None:
                # Now evaluat ethat linear layer output
                self.assertAlmostEqual(
                    linear_output_lo,
                    truncated_network(lo_input_val)
                    [linear_output_row_index].item())
            if up_input_val is not None:
                self.assertAlmostEqual(
                    linear_output_up,
                    truncated_network(up_input_val)
                    [linear_output_row_index].item())

        # Now take many sample inputs, check if the linear output is within
        # the bounds.
        network_input_guesses = utils.uniform_sample_in_box(
            torch.from_numpy(network_input_lo),
            torch.from_numpy(network_input_up), 1000)
        network_input_samples = network_input_guesses[
            input_checker(network_input_guesses), :]
        # Now compute the linear output for all these samples.
        linear_inputs = network_input_samples
        for layer in range(layer_index):
            linear_outputs = dut.model[2 * layer](linear_inputs)
            relu_outputs = dut.model[2 * layer + 1](linear_outputs)
            linear_inputs = relu_outputs
        linear_outputs = dut.model[2 * layer_index](
            linear_inputs)[:, linear_output_row_index]
        np.testing.assert_array_less(linear_outputs.detach().numpy(),
                                     linear_output_up + 1E-6)
        np.testing.assert_array_less(linear_output_lo - 1E-6,
                                     linear_outputs.detach().numpy())
        # Propagate the bounds through IA. The bounds from MILP/LP should be
        # tighter than IA.
        bij = dut.model[2 * layer_index].bias[linear_output_row_index].reshape(
            (1, )) if dut.model[2 * layer_index].bias is not None else\
            torch.tensor([0.], dtype=self.dtype)
        Wij = dut.model[2 *
                        layer_index].weight[linear_output_row_index].reshape(
                            (1, -1))
        if layer_index == 0:
            linear_output_lo_ia, linear_output_up_ia =\
                mip_utils.compute_range_by_IA(
                    Wij, bij, torch.from_numpy(network_input_lo),
                    torch.from_numpy(network_input_up))
            # At the input layer, the MILP/LP bounds should be the same as the
            # IA bounds, if there is no additional constraints on the network
            # input.
            if create_prog_callback is None:
                self.assertAlmostEqual(linear_output_lo_ia[0].item(),
                                       linear_output_lo)
                self.assertAlmostEqual(linear_output_up_ia[0].item(),
                                       linear_output_up)
            else:
                self.assertLessEqual(linear_output_lo_ia[0].item(),
                                     linear_output_lo)
                self.assertGreaterEqual(linear_output_up_ia[0].item(),
                                        linear_output_up)
        else:
            linear_input_lo = dut.model[2 * layer_index - 1](torch.from_numpy(
                previous_neuron_input_lo[dut.relu_unit_index[layer_index -
                                                             1]]))
            linear_input_up = dut.model[2 * layer_index - 1](torch.from_numpy(
                previous_neuron_input_up[dut.relu_unit_index[layer_index -
                                                             1]]))
            linear_output_lo_ia, linear_output_up_ia =\
                mip_utils.compute_range_by_IA(
                    Wij, bij, linear_input_lo, linear_input_up)
            self.assertLessEqual(linear_output_lo_ia[0].item(),
                                 linear_output_lo)
            self.assertGreaterEqual(linear_output_up_ia[0].item(),
                                    linear_output_up)

        return linear_output_lo, linear_output_up

    def given_relu_test(self, relu, network_input_lo, network_input_up,
                        create_prog_callback, input_checker, lp_relaxation):
        dut = relu_to_optimization.ReLUFreePattern(relu, self.dtype)
        previous_neuron_input_lo = np.zeros((dut.num_relu_units, ))
        previous_neuron_input_up = np.zeros((dut.num_relu_units, ))

        # First test the input layer.
        for i in range(self.relu_with_bias[0].out_features):
            previous_neuron_input_lo[
                dut.relu_unit_index[0][i]], previous_neuron_input_up[
                    dut.relu_unit_index[0][i]] = self.linear_output_tester(
                        dut, 0, i, previous_neuron_input_lo,
                        previous_neuron_input_up, network_input_lo,
                        network_input_up, create_prog_callback, input_checker,
                        lp_relaxation)
        # Now test the second layer.
        for i in range(self.relu_with_bias[2].out_features):
            previous_neuron_input_lo[
                dut.relu_unit_index[1][i]], previous_neuron_input_up[
                    dut.relu_unit_index[1][i]] = self.linear_output_tester(
                        dut, 1, i, previous_neuron_input_lo,
                        previous_neuron_input_up, network_input_lo,
                        network_input_up, create_prog_callback, input_checker,
                        lp_relaxation)

    def test_relu_with_bias(self):
        def checker(x):
            return torch.ones(x.shape[0], dtype=torch.bool)

        network_input_lo = np.array([-2., -3.])
        network_input_up = np.array([-1., 2.])
        for lp_relaxation in (True, False):
            self.given_relu_test(self.relu_with_bias,
                                 network_input_lo,
                                 network_input_up,
                                 create_prog_callback=None,
                                 input_checker=checker,
                                 lp_relaxation=lp_relaxation)
        network_input_lo = np.array([3., 5.])
        network_input_up = np.array([3.1, 10.])
        for lp_relaxation in (True, False):
            self.given_relu_test(self.relu_with_bias,
                                 network_input_lo,
                                 network_input_up,
                                 create_prog_callback=None,
                                 input_checker=checker,
                                 lp_relaxation=lp_relaxation)

    def test_relu_with_bias_create_prog_callback(self):
        # Create a simple program with some constraints.
        def create_prog_callback():
            prog = gurobi_torch_mip.GurobiTorchMILP(self.dtype)
            input_var = prog.addVars(2,
                                     lb=-gurobipy.GRB.INFINITY,
                                     ub=gurobipy.GRB.INFINITY)
            prog.addLConstr([torch.tensor([1., 1.], dtype=self.dtype)],
                            [input_var],
                            rhs=2.,
                            sense=gurobipy.GRB.LESS_EQUAL)
            return prog, input_var

        def checker(x):
            return torch.sum(x, dim=1) <= 2.

        network_input_lo = np.array([0.1, -0.2])
        network_input_up = np.array([2., 1.])
        for lp_relaxation in (True, False):
            self.given_relu_test(self.relu_with_bias, network_input_lo,
                                 network_input_up, create_prog_callback,
                                 checker, lp_relaxation)

    def test_relu_no_bias(self):
        def checker(x):
            return torch.ones(x.shape[0], dtype=torch.bool)

        network_input_lo = np.array([-2., -3.])
        network_input_up = np.array([-1., 2.])
        for lp_relaxation in (True, False):
            self.given_relu_test(self.relu_no_bias,
                                 network_input_lo,
                                 network_input_up,
                                 create_prog_callback=None,
                                 input_checker=checker,
                                 lp_relaxation=lp_relaxation)
        network_input_lo = np.array([3., 5.])
        network_input_up = np.array([3.1, 10.])
        for lp_relaxation in (True, False):
            self.given_relu_test(self.relu_no_bias,
                                 network_input_lo,
                                 network_input_up,
                                 create_prog_callback=None,
                                 input_checker=checker,
                                 lp_relaxation=lp_relaxation)


class TestComputeLayerBound(TestComputeLinearOutputBoundByOptimization):
    def bound_tester(self, relu_network, x_lo, x_up, method):
        dut = relu_to_optimization.ReLUFreePattern(relu_network, self.dtype)
        z_pre_relu_lo, z_pre_relu_up, z_post_relu_lo, z_post_relu_up =\
            dut._compute_layer_bound(x_lo, x_up, method)
        # Now take many samples inside x_lo <= x <= x_up, compute the neuron
        # values z and compare with the bound.
        x_samples = utils.uniform_sample_in_box(x_lo, x_up, 100)
        linear_inputs = x_samples
        for layer in range(len(dut.relu_unit_index)):
            linear_outputs = dut.model[2 * layer](linear_inputs)
            z_indices = dut.relu_unit_index[layer]
            np.testing.assert_array_less(
                linear_outputs.detach().numpy(),
                np.repeat(z_pre_relu_up[z_indices].detach().numpy().reshape(
                    (1, -1)),
                          linear_outputs.shape[0],
                          axis=0) + 1E-6)
            np.testing.assert_array_less(
                np.repeat(z_pre_relu_lo[z_indices].detach().numpy().reshape(
                    (1, -1)),
                          linear_outputs.shape[0],
                          axis=0) - 1E-6,
                linear_outputs.detach().numpy())
            relu_outputs = dut.model[2 * layer + 1](linear_outputs)
            np.testing.assert_array_less(
                relu_outputs.detach().numpy(),
                np.repeat(z_post_relu_up[z_indices].detach().numpy().reshape(
                    (1, -1)),
                          relu_outputs.shape[0],
                          axis=0) + 1E-6)
            np.testing.assert_array_less(
                np.repeat(z_post_relu_lo[z_indices].detach().numpy().reshape(
                    (1, -1)),
                          relu_outputs.shape[0],
                          axis=0) - 1E-6,
                relu_outputs.detach().numpy())
            linear_inputs = relu_outputs

    def network_tester(self, relu_network):
        x_lo = torch.tensor([-2., -3.], dtype=self.dtype)
        x_up = torch.tensor([-1., 2.], dtype=self.dtype)
        for method in list(mip_utils.PropagateBoundsMethod):
            self.bound_tester(relu_network, x_lo, x_up, method)
        dut = relu_to_optimization.ReLUFreePattern(relu_network, self.dtype)
        z_pre_relu_lo_ia, z_pre_relu_up_ia, z_post_relu_lo_ia,\
            z_post_relu_up_ia = dut._compute_layer_bound(
                x_lo, x_up, mip_utils.PropagateBoundsMethod.IA)
        z_pre_relu_lo_lp, z_pre_relu_up_lp, z_post_relu_lo_lp,\
            z_post_relu_up_lp = dut._compute_layer_bound(
                x_lo, x_up, mip_utils.PropagateBoundsMethod.LP)
        z_pre_relu_lo_mip, z_pre_relu_up_mip, z_post_relu_lo_mip,\
            z_post_relu_up_mip = dut._compute_layer_bound(
                x_lo, x_up, mip_utils.PropagateBoundsMethod.MIP)
        np.testing.assert_array_less(z_pre_relu_lo_ia.detach().numpy(),
                                     z_pre_relu_lo_lp.detach().numpy() + 1E-6)
        np.testing.assert_array_less(z_pre_relu_up_lp.detach().numpy(),
                                     z_pre_relu_up_ia.detach().numpy() + 1E-6)
        np.testing.assert_array_less(z_post_relu_lo_ia.detach().numpy(),
                                     z_post_relu_lo_lp.detach().numpy() + 1E-6)
        np.testing.assert_array_less(z_post_relu_up_lp.detach().numpy(),
                                     z_post_relu_up_ia.detach().numpy() + 1E-6)
        np.testing.assert_array_less(z_pre_relu_lo_lp.detach().numpy(),
                                     z_pre_relu_lo_mip.detach().numpy() + 1E-6)
        np.testing.assert_array_less(z_pre_relu_up_mip.detach().numpy(),
                                     z_pre_relu_up_lp.detach().numpy() + 1E-6)
        np.testing.assert_array_less(
            z_post_relu_lo_lp.detach().numpy(),
            z_post_relu_lo_mip.detach().numpy() + 1E-6)
        np.testing.assert_array_less(z_post_relu_up_mip.detach().numpy(),
                                     z_post_relu_up_lp.detach().numpy() + 1E-6)

    def test(self):
        self.network_tester(self.relu_with_bias)
        self.network_tester(self.relu_no_bias)

    def compute_network_output_bounds_tester(self, network, x_lo, x_up,
                                             method):
        dut = relu_to_optimization.ReLUFreePattern(network, self.dtype)
        # I do not want to test when the last layer is relu. We don't use that
        # in practice.
        z_pre_relu_lo, z_pre_relu_up, z_post_relu_lo, z_post_relu_up =\
            dut._compute_layer_bound(x_lo, x_up, method)
        output_lo, output_up = dut._compute_network_output_bounds(
            z_pre_relu_lo, z_pre_relu_up, x_lo, x_up, method)
        if method == mip_utils.PropagateBoundsMethod.IA:
            output_lo_expected, output_up_expected =\
                mip_utils.propagate_bounds(
                    dut.model[-1], z_post_relu_lo[dut.relu_unit_index[-1]],
                    z_post_relu_up[dut.relu_unit_index[-1]])
        else:
            output_lo_expected = torch.empty((dut.model[-1].out_features, ),
                                             dtype=self.dtype)
            output_up_expected = torch.empty((dut.model[-1].out_features, ),
                                             dtype=self.dtype)
            if method == mip_utils.PropagateBoundsMethod.LP:
                lp_relaxation = True
            elif method == mip_utils.PropagateBoundsMethod.MIP:
                lp_relaxation = False
            for j in range(dut.model[-1].out_features):
                output_lo_expected[j], output_up_expected[
                    j], _, _ = dut.\
                        _compute_linear_output_bound_by_optimization(
                            int((len(dut.model) - 1) / 2),
                            j,
                            z_pre_relu_lo.detach().numpy(),
                            z_pre_relu_up.detach().numpy(),
                            x_lo.detach().numpy(),
                            x_up.detach().numpy(),
                            create_prog_callback=None,
                            lp_relaxation=lp_relaxation)
        np.testing.assert_allclose(output_lo.detach().numpy(),
                                   output_lo_expected.detach().numpy())
        np.testing.assert_allclose(output_up.detach().numpy(),
                                   output_up_expected.detach().numpy())
        # Take many sampled inputs, make sure the outputs are in the bounds.
        input_samples = utils.uniform_sample_in_box(x_lo, x_up, 1000)
        output_samples = dut.model(input_samples)
        np.testing.assert_array_less(
            output_samples.detach().numpy(),
            np.repeat(output_up.detach().numpy().reshape((1, -1)),
                      output_samples.shape[0],
                      axis=0) + 1E-6)
        np.testing.assert_array_less(
            np.repeat(output_lo.detach().numpy().reshape((1, -1)),
                      output_samples.shape[0],
                      axis=0) - 1E-6,
            output_samples.detach().numpy())

    def test_compute_network_output_bounds(self):
        x_lo = torch.tensor([-2., -1.], dtype=self.dtype)
        x_up = torch.tensor([-1., 2.], dtype=self.dtype)
        for method in list(mip_utils.PropagateBoundsMethod):
            self.compute_network_output_bounds_tester(self.relu_with_bias,
                                                      x_lo, x_up, method)
            self.compute_network_output_bounds_tester(self.relu_no_bias, x_lo,
                                                      x_up, method)


if __name__ == "__main__":
    unittest.main()
