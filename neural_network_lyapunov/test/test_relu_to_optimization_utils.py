import neural_network_lyapunov.relu_to_optimization_utils as\
    relu_to_optimization_utils
import neural_network_lyapunov.mip_utils as mip_utils
import neural_network_lyapunov.utils as utils
import neural_network_lyapunov.gurobi_torch_mip as gurobi_torch_mip

import torch
import torch.nn as nn
import numpy as np
import unittest
import gurobipy


class TestAddConstraintByNeuron(unittest.TestCase):
    def constraint_test(self, Wij, bij, relu_layer, neuron_input_lo,
                        neuron_input_up):
        Ain_linear_input, Ain_neuron_output, Ain_binary, rhs_in,\
            Aeq_linear_input, Aeq_neuron_output, Aeq_binary, rhs_eq,\
            neuron_output_lo, neuron_output_up = \
            relu_to_optimization_utils._add_constraint_by_neuron(
                Wij, bij, relu_layer, neuron_input_lo, neuron_input_up)

        neuron_output_lo_expected, neuron_output_up_expected =\
            mip_utils.propagate_bounds(
                relu_layer, neuron_input_lo, neuron_input_up)
        np.testing.assert_allclose(neuron_output_lo.detach().numpy(),
                                   neuron_output_lo_expected.detach().numpy())
        np.testing.assert_allclose(neuron_output_up.detach().numpy(),
                                   neuron_output_up_expected.detach().numpy())
        # Now solve an optimization problem with the constraints, verify that
        # the solution is the output of the neuron.
        linear_output_val_samples = utils.uniform_sample_in_box(
            neuron_input_lo, neuron_input_up, 30)
        for i in range(linear_output_val_samples.shape[0]):
            model = gurobi_torch_mip.GurobiTorchMIP(torch.float64)
            linear_input = model.addVars(Wij.numel(),
                                         lb=-gurobipy.GRB.INFINITY)
            neuron_output = model.addVars(1, lb=-gurobipy.GRB.INFINITY)
            binary = model.addVars(1, vtype=gurobipy.GRB.BINARY)
            model.addMConstrs(
                [Ain_linear_input, Ain_neuron_output, Ain_binary],
                [linear_input, neuron_output, binary],
                b=rhs_in,
                sense=gurobipy.GRB.LESS_EQUAL)
            model.addMConstrs(
                [Aeq_linear_input, Aeq_neuron_output, Aeq_binary],
                [linear_input, neuron_output, binary],
                b=rhs_eq,
                sense=gurobipy.GRB.EQUAL)
            model.addMConstrs([Wij.reshape((1, -1))], [linear_input],
                              b=linear_output_val_samples[i] - bij,
                              sense=gurobipy.GRB.EQUAL)
            model.gurobi_model.setParam(gurobipy.GRB.Param.OutputFlag, False)
            model.gurobi_model.optimize()

            self.assertEqual(model.gurobi_model.status,
                             gurobipy.GRB.Status.OPTIMAL)
            linear_input_sol = torch.tensor([v.x for v in linear_input],
                                            dtype=Wij.dtype)
            self.assertAlmostEqual((Wij @ linear_input_sol + bij).item(),
                                   linear_output_val_samples[i].item())
            neuron_output_expected = relu_layer(linear_output_val_samples[i])
            self.assertAlmostEqual(neuron_output[0].x,
                                   neuron_output_expected.item())
            self.assertAlmostEqual(
                binary[0].x, int(linear_output_val_samples[i].item() >= 0))
            self.assertLess(neuron_output_expected.item(),
                            neuron_output_up[0].item() + 1E-6)
            self.assertLess(neuron_output_lo[0].item() - 1E-6,
                            neuron_output_expected.item())

    def test(self):
        dtype = torch.float64
        Wij = torch.tensor([1., 2., -2.], dtype=dtype)
        bij = torch.tensor([2.], dtype=dtype)
        no_bias_bij = torch.tensor([0.], dtype=dtype)
        relu_layer = torch.nn.ReLU()
        leaky_relu_layer = torch.nn.LeakyReLU(0.1)
        linear_input_lo = torch.tensor([-0.5, -2., 1.2], dtype=dtype)
        linear_input_up = torch.tensor([-0.2, 1.5, 2.3], dtype=dtype)
        neuron_input_lo, neuron_input_up = mip_utils.compute_range_by_IA(
            Wij.reshape((1, -1)), bij.reshape((1, )), linear_input_lo,
            linear_input_up)
        self.constraint_test(Wij, bij, relu_layer, neuron_input_lo,
                             neuron_input_up)
        self.constraint_test(Wij, no_bias_bij, relu_layer, neuron_input_lo,
                             neuron_input_up)
        self.constraint_test(Wij, bij, leaky_relu_layer, neuron_input_lo,
                             neuron_input_up)
        self.constraint_test(Wij, no_bias_bij, leaky_relu_layer,
                             neuron_input_lo, neuron_input_up)


class TestAddConstraintByLayer(unittest.TestCase):
    def setUp(self):
        self.dtype = torch.float64
        self.linear_with_bias = nn.Linear(2, 4, bias=True)
        self.linear_with_bias.weight.data = torch.tensor(
            [[1., 2.], [-2., 1.], [-1., 4], [4., -5.]], dtype=self.dtype)
        self.linear_with_bias.bias.data = torch.tensor([0.5, -1., 1.5, -3],
                                                       dtype=self.dtype)
        self.linear_no_bias = nn.Linear(2, 4, bias=False)
        self.linear_no_bias.weight = self.linear_with_bias.weight
        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.1)

    def constraint_test(self, linear_layer, relu_layer, z_curr_lo, z_curr_up):
        linear_output_lo, linear_output_up = mip_utils.propagate_bounds(
            linear_layer, z_curr_lo, z_curr_up)
        Ain_z_curr, Ain_z_next, Ain_binary, rhs_in, Aeq_z_curr, Aeq_z_next,\
            Aeq_binary, rhs_eq, z_next_lo, z_next_up = \
            relu_to_optimization_utils._add_constraint_by_layer(
                linear_layer, relu_layer, linear_output_lo, linear_output_up)
        z_next_lo_expected, z_next_up_expected = mip_utils.propagate_bounds(
            relu_layer, linear_output_lo, linear_output_up)
        np.testing.assert_allclose(z_next_lo.detach().numpy(),
                                   z_next_lo_expected.detach().numpy())
        np.testing.assert_allclose(z_next_up.detach().numpy(),
                                   z_next_up_expected.detach().numpy())
        # Maker sure we tested all cases, that the relu can be
        # 1. Both active or inactive.
        # 2. Only inactive.
        # 3. Only active.
        # Now form an optimization problem satisfying these added constraints,
        # and check the solution matches with evaluating the ReLU.
        z_curr_val = utils.uniform_sample_in_box(z_curr_lo, z_curr_up, 40)
        for i in range(z_curr_val.shape[0]):
            model = gurobi_torch_mip.GurobiTorchMILP(self.dtype)
            z_curr = model.addVars(linear_layer.in_features,
                                   lb=-gurobipy.GRB.INFINITY)
            z_next = model.addVars(linear_layer.out_features,
                                   lb=-gurobipy.GRB.INFINITY)
            beta = model.addVars(linear_layer.out_features,
                                 vtype=gurobipy.GRB.BINARY)
            model.addMConstrs(
                [torch.eye(linear_layer.in_features, dtype=self.dtype)],
                [z_curr],
                b=z_curr_val[i],
                sense=gurobipy.GRB.EQUAL)
            model.addMConstrs([Ain_z_curr, Ain_z_next, Ain_binary],
                              [z_curr, z_next, beta],
                              b=rhs_in,
                              sense=gurobipy.GRB.LESS_EQUAL)
            model.addMConstrs([Aeq_z_curr, Aeq_z_next, Aeq_binary],
                              [z_curr, z_next, beta],
                              b=rhs_eq,
                              sense=gurobipy.GRB.EQUAL)
            model.gurobi_model.setParam(gurobipy.GRB.Param.OutputFlag, False)
            model.gurobi_model.optimize()
            with torch.no_grad():
                self.assertEqual(model.gurobi_model.status,
                                 gurobipy.GRB.Status.OPTIMAL)
                linear_output = linear_layer(z_curr_val[i])
                z_next_val_expected = relu_layer(linear_output)
                beta_val_expected = np.array([
                    1 if linear_output[j] > 0 else 0
                    for j in range(linear_layer.out_features)
                ])
                z_next_val = np.array(
                    [z_next[j].x for j in range(len(z_next))])
                beta_val = np.array([beta[j].x for j in range(len(beta))])
                np.testing.assert_allclose(
                    z_next_val,
                    z_next_val_expected.detach().numpy())
                np.testing.assert_allclose(beta_val, beta_val_expected)

        # Sample many z_curr within the limits, make sure they all satisfy the
        # constraints.
        with torch.no_grad():
            z_curr_val = utils.uniform_sample_in_box(z_curr_lo, z_curr_up, 20)
            for i in range(z_curr_val.shape[0]):
                linear_output = linear_layer(z_curr_val[i])
                beta_val = torch.tensor([
                    1 if linear_output[j] > 0 else 0
                    for j in range(linear_output.shape[0])
                ],
                                        dtype=self.dtype)
                z_next_val = relu_layer(linear_output)
                np.testing.assert_array_less(
                    (Ain_z_curr @ z_curr_val[i] + Ain_z_next @ z_next_val +
                     Ain_binary @ beta_val).detach().numpy().squeeze(),
                    rhs_in.detach().numpy().squeeze() + 1E-6)

    def constraint_gradient_test(self, linear_layer, relu_layer, z_curr_lo,
                                 z_curr_up):
        linear_output_lo, linear_output_up = mip_utils.propagate_bounds(
            linear_layer, z_curr_lo, z_curr_up)
        Ain_z_curr, Ain_z_next, Ain_binary, rhs_in, Aeq_z_curr, Aeq_z_next,\
            Aeq_binary, rhs_eq, z_next_lo, z_next_up =\
            relu_to_optimization_utils._add_constraint_by_layer(
                linear_layer, relu_layer, linear_output_lo, linear_output_up)

        (Ain_z_curr.sum() + Ain_z_next.sum() + Ain_binary.sum() +
         rhs_in.sum() + Aeq_z_curr.sum() + Aeq_z_next.sum() +
         Aeq_binary.sum() + rhs_eq.sum() + linear_output_lo.sum() +
         linear_output_up.sum() + z_next_lo.sum() +
         z_next_up.sum()).backward()

        linear_layer_weight_grad = linear_layer.weight.grad.clone()
        if linear_layer.bias is not None:
            linear_layer_bias_grad = linear_layer.bias.grad.clone()
        input_lo_grad = z_curr_lo.grad.clone()
        input_up_grad = z_curr_up.grad.clone()

        def eval_fun(linear_layer_weight, linear_layer_bias, input_lo_np,
                     input_up_np):
            linear_layer.weight.data = torch.from_numpy(
                linear_layer_weight.reshape(linear_layer.weight.data.shape))
            if linear_layer.bias is not None:
                linear_layer.bias.data = torch.from_numpy(linear_layer_bias)
            with torch.no_grad():
                linear_output_lo, linear_output_up =\
                    mip_utils.propagate_bounds(
                        linear_layer, torch.from_numpy(input_lo_np),
                        torch.from_numpy(input_up_np))
                Ain_z_curr, Ain_z_next, Ain_binary, rhs_in, Aeq_z_curr,\
                    Aeq_z_next, Aeq_binary, rhs_eq, z_next_lo, z_next_up =\
                    relu_to_optimization_utils._add_constraint_by_layer(
                        linear_layer, relu_layer, linear_output_lo,
                        linear_output_up)

                return np.array([
                    (Ain_z_curr.sum() + Ain_z_next.sum() + Ain_binary.sum() +
                     rhs_in.sum() + Aeq_z_curr.sum() + Aeq_z_next.sum() +
                     Aeq_binary.sum() + rhs_eq.sum() + linear_output_lo.sum() +
                     linear_output_up.sum() + z_next_lo.sum() +
                     z_next_up.sum()).detach()
                ])

        bias_val = linear_layer.bias.data.detach().numpy() if\
            linear_layer.bias is not None else np.zeros((2,))

        numerical_grads = utils.compute_numerical_gradient(
            eval_fun,
            linear_layer.weight.data.detach().numpy().reshape((-1, )),
            bias_val,
            z_curr_lo.detach().numpy(),
            z_curr_up.detach().numpy())
        np.testing.assert_allclose(numerical_grads[0],
                                   linear_layer_weight_grad.reshape((1, -1)))
        if linear_layer.bias is not None:
            np.testing.assert_allclose(numerical_grads[1].reshape((-1, )),
                                       linear_layer_bias_grad)
        np.testing.assert_allclose(numerical_grads[2].squeeze(), input_lo_grad)
        np.testing.assert_allclose(numerical_grads[3].squeeze(), input_up_grad)

    def test_with_bias_relu(self):
        z_curr_lo = torch.tensor([-1., 2.],
                                 dtype=self.dtype,
                                 requires_grad=True)
        z_curr_up = torch.tensor([2., 4.],
                                 dtype=self.dtype,
                                 requires_grad=True)
        self.constraint_test(self.linear_with_bias, self.relu, z_curr_lo,
                             z_curr_up)
        self.constraint_gradient_test(self.linear_with_bias, self.relu,
                                      z_curr_lo, z_curr_up)

    def test_with_bias_leaky_relu1(self):
        # The input bounds allow the output to be
        # 1. Either active or inactive.
        # 2. Only active.
        # 3. Only inactive.
        z_curr_lo = torch.tensor([-1., 2.],
                                 dtype=self.dtype,
                                 requires_grad=True)
        z_curr_up = torch.tensor([2., 4.],
                                 dtype=self.dtype,
                                 requires_grad=True)
        self.constraint_test(self.linear_with_bias, self.leaky_relu, z_curr_lo,
                             z_curr_up)
        self.constraint_gradient_test(self.linear_with_bias, self.leaky_relu,
                                      z_curr_lo, z_curr_up)

    def test_with_bias_leaky_relu2(self):
        # Only a single output
        z_curr_lo = torch.tensor([-1., 2.],
                                 dtype=self.dtype,
                                 requires_grad=True)
        z_curr_up = torch.tensor([3., 5], dtype=self.dtype, requires_grad=True)
        linear_layer = nn.Linear(2, 1, bias=True)
        linear_layer.weight.data = torch.tensor([[1, 3]], dtype=self.dtype)
        # Three different cases
        # 1. The ReLU is always active.
        # 2. The ReLU is always inactive.
        # 3. The ReLU could be either active or inactive.
        for bias_val in [5, -20, -4]:
            linear_layer.bias.data = torch.tensor([bias_val], dtype=self.dtype)
            self.constraint_test(linear_layer, self.leaky_relu, z_curr_lo,
                                 z_curr_up)
            self.constraint_gradient_test(linear_layer, self.leaky_relu,
                                          z_curr_lo, z_curr_up)
            linear_layer.weight.grad.zero_()
            linear_layer.bias.grad.zero_()
            z_curr_lo.grad.zero_()
            z_curr_up.grad.zero_()

    def test_no_bias_relu(self):
        z_curr_lo = torch.tensor([-1., 2.],
                                 dtype=self.dtype,
                                 requires_grad=True)
        z_curr_up = torch.tensor([2., 4.],
                                 dtype=self.dtype,
                                 requires_grad=True)
        self.constraint_test(self.linear_no_bias, self.relu, z_curr_lo,
                             z_curr_up)
        self.constraint_gradient_test(self.linear_no_bias, self.relu,
                                      z_curr_lo, z_curr_up)

    def test_no_bias_leaky_relu(self):
        z_curr_lo = torch.tensor([-1., 2.],
                                 dtype=self.dtype,
                                 requires_grad=True)
        z_curr_up = torch.tensor([2., 4.],
                                 dtype=self.dtype,
                                 requires_grad=True)
        self.constraint_test(self.linear_no_bias, self.leaky_relu, z_curr_lo,
                             z_curr_up)
        self.constraint_gradient_test(self.linear_no_bias, self.leaky_relu,
                                      z_curr_lo, z_curr_up)


class TestAddLinearRelaxationByLayer(unittest.TestCase):
    def setUp(self):
        self.dtype = torch.float64

    def add_linear_relaxation_by_layer_tester(self, linear_layer, relu_layer,
                                              linear_input_lo,
                                              linear_input_up):
        relu_input_lo, relu_input_up = mip_utils.propagate_bounds(
            linear_layer, linear_input_lo, linear_input_up)
        prog = gurobi_torch_mip.GurobiTorchMILP(self.dtype)
        linear_input_var = prog.addVars(linear_layer.in_features,
                                        lb=-gurobipy.GRB.INFINITY)
        relu_output_var, binary_relax = \
            relu_to_optimization_utils._add_linear_relaxation_by_layer(
                prog, linear_layer, relu_layer, linear_input_var,
                relu_input_lo, relu_input_up)
        self.assertEqual(len(relu_output_var), linear_layer.out_features)
        self.assertEqual(len(binary_relax), linear_layer.out_features)
        for i in range(linear_layer.out_features):
            self.assertEqual(relu_output_var[i].vtype, gurobipy.GRB.CONTINUOUS)
            self.assertEqual(binary_relax[i].vtype, gurobipy.GRB.CONTINUOUS)
            self.assertEqual(binary_relax[i].lb, 0.)
            self.assertEqual(binary_relax[i].ub, 1.)
        # Now take a sampled value of the input, compute the output. Make sure
        # the output falls within the convex hull of the ReLU function.
        input_samples = utils.uniform_sample_in_box(linear_input_lo,
                                                    linear_input_up, 100)
        for i in range(input_samples.shape[0]):
            for j in range(linear_layer.in_features):
                linear_input_var[j].lb = input_samples[i][j].item()
                linear_input_var[j].ub = input_samples[i][j].item()
                prog.gurobi_model.update()
            prog.gurobi_model.setParam(gurobipy.GRB.Param.OutputFlag, False)
            prog.gurobi_model.optimize()
            relu_output_sol = np.array([v.x for v in relu_output_var])
            relu_input = linear_layer(input_samples[i])
            relu_actual_output = relu_layer(relu_input)
            for j in range(linear_layer.out_features):
                # If the relu input lower and upper are both >= 0 or both <= 0,
                # then the solution to relu_output_var is the same as the
                # actual output of the ReLU layer.
                if relu_input_up[j] <= 0 or relu_input_lo[j] >= 0:
                    self.assertAlmostEqual(relu_output_sol[j],
                                           relu_actual_output[j].item())
                else:
                    # The set of LP solution is the intersection of the
                    # line output=relu_actual_output, and the triangle with
                    # three vertices (0, 0), (relu_input_up, relu_input_up),
                    # (relu_input_lo, relu(relu_input_low))
                    self.assertGreaterEqual(
                        relu_output_sol[j],
                        relu_actual_output[j].item() - 1E-6)
                    self.assertLessEqual(
                        relu_output_sol[j],
                        ((relu_input_up[j] - relu_layer(relu_input_lo[j])) /
                         (relu_input_up[j] - relu_input_lo[j]) *
                         (relu_input[j] - relu_input_lo[j]) +
                         relu_layer(relu_input_lo[j])).item() + 1E-6)

    def test(self):
        linear_layer = torch.nn.Linear(2, 5)
        linear_layer.weight.data = torch.tensor(
            [[1, 0], [0, 1], [-1, -2], [2, 1], [1, -2]], dtype=self.dtype)
        linear_layer.bias.data = torch.tensor([1, -2, -3, 4, 5],
                                              dtype=self.dtype)
        self.add_linear_relaxation_by_layer_tester(
            linear_layer,
            torch.nn.LeakyReLU(negative_slope=0.1),
            linear_input_lo=torch.tensor([-2, -1], dtype=self.dtype),
            linear_input_up=torch.tensor([2, 1], dtype=self.dtype))
        self.add_linear_relaxation_by_layer_tester(
            linear_layer,
            torch.nn.LeakyReLU(negative_slope=0.1),
            linear_input_lo=torch.tensor([2, 1], dtype=self.dtype),
            linear_input_up=torch.tensor([3, 4], dtype=self.dtype))
        self.add_linear_relaxation_by_layer_tester(
            linear_layer,
            torch.nn.LeakyReLU(negative_slope=0.1),
            linear_input_lo=torch.tensor([-2, -3], dtype=self.dtype),
            linear_input_up=torch.tensor([-1, -2], dtype=self.dtype))


if __name__ == "__main__":
    unittest.main()
