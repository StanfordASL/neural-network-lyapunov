# -*- coding: utf-8 -*-
import queue

import torch
import torch.nn as nn
import numpy as np
import gurobipy
from typing import Tuple

import neural_network_lyapunov.utils as utils
import neural_network_lyapunov.mip_utils as mip_utils
import neural_network_lyapunov.gurobi_torch_mip as gurobi_torch_mip
import neural_network_lyapunov.relu_to_optimization_utils as\
    relu_to_optimization_utils


def ComputeReLUActivationPattern(model_relu, x):
    """
    For a given input x to a ReLU (including leaky ReLU) network, returns the
    activation pattern for this input.
    @param model_relu A ReLU network (constructed by Sequential Linear and
    ReLU units).
    @param x A numpy array, the input to the network.
    @return activation_pattern A list of length num_relu_layers.
    activation_pattern[i] is a list of 0/1, with size equal to the number of
    ReLU units in the i'th layer. If activation_pattern[i][j] = 1, then in the
    i'th layer, the j'th unit is activated.
    """
    activation_pattern = []
    layer_x = x
    for layer in model_relu:
        if (isinstance(layer, nn.Linear)):
            layer_x = layer.forward(layer_x)
        elif isinstance(layer, nn.ReLU) or isinstance(layer, nn.LeakyReLU):
            activation_pattern.append([relu_x >= 0 for relu_x in layer_x])
            layer_x = layer.forward(layer_x)
    return activation_pattern


def compute_all_relu_activation_patterns(relu, x):
    """
    Similar to ComputeReLUActionPattern(), but return a list of activation
    patterns (when the input to a ReLU unit is 0, that ReLU unit could be
    regarded as both active or inactive, which could give different gradient).
    @param relu A (leaky) ReLU network.
    @param x A pytorch tensor. The input to the relu network.
    """
    patterns = queue.Queue()
    patterns.put([])
    layer_x = x
    for layer in relu:
        if (isinstance(layer, nn.Linear)):
            layer_x = layer.forward(layer_x)
        elif isinstance(layer, nn.ReLU) or isinstance(layer, nn.LeakyReLU):
            # layer_patterns include all possible activation pattern in this
            # layer.
            layer_patterns = queue.Queue()
            layer_patterns.put([])
            for i in range(layer_x.numel()):
                layer_patterns_len = layer_patterns.qsize()
                for _ in range(layer_patterns_len):
                    front = layer_patterns.get()
                    if layer_x[i] > 0:
                        front_clone = front.copy()
                        front_clone.append(True)
                        layer_patterns.put(front_clone)
                    elif layer_x[i] < 0:
                        front_clone = front.copy()
                        front_clone.append(False)
                        layer_patterns.put(front_clone)
                    else:
                        front_clone = front.copy()
                        front_clone.append(True)
                        layer_patterns.put(front_clone)
                        front_clone = front.copy()
                        front_clone.append(False)
                        layer_patterns.put(front_clone)

            layer_x = layer.forward(layer_x)

            patterns_len = patterns.qsize()
            layer_patterns_len = layer_patterns.qsize()
            for _ in range(patterns_len):
                patterns_front = patterns.get()
                for _ in range(layer_patterns_len):
                    patterns_front_clone = patterns_front.copy()
                    layer_patterns_front = layer_patterns.get()
                    patterns_front_clone.append(layer_patterns_front)
                    patterns.put(patterns_front_clone)
                    layer_patterns.put(layer_patterns_front)
    # convert patterns from queue to list.
    patterns_list = [None] * patterns.qsize()
    for i in range(len(patterns_list)):
        patterns_list[i] = patterns.get()
    return patterns_list


def relu_activation_binary_to_pattern(relu_network, activation_binary):
    """
    Given a numpy array of 0 and 1 representing the activation of each ReLU
    unit in the network, returns a list of lists as the activation pattern.
    The returned activation pattern can be used in ReLUGivenActivationPattern
    @param relu_network A fully connected network with ReLU units.
    @param activation_binary A numpy array. The length of this array is the
    same as the number of ReLU units in the network.
    @return activation pattern activation_pattern[i][j] is True if the j'th
    unit on the i'th layer is actived, False otherwise.
    """
    assert (isinstance(relu_network, nn.Sequential))
    assert (isinstance(activation_binary, np.ndarray))
    assert (np.all(
        np.logical_or(activation_binary == 1, activation_binary == 0)))
    assert (isinstance(relu_network[-1], nn.Linear))
    num_relu_layers = int((len(relu_network) - 1) / 2)
    total_relu_units = np.sum(
        np.array([
            relu_network[2 * i].out_features for i in range(num_relu_layers)
        ]))
    assert (activation_binary.shape == (total_relu_units, ))
    activation_pattern = [None] * num_relu_layers
    relu_unit_count = 0
    for i in range(num_relu_layers):
        num_relu_in_layer = relu_network[2 * i].out_features
        activation_pattern[i] = list(
            activation_binary[relu_unit_count:relu_unit_count +
                              num_relu_in_layer] == 1)
        relu_unit_count += num_relu_in_layer
    return activation_pattern


def set_activation_warmstart(relu, beta, x_warmstart):
    """
    initializes the binary variables beta with the activation pattern
    generated by x_warmstart. beta should have the same order as
    ComputeReLUActivationPattern (as would be returned by output_constraints)
    @param relu (leaky) relu neural network correspoding to the binary
    variables to warmstart
    @param beta list of binary variables (from gurobipy) corresponding to the
    activations of the lyapunov neural network (same order established by
    output_constraints)
    @param x_warmstart tensor of size self.system.x_dim. beta is then
    warmstarted using the activation pattern produced by using
    x_warmstart as the input of the lyapunov neural network
    """
    activation = ComputeReLUActivationPattern(relu, x_warmstart)
    unit_counter = 0
    for layer in activation:
        for unit in layer:
            if unit:
                beta[unit_counter].start = 1.
            else:
                beta[unit_counter].start = 0.
            unit_counter += 1


def ReLUGivenActivationPattern(model_relu, x_size, activation_pattern, dtype):
    """
    Given a ReLU network, and a given activation pattern, the ReLU network can
    be represented as
    ReLU(x) = gᵀx+h, while P*x ≤ q
    we return the quantity g, h, P and q.
    @param model_relu A ReLU network.
    @param x_size The size of the input x.
    @param activation_pattern A list of length num_relu_layers.
    activation_pattern[i] is a list of True/False, with size equal to the
    number of ReLU units in the i'th layer. If activation_pattern[i][j] = True,
    then in the i'th layer, the j'th unit is activated.
    @return (g, h, P, q)  g is a column vector, h is a scalar. P is a 2D
    matrix, and q is a column vector
    """
    A_layer = torch.eye(x_size, dtype=dtype)
    b_layer = torch.zeros((x_size, 1), dtype=dtype)
    relu_layer_count = 0
    num_linear_layer_output = None
    P = torch.empty((0, x_size), dtype=dtype)
    q = torch.empty((0, 1), dtype=dtype)
    for layer in model_relu:
        if (isinstance(layer, nn.Linear)):
            A_layer = layer.weight @ A_layer
            b_layer = layer.weight @ b_layer + \
                (layer.bias.reshape((-1, 1)) if layer.bias is not None else
                 torch.zeros((layer.weight.shape[0], 1), dtype=dtype))
            num_linear_layer_output = layer.weight.shape[0]
        elif (isinstance(layer, nn.ReLU) or isinstance(layer, nn.LeakyReLU)):
            assert (len(activation_pattern[relu_layer_count]) ==
                    num_linear_layer_output)
            for row, activation_flag in enumerate(
                    activation_pattern[relu_layer_count]):
                if activation_flag:
                    # If this row is active, then A.row(i) * x + b(i) >= 0
                    # is the same as -A.row(i) * x <= b(i)
                    P = torch.cat([P, -A_layer[row].reshape((1, -1))], dim=0)
                    q = torch.cat([q, b_layer[row].reshape((1, -1))], dim=0)
                else:
                    # If this row is inactive, then A.row(i) * x + b(i) <= 0
                    # is the same as A.row(i) * x <= -b(i)
                    P = torch.cat([P, A_layer[row].reshape((1, -1))], dim=0)
                    q = torch.cat([q, -b_layer[row].reshape((1, -1))], dim=0)
                    if (isinstance(layer, nn.ReLU)):
                        A_layer[row] = 0
                        b_layer[row] = 0
                    elif (isinstance(layer, nn.LeakyReLU)):
                        A_layer[row] *= layer.negative_slope
                        b_layer[row] *= layer.negative_slope
            relu_layer_count += 1
        else:
            raise Exception(
                "The ReLU network only allows ReLU and linear units.")
    g = A_layer.T
    h = b_layer
    return (g, h, P, q)


class ReLUMixedIntegerConstraintsReturn(
        gurobi_torch_mip.MixedIntegerConstraintsReturn):
    def __init__(self):
        super(ReLUMixedIntegerConstraintsReturn, self).__init__()
        self.nn_input_lo = None
        self.nn_input_up = None
        self.nn_output_lo = None
        self.nn_output_up = None
        self.relu_input_lo = None
        self.relu_input_up = None
        self.relu_output_lo = None
        self.relu_output_up = None


class ReLUGradientTimesVecMixedIntegerConstraintsReturn(
        gurobi_torch_mip.MixedIntegerConstraintsReturn):
    """
    The output of ReLUFreePattern.output_gradient_times_vector.
    """
    def __init__(self):
        super(ReLUGradientTimesVecMixedIntegerConstraintsReturn,
              self).__init__()
        # The lower bound of the slack variable z (Note z is NOT the ReLU unit
        # input/output). z_lo[0]/z_up[0] is the bound on the input vector.
        self.z_lo = None
        self.z_up = None
        # The lower and upper bound of Wi * zi, where Wi is the weight matrix
        # of the i'th linear layer. Wz_lo[-1]/Wz_up[-1] is the bound on
        # ∂ϕ/∂x * y
        self.Wz_lo = None
        self.Wz_up = None


class ReLUFreePattern:
    """
    The output of ReLU network is a piecewise linear function of the input.
    We will formulate the ReLU network using mixed-integer linear constraint.
    """
    def __init__(self, model, dtype):
        """
        @param model A ReLU network.
        """
        # We index each ReLU unit in the network. The units on the i'th layer
        # come before these on the i+1'th layer. self.relu_unit_index[i][j] is
        # the index of the j'th ReLU unit on the i'th layer.

        # Note that self.model is a reference to @p model. If @p model is
        # changed outside of ReLUFreePattern class, self.model would also
        # change accordingly.
        self.model = model
        # self.relu_unit_index[i][j] records the index of the j'th unit on
        # the i'th ReLU layer.
        self.relu_unit_index = []
        self.num_relu_units = 0
        self.dtype = dtype
        layer_count = 0
        assert (isinstance(model[-1], nn.Linear))
        for layer in self.model:
            if (isinstance(layer, nn.Linear)):
                self.relu_unit_index.append(
                    list(
                        range(self.num_relu_units,
                              self.num_relu_units + layer.weight.shape[0])))
                self.num_relu_units += layer.weight.shape[0]
                if layer_count == 0:
                    self.x_size = layer.weight.shape[1]
                layer_count += 1
            elif (isinstance(layer, nn.ReLU)
                  or isinstance(layer, nn.LeakyReLU)):
                pass
            else:
                raise Exception("Only accept linear or relu layer")

        # The last linear layer is not connected to a ReLU layer.
        self.num_relu_units -= len(self.relu_unit_index[-1])
        self.relu_unit_index = self.relu_unit_index[:-1]

    def strengthen_mip_at_point(self, pt: tuple,
                                linear_inputs_lo: torch.Tensor,
                                linear_inputs_up: torch.Tensor):
        """
        Given a point (linear_inputs, relu_activations), strengthen the big-M
        formulation by adding the constraints with the most violation evaluated
        at this point. Check Strong mixed-integer programming formulations for
        trained neural networks.
        """
        linear_inputs = pt[0]
        relu_activations = pt[1]

        Ain_input = []
        Ain_slack = []
        Ain_binary = []
        rhs_in = []

        # Go through each layer and each unit, strengthen the big-M
        # formulation for that unit.
        linear_input_count = 0
        for relu_layer_count in range(len(self.relu_unit_index)):
            linear_layer = self.model[2 * relu_layer_count]
            relu_layer = self.model[2 * relu_layer_count + 1]
            negative_slope = relu_layer.negative_slope if isinstance(
                relu_layer, torch.nn.LeakyReLU) else 0.
            assert (len(self.relu_unit_index[relu_layer_count]) ==
                    linear_layer.out_features)
            linear_input_indices = np.arange(
                linear_input_count,
                linear_input_count + linear_layer.in_features).tolist()
            linear_input_count += linear_layer.in_features
            linear_input = linear_inputs[linear_input_indices]
            linear_input_lo = linear_inputs_lo[linear_input_indices]
            linear_input_up = linear_inputs_up[linear_input_indices]
            for j in range(len(self.relu_unit_index[relu_layer_count])):
                beta_hat = relu_activations[
                    self.relu_unit_index[relu_layer_count][j]]
                w = linear_layer.weight.data[j]
                b = linear_layer.bias.data[
                    j] if linear_layer.bias is not None else torch.tensor(
                        0, dtype=self.dtype)
                index_set = mip_utils.find_index_set_to_strengthen(
                    w, linear_input_lo, linear_input_up, linear_input,
                    beta_hat)
                x_coeff, binary_coeff, constant = \
                    mip_utils.strengthen_relu_mip_w_indices(
                        negative_slope, w, b, linear_input_lo,
                        linear_input_up, index_set)
                assert (x_coeff.shape == (len(linear_input_indices), ))
                relu_output = linear_inputs[linear_input_count + j]
                if relu_output > x_coeff @ linear_input +\
                        binary_coeff * beta_hat + constant + 1E-12:
                    # This additional constraint is violated, append it to
                    # existing constraints, add the constraint relu_output -
                    # x_coeff * linear_input - binary_coeff * relu_activation
                    # <= constant
                    slack_coeff = torch.zeros((self.num_relu_units, ),
                                              dtype=self.dtype)
                    slack_coeff[self.relu_unit_index[relu_layer_count][j]] = 1.
                    if relu_layer_count == 0:
                        # This is the input layer.
                        Ain_input.append(-x_coeff.reshape((1, -1)))
                    else:
                        # This is not the input layer.
                        Ain_input.append(
                            torch.zeros((1, self.model[0].in_features),
                                        dtype=self.dtype))
                        slack_coeff[self.relu_unit_index[relu_layer_count -
                                                         1]] = -x_coeff
                    Ain_slack.append(slack_coeff.reshape((1, -1)))
                    binary_all_coeff = torch.zeros((1, self.num_relu_units),
                                                   dtype=self.dtype)
                    binary_all_coeff[0, self.relu_unit_index[relu_layer_count]
                                     [j]] = -binary_coeff
                    Ain_binary.append(binary_all_coeff)
                    rhs_in.append(constant)
        if len(Ain_input) == 0:
            return None, None, None, None
        return torch.cat(Ain_input, dim=0), torch.cat(
            Ain_slack, dim=0), torch.cat(Ain_binary, dim=0),\
            torch.stack(rhs_in)

    def strengthen_relu_mip_at_solution(
            self, prog: gurobi_torch_mip.GurobiTorchMIP, x_var: list,
            slack_var: list, binary_var: list,
            relu_mip_cnstr_return: ReLUMixedIntegerConstraintsReturn):
        """
        We typically strengthen the big-M MIP formulation of the ReLU network
        with the following idea:
        1. Relax the binary variables to continuous variables within [0, 1].
        2. Solve this LP relaxation to optimality.
        3. Find the most violated constraint in the ideal formulation,
           evaluated at the solution from step 2. Add that constraint to the
           program.
        4. Go to step 2, loop for several iterations.
        This function implements step 3 of this algorithm.
        """
        assert (isinstance(prog, gurobi_torch_mip.GurobiTorchMIP))
        assert (isinstance(x_var, list))
        assert (isinstance(slack_var, list))
        assert (isinstance(binary_var, list))
        assert (isinstance(relu_mip_cnstr_return,
                           ReLUMixedIntegerConstraintsReturn))
        linear_inputs = torch.tensor([v.x for v in x_var] +
                                     [v.x for v in slack_var],
                                     dtype=prog.dtype)
        relu_activations = torch.tensor([v.x for v in binary_var],
                                        dtype=prog.dtype)
        linear_inputs_lo = torch.cat((relu_mip_cnstr_return.nn_input_lo,
                                      relu_mip_cnstr_return.relu_output_lo))
        linear_inputs_up = torch.cat((relu_mip_cnstr_return.nn_input_up,
                                      relu_mip_cnstr_return.relu_output_up))
        Ain_x, Ain_slack, Ain_binary, rhs_in = self.strengthen_mip_at_point(
            (linear_inputs, relu_activations), linear_inputs_lo,
            linear_inputs_up)
        if Ain_x is not None:
            prog.addMConstrs([Ain_x, Ain_slack, Ain_binary],
                             [x_var, slack_var, binary_var],
                             b=rhs_in,
                             sense=gurobipy.GRB.LESS_EQUAL,
                             name="relu strengthened")

    def _compute_linear_output_bound_by_optimization(
            self, layer_index, linear_output_row_index,
            previous_neuron_input_lo: np.ndarray,
            previous_neuron_input_up: np.ndarray, network_input_lo: np.ndarray,
            network_input_up: np.ndarray, create_prog_callback, binary_var_type
    ) -> Tuple[float, float, torch.Tensor, torch.Tensor]:
        """
        Compute the range of a linear layer output.
        We could solve an optimization problem (LP or MILP) to find (relaxed)
        bound of the linear layer output.
        The idea is to convert the neural network to constraints. If we use
        a binary variable for each neuron activation, then we have an MILP to
        compute the exact bound. If we relax the binary variable to continuous
        variable in the range [0, 1], then we have an LP to compute the relaxed
        bound. The LP approach is explained in
        Evaluating Robustness of Neural Networks with Mixed Integer Programming
        by Vincent Tjeng, Kai Xiao and Russ Tedrake.
        @param layer_index layer 0 is the first linear layer, layer 1 is the
        second linear layer, etc.
        @param linear_output_row_index The row of the output in that linear
        layer.
        @param create_prog_callback A function that returns a GurobiTorchMIP
        object and the variable for the network input. The returned
        GurobiTorchMIP object contains the additional constraints imposed on
        network_input, that are not included in this function
        _compute_linear_output_bound_by_optimization. If set to None, then we
        create an empty GurobiTorchMIP at the beginning of this program by
        ourselves.
        """
        assert (isinstance(previous_neuron_input_lo, np.ndarray))
        assert (isinstance(previous_neuron_input_up, np.ndarray))
        assert (isinstance(network_input_lo, np.ndarray))
        assert (isinstance(network_input_up, np.ndarray))
        input_dim = self.model[0].in_features
        if create_prog_callback is None:
            prog = gurobi_torch_mip.GurobiTorchMILP(self.dtype)
            network_input = prog.addVars(input_dim,
                                         lb=torch.from_numpy(network_input_lo),
                                         ub=torch.from_numpy(network_input_up))
        else:
            assert (callable(create_prog_callback))
            prog, network_input = create_prog_callback()
            assert (isinstance(prog, gurobi_torch_mip.GurobiTorchMILP))
            assert (isinstance(network_input, list))
            assert (len(network_input) == input_dim)
            prog.addMConstrs([torch.eye(input_dim, dtype=self.dtype)],
                             [network_input],
                             b=torch.from_numpy(network_input_up),
                             sense=gurobipy.GRB.LESS_EQUAL)
            prog.addMConstrs([torch.eye(input_dim, dtype=self.dtype)],
                             [network_input],
                             b=torch.from_numpy(network_input_lo),
                             sense=gurobipy.GRB.GREATER_EQUAL)
        z_curr = network_input
        for layer in range(layer_index):
            linear_layer = self.model[2 * layer]
            relu_layer = self.model[2 * layer + 1]
            z_next, binary_relax = \
                relu_to_optimization_utils._add_constraint_to_program_by_layer(
                    prog, linear_layer, relu_layer, z_curr,
                    torch.from_numpy(
                        previous_neuron_input_lo[self.relu_unit_index[layer]]),
                    torch.from_numpy(
                        previous_neuron_input_up[self.relu_unit_index[layer]]),
                    binary_var_type=binary_var_type)
            z_curr = z_next

        # Now optimize the bound on the neuron input as Wij @ z_curr + bij
        Wij = self.model[2 * layer_index].weight[linear_output_row_index]
        bij = self.model[2 * layer_index].bias[
            linear_output_row_index] if self.model[2*layer_index].bias is not\
            None else torch.tensor(0, dtype=self.dtype)
        prog.setObjective([Wij], [z_curr],
                          constant=bij,
                          sense=gurobipy.GRB.MAXIMIZE)
        prog.gurobi_model.setParam(gurobipy.GRB.Param.OutputFlag, False)
        prog.gurobi_model.setParam(gurobipy.GRB.Param.DualReductions, 0)
        prog.gurobi_model.optimize()
        # The value of the network input that gives the upper bound.
        up_input_val = None
        if prog.gurobi_model.status == gurobipy.GRB.Status.OPTIMAL:
            linear_output_up = prog.gurobi_model.ObjVal
            up_input_val = torch.tensor([v.x for v in network_input],
                                        dtype=self.dtype)
        elif prog.gurobi_model.status == gurobipy.GRB.Status.UNBOUNDED:
            linear_output_up = np.inf
        elif prog.gurobi_model.status == gurobipy.GRB.Status.INFEASIBLE:
            linear_output_up = -np.inf

        prog.setObjective([Wij], [z_curr],
                          constant=bij,
                          sense=gurobipy.GRB.MINIMIZE)
        prog.gurobi_model.optimize()
        # The value of the network input that gives the lower bound.
        lo_input_val = None
        if prog.gurobi_model.status == gurobipy.GRB.Status.OPTIMAL:
            linear_output_lo = prog.gurobi_model.ObjVal
            lo_input_val = torch.tensor([v.x for v in network_input],
                                        dtype=self.dtype)
        elif prog.gurobi_model.status == gurobipy.GRB.Status.UNBOUNDED:
            linear_output_lo = -np.inf
        elif prog.gurobi_model.status == gurobipy.GRB.Status.INFEASIBLE:
            linear_output_lo = np.inf
        return linear_output_lo, linear_output_up, lo_input_val, up_input_val

    def _compute_layer_bound(self,
                             x_lo,
                             x_up,
                             method: mip_utils.PropagateBoundsMethod,
                             create_prog_callback=None):
        """
        Compute the input and output bounds of each ReLU neurons.
        """
        linear_layer_input_lo = x_lo.clone()
        linear_layer_input_up = x_up.clone()
        z_pre_relu_lo = torch.empty((self.num_relu_units, ), dtype=self.dtype)
        z_pre_relu_up = torch.empty((self.num_relu_units, ), dtype=self.dtype)
        z_post_relu_lo = torch.empty((self.num_relu_units, ), dtype=self.dtype)
        z_post_relu_up = torch.empty((self.num_relu_units, ), dtype=self.dtype)
        for layer_count in range(len(self.relu_unit_index)):
            z_indices = self.relu_unit_index[layer_count]
            bias = self.model[2 * layer_count].bias if self.model[
                2 * layer_count].bias is not None else torch.zeros(
                    (self.model[2 * layer_count].out_features, ),
                    dtype=self.dtype)
            if method == mip_utils.PropagateBoundsMethod.IA:
                z_pre_relu_lo[z_indices], z_pre_relu_up[z_indices] = \
                    mip_utils.compute_range_by_IA(
                        self.model[2 * layer_count].weight, bias,
                        linear_layer_input_lo, linear_layer_input_up)
            else:
                if method == mip_utils.PropagateBoundsMethod.LP:
                    binary_var_type = gurobi_torch_mip.BINARYRELAX
                elif method in (mip_utils.PropagateBoundsMethod.MIP,
                                mip_utils.PropagateBoundsMethod.IA_MIP):
                    binary_var_type = gurobipy.GRB.BINARY
                else:
                    raise Exception(
                        "_compute_layer_bound: unknown bound propagation" +
                        " method")

                for j in range(self.model[2 * layer_count].out_features):
                    neuron_index = self.relu_unit_index[layer_count][j]
                    z_pre_relu_lo[neuron_index], z_pre_relu_up[
                        neuron_index], _, _ = \
                        self._compute_linear_output_bound_by_optimization(
                            layer_count, j,
                            z_pre_relu_lo.detach().numpy(),
                            z_pre_relu_up.detach().numpy(),
                            x_lo.detach().numpy(),
                            x_up.detach().numpy(),
                            create_prog_callback, binary_var_type)
                if method == mip_utils.PropagateBoundsMethod.IA_MIP:
                    # We also compute the bounds by IA. If the IA bound is
                    # close to the MIP bound, then we use the IA bound.
                    # Otherwise, we relax the MIP bound a little bit to make
                    # sure this relaxed bound will not be active.
                    z_pre_relu_lo_ia, z_pre_relu_up_ia = \
                        mip_utils.compute_range_by_IA(
                            self.model[2 * layer_count].weight, bias,
                            linear_layer_input_lo, linear_layer_input_up)
                    for j in range(self.model[2 * layer_count].out_features):
                        neuron_index = self.relu_unit_index[layer_count][j]
                        if torch.abs(z_pre_relu_lo[neuron_index] -
                                     z_pre_relu_lo_ia[j]) < 1E-4:
                            z_pre_relu_lo[neuron_index] = z_pre_relu_lo_ia[j]
                        else:
                            if z_pre_relu_lo[neuron_index] > 0:
                                # If the MIP lower bound is > 0, then the ReLU
                                # unit should always be active. We reduce this
                                # MIP lower bound a little bit (but still
                                # remains positive)
                                z_pre_relu_lo[neuron_index] *= 0.99
                                z_pre_relu_lo[
                                    neuron_index] += 0.01 * torch.clamp(
                                        z_pre_relu_lo_ia[j], max=0.).detach()
                            else:
                                z_pre_relu_lo[neuron_index] *= 0.99
                                z_pre_relu_lo[
                                    neuron_index] += 0.01 * z_pre_relu_lo_ia[
                                        j].detach()
                        if torch.abs(z_pre_relu_up[neuron_index] -
                                     z_pre_relu_up_ia[j]) < 1E-4:
                            z_pre_relu_up[neuron_index] = z_pre_relu_up_ia[j]
                        else:
                            if z_pre_relu_up[neuron_index] <= 0:
                                z_pre_relu_up[neuron_index] *= 0.99
                                z_pre_relu_up[
                                    neuron_index] += 0.01 * torch.clamp(
                                        z_pre_relu_up_ia[j], max=0.).detach()
                            else:
                                z_pre_relu_up[neuron_index] *= 0.99
                                z_pre_relu_up[
                                    neuron_index] += 0.01 * z_pre_relu_up_ia[
                                        j].detach()

            z_post_relu_lo[z_indices], z_post_relu_up[
                z_indices] = mip_utils.propagate_bounds(
                    self.model[2 * layer_count + 1], z_pre_relu_lo[z_indices],
                    z_pre_relu_up[z_indices])
            linear_layer_input_lo = z_post_relu_lo[z_indices].clone()
            linear_layer_input_up = z_post_relu_up[z_indices].clone()
        return z_pre_relu_lo, z_pre_relu_up, z_post_relu_lo, z_post_relu_up

    def _compute_network_output_bounds(self,
                                       previous_neuron_input_lo,
                                       previous_neuron_input_up,
                                       network_input_lo,
                                       network_input_up,
                                       method,
                                       create_prog_callback=None):
        if method == mip_utils.PropagateBoundsMethod.IA:
            linear_input_lo, linear_input_up = mip_utils.propagate_bounds(
                self.model[-2],
                previous_neuron_input_lo[self.relu_unit_index[-1]],
                previous_neuron_input_up[self.relu_unit_index[-1]])
            linear_output_lo, linear_output_up = \
                mip_utils.propagate_bounds(
                    self.model[-1], linear_input_lo, linear_input_up)
            return linear_output_lo, linear_output_up
        else:
            if method == mip_utils.PropagateBoundsMethod.LP:
                binary_var_type = gurobi_torch_mip.BINARYRELAX
            elif method in (mip_utils.PropagateBoundsMethod.MIP,
                            mip_utils.PropagateBoundsMethod.IA_MIP):
                binary_var_type = gurobipy.GRB.BINARY
            else:
                raise Exception(
                    "_compute_network_output_bounds: unknown bound " +
                    "propagate method.")
            linear_output_lo = torch.empty((self.model[-1].out_features, ),
                                           dtype=self.dtype)
            linear_output_up = torch.empty((self.model[-1].out_features, ),
                                           dtype=self.dtype)
            if method == mip_utils.PropagateBoundsMethod.IA_MIP:
                linear_output_lo_ia, linear_output_up_ia = \
                    self._compute_network_output_bounds(
                        previous_neuron_input_lo, previous_neuron_input_up,
                        network_input_lo, network_input_up,
                        mip_utils.PropagateBoundsMethod.IA)
            for i in range(self.model[-1].out_features):
                linear_output_lo[i], linear_output_up[
                    i], _, _ = \
                        self._compute_linear_output_bound_by_optimization(
                            int((len(self.model) - 1) / 2),
                            i,
                            previous_neuron_input_lo.detach().numpy(),
                            previous_neuron_input_up.detach().numpy(),
                            network_input_lo.detach().numpy(),
                            network_input_up.detach().numpy(),
                            create_prog_callback,
                            binary_var_type)
                if method == mip_utils.PropagateBoundsMethod.IA_MIP:
                    if torch.abs(linear_output_lo[i] -
                                 linear_output_lo_ia[i]) < 1E-4:
                        linear_output_lo[i] = linear_output_lo_ia[i]
                    else:
                        linear_output_lo[i] *= 0.99
                        linear_output_lo[i] += 0.01 * linear_output_lo_ia[i]
                    if torch.abs(linear_output_up[i] -
                                 linear_output_up_ia[i]) < 1E-3:
                        linear_output_up[i] = linear_output_up_ia[i].detach()
                    else:
                        linear_output_up[i] *= 0.99
                        linear_output_up[
                            i] += 0.01 * linear_output_up_ia[i].detach()

            return linear_output_lo, linear_output_up

    def _output_constraint_given_bounds(self, z_pre_relu_lo, z_pre_relu_up,
                                        x_lo, x_up):
        assert (isinstance(z_pre_relu_lo, torch.Tensor))
        assert (isinstance(z_pre_relu_up, torch.Tensor))
        assert (isinstance(x_lo, torch.Tensor))
        assert (isinstance(x_up, torch.Tensor))
        # Each ReLU unit introduces at most 4 inequality constraints.
        Ain_input = torch.zeros(
            (4 * self.num_relu_units + 2 * self.x_size, self.x_size),
            dtype=self.dtype)
        Ain_slack = torch.zeros(
            (4 * self.num_relu_units + 2 * self.x_size, self.num_relu_units),
            dtype=self.dtype)
        Ain_binary = torch.zeros(
            (4 * self.num_relu_units + 2 * self.x_size, self.num_relu_units),
            dtype=self.dtype)
        rhs_in = torch.empty((4 * self.num_relu_units + 2 * self.x_size, ),
                             dtype=self.dtype)
        # Each ReLU unit introduces at most 2 equality constraints.
        Aeq_input = torch.zeros((2 * self.num_relu_units, self.x_size),
                                dtype=self.dtype)
        Aeq_slack = torch.zeros((2 * self.num_relu_units, self.num_relu_units),
                                dtype=self.dtype)
        Aeq_binary = torch.zeros(
            (2 * self.num_relu_units, self.num_relu_units), dtype=self.dtype)
        rhs_eq = torch.empty((2 * self.num_relu_units, ), dtype=self.dtype)

        eq_constr_count = 0
        ineq_constr_count = 0

        # First add the constraint on the input x_lo <= x <= x_up
        Ain_input[:self.x_size] = torch.eye(self.x_size, dtype=self.dtype)
        Ain_input[self.x_size:2 *
                  self.x_size] = -torch.eye(self.x_size, dtype=self.dtype)
        rhs_in[:self.x_size] = x_up
        rhs_in[self.x_size:2 * self.x_size] = -x_lo
        ineq_constr_count += 2 * self.x_size

        for layer_count in range(len(self.relu_unit_index)):
            Ain_z_curr, Ain_z_next, Ain_binary_layer, rhs_in_layer,\
                Aeq_z_curr, Aeq_z_next, Aeq_binary_layer, rhs_eq_layer, _, _ =\
                relu_to_optimization_utils._add_constraint_by_layer(
                    self.model[2*layer_count], self.model[2*layer_count+1],
                    z_pre_relu_lo[self.relu_unit_index[layer_count]],
                    z_pre_relu_up[self.relu_unit_index[layer_count]])
            if layer_count == 0:
                # For the input layer, z_curr is the network input.
                if Ain_z_curr.shape[0] > 0:
                    Ain_input[ineq_constr_count:ineq_constr_count +
                              Ain_z_curr.shape[0]] = Ain_z_curr
                if Aeq_z_curr.shape[0] > 0:
                    Aeq_input[eq_constr_count:eq_constr_count +
                              Aeq_z_curr.shape[0]] = Aeq_z_curr
            else:
                if Ain_z_curr.shape[0] > 0:
                    Ain_slack[ineq_constr_count:ineq_constr_count +
                              Ain_z_curr.shape[0],
                              self.relu_unit_index[layer_count -
                                                   1]] = Ain_z_curr
                if Aeq_z_curr.shape[0] > 0:
                    Aeq_slack[eq_constr_count:eq_constr_count +
                              Aeq_z_curr.shape[0],
                              self.relu_unit_index[layer_count -
                                                   1]] = Aeq_z_curr
            if Ain_z_curr.shape[0] > 0:
                Ain_slack[ineq_constr_count:ineq_constr_count +
                          Ain_z_curr.shape[0],
                          self.relu_unit_index[layer_count]] = Ain_z_next
                Ain_binary[
                    ineq_constr_count:ineq_constr_count + Ain_z_curr.shape[0],
                    self.relu_unit_index[layer_count]] = Ain_binary_layer
                rhs_in[ineq_constr_count:ineq_constr_count +
                       Ain_z_curr.shape[0]] = rhs_in_layer
                ineq_constr_count += Ain_z_curr.shape[0]
            if Aeq_z_curr.shape[0] > 0:
                Aeq_slack[eq_constr_count:eq_constr_count +
                          Aeq_z_curr.shape[0],
                          self.relu_unit_index[layer_count]] = Aeq_z_next
                Aeq_binary[
                    eq_constr_count:eq_constr_count + Aeq_z_curr.shape[0],
                    self.relu_unit_index[layer_count]] = Aeq_binary_layer
                rhs_eq[eq_constr_count:eq_constr_count +
                       Aeq_z_curr.shape[0]] = rhs_eq_layer
                eq_constr_count += Aeq_z_curr.shape[0]

        mip_constr_return = ReLUMixedIntegerConstraintsReturn()
        # The output is the last linear layer applied on the last chunk of z.
        mip_constr_return.Aout_slack = torch.zeros(
            (self.model[-1].out_features, self.num_relu_units),
            dtype=self.dtype)
        mip_constr_return.Aout_slack[:, self.relu_unit_index[-1]] = \
            self.model[-1].weight.clone()
        if self.model[-1].bias is None:
            mip_constr_return.Cout = torch.zeros(
                (self.model[-1].out_features, ), dtype=self.dtype)
        else:
            mip_constr_return.Cout = self.model[-1].bias.clone()

        mip_constr_return.Ain_input = Ain_input[:ineq_constr_count]
        mip_constr_return.Ain_slack = Ain_slack[:ineq_constr_count]
        mip_constr_return.Ain_binary = Ain_binary[:ineq_constr_count]
        mip_constr_return.rhs_in = rhs_in[:ineq_constr_count]
        mip_constr_return.Aeq_input = Aeq_input[:eq_constr_count]
        mip_constr_return.Aeq_slack = Aeq_slack[:eq_constr_count]
        mip_constr_return.Aeq_binary = Aeq_binary[:eq_constr_count]
        mip_constr_return.rhs_eq = rhs_eq[:eq_constr_count]
        mip_constr_return.nn_input_lo = x_lo
        mip_constr_return.nn_input_up = x_up
        mip_constr_return.relu_input_lo = z_pre_relu_lo
        mip_constr_return.relu_input_up = z_pre_relu_up
        return mip_constr_return

    def output_constraint(self, x_lo, x_up,
                          method: mip_utils.PropagateBoundsMethod):
        """
        The output of (leaky) ReLU network is a piecewise linear function of
        the input.
        ReLU(x) = wₙᵀzₙ+bₙ
        s.t zᵢ₊₁ = max{0, Wᵢzᵢ+bᵢ}
            z₀ = x
        Hence we can formulate the ReLU network as a linear function subject to
        mixed-integer linear constraints
        If we define the bound (Wᵢzᵢ)(j) + bᵢ(j) as [zₗₒ, zᵤₚ], then depending
        on the bounds, there are 3 cases:
        case 1
            If  zₗₒ < 0 <  zᵤₚ, we don't know if the ReLU unit is always
            active, then the constraints are
            zᵢ₊₁(j) ≥ 0
            zᵢ₊₁(j) ≥ (Wᵢzᵢ)(j)+bᵢ(j)
            zᵢ₊₁(j) ≤ zᵤₚβᵢ(j)
            (Wᵢzᵢ)(j) + bᵢ(j) - zᵢ₊₁(j) + zₗₒβᵢ(j) ≥ zₗₒ
            This formulation is explained in
            replace_relu_with_mixed_integer_constraint()
            Similarly if the layer is a leaky relu layer, then we replace the
            leaky relu output using
            replace_leaky_relu_with_mixed_integer_constraint()
        case 2
            If 0 <= zₗₒ <= zᵤₚ, the ReLU unit is always active,  then
            the constraints are
            zᵢ₊₁(j) = (Wᵢzᵢ)(j) + bᵢ(j)
            βᵢ(j) = 1
        case 3
            zₗₒ <= zᵤₚ <= 0, the ReLU unit is always inactive, then the
            constraint are
            zᵢ₊₁(j) = 0
            βᵢ(j) = 0
        where βᵢ(j) is a binary variable,
        such that βᵢ(j) = 1 means that the j'th ReLU unit on the i'th layer
        is active.
        Moreover, we impose the constraint that x_lo <= x <= x_up
        We will write the constraint in a more concise way
        Ain1 * x + Ain2 * z + Ain3 * β <= rhs_in  (case 1)
        Aeq1 * x + Aeq2 * z + Aeq3 * β = rhs_eq   (case 2 and 3)
        ReLU(x) = Aₒᵤₜ*z + bₒᵤₜ
        where z, β are the "flat" column vectors, z = [z₁; z₂;...;zₙ],
        β = [β₀; β₁; ...; βₙ₋₁].
        @param x_lo A 1-D vector, the lower bound of input x.
        @param x_up A 1-D vector, the upper bound of input x.
        @param method The method to propagate the bounds of the inputs in each
        layer. If you want to take the gradient w.r.t the parameter in this
        network, then use PropagateBoundsMethod.IA, otherwise use
        PropagateBoundsMethod.LP.
        @return (Ain1, Ain2, Ain3, rhs_in, Aeq1, Aeq2, Aeq3, rhs_eq, A_out,
        b_out, z_pre_relu_lo, z_pre_relu_up, output_lo, output_up)
        Ain1, Ain2, Ain3, Aeq1, Aeq2, Aeq3, A_out are matrices, rhs_in, rhs_eq,
        b_out column vectors. z_pre_relu_lo and z_pre_relu_up are 1-D vectors.
        When the output is a scalar, then A_out is a vector, and b_out is a
        scalar.
        Notice that z_pre_relu_lo[i] and z_pre_relu_up[i] are the bounds of
        z[i] BEFORE applying the ReLU activation function, note these are NOT
        the bounds on z.
        """
        assert (x_lo.dtype == self.dtype)
        assert (x_up.dtype == self.dtype)
        assert (len(x_lo.shape) == 1)
        assert (len(x_up.shape) == 1)
        assert (torch.all(torch.le(x_lo, x_up)))

        z_pre_relu_lo, z_pre_relu_up, z_post_relu_lo, z_post_relu_up =\
            self._compute_layer_bound(x_lo, x_up, method)
        output_lo, output_up = self._compute_network_output_bounds(
            z_pre_relu_lo, z_pre_relu_up, x_lo, x_up, method)
        mip_constr_return = self._output_constraint_given_bounds(
            z_pre_relu_lo, z_pre_relu_up, x_lo, x_up)
        mip_constr_return.nn_input_lo = x_lo
        mip_constr_return.nn_input_up = x_up
        mip_constr_return.nn_output_lo = output_lo
        mip_constr_return.nn_output_up = output_up
        mip_constr_return.relu_input_lo = z_pre_relu_lo
        mip_constr_return.relu_input_up = z_pre_relu_up
        mip_constr_return.relu_output_lo = z_post_relu_lo
        mip_constr_return.relu_output_up = z_post_relu_up
        return mip_constr_return

    def compute_relu_unit_outputs_and_activation(self, x):
        """
        This is a utility function for output_constraint(). Given a network
        input x, this function computes the vector containing each ReLU unit
        output z, and the activation of each ReLU unit β.
        @param x The input to the ReLU network. A 1-D tensor.
        @return (z, β, output) z, β are column vectors. Refer to
        output_constraint()
        function for their definition.
        """
        z = torch.empty((self.num_relu_units, 1), dtype=self.dtype)
        beta = torch.empty((self.num_relu_units, 1), dtype=self.dtype)
        relu_unit_count = 0
        z_layer = x
        for layer in self.model:
            if (isinstance(layer, nn.Linear)):
                z_layer = layer.forward(z_layer)
            elif (isinstance(layer, nn.ReLU)
                  or isinstance(layer, nn.LeakyReLU)):
                for i in range(z_layer.numel()):
                    beta[relu_unit_count + i][0] = 1 if z_layer[i] > 0 else 0
                z_layer = layer.forward(z_layer)
                z[relu_unit_count:relu_unit_count +
                  z_layer.numel()] = z_layer.reshape((-1, 1))
                relu_unit_count += z_layer.numel()
            else:
                raise Exception("compute_relu_unit_outputs_and_activation: " +
                                " only supports linear, relu or leaky relu " +
                                "units.")
        # The output layer
        if len(z_layer) <= 1:
            output = z_layer.item()
        else:
            output = z_layer.squeeze()

        return (z, beta, output)

    def compute_alpha_index(self, relu_unit_layer_indices):
        """
        Compute the index of α given an activation path (one and only one
        active ReLU unit on each layer). You could refer to output_gradient()
        function for the definition of α. Given a tuple (i₀, i₁, ..., iₙ₋₁)
        meaning that on the k'th layer, iₖ'th ReLU unit is active, we give
        the index of α[i₀][i₁]...[iₙ₋₁] in the vector α.
        @param relu_unit_layer_indices A tuple of length num_ReLU_layers in the
        ReLU network. relu_unit_layer_indices[i] is the active ReLU unit on the
        i'th layer.
        """
        assert (len(relu_unit_layer_indices) == len(self.relu_unit_index))
        index = 0
        for i in range(len(self.relu_unit_index)):
            # Go through each layer
            assert (relu_unit_layer_indices[i] >= 0 and
                    relu_unit_layer_indices[i] < len(self.relu_unit_index[i]))
            index = index * \
                len(self.relu_unit_index[i]) + relu_unit_layer_indices[i]
        return index

    def output_gradient(self):
        """
        The ReLU network output is a piecewise linear function of the input x.
        Hence the gradient of the output w.r.t the input can be expressed as
        a linear function of some binary variable α, namely
        ∂ReLU(x)/∂x = αᵀM
        with the additional linear constraint
        B1 * α + B2 * β ≤ d
        where β are also binary variables, β = [β₀; β₁; ...; βₙ₋₁], with
        βᵢ(j) = 1 if the j'th ReLU unit on the i'th layer is active.

        To understand why this is true, notice that the gradient could be
        written as
        ∂ReLU(x)/∂x = wₙᵀ * diag(βₙ₋₁) * Wₙ₋₁ * diag(βₙ₋₂) * Wₙ₋₂ * ... *
                      diag(β₀) * W₀
        if the output is not connected to a ReLU unit. If the output is
        connected to a ReLU unit, then we multiply then
        β = [β₀; β₁; ...; βₙ₋₁; βₙ], and the gradient is
        ∂ReLU(x)/∂x = βₙ * wₙᵀ * diag(βₙ₋₁) * Wₙ₋₁ * diag(βₙ₋₂) * Wₙ₋₂ * ... *
                      diag(β₀) * W₀
        In either case, the gradient is a multilinear function of β, and we
        will introduce new binary variable α to represent the product of β.
        We define α[i₀][i₁]...[iₙ₋₁]=β₀[i₀]*β₁[i₁]*...*βₙ₋₁[iₙ₋₁] and the
        gradient ∂ReLU(x)/∂x could be written as a linear function of α.
        To impose the constraint that α is the product of β, we introduce the
        following linear constraints
        α[i₀][i₁]...[iₙ₋₁] ≤ βₖ[iₖ] ∀ k=0,...,n-1
        α[i₀][i₁]...[iₙ₋₁] ≥ β₀[i₀] + ... + βₙ₋₁[iₙ₋₁] - (n-1)
        @param model A ReLU network. This network must have the same structure
        as the network in the class constructor (but the weights can be
        different).
        @return (M, B1, B2, d) A, B1, B2 are 2-D matrices, d is a column
        vector.
        """
        """
        In order to compute the gradient
        ∂ReLU(x)/∂x = wₙᵀ * diag(βₙ₋₁) * Wₙ₋₁ * diag(βₙ₋₂) * Wₙ₋₂ * ...
                      * diag(β₀) * W₀
        Notice that the gradient ∂ReLU(x)/∂x can be written as a multilinear
        polynomial of β. We will replace all the product of β terms with a new
        binary variable α. If we define
        α[i₀][i₁]...[iₙ₋₁] = ∏ₖ(βₖ[iₖ])
        and the size of α is ∏ₖ(# of ReLU units in layer k).
        Then we can write the output gradient as a linear function of α, we
        write the gradient ∂ReLU(x)/∂x = αᵀM, where M is a big matrix depending
        on the network weight w. In order to enforce the condition
        α[i₀][i₁]...[iₙ₋₁] = ∏ₖ(βₖ[iₖ]), we introduce the linear constraints
        α[i₀][i₁]...[iₙ₋₁] ≤ βₖ[iₖ] ∀ k=0,...,n-1
        α[i₀][i₁]...[iₙ₋₁] ≥ β₀[i₀] + ... + βₙ₋₁[iₙ₋₁] - (n-1)
        We will write these linear constraints as
        B1 * α + B2 * β ≤ d

        @return (M, B1, B2, d) M, B1, B2 are matrices, d is a column vector.
        """

        # LinearUnitGradient stores a length-k tuple of ReLU unit indices (one
        # ReLU unit per layer) and the gradient of the k+1'th linear layer
        # output,
        # when only the ReLU units with the given indices are active. For
        # example
        # if LinearUnitGradient.activated_relu_indices = (2, 4), then
        # LinearUnitGradient.gradient stores the gradient of the 3rd linear
        # layer output, when only the 2nd unit on the first layer, and 4th unit
        # on the second layer are active. Note that the value of
        # LinearUnitGradient.gradient is a matrix. The i'th row of this matrix
        # is the gradient of the i'th entry of the linear output.
        class LinearUnitGradient:
            def __init__(self, activated_relu_indices, gradient):
                self.activated_relu_indices = activated_relu_indices
                self.gradient = gradient

        # layer_linear_unit_gradients stores all the LinearUnitGradient for
        # this linear layer.
        num_alpha = np.prod(
            np.array([
                len(layer_relu_unit_index)
                for layer_relu_unit_index in self.relu_unit_index
            ]))

        layer_linear_unit_gradients = queue.Queue(maxsize=num_alpha)
        layer_count = 0
        for layer in self.model:
            if (isinstance(layer, nn.Linear)):
                if layer_count == 0:
                    linear_unit_gradient = LinearUnitGradient(
                        (), layer.weight.clone())
                    layer_linear_unit_gradients.put(linear_unit_gradient)
                else:
                    queue_size = layer_linear_unit_gradients.qsize()
                    last_layer_linear_unit_gradient_count = 0
                    while (last_layer_linear_unit_gradient_count < queue_size):
                        last_layer_linear_unit_gradient =\
                            layer_linear_unit_gradients.get()
                        last_layer_linear_unit_gradient_count += 1
                        for layer_relu_unit_index in \
                                range(len(
                                      self.relu_unit_index[layer_count - 1])):
                            # We append the ReLU unit with index
                            # layer_relu_unit_index
                            # to the activated path
                            linear_unit_gradient = LinearUnitGradient(
                                last_layer_linear_unit_gradient.
                                activated_relu_indices +
                                (layer_relu_unit_index, ),
                                layer.weight[:, layer_relu_unit_index].reshape(
                                    (-1, 1)) @ last_layer_linear_unit_gradient.
                                gradient[layer_relu_unit_index].reshape(
                                    (1, -1)))
                            layer_linear_unit_gradients.put(
                                linear_unit_gradient)

            elif (isinstance(layer, nn.ReLU)):
                layer_count += 1
            else:
                raise Exception("output_gradient: currently only " +
                                "supports linear and ReLU layers.")

        # Now loop through layer_linear_unit_gradients, fill in the matrix M,
        # and add the linear equality constraint B1 * α + B2 * β ≤ d
        M = torch.empty((num_alpha, self.x_size), dtype=self.dtype)
        num_ineq = num_alpha * (len(self.relu_unit_index) + 1)
        B1 = torch.zeros((num_ineq, num_alpha), dtype=self.dtype)
        B2 = torch.zeros((num_ineq, self.num_relu_units), dtype=self.dtype)
        d = torch.zeros((num_ineq, 1), dtype=self.dtype)
        ineq_constraint_count = 0
        while (not layer_linear_unit_gradients.empty()):
            layer_linear_unit_gradient = layer_linear_unit_gradients.get()
            alpha_index = self.compute_alpha_index(
                layer_linear_unit_gradient.activated_relu_indices)
            M[alpha_index] = layer_linear_unit_gradient.gradient
            # Now add the constraint
            # α[i₀][i₁]...[iₙ₋₁] ≤ βₖ[iₖ] ∀ k=0,...,n-1
            # α[i₀][i₁]...[iₙ₋₁] ≥ β₀[i₀] + ... + βₙ₋₁[iₙ₋₁] - (n-1)
            num_layers = len(self.relu_unit_index)
            B1[ineq_constraint_count + num_layers][alpha_index] = -1.
            d[ineq_constraint_count + num_layers][0] = num_layers - 1.
            for layer_count in range(num_layers):
                B1[ineq_constraint_count + layer_count][alpha_index] = 1.
                beta_index = self.relu_unit_index[layer_count][
                    layer_linear_unit_gradient.
                    activated_relu_indices[layer_count]]
                d[ineq_constraint_count + layer_count][0] = 0.
                B2[ineq_constraint_count + layer_count][beta_index] = -1.
                B2[ineq_constraint_count + num_layers][beta_index] = 1
            ineq_constraint_count += num_layers + 1

        return (M, B1, B2, d)

    def output_gradient_times_vector(
            self, vector_lower,
            vector_upper) -> ReLUGradientTimesVecMixedIntegerConstraintsReturn:
        """
        We want to compute the gradient of the network ∂ReLU(x)/∂x times a
        vector y: ∂ReLU(x)/∂x  * y, and reformulate this product as
        mixed-integer linear constraints.
        We assume that the leaky relu unit has negative slope c (c = 0 for
        ReLU unit). And we define a matrix
        M(β, c) = c*I + (1-c)*diag(β)
        We notice that
        ∂ReLU(x)/∂x = Wₙ * M(βₙ₋₁, c)  * Wₙ₋₁ * M(βₙ₋₂, c) * Wₙ₋₂ * ...
                      * M(β₀, c) * W₀
        So we introduce slack variable z, such that
        z₀ = y
        zᵢ₊₁ = M(βᵢ, c)*Wᵢ*zᵢ, i = 0, ..., n-1    (1)
        The constraint (1) can be replaced by mixed-integer linear constraint
        on zᵢ and βᵢ.

        We write ∂ReLU(x)/∂x * y as
        Aₒᵤₜ * slack
        with the additional constraint
        A_y * y + A_slack * slack + A_beta * β ≤ rhs
        where slack = [z₁; z₂;...;zₙ]
        Note that we do NOT require that β is the right activation pattern for
        the input x. This constraint should be imposed in output() function.
        @param vector_lower The lower bound of the vector y.
        @param vector_upper The upper bound of the vector y.
        @return mip_cnstr_return that contains the output
        ∂ReLU(x)/∂x * y = Aₒᵤₜ * z
        and the linear inequality constraint
        A_y * y + A_slack * slack + A_beta * β ≤ rhs
        """
        utils.check_shape_and_type(vector_lower, (self.x_size, ), self.dtype)
        utils.check_shape_and_type(vector_upper, (self.x_size, ), self.dtype)
        A_y = torch.zeros((4 * self.num_relu_units, self.x_size),
                          dtype=self.dtype)
        A_slack = torch.zeros((4 * self.num_relu_units, self.num_relu_units),
                              dtype=self.dtype)
        A_beta = torch.zeros((4 * self.num_relu_units, self.num_relu_units),
                             dtype=self.dtype)
        rhs = torch.zeros(4 * self.num_relu_units, dtype=self.dtype)
        z_lo = torch.empty(self.num_relu_units, dtype=self.dtype)
        z_up = torch.empty(self.num_relu_units, dtype=self.dtype)
        layer_count = 0
        ineq_count = 0
        A_out = torch.zeros((self.model[-1].out_features, self.num_relu_units),
                            dtype=self.dtype)
        z_lo, z_up, Wz_lo, Wz_up = self._compute_Wz_bounds_IA(
            vector_lower, vector_upper)
        linear_layer_count = 0
        for layer in self.model:
            if (isinstance(layer, nn.Linear)):
                Wi = layer.weight
                if layer_count == len(self.relu_unit_index):
                    # If this is the last linear layer, and this linear layer
                    # is the output layer, then we set A_out. Otherwise, we
                    # append inequality constraints.
                    A_out[:, self.relu_unit_index[layer_count - 1]] = \
                        layer.weight
                linear_layer_count += 1
            elif isinstance(layer, nn.ReLU) or isinstance(layer, nn.LeakyReLU):
                for j in range(len(self.relu_unit_index[layer_count])):
                    if isinstance(layer, nn.ReLU):
                        A_pre, A_z_next, A_beta_i, rhs_i = utils.\
                            replace_binary_continuous_product(
                                Wz_lo[linear_layer_count-1][j],
                                Wz_up[linear_layer_count-1][j],
                                dtype=self.dtype)
                    else:
                        A_pre, A_z_next, A_beta_i, rhs_i = utils.\
                            leaky_relu_gradient_times_x(
                                Wz_lo[linear_layer_count-1][j],
                                Wz_up[linear_layer_count-1][j],
                                layer.negative_slope, dtype=self.dtype)
                    if layer_count == 0:
                        A_y[ineq_count:ineq_count+4] =\
                            A_pre.reshape((-1, 1)) @\
                            Wi[j].reshape((1, -1))
                    else:
                        A_slack[ineq_count:ineq_count+4,
                                self.relu_unit_index[layer_count - 1]] =\
                            A_pre.reshape((-1, 1)) @\
                            Wi[j].reshape((1, -1))
                    A_slack[ineq_count:ineq_count+4,
                            self.relu_unit_index[layer_count][j]] =\
                        A_z_next.squeeze()
                    A_beta[ineq_count:ineq_count+4,
                           self.relu_unit_index[layer_count][j]] =\
                        A_beta_i.squeeze()
                    rhs[ineq_count:ineq_count + 4] = rhs_i
                    ineq_count += 4
                layer_count += 1

            else:
                raise Exception("output_gradient_times_vector: we currently " +
                                "only support linear and ReLU units.")
        result = ReLUGradientTimesVecMixedIntegerConstraintsReturn()
        result.Aout_slack = A_out
        result.Ain_input = A_y
        result.Ain_slack = A_slack
        result.Ain_binary = A_beta
        result.rhs_in = rhs
        result.z_lo = z_lo
        result.z_up = z_up
        result.Wz_lo = Wz_lo
        result.Wz_up = Wz_up
        return result

    def _compute_Wz_bounds_optimization(self, prog: gurobipy.Model, z0: list,
                                        beta: list):
        """
        This function is used in output_gradient_times_vector. It computes
        the lower/upper bounds of Wᵢ*z through optimization, where zᵢ is
        defined recursively as
        zᵢ₊₁ = M(βᵢ, c)*Wᵢ*zᵢ, i = 0, ..., n-1    (1)
        and M is the matrix defined as
        M(β, c) = c*I + (1-c)*diag(β)
        Note that unlike _compute_Wz_bounds_IA(), the bounds computed in this
        approach does NOT carry the gradient.

        Args:
          prog, z0, beta: prog is an optimization program, it contains all the
          constraints that you want to impose on z₀, β except
          zᵢ₊₁ = M(βᵢ, c)*Wᵢ*zᵢ, i = 0, ..., n-1

        Return:
          z_lo, z_up: The lower and upper bound of z.
          Wz_lo, Wz_up: Wz_lo[i]/Wz_up[i] is the lower/upper bound of Wᵢ*zᵢ
          z: The decision variables of z.
        """
        assert (isinstance(prog, gurobipy.Model))
        assert (isinstance(z0, list))
        assert (isinstance(beta, list))
        # Loop through each layer
        num_linear_layers = int((len(self.model) + 1) / 2)
        z = [None] * num_linear_layers
        z[0] = z0
        z_lo = [None] * num_linear_layers
        z_up = [None] * num_linear_layers
        Wz_lo = [None] * num_linear_layers
        Wz_up = [None] * num_linear_layers

        # First compute the bounds of z₀
        z_lo[0] = torch.empty((self.model[0].in_features, ), dtype=self.dtype)
        z_up[0] = torch.empty((self.model[0].in_features, ), dtype=self.dtype)
        for i in range(len(z0)):
            prog.setObjective(gurobipy.LinExpr(1, z0[i]),
                              sense=gurobipy.GRB.MAXIMIZE)
            prog.optimize()
            assert (prog.status == gurobipy.GRB.Status.OPTIMAL)
            z_up[0][i] = prog.ObjVal
            prog.setObjective(gurobipy.LinExpr(1, z0[i]),
                              sense=gurobipy.GRB.MINIMIZE)
            prog.optimize()
            assert (prog.status == gurobipy.GRB.Status.OPTIMAL)
            z_lo[0][i] = prog.ObjVal

        # Now compute the bounds of z and Wz.
        for linear_layer_count in range(num_linear_layers):
            linear_layer = self.model[2 * linear_layer_count]
            # Compute the bounds of Wᵢ(j, :)*zᵢ
            Wz_lo[linear_layer_count] = torch.empty(
                (linear_layer.out_features, ), dtype=self.dtype)
            Wz_up[linear_layer_count] = torch.empty(
                (linear_layer.out_features, ), dtype=self.dtype)
            if linear_layer_count < num_linear_layers - 1:
                z[linear_layer_count + 1] = [None] * linear_layer.out_features
                z_lo[linear_layer_count + 1] = torch.empty(
                    (linear_layer.out_features, ), dtype=self.dtype)
                z_up[linear_layer_count + 1] = torch.empty(
                    (linear_layer.out_features, ), dtype=self.dtype)
            for j in range(linear_layer.out_features):
                Wizi = gurobipy.LinExpr(
                    linear_layer.weight.data[j, :].tolist(),
                    z[linear_layer_count])
                prog.setObjective(Wizi, sense=gurobipy.GRB.MAXIMIZE)
                prog.optimize()
                assert (prog.status == gurobipy.GRB.Status.OPTIMAL)
                Wz_up[linear_layer_count][j] = prog.ObjVal
                prog.setObjective(Wizi, sense=gurobipy.GRB.MINIMIZE)
                prog.optimize()
                assert (prog.status == gurobipy.GRB.Status.OPTIMAL)
                Wz_lo[linear_layer_count][j] = prog.ObjVal
                # Compute the bounds of
                # zᵢ₊₁(j) =((1−c)βᵢ(j)+c)*Wᵢ(j, :)*zᵢ
                if linear_layer_count < num_linear_layers - 1:
                    relu_layer = self.model[2 * linear_layer_count + 1]
                    if isinstance(relu_layer, nn.ReLU):
                        A_pre, A_z_next, A_beta_i, rhs_i = \
                            utils.replace_binary_continuous_product(
                                Wz_lo[linear_layer_count][j],
                                Wz_up[linear_layer_count][j],
                                dtype=self.dtype)
                    elif isinstance(relu_layer, nn.LeakyReLU):
                        A_pre, A_z_next, A_beta_i, rhs_i =\
                            utils.leaky_relu_gradient_times_x(
                                Wz_lo[linear_layer_count][j],
                                Wz_up[linear_layer_count][j],
                                relu_layer.negative_slope,
                                dtype=self.dtype)
                    A_z = A_pre.reshape(
                        (-1, 1)) @ linear_layer.weight.data[j, :].reshape(
                            (1, -1))
                    z[linear_layer_count +
                      1][j] = prog.addVar(lb=-gurobipy.GRB.INFINITY)
                    A_relu = torch.hstack((A_z, A_z_next.reshape(
                        (-1, 1)), A_beta_i.reshape((-1, 1))))
                    var_relu = z[linear_layer_count] + [
                        z[linear_layer_count + 1][j]
                    ] + [beta[self.relu_unit_index[linear_layer_count][j]]]
                    prog.addMConstrs(A_relu.detach().numpy(),
                                     var_relu,
                                     sense=gurobipy.GRB.LESS_EQUAL,
                                     b=rhs_i.detach().numpy().squeeze())
                    z_next_j_objective = gurobipy.LinExpr(
                        z[linear_layer_count + 1][j])
                    prog.setObjective(z_next_j_objective,
                                      sense=gurobipy.GRB.MAXIMIZE)
                    prog.optimize()
                    assert (prog.status == gurobipy.GRB.Status.OPTIMAL)
                    z_up[linear_layer_count + 1][j] = prog.ObjVal
                    prog.setObjective(z_next_j_objective,
                                      sense=gurobipy.GRB.MINIMIZE)
                    prog.optimize()
                    assert (prog.status == gurobipy.GRB.Status.OPTIMAL)
                    z_lo[linear_layer_count + 1][j] = prog.ObjVal
        return z_lo, z_up, Wz_lo, Wz_up, z

    def _compute_Wz_bounds_IA(self, vector_lower: torch.Tensor,
                              vector_upper: torch.Tensor):
        """
        This function is used in output_gradient_times_vector. It computes
        the lower/upper bounds of Wᵢ*z through interval arithemetics (IA)ᵢ,
        where zᵢ is defined recursively as
        vector_lower <= z₀ <= vector_upper
        zᵢ₊₁ = M(βᵢ, c)*Wᵢ*zᵢ, i = 0, ..., n-1    (1)
        and M is the matrix defined as
        M(β, c) = c*I + (1-c)*diag(β)
        Refer to output_gradient_times_vector() for more details.

        Returns:
            z_lo, z_up: z_lo[i]/z_up[i] is the lower/upper bound of zᵢ
            Wz_lo, Wz_up: Wz_lo[i]/Wz_up[i] is the lower/upper bound of Wᵢ*zᵢ
        """
        num_linear_layers = int((len(self.model) + 1) / 2)
        z_lo = [None] * num_linear_layers
        z_up = [None] * num_linear_layers
        z_lo[0] = vector_lower
        z_up[0] = vector_upper
        Wz_lo = [None] * num_linear_layers
        Wz_up = [None] * num_linear_layers
        for layer_count in range(num_linear_layers):
            # First compute the range of (Wᵢ*zᵢ)(j)
            linear_layer = self.model[2 * layer_count]
            assert (isinstance(linear_layer, torch.nn.Linear))
            Wizi_lo = torch.zeros(linear_layer.weight.shape[0],
                                  dtype=self.dtype)
            Wizi_up = torch.zeros(linear_layer.weight.shape[0],
                                  dtype=self.dtype)

            for j in range(linear_layer.weight.shape[0]):
                for k in range(linear_layer.weight.shape[1]):
                    if linear_layer.weight[j][k] > 0:
                        Wizi_lo[j] += linear_layer.weight[j][k] * z_lo[
                            layer_count][k]
                        Wizi_up[j] += linear_layer.weight[j][k] * z_up[
                            layer_count][k]
                    else:
                        Wizi_lo[j] += linear_layer.weight[j][k] * z_up[
                            layer_count][k]
                        Wizi_up[j] += linear_layer.weight[j][k] * z_lo[
                            layer_count][k]
            Wz_lo[layer_count] = Wizi_lo
            Wz_up[layer_count] = Wizi_up

            # Now compute the range of zᵢ₊₁ = M(βᵢ, c)*Wᵢ*zᵢ
            if layer_count < num_linear_layers - 1:
                relu_layer = self.model[2 * layer_count + 1]
                # M(βᵢ, c) is a diagonal matrix, with diagonal entries being
                # either c or 1. So the lower/upper bound of zᵢ₊₁ is either
                # the lower/upper bound of Wᵢ*zᵢ, or c times the lower/upper
                # bounds.
                if isinstance(relu_layer, nn.ReLU):
                    z_lo[layer_count + 1] = torch.min(
                        torch.zeros_like(Wz_lo[layer_count], dtype=self.dtype),
                        Wz_lo[layer_count])
                    z_up[layer_count + 1] = torch.max(
                        torch.zeros_like(Wz_up[layer_count], dtype=self.dtype),
                        Wz_up[layer_count])
                elif isinstance(relu_layer, nn.LeakyReLU):
                    c = relu_layer.negative_slope
                    assert (c >= 0)
                    z_lo[layer_count + 1], _ = torch.min(torch.cat(
                        (Wz_lo[layer_count].reshape(
                            (1, -1)), c * Wz_lo[layer_count].reshape((1, -1))),
                        dim=0),
                                                         axis=0)
                    z_up[layer_count + 1], _ = torch.max(torch.cat(
                        (Wz_up[layer_count].reshape(
                            (1, -1)), c * Wz_up[layer_count].reshape((1, -1))),
                        dim=0),
                                                         axis=0)
                else:
                    raise Exception(
                        "compute_Wz_bounds_IA(): this layer should be either" +
                        "ReLU or leaky ReLU")
        return z_lo, z_up, Wz_lo, Wz_up
