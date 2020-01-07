# -*- coding: utf-8 -*-
import queue

import torch
import torch.nn as nn
import numpy as np

import robust_value_approx.utils as utils


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
                layer.bias.reshape((-1, 1))
            num_linear_layer_output = layer.weight.shape[0]
        elif (isinstance(layer, nn.ReLU) or isinstance(layer, nn.LeakyReLU)):
            assert(len(activation_pattern[relu_layer_count])
                   == num_linear_layer_output)
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
        self.relu_unit_index = []
        self.num_relu_units = 0
        self.dtype = dtype
        layer_count = 0
        for layer in model:
            if (isinstance(layer, nn.Linear)):
                self.relu_unit_index.append(list(range(
                    self.num_relu_units,
                    self.num_relu_units + layer.weight.shape[0])))
                self.num_relu_units += layer.weight.shape[0]
                self.last_layer_is_relu = False
                if layer_count == 0:
                    self.x_size = layer.weight.shape[1]
                layer_count += 1
            elif isinstance(layer, nn.ReLU) or isinstance(layer, nn.LeakyReLU):
                self.last_layer_is_relu = True
            else:
                raise Exception("Only accept linear or relu layer")

        if not self.last_layer_is_relu:
            # The last linear layer is not connected to a ReLU layer.
            self.num_relu_units -= len(self.relu_unit_index[-1])
            self.relu_unit_index = self.relu_unit_index[:-1]

    def output_constraint(self, model, x_lo, x_up):
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
        ReLU(x) = aₒᵤₜᵀz + bₒᵤₜ
        where z, β are the "flat" column vectors, z = [z₁; z₂;...;zₙ],
        β = [β₀; β₁; ...; βₙ₋₁].
        @param model A ReLU network. This network must have the same structure
        as the network in the class constructor (but the weights can be
        different).
        @param x_lo A 1-D vector, the lower bound of input x.
        @param x_up A 1-D vector, the upper bound of input x.
        @return (Ain1, Ain2, Ain3, rhs_in, Aeq1, Aeq2, Aeq3, rhs_eq, a_out,
        b_out, z_pre_relu_lo, z_pre_relu_up)
        Ain1, Ain2, Ain3, Aeq1, Aeq2, Aeq3 are matrices, rhs_in, rhs_eq, a_out
        column vectors, b_out is a scalar. z_pre_relu_lo and z_pre_relu_up are
        1-D vectors.
        Notice that z_pre_relu_lo[i] and z_pre_relu_up[i] are the bounds of
        z[i] BEFORE applying the ReLU activation function, note these are NOT
        the bounds on z.
        """
        assert(x_lo.dtype == self.dtype)
        assert(x_up.dtype == self.dtype)
        assert(len(x_lo.shape) == 1)
        assert(len(x_up.shape) == 1)
        assert(torch.all(torch.le(x_lo, x_up)))

        # Each ReLU unit introduces at most 4 inequality constraints.
        Ain1 = torch.zeros((4 * self.num_relu_units + 2 * self.x_size,
                            self.x_size), dtype=self.dtype)
        Ain2 = torch.zeros((4 * self.num_relu_units + 2 * self.x_size,
                            self.num_relu_units), dtype=self.dtype)
        Ain3 = torch.zeros((4 * self.num_relu_units + 2 * self.x_size,
                            self.num_relu_units), dtype=self.dtype)
        rhs_in = torch.empty((4 * self.num_relu_units + 2 * self.x_size, 1),
                             dtype=self.dtype)
        # Each ReLU unit introduces at most 2 equality constraints.
        Aeq1 = torch.zeros((2 * self.num_relu_units, self.x_size),
                           dtype=self.dtype)
        Aeq2 = torch.zeros((2 * self.num_relu_units, self.num_relu_units),
                           dtype=self.dtype)
        Aeq3 = torch.zeros((2 * self.num_relu_units, self.num_relu_units),
                           dtype=self.dtype)
        rhs_eq = torch.empty((2 * self.num_relu_units, 1), dtype=self.dtype)

        eq_constraint_count = 0
        ineq_constraint_count = 0
        # First add the constraint x_lo <= x <= x_up
        Ain1[:self.x_size] = torch.eye(self.x_size, dtype=self.dtype)
        rhs_in[:self.x_size] = x_up.reshape((-1, 1))
        Ain1[self.x_size: 2*self.x_size] =\
            -torch.eye(self.x_size, dtype=self.dtype)
        rhs_in[self.x_size:2*self.x_size] = -x_lo.reshape((-1, 1))
        ineq_constraint_count = 2 * self.x_size
        layer_count = 0
        z_pre_relu_lo = [None] * self.num_relu_units
        z_pre_relu_up = [None] * self.num_relu_units
        z_post_relu_lo = [None] * self.num_relu_units
        z_post_relu_up = [None] * self.num_relu_units
        for layer in model:
            if (isinstance(layer, nn.Linear)):
                Wi = layer.weight
                bi = layer.bias
                if (layer_count < len(self.relu_unit_index)):
                    for j in range(len(self.relu_unit_index[layer_count])):
                        # First compute zᵤₚ, zₗₒ as the bounds for
                        # (Wᵢzᵢ)(j) + bᵢ(j)
                        z_bound_index = self.relu_unit_index[layer_count][j]
                        # z0 is the input x.
                        if layer_count == 0:
                            zi_size = self.x_size
                            zi_lo = x_lo.clone()
                            zi_up = x_up.clone()
                        else:
                            zi_size = len(
                                self.relu_unit_index[layer_count - 1])
                            # Note that z_post_relu_lo and z_post_relu_up are
                            # lists, not torch tensors (because Tensor[]
                            # operator is an inplace operator, which causes
                            # the pytorch autograd to fail. So we have to use
                            # a for loop below to copy z_post_relu_lo to zi_lo,
                            # and z_post_relu_up to zi_up.
                            zi_lo = torch.empty(len(self.relu_unit_index[
                                layer_count-1]), dtype=self.dtype)
                            zi_up = torch.empty(len(self.relu_unit_index[
                                layer_count-1]), dtype=self.dtype)
                            for k in range(len(self.relu_unit_index[
                                    layer_count-1])):
                                zi_lo[k] = z_post_relu_lo[self.relu_unit_index[
                                    layer_count-1][k]].clone()
                                zi_up[k] = z_post_relu_up[self.relu_unit_index[
                                    layer_count-1][k]].clone()
                        mask1 = torch.where(layer.weight[j] > 0)[0]
                        mask2 = torch.where(layer.weight[j] <= 0)[0]
                        z_pre_relu_lo[z_bound_index] = layer.weight[j][mask1] \
                            @ zi_lo[mask1].reshape((-1)) + \
                            layer.weight[j][mask2] @ \
                            zi_up[mask2].reshape((-1)) + layer.bias[j]
                        z_pre_relu_up[z_bound_index] = layer.weight[j][mask1] \
                            @ zi_up[mask1].reshape((-1)) + \
                            layer.weight[j][mask2] @ \
                            zi_lo[mask2].reshape((-1)) + layer.bias[j]
                        assert(z_pre_relu_lo[z_bound_index] <=
                               z_pre_relu_up[z_bound_index])

                else:
                    # This is for the output layer when the output layer
                    # doesn't have a ReLU unit.
                    assert(not self.last_layer_is_relu)
                    a_out = torch.zeros((self.num_relu_units,),
                                        dtype=self.dtype)
                    for k in range(len(self.relu_unit_index[layer_count - 1])):
                        a_out[self.relu_unit_index[layer_count - 1][k]] =\
                            layer.weight[0][k]
                    b_out = layer.bias.item()

            elif isinstance(layer, nn.ReLU) or isinstance(layer, nn.LeakyReLU):
                # The ReLU network can potentially change the bound on z.
                negative_slope = layer.negative_slope if\
                    isinstance(layer, nn.LeakyReLU) else 0.
                for j in range(len(self.relu_unit_index[layer_count])):
                    z_bound_index = self.relu_unit_index[layer_count][j]
                    if z_pre_relu_lo[z_bound_index] < 0 and\
                            z_pre_relu_up[z_bound_index] > 0:
                        if (isinstance(layer, nn.ReLU)):
                            (A_relu_input, A_relu_output, A_relu_beta,
                                relu_rhs) = utils.\
                                replace_relu_with_mixed_integer_constraint(
                                z_pre_relu_lo[z_bound_index],
                                z_pre_relu_up[z_bound_index], self.dtype)
                        elif isinstance(layer, nn.LeakyReLU):
                            A_relu_input, A_relu_output, A_relu_beta, relu_rhs\
                                = utils.\
                                replace_leaky_relu_mixed_integer_constraint(
                                    layer.negative_slope,
                                    z_pre_relu_lo[z_bound_index],
                                    z_pre_relu_up[z_bound_index], self.dtype)
                        if layer_count == 0:
                            # If this layer is the input layer, then the
                            # constraint is
                            # A_relu_input * ((Wᵢx)(j)+bᵢ(j))
                            # A_relu_output * zᵢ₊₁(j) +
                            # A_relu_beta * βᵢ(j) <= relu_rhs
                            Ain1[ineq_constraint_count:
                                 ineq_constraint_count + 4] =\
                                     A_relu_input.reshape((-1, 1))\
                                     @ Wi[j].reshape((1, -1))
                        else:
                            # If this layer is not the input layer, then
                            # the constraint is
                            # A_relu_input * ((Wᵢzᵢ)(j)+bᵢ(j)) +
                            # A_relu_output * zᵢ₊₁(j) +
                            # A_relu_beta * βᵢ(j) <= relu_rhs
                            Ain2[ineq_constraint_count:
                                 ineq_constraint_count+4,
                                 self.relu_unit_index[layer_count - 1]] =\
                                A_relu_input.reshape((-1, 1)) @\
                                Wi[j].reshape((1, -1))
                        Ain2[ineq_constraint_count:ineq_constraint_count+4,
                             self.relu_unit_index[layer_count][j]] =\
                            A_relu_output.squeeze()
                        Ain3[ineq_constraint_count:ineq_constraint_count+4,
                             self.relu_unit_index[layer_count][j]] =\
                            A_relu_beta.squeeze()
                        rhs_in[ineq_constraint_count: ineq_constraint_count
                               + 4] = relu_rhs.reshape((-1, 1)) -\
                            A_relu_input.reshape((-1, 1)) * bi[j]
                        ineq_constraint_count += 4
                        relu_unit_index_ij =\
                            self.relu_unit_index[layer_count][j]
                        if (negative_slope >= 0):
                            z_post_relu_lo[relu_unit_index_ij] =\
                                negative_slope * z_pre_relu_lo[
                                    relu_unit_index_ij]
                            z_post_relu_up[relu_unit_index_ij] = \
                                z_pre_relu_up[relu_unit_index_ij].clone()
                        else:
                            z_post_relu_lo[relu_unit_index_ij] = \
                                torch.tensor(0., dtype=self.dtype)
                            z_post_relu_up[relu_unit_index_ij] = torch.max(
                                negative_slope * z_pre_relu_lo[
                                    relu_unit_index_ij], z_pre_relu_up[
                                        relu_unit_index_ij])
                    elif z_pre_relu_lo[z_bound_index] >= 0:
                        # Case 2, introduce 2 equality constraints
                        # zᵢ₊₁(j) = (Wᵢzᵢ)(j) + bᵢ(j)
                        Aeq2[eq_constraint_count][
                            self.relu_unit_index[layer_count][j]] = 1.
                        if layer_count == 0:
                            Aeq1[eq_constraint_count] = -Wi[j]
                        else:
                            for k in range(zi_size):
                                Aeq2[eq_constraint_count][
                                    self.relu_unit_index[layer_count-1]
                                    [k]] = -Wi[j][k]
                        rhs_eq[eq_constraint_count] = bi[j].clone()
                        eq_constraint_count += 1
                        # βᵢ(j) = 1
                        Aeq3[eq_constraint_count][
                            self.relu_unit_index[layer_count][j]] = 1.
                        rhs_eq[eq_constraint_count] = 1.
                        eq_constraint_count += 1
                        relu_unit_index_ij =\
                            self.relu_unit_index[layer_count][j]
                        z_post_relu_lo[relu_unit_index_ij] =\
                            z_pre_relu_lo[relu_unit_index_ij]
                        z_post_relu_up[relu_unit_index_ij] =\
                            z_pre_relu_up[relu_unit_index_ij]
                    else:
                        # Case 3, introduce 2 equality constraints
                        # zᵢ₊₁(j) = negative_slope * ((Wᵢzᵢ)(j) + bᵢ(j))
                        if (isinstance(layer, nn.ReLU)):
                            Aeq2[eq_constraint_count][
                                self.relu_unit_index[layer_count][j]] = 1.
                            rhs_eq[eq_constraint_count] = 0.
                            eq_constraint_count += 1
                            # βᵢ(j) = 0
                            Aeq3[eq_constraint_count][
                                self.relu_unit_index[layer_count][j]] = 1.
                            rhs_eq[eq_constraint_count] = 0.
                            eq_constraint_count += 1
                        elif isinstance(layer, nn.LeakyReLU):
                            # zᵢ₊₁(j) = negative_slope * ((Wᵢzᵢ)(j) + bᵢ(j))
                            Aeq2[eq_constraint_count][
                                self.relu_unit_index[layer_count][j]] = 1.
                            if layer_count == 0:
                                Aeq1[eq_constraint_count] =\
                                    -layer.negative_slope * Wi[j]
                            else:
                                for k in range(zi_size):
                                    Aeq2[eq_constraint_count][
                                        self.relu_unit_index[layer_count-1]
                                        [k]] = -layer.negative_slope * Wi[j][k]
                            rhs_eq[eq_constraint_count] =\
                                layer.negative_slope * bi[j].clone()
                            eq_constraint_count += 1
                            # βᵢ(j) = 0
                            Aeq3[eq_constraint_count][
                                self.relu_unit_index[layer_count][j]] = 1.
                            rhs_eq[eq_constraint_count] = 0.
                            eq_constraint_count += 1
                        relu_unit_index_ij =\
                            self.relu_unit_index[layer_count][j]
                        if negative_slope >= 0:
                            z_post_relu_lo[relu_unit_index_ij] =\
                                negative_slope *\
                                z_pre_relu_lo[relu_unit_index_ij]
                            z_post_relu_up[relu_unit_index_ij] =\
                                negative_slope *\
                                z_pre_relu_up[relu_unit_index_ij]
                        else:
                            z_post_relu_lo[relu_unit_index_ij] =\
                                negative_slope *\
                                z_pre_relu_up[relu_unit_index_ij]
                            z_post_relu_up[relu_unit_index_ij] =\
                                negative_slope *\
                                z_pre_relu_lo[relu_unit_index_ij]

                layer_count += 1
        if self.last_layer_is_relu:
            a_out = torch.zeros((self.num_relu_units,), dtype=self.dtype)
            a_out[-1] = 1
            b_out = 0
        Ain1 = Ain1[:ineq_constraint_count]
        Ain2 = Ain2[:ineq_constraint_count]
        Ain3 = Ain3[:ineq_constraint_count]
        rhs_in = rhs_in[:ineq_constraint_count]
        Aeq1 = Aeq1[:eq_constraint_count]
        Aeq2 = Aeq2[:eq_constraint_count]
        Aeq3 = Aeq3[:eq_constraint_count]
        rhs_eq = rhs_eq[:eq_constraint_count]

        return(Ain1, Ain2, Ain3, rhs_in, Aeq1, Aeq2, Aeq3, rhs_eq, a_out,
               b_out, z_pre_relu_lo, z_pre_relu_up, z_post_relu_lo,
               z_post_relu_up)

    def compute_relu_unit_outputs_and_activation(self, model, x):
        """
        This is a utility function for output_constraint(). Given a network
        input x, this function computes the vector containing each ReLU unit
        output z, and the activation of each ReLU unit β.
        @param model A ReLU network. This network must have the same structure
        as the network in the class constructor (but the weights can be
        different).
        @param x The input to the ReLU network. A 1-D tensor.
        @return (z, β, output) z, β are column vectors. Refer to
        output_constraint()
        function for their definition.
        """
        z = torch.empty((self.num_relu_units, 1), dtype=self.dtype)
        beta = torch.empty((self.num_relu_units, 1), dtype=self.dtype)
        relu_unit_count = 0
        z_layer = x
        for layer in model:
            if (isinstance(layer, nn.Linear)):
                z_layer = layer.forward(z_layer)
            elif (isinstance(layer, nn.ReLU) or
                  isinstance(layer, nn.LeakyReLU)):
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
        output = z_layer.item()

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
        assert(len(relu_unit_layer_indices) == len(self.relu_unit_index))
        index = 0
        for i in range(len(self.relu_unit_index)):
            # Go through each layer
            assert (relu_unit_layer_indices[i] >= 0 and
                    relu_unit_layer_indices[i] < len(
                self.relu_unit_index[i]))
            index = index * \
                len(self.relu_unit_index[i]) + relu_unit_layer_indices[i]
        return index

    def output_gradient(self, model):
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

        @param model A ReLU network. This network must have the same structure
        as the network in the class constructor (but the weights can be
        different).
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
        num_alpha = np.prod(np.array(
            [len(layer_relu_unit_index) for layer_relu_unit_index
             in self.relu_unit_index]))

        layer_linear_unit_gradients = queue.Queue(maxsize=num_alpha)
        layer_count = 0
        for layer in model:
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
                                (layer_relu_unit_index,),
                                layer.weight[:, layer_relu_unit_index].
                                reshape((-1, 1))
                                @ last_layer_linear_unit_gradient.
                                gradient[layer_relu_unit_index].
                                reshape((1, -1)))
                            layer_linear_unit_gradients.put(
                                linear_unit_gradient)

            elif (isinstance(layer, nn.ReLU)):
                layer_count += 1
            else:
                raise Exception("output_gradient: currently only " +
                                "supports linear and ReLU layers.")

        if self.last_layer_is_relu:
            # Now append 0 to the end of the beta index (the last layer has
            # only one ReLU unit, so its index is always 0).
            queue_size = layer_linear_unit_gradients.qsize()
            last_layer_linear_unit_gradient_count = 0
            while (last_layer_linear_unit_gradient_count < queue_size):
                last_layer_linear_unit_gradient =\
                    layer_linear_unit_gradients.get()
                last_layer_linear_unit_gradient_count += 1
                linear_unit_gradient = LinearUnitGradient(
                    last_layer_linear_unit_gradient.
                    activated_relu_indices + (0,),
                    last_layer_linear_unit_gradient.gradient)
                layer_linear_unit_gradients.put(linear_unit_gradient)

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

    def output_gradient_times_vector(self, model, vector_lower, vector_upper):
        """
        We want to compute the gradient of the network ∂ReLU(x)/∂x times a
        vector y: ∂ReLU(x)/∂x  * y, and reformulate this product as
        mixed-integer linear constraints.
        We assume that the leaky relu unit has negative slope c (c = 0 for
        ReLU unit). And we define a matrix
        M(β, c) = c + (1-c)*diag(β)
        We notice that
        ∂ReLU(x)/∂x = wₙᵀ * M(βₙ₋₁, c) +  * Wₙ₋₁ * M(βₙ₋₂, c) * Wₙ₋₂ * ...
                      * M(β₀, c) * W₀
        So we introduce slack variable z, such that
        z₀ = y
        zᵢ₊₁ = M(βᵢ, c)*Wᵢ*zᵢ, i = 0, ..., n-1    (1)
        The constraint (1) can be replaced by mixed-integer linear constraint
        on zᵢ and βᵢ.
        If the ReLU network has a ReLU unit for the output layer, then
        ∂ReLU(x)/∂x * y = zₙ₊₁
        otherwise
        ∂ReLU(x)/∂x * y = wₙᵀzₙ

        We write ∂ReLU(x)/∂x * y as
        aₒᵤₜᵀz
        with the additional constraint
        A_y * y + A_z * z + A_beta * β ≤ rhs
        where z = [z₁; z₂;...;zₙ]
        Note that we do NOT require that β is the right activation pattern for
        the input x. This constraint should be imposed in output() function.
        @param model A ReLU network. This network must have the same structure
        as the network in the class constructor (but the weights can be
        different).
        @param vector_lower The lower bound of the vector y.
        @param vector_upper The upper bound of the vector y.
        @return (a_out, A_y, A_z, A_beta, rhs, z_lo, z_up) z_lo and z_up are
        the propagated bounds on z, based on the bounds on y.
        """
        utils.check_shape_and_type(vector_lower, (self.x_size,), self.dtype)
        utils.check_shape_and_type(vector_upper, (self.x_size,), self.dtype)
        A_y = torch.zeros((4 * self.num_relu_units, self.x_size),
                          dtype=self.dtype)
        A_z = torch.zeros((4 * self.num_relu_units, self.num_relu_units),
                          dtype=self.dtype)
        A_beta = torch.zeros((4 * self.num_relu_units, self.num_relu_units),
                             dtype=self.dtype)
        rhs = torch.zeros(4 * self.num_relu_units, dtype=self.dtype)
        z_lo = torch.empty(self.num_relu_units, dtype=self.dtype)
        z_up = torch.empty(self.num_relu_units, dtype=self.dtype)
        layer_count = 0
        zi_lo = vector_lower
        zi_up = vector_upper
        ineq_count = 0
        a_out = torch.zeros(self.num_relu_units, dtype=self.dtype)
        if self.last_layer_is_relu:
            a_out[self.relu_unit_index[-1]] = 1
        for layer in model:
            if (isinstance(layer, nn.Linear)):
                if layer_count == len(self.relu_unit_index) and not\
                        self.last_layer_is_relu:
                    # If this is the last linear layer, and this linear layer
                    # is the output layer, then we set a_out. Otherwise, we
                    # append inequality constraints.
                    a_out[self.relu_unit_index[layer_count - 1]] =\
                        layer.weight[0]
                else:
                    # First compute the range of (Wᵢ*zᵢ)(j)
                    Wizi_lo = torch.zeros(layer.weight.shape[0],
                                          dtype=self.dtype)
                    Wizi_up = torch.zeros(layer.weight.shape[0],
                                          dtype=self.dtype)

                    # Now impose the constraint zᵢ₊₁ = diag(βᵢ)*Wᵢ*zᵢ
                    # We do this by replacing zᵢ₊₁(j) = βᵢ(j)*(Wᵢ*zᵢ)(j) with
                    # mixed-integer linear constraints.
                    for j in range(layer.weight.shape[0]):
                        for k in range(layer.weight.shape[1]):
                            if layer.weight[j][k] > 0:
                                Wizi_lo[j] += layer.weight[j][k] * zi_lo[k]
                                Wizi_up[j] += layer.weight[j][k] * zi_up[k]
                            else:
                                Wizi_lo[j] += layer.weight[j][k] * zi_up[k]
                                Wizi_up[j] += layer.weight[j][k] * zi_lo[k]
                        (A_pre, A_z_next, A_beta_i, rhs_i) =\
                            utils.replace_binary_continuous_product(
                                Wizi_lo[j], Wizi_up[j], dtype=self.dtype)
                        if layer_count == 0:
                            A_y[ineq_count:ineq_count+4] =\
                                A_pre.reshape((-1, 1)) @\
                                layer.weight[j].reshape((1, -1))
                        else:
                            A_z[ineq_count:ineq_count+4,
                                self.relu_unit_index[layer_count - 1]] =\
                                A_pre.reshape((-1, 1)) @\
                                layer.weight[j].reshape((1, -1))
                        A_z[ineq_count:ineq_count+4,
                            self.relu_unit_index[layer_count][j]] =\
                            A_z_next.squeeze()
                        A_beta[ineq_count:ineq_count+4,
                               self.relu_unit_index[layer_count][j]] =\
                            A_beta_i.squeeze()
                        rhs[ineq_count:ineq_count+4] = rhs_i
                        ineq_count += 4
            elif(isinstance(layer, nn.ReLU)):
                zi_lo = torch.min(
                    torch.zeros(len(self.relu_unit_index[layer_count]),
                                dtype=self.dtype),
                    Wizi_lo)
                zi_up = torch.max(
                    torch.zeros(len(self.relu_unit_index[layer_count]),
                                dtype=self.dtype),
                    Wizi_up)
                z_lo[self.relu_unit_index[layer_count]] = zi_lo
                z_up[self.relu_unit_index[layer_count]] = zi_up
                layer_count += 1
            else:
                raise Exception("output_gradient_times_vector: we currently " +
                                "only support linear and ReLU units.")
        return (a_out, A_y, A_z, A_beta, rhs, z_lo, z_up)
