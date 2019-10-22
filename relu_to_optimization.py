# -*- coding: utf-8 -*-
import queue

import torch
import torch.nn as nn
import numpy as np

import utils


def ComputeReLUActivationPattern(model_relu, x):
    """
    For a given input x to a ReLU network, returns the activation pattern for
    this input.
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
        elif isinstance(layer, nn.ReLU):
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
            A_layer = layer.weight.data @ A_layer
            b_layer = layer.weight.data @ b_layer + \
                layer.bias.data.reshape((-1, 1))
            num_linear_layer_output = layer.weight.data.shape[0]
        elif (isinstance(layer, nn.ReLU)):
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
                    A_layer[row] = 0
                    b_layer[row] = 0
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
                    self.num_relu_units + layer.weight.data.shape[0])))
                self.num_relu_units += layer.weight.data.shape[0]
                if layer_count == 0:
                    self.x_size = layer.weight.data.shape[1]
                layer_count += 1

        # The last linear layer is not connected to a ReLU layer.
        self.num_relu_units -= len(self.relu_unit_index[-1])
        self.relu_unit_index = self.relu_unit_index[:-1]

    def output_constraint(self, model, x_lo, x_up):
        """
        The output of ReLU network is a piecewise linear function of the input.
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
        We will write the constraint in a more concise way
        Ain1 * x + Ain2 * z + Ain3 * β <= rhs_in  (case 1)
        Aeq1 * x + Aeq2 * z + Aeq3 * β = rhs_eq   (case 2 and 3)
        ReLU(x) = aₒᵤₜᵀz + bₒᵤₜ
        where z, β are the "flat" column vectors, z = [z₁; z₂;...;zₙ],
        β = [β₀; β₁; ...; βₙ₋₁]. Note that the network output zₙ is the LAST
        entry of z.
        @param model A ReLU network. This network must have the same structure
        as the network in the class constructor (but the weights can be
        different).
        @param x_lo A 1-D vector, the lower bound of input x.
        @param x_up A 1-D vector, the upper bound of input x.
        @return (Ain1, Ain2, Ain3, rhs_in, Aeq1, Aeq2, Aeq3, rhs_eq, a_out,
        b_out, z_lo, z_up)
        Ain1, Ain2, Ain3, Aeq1, Aeq2, Aeq3 are matrices, rhs_in, rhs_eq, a_out
        column vectors, b_out is a scalar. z_lo and z_up are 1-D vectors.
        Notice that z_lo[i] and z_up[i] are the bounds of z[i] BEFORE
        applying the ReLU activation function.
        """
        assert(x_lo.dtype == self.dtype)
        assert(x_up.dtype == self.dtype)
        assert(len(x_lo.shape) == 1)
        assert(len(x_up.shape) == 1)
        assert(torch.all(torch.le(x_lo, x_up)))

        # Each ReLU unit introduces at most 4 inequality constraints.
        Ain1 = torch.zeros((4 * self.num_relu_units, self.x_size),
                           dtype=self.dtype)
        Ain2 = torch.zeros((4 * self.num_relu_units, self.num_relu_units),
                           dtype=self.dtype)
        Ain3 = torch.zeros((4 * self.num_relu_units, self.num_relu_units),
                           dtype=self.dtype)
        rhs_in = torch.empty((4 * self.num_relu_units, 1), dtype=self.dtype)
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
        layer_count = 0
        z_lo = torch.empty(self.num_relu_units, dtype=self.dtype)
        z_up = torch.empty(self.num_relu_units, dtype=self.dtype)
        z_lo_after = torch.empty(self.num_relu_units, dtype=self.dtype)
        z_up_after = torch.empty(self.num_relu_units, dtype=self.dtype)
        for layer in model:
            if (isinstance(layer, nn.Linear)):
                if (layer_count < len(self.relu_unit_index)):
                    for j in range(len(self.relu_unit_index[layer_count])):
                        # First compute zᵤₚ, zₗₒ as the bounds for
                        # (Wᵢzᵢ)(j) + bᵢ(j)
                        z_bound_index = self.relu_unit_index[layer_count][j]
                        z_lo[z_bound_index] = layer.bias.data[j].clone()
                        z_up[z_bound_index] = layer.bias.data[j].clone()
                        # z0 is the input x.
                        if layer_count == 0:
                            zi_size = self.x_size
                            zi_lo = x_lo
                            zi_up = x_up
                        else:
                            zi_size = len(
                                self.relu_unit_index[layer_count - 1])
                            zi_lo = [z_lo_after[z_index]
                                     for z_index in
                                     self.relu_unit_index[layer_count - 1]]
                            zi_up = [z_up_after[z_index]
                                     for z_index in
                                     self.relu_unit_index[layer_count - 1]]
                        for k in range(zi_size):
                            if (layer.weight.data[j][k] > 0):
                                z_lo[z_bound_index] += \
                                    layer.weight.data[j][k] * zi_lo[k]
                                z_up[z_bound_index] +=\
                                    layer.weight.data[j][k] * zi_up[k]
                            else:
                                z_lo[z_bound_index] +=\
                                    layer.weight.data[j][k] * zi_up[k]
                                z_up[z_bound_index] +=\
                                    layer.weight.data[j][k] * zi_lo[k]
                        assert(z_lo[z_bound_index] <= z_up[z_bound_index])
                        if z_lo[z_bound_index] < 0 and z_up[z_bound_index] > 0:
                            (A_relu_input, A_relu_output, A_relu_beta,
                                relu_rhs) = utils.\
                                replace_relu_with_mixed_integer_constraint(
                                z_lo[z_bound_index],
                                z_up[z_bound_index], self.dtype)
                            if layer_count == 0:
                                # If this layer is the input layer, then the
                                # constraint is
                                # A_relu_input * ((Wᵢx)(j)+bᵢ(j))
                                # A_relu_output * zᵢ₊₁(j) +
                                # A_relu_beta * βᵢ(j) <= relu_rhs
                                Ain1[ineq_constraint_count:
                                     ineq_constraint_count + 4] = A_relu_input\
                                    @ layer.weight.data[j].reshape((1, -1))
                            else:
                                # If this layer is not the input layer, then
                                # the constraint is
                                # A_relu_input * ((Wᵢzᵢ)(j)+bᵢ(j)) +
                                # A_relu_output * zᵢ₊₁(j) +
                                # A_relu_beta * βᵢ(j) <= relu_rhs
                                Ain2[ineq_constraint_count:
                                     ineq_constraint_count+4,
                                     self.relu_unit_index[layer_count - 1]] =\
                                    A_relu_input @\
                                    layer.weight.data[j].reshape((1, -1))
                            Ain2[ineq_constraint_count:ineq_constraint_count+4,
                                 self.relu_unit_index[layer_count][j]] =\
                                A_relu_output.squeeze()
                            Ain3[ineq_constraint_count:ineq_constraint_count+4,
                                 self.relu_unit_index[layer_count][j]] =\
                                A_relu_beta.squeeze()
                            rhs_in[ineq_constraint_count: ineq_constraint_count
                                   + 4] =\
                                relu_rhs - A_relu_input * layer.bias.data[j]
                            ineq_constraint_count += 4
                        elif z_lo[z_bound_index] >= 0:
                            # Case 2, introduce 2 equality constraints
                            # zᵢ₊₁(j) = (Wᵢzᵢ)(j) + bᵢ(j)
                            Aeq2[eq_constraint_count][
                                self.relu_unit_index[layer_count][j]] = 1.
                            if layer_count == 0:
                                Aeq1[eq_constraint_count] = - \
                                    layer.weight.data[j]
                            else:
                                for k in range(zi_size):
                                    Aeq2[eq_constraint_count][
                                        self.relu_unit_index[layer_count-1]
                                        [k]] = -layer.weight.data[j][k]
                            rhs_eq[eq_constraint_count] =\
                                layer.bias.data[j].clone()
                            eq_constraint_count += 1
                            # βᵢ(j) = 1
                            Aeq3[eq_constraint_count][
                                self.relu_unit_index[layer_count][j]] = 1.
                            rhs_eq[eq_constraint_count] = 1.
                            eq_constraint_count += 1
                        else:
                            # Case 3, introduce 2 equality constraints
                            # zᵢ₊₁(j) = 0
                            Aeq2[eq_constraint_count][
                                self.relu_unit_index[layer_count][j]] = 1.
                            rhs_eq[eq_constraint_count] = 0.
                            eq_constraint_count += 1
                            # βᵢ(j) = 0
                            Aeq3[eq_constraint_count][
                                self.relu_unit_index[layer_count][j]] = 1.
                            rhs_eq[eq_constraint_count] = 0.
                            eq_constraint_count += 1

                else:
                    # output layer.
                    a_out = torch.zeros((self.num_relu_units, 1),
                                        dtype=self.dtype)
                    for k in range(len(self.relu_unit_index[layer_count - 1])):
                        a_out[self.relu_unit_index[layer_count - 1][k]
                              ][0] = layer.weight.data[0][k]
                    b_out = layer.bias.item()

            elif (isinstance(layer, nn.ReLU)):
                # The ReLU network can potentially change the bound on z.
                for j in range(len(self.relu_unit_index[layer_count])):
                    z_lo_after[self.relu_unit_index[layer_count][j]] =\
                        torch.max(
                        z_lo[self.relu_unit_index[layer_count][j]],
                        torch.tensor(0., dtype=self.dtype))
                    z_up_after[self.relu_unit_index[layer_count][j]] =\
                        torch.max(
                        z_up[self.relu_unit_index[layer_count][j]],
                        torch.tensor(0., dtype=self.dtype))

                layer_count += 1
        Ain1 = Ain1[:ineq_constraint_count]
        Ain2 = Ain2[:ineq_constraint_count]
        Ain3 = Ain3[:ineq_constraint_count]
        rhs_in = rhs_in[:ineq_constraint_count]
        Aeq1 = Aeq1[:eq_constraint_count]
        Aeq2 = Aeq2[:eq_constraint_count]
        Aeq3 = Aeq3[:eq_constraint_count]
        rhs_eq = rhs_eq[:eq_constraint_count]

        return(Ain1, Ain2, Ain3, rhs_in, Aeq1, Aeq2, Aeq3, rhs_eq, a_out,
               b_out, z_lo, z_up)

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
            elif (isinstance(layer, nn.ReLU)):
                for i in range(z_layer.numel()):
                    beta[relu_unit_count + i][0] = 1 if z_layer[i] > 0 else 0
                z_layer = layer.forward(z_layer)
                z[relu_unit_count:relu_unit_count +
                    z_layer.numel()] = z_layer.reshape((-1, 1))
                relu_unit_count += z_layer.numel()
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
        This is a multilinear function of β, and we will introduce new binary
        variable α to represent the product of β.
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

        @param model A ReLU network
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
                        (), layer.weight.data.clone())
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
                                layer.weight.data[:, layer_relu_unit_index].
                                reshape((-1, 1))
                                @ last_layer_linear_unit_gradient.
                                gradient[layer_relu_unit_index].
                                reshape((1, -1)))
                            layer_linear_unit_gradients.put(
                                linear_unit_gradient)

            elif (isinstance(layer, nn.ReLU)):
                layer_count += 1

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
