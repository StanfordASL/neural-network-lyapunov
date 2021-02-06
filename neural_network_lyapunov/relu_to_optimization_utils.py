import torch
import torch.nn as nn
import gurobipy
import neural_network_lyapunov.utils as utils
import neural_network_lyapunov.mip_utils as mip_utils
import neural_network_lyapunov.gurobi_torch_mip as gurobi_torch_mip


def _add_constraint_by_neuron(
    Wij: torch.Tensor,
    bij: torch.Tensor,
    relu_layer,
    neuron_input_lo: torch.Tensor,
    neuron_input_up: torch.Tensor,
):
    """
    This function adds the constraint on
    zᵢ₊₁(j) = leaky_relu((Wᵢzᵢ+bᵢ)(j))
    between zᵢ₊₁(j), zᵢ, and the (binary) slack variables.
    """
    dtype = Wij.dtype
    assert (len(Wij.shape) == 1)
    if neuron_input_lo < 0 and neuron_input_up > 0:
        # The ReLU unit can be either active or inactive. Add the mixed-integer
        # linear constraints the ReLU unit.
        if isinstance(relu_layer, nn.ReLU):
            A_relu_input, A_relu_output, A_relu_beta, relu_rhs = \
                utils.replace_relu_with_mixed_integer_constraint(
                    neuron_input_lo, neuron_input_up, dtype)
        elif isinstance(relu_layer, nn.LeakyReLU):
            A_relu_input, A_relu_output, A_relu_beta, relu_rhs = \
                utils.replace_leaky_relu_mixed_integer_constraint(
                    relu_layer.negative_slope, neuron_input_lo,
                    neuron_input_up, dtype)
        # The constraint is A_relu_input * ((Wᵢzᵢ)(j)+bᵢ(j)) +
        # A_relu_output * zᵢ₊₁(j) + A_relu_beta * βᵢ(j) <= relu_rhs
        Ain_linear_input = A_relu_input.reshape((-1, 1)) @ Wij.reshape((1, -1))
        Ain_neuron_output = A_relu_output.reshape((-1, 1))
        Ain_binary = A_relu_beta.reshape((-1, 1))
        rhs_in = relu_rhs - A_relu_input * bij
        Aeq_linear_input = torch.empty((0, Wij.numel()), dtype=dtype)
        Aeq_neuron_output = torch.empty((0, 1), dtype=dtype)
        Aeq_binary = torch.empty((0, 1), dtype=dtype)
        rhs_eq = torch.empty((0, ), dtype=dtype)
    else:
        # The (leaky) ReLU is always active, or always inactive. If
        # the lower bound output_lo[j] >= 0, then it is always active,
        # and we add two linear equality constraints
        # zᵢ₊₁(j) = (Wᵢzᵢ)(j) + bᵢ(j) and βᵢ(j) = 1
        # If the upper bound output_up[j] <= 0, then it is always
        # inactive, and we add to linear equality constraints
        # zᵢ₊₁(j) = c*((Wᵢzᵢ)(j) + bᵢ(j)) and βᵢ(j) = 0
        if neuron_input_lo >= 0:
            slope = 1.
            binary_value = 1
        elif neuron_input_up <= 0:
            slope = relu_layer.negative_slope if isinstance(
                relu_layer, nn.LeakyReLU) else 0.
            binary_value = 0.
        Aeq_linear_input = torch.cat((-slope * Wij.reshape(
            (1, -1)), torch.zeros((1, Wij.numel()), dtype=dtype)),
                                     dim=0)
        Aeq_neuron_output = torch.tensor([[1.], [0]], dtype=dtype)
        Aeq_binary = torch.tensor([[0.], [1.]], dtype=dtype)
        rhs_eq = torch.stack(
            (slope * bij, torch.tensor(binary_value, dtype=dtype)))
        Ain_linear_input = torch.empty((0, Wij.numel()), dtype=dtype)
        Ain_neuron_output = torch.empty((0, 1), dtype=dtype)
        Ain_binary = torch.empty((0, 1), dtype=dtype)
        rhs_in = torch.empty((0, ), dtype=dtype)
    neuron_output_lo, neuron_output_up = mip_utils.propagate_bounds(
        relu_layer, neuron_input_lo, neuron_input_up)
    return Ain_linear_input, Ain_neuron_output, Ain_binary, rhs_in,\
        Aeq_linear_input, Aeq_neuron_output, Aeq_binary, rhs_eq,\
        neuron_output_lo, neuron_output_up


def _add_constraint_by_layer(linear_layer, relu_layer,
                             linear_output_lo: torch.Tensor,
                             linear_output_up: torch.Tensor):
    """
    This function will be called inside output_constraint(). We group the
    network layers by a linear layer followed by a (leaky) ReLU layer,
    these two layers together impose a condition
    zᵢ₊₁ = leaky_relu(Wᵢzᵢ+bᵢ)
    We add this condition as mixed-integer linear constraints between zᵢ,
    zᵢ₊₁, and the binary variable representing the activation of ReLU
    units.
    @param linear_output_lo The lower bound of the linear layer output.
    @param linear_output_up The upper bound of the linear layer output.
    """
    assert (isinstance(linear_layer, nn.Linear))
    assert (isinstance(linear_output_lo, torch.Tensor))
    assert (isinstance(linear_output_up, torch.Tensor))
    dtype = linear_layer.weight.data.dtype
    Ain_z_curr = []
    Ain_z_next = []
    Ain_binary = []
    rhs_in = []
    Aeq_z_curr = []
    Aeq_z_next = []
    Aeq_binary = []
    rhs_eq = []
    z_next_lo = []
    z_next_up = []
    bias = linear_layer.bias if linear_layer.bias is not None else \
        torch.zeros((linear_layer.out_features,), dtype=dtype)
    for j in range(linear_layer.out_features):
        Ain_linear_input, Ain_neuron_output, Ain_binary_j, rhs_in_j,\
            Aeq_linear_input, Aeq_neuron_output, Aeq_binary_j, rhs_eq_j,\
            neuron_output_lo, neuron_output_up = _add_constraint_by_neuron(
                linear_layer.weight[j], bias[j], relu_layer,
                linear_output_lo[j], linear_output_up[j])
        Ain_z_curr.append(Ain_linear_input)
        Ain_z_next.append(
            torch.zeros((rhs_in_j.numel(), linear_layer.out_features),
                        dtype=dtype))
        Ain_z_next[-1][:, j] = Ain_neuron_output.reshape((-1, ))
        Ain_binary.append(
            torch.zeros((rhs_in_j.numel(), linear_layer.out_features),
                        dtype=dtype))
        Ain_binary[-1][:, j] = Ain_binary_j.reshape((-1, ))
        rhs_in.append(rhs_in_j)
        Aeq_z_curr.append(Aeq_linear_input)
        Aeq_z_next.append(
            torch.zeros((rhs_eq_j.numel(), linear_layer.out_features),
                        dtype=dtype))
        Aeq_z_next[-1][:, j] = Aeq_neuron_output.reshape((-1, ))
        Aeq_binary.append(
            torch.zeros((rhs_eq_j.numel(), linear_layer.out_features),
                        dtype=dtype))
        Aeq_binary[-1][:, j] = Aeq_binary_j.reshape((-1, ))
        rhs_eq.append(rhs_eq_j)
        z_next_lo.append(neuron_output_lo.squeeze())
        z_next_up.append(neuron_output_up.squeeze())
    return torch.cat(Ain_z_curr, dim=0), torch.cat(Ain_z_next, dim=0),\
        torch.cat(Ain_binary, dim=0), torch.cat(rhs_in, dim=0),\
        torch.cat(Aeq_z_curr, dim=0), torch.cat(Aeq_z_next, dim=0),\
        torch.cat(Aeq_binary, dim=0), torch.cat(rhs_eq, dim=0),\
        torch.stack(z_next_lo), torch.stack(z_next_up)


def _add_linear_relaxation_by_layer(prog: gurobi_torch_mip.GurobiTorchMIP,
                                    linear_layer, relu_layer,
                                    linear_input_var: list,
                                    relu_input_lo: torch.Tensor,
                                    relu_input_up: torch.Tensor):
    """
    Add the linear relaxation of constraint output = relu(W * input + b)
    The relu input W*input+b is within the bound relu_input_lo to relu_input_up
    """
    assert (isinstance(prog, gurobi_torch_mip.GurobiTorchMIP))
    assert (isinstance(relu_input_lo, torch.Tensor))
    assert (isinstance(relu_input_up, torch.Tensor))
    relu_output_var = prog.addVars(linear_layer.out_features,
                                   lb=-gurobipy.GRB.INFINITY)
    binary_relax = prog.addVars(linear_layer.out_features, lb=0., ub=1.)
    for j in range(linear_layer.out_features):
        bij = torch.tensor(
            0., dtype=linear_layer.weight.dtype
        ) if linear_layer.bias is None else linear_layer.bias[j]
        Ain_linear_input, Ain_neuron_output, Ain_neuron_binary, rhs_in,\
            Aeq_linear_input, Aeq_neuron_output, Aeq_neuron_binary, rhs_eq, _,\
            _ = _add_constraint_by_neuron(
                linear_layer.weight[j], bij, relu_layer, relu_input_lo[j],
                relu_input_up[j])
        prog.addMConstrs(
            [Ain_linear_input, Ain_neuron_output, Ain_neuron_binary],
            [linear_input_var, [relu_output_var[j]], [binary_relax[j]]],
            b=rhs_in,
            sense=gurobipy.GRB.LESS_EQUAL)
        prog.addMConstrs(
            [Aeq_linear_input, Aeq_neuron_output, Aeq_neuron_binary],
            [linear_input_var, [relu_output_var[j]], [binary_relax[j]]],
            b=rhs_eq,
            sense=gurobipy.GRB.EQUAL)
    return relu_output_var, binary_relax
