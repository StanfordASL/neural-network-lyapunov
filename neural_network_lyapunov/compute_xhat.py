"""
Sometimes we do not need all the states to converge to the equilibrium state
x*, instead we only want a partial state x[xhat_indices] to converge to
x*[xhat_indices]. To do so we define x̂ as a vector with the same dimension as
x, and x̂[i] = x*[i] if i is in xhat_indices, otherwise x̂[i] = x[i].
"""
import torch
import gurobipy
import neural_network_lyapunov.gurobi_torch_mip as gurobi_torch_mip
import neural_network_lyapunov.relu_to_optimization as relu_to_optimization
import neural_network_lyapunov.mip_utils as mip_utils


def _get_xhat_indices(x_dim, xhat_indices) -> tuple:
    """
    @return (xhat_indices, xhat_complement_indices)
    if xhat_indices is None, then return ([0, 1, ..., x_dim-1], [])
    else xhat_complement_indices are the indices in [0, 1, ..., x_dim-1] that
    are not in xhat_indices.
    """
    if xhat_indices is None:
        return list(range(x_dim)), []
    else:
        return xhat_indices, list(set(range(x_dim)) - set(xhat_indices))


def _get_xhat_val(x_val: torch.Tensor, x_equilibrium: torch.Tensor,
                  xhat_indices: list) -> torch.Tensor:
    """
    x̂ is a vector with the same dimension as x, x̂[i] = x*[i] if i is in
    xhat_indices, otherwise x̂[i] = x[i].
    @param xhat_indices x̂[i] = x*[i] if i is in xhat_indices, otherwise
    x̂[i] = x[i]. xhat_indices=None means x̂=x*
    """
    if xhat_indices is None:
        return x_equilibrium
    else:
        x_dim = x_equilibrium.shape[0]
        xhat_indices, xhat_comp_indices = _get_xhat_indices(
            x_dim, xhat_indices)
        if len(x_val.shape) == 1:
            xhat = x_equilibrium.clone()
            xhat[xhat_comp_indices] = x_val[xhat_comp_indices]
        elif len(x_val.shape) == 2:
            xhat = x_val.clone()
            xhat[:, xhat_indices] = x_equilibrium[xhat_indices]
        return xhat


def _get_xbar_indices(x_dim, xbar_indices):
    """
    If xbar_indices=None, then return [0, 1, ..., x_dim-1]
    else return xbar_indices
    """
    if xbar_indices is None:
        return list(range(x_dim))
    return xbar_indices


def _compute_network_at_xhat(
        mip: gurobi_torch_mip.GurobiTorchMIP, x_var: list, x_equilibrium,
        relu_free_pattern: relu_to_optimization.ReLUFreePattern,
        xhat_indices: list, x_lb: torch.Tensor, x_ub: torch.Tensor,
        method: mip_utils.PropagateBoundsMethod, lp_relaxation: bool):
    """
    Add the mixed-integer linear constraints between ϕ(x̂) and x, where
    x̂[i] = x*[i] if i is in xhat_indices, otherwise x̂[i] = x[i].
    @param mip The mixed-integer program to which the constraints are added.
    @param x_var The variables representing x in mip. These variables must
    have been registered to @p mip already.
    @param x_equilibrium x* in the documentation above.
    @param relu_free_pattern. The network ϕ is encoded in this object.
    @param xhat_indices The indices of entries in x̂ that equals to x*.
    @param x_lb The lower bound on the variable x
    @param x_ub The upper bound on the variable x
    """
    assert (isinstance(mip, gurobi_torch_mip.GurobiTorchMIP))
    assert (isinstance(relu_free_pattern,
                       relu_to_optimization.ReLUFreePattern))
    assert (isinstance(xhat_indices, list))
    assert (isinstance(method, mip_utils.PropagateBoundsMethod))
    assert (isinstance(lp_relaxation, bool))
    xhat = [var for var in x_var]
    xhat_lb = x_lb.clone()
    xhat_ub = x_ub.clone()
    for i in xhat_indices:
        xhat[i] = mip.addVars(1,
                              lb=-gurobipy.GRB.INFINITY,
                              ub=gurobipy.GRB.INFINITY,
                              vtype=gurobipy.GRB.CONTINUOUS,
                              name=f"xhat[{i}]")[0]
        xhat_lb[i] = x_equilibrium[i]
        xhat_ub[i] = x_equilibrium[i]
    mip_cnstr_return = relu_free_pattern.output_constraint(
        xhat_lb, xhat_ub, method)
    relu_z, relu_beta = mip.add_mixed_integer_linear_constraints(
        mip_cnstr_return, xhat, None, "relu_xhat_slack", "relu_xhat_binary",
        "relu_xhat_ineq", "relu_xhat_eq", "", lp_relaxation)
    return (relu_z, relu_beta, mip_cnstr_return.Aout_slack,
            mip_cnstr_return.Cout, xhat, mip_cnstr_return.nn_output_lo,
            mip_cnstr_return.nn_output_up)
