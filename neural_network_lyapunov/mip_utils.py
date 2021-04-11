import torch
import numpy as np
import gurobipy
import enum
import warnings


def strengthen_leaky_relu_mip_constraint(c: float, w: torch.Tensor,
                                         b: torch.Tensor, lo: torch.Tensor,
                                         up: torch.Tensor, indices: set):
    """
    We strengthen the big-M formulation of the leaky ReLU unit
    y = max(c*wᵀx+b, wᵀx+b), lo <= x <= up
    with the constraint
    y <= bc + b(1-c)β + ∑ i∈ℑ (wᵢxᵢ−(1−c)(1−β)wᵢL̅ᵢ) + ∑i∉ℑ(cwᵢxᵢ+(1−c)βwᵢU̅ᵢ)
    Refer to the pdf file from doc/ideal_formulation.tex for more details.
    Note that this function should be called only when the input bounds to the
    leaky ReLU unit includes 0, that we have to impose a mixed-integer linear
    constraints with a binary variable to indicate the activeness of the
    neuron.
    @param c The negative slope of the leaky relu unit.
    @param w A vector
    @param b a scalar
    @param lo The lower bound of x.
    @param up The upper bound of x.
    @param indices The set equals to the index set ℑ.
    @return (x_coeff, binary_coeff, y_coeff, rhs) We return the constraint as
    x_coeff * x + binary_coeff * β + y_coeff * y <= rhs
    """
    assert (isinstance(c, float))
    assert (c >= 0 and c < 1)
    assert (isinstance(w, torch.Tensor))
    assert (len(w.shape) == 1)
    assert (isinstance(b, torch.Tensor))
    assert (len(b.shape) == 0)
    assert (isinstance(lo, torch.Tensor))
    assert (isinstance(up, torch.Tensor))
    assert (lo.shape == w.shape)
    assert (up.shape == lo.shape)
    assert (isinstance(indices, set))
    dtype = w.dtype
    rhs = b * c
    y_coeff = torch.tensor(1, dtype=dtype)
    nx = w.shape[0]
    not_indices = set(range(nx)) - indices
    x_coeff = torch.zeros((nx, ), dtype=dtype)
    binary_coeff = -b * (1 - c)
    for i in indices:
        if w[i] >= 0:
            lo_bar_i = lo[i]
        else:
            lo_bar_i = up[i]
        x_coeff[i] -= w[i]
        rhs += -(1 - c) * w[i] * lo_bar_i
        binary_coeff -= (1 - c) * w[i] * lo_bar_i
    for i in not_indices:
        if w[i] >= 0:
            up_bar_i = up[i]
        else:
            up_bar_i = lo[i]
        x_coeff[i] -= c * w[i]
        binary_coeff -= (1 - c) * w[i] * up_bar_i
    return (x_coeff, binary_coeff, y_coeff, rhs)


def compute_range_by_lp(A: np.ndarray, b: np.ndarray, x_lb: np.ndarray,
                        x_ub: np.ndarray, C: np.ndarray,
                        d: np.ndarray) -> (np.ndarray, np.ndarray):
    """
    Given x_lb <= x <= x_ub and C * x <= d, compute the range of y = A * x + b
    through linear programming
    Note that if C * x <= d is not imposed, then this function returns the
    same bound as interval arithmetics (because the optimal solution of
    an LP occurs at the vertices of the bounding box x_lb <= x <= x_ub).
    @param x_lb Either np.ndarray or None
    @param x_ub Either np.ndarray or None
    @param C Either np.ndarray or None
    @param d Either np.ndarray or None
    """
    assert (isinstance(A, np.ndarray))
    assert (isinstance(b, np.ndarray))
    assert (isinstance(x_lb, np.ndarray) or x_lb is None)
    assert (isinstance(x_ub, np.ndarray) or x_ub is None)
    assert (isinstance(C, np.ndarray) or C is None)
    assert (isinstance(d, np.ndarray) or d is None)

    if (C is None and d is None):
        warnings.warn(
            "Compute_range_by_lp with empty C*x<=d constraint. This is the "
            "same as calling compute_range_by_IA")

    y_dim = A.shape[0]
    x_dim = A.shape[1]
    if x_lb is None:
        x_lb = np.full((x_dim, ), -np.inf)
    if x_ub is None:
        x_ub = np.full((x_dim, ), np.inf)
    y_lb = np.empty(y_dim)
    y_ub = np.empty(y_dim)
    model = gurobipy.Model()
    x = model.addMVar(x_dim, lb=x_lb, ub=x_ub)
    if C is not None:
        model.addMConstrs(C, x, gurobipy.GRB.LESS_EQUAL, d)
    for i in range(y_dim):
        # First find the upper bound.
        model.setMObjective(Q=None,
                            c=A[i],
                            constant=b[i],
                            xQ_L=None,
                            xQ_R=None,
                            xc=x,
                            sense=gurobipy.GRB.MAXIMIZE)
        model.update()
        model.setParam(gurobipy.GRB.Param.OutputFlag, False)
        model.optimize()
        if model.status == gurobipy.GRB.Status.OPTIMAL:
            y_ub[i] = model.ObjVal
        elif model.status == gurobipy.GRB.Status.INFEASIBLE:
            y_ub[i] = -np.inf
        elif model.status == gurobipy.GRB.Status.UNBOUNDED:
            y_ub[i] = np.inf
        else:
            raise Exception("compute_range_by_lp: unknown status.")

        # Now find the lower bound.
        model.setMObjective(Q=None,
                            c=A[i],
                            constant=b[i],
                            xQ_L=None,
                            xQ_R=None,
                            xc=x,
                            sense=gurobipy.GRB.MINIMIZE)
        model.update()
        model.setParam(gurobipy.GRB.Param.OutputFlag, False)
        model.optimize()
        if model.status == gurobipy.GRB.Status.OPTIMAL:
            y_lb[i] = model.ObjVal
        elif model.status == gurobipy.GRB.Status.INFEASIBLE:
            y_lb[i] = np.inf
        elif model.status == gurobipy.GRB.Status.UNBOUNDED:
            y_lb[i] = -np.inf
        else:
            raise Exception("compute_range_by_lp: unknown status.")
    return (y_lb, y_ub)


def compute_range_by_IA(
    A: torch.Tensor, b: torch.Tensor, x_lb: torch.Tensor, x_ub: torch.Tensor)\
        -> (torch.Tensor, torch.Tensor):
    """
    Given x_lb <= x <= x_ub, compute the bounds on A * x + b by interval
    arithmetics (IA). Notice that this allows the computed bounds to be
    differentiable w.r.t the input bounds x_lb, x_ub and parameter A, b.
    """
    assert (isinstance(A, torch.Tensor))
    assert (isinstance(b, torch.Tensor))
    output_dim = A.shape[0]
    x_dim = A.shape[1]
    assert (b.shape == (output_dim, ))
    assert (isinstance(x_lb, torch.Tensor))
    assert (isinstance(x_ub, torch.Tensor))
    assert (x_lb.shape == (x_dim, ))
    assert (x_ub.shape == (x_dim, ))
    output_lb = torch.empty(b.shape, dtype=b.dtype)
    output_ub = torch.empty(b.shape, dtype=b.dtype)

    for i in range(output_dim):
        mask1 = torch.where(A[i] > 0)[0]
        mask2 = torch.where(A[i] <= 0)[0]
        output_lb[i] = A[i][mask1] @ x_lb[mask1] + A[i][mask2] @ x_ub[mask2]\
            + b[i]
        output_ub[i] = A[i][mask1] @ x_ub[mask1] + A[i][mask2] @ x_lb[mask2]\
            + b[i]
    return output_lb, output_ub


class PropagateBoundsMethod(enum.Enum):
    IA = 1
    LP = 2
    MIP = 3


def propagate_bounds(layer, input_lo, input_up):
    """
    Given the bound of the layer's input, find the bound of the output.
    @param method Either use interval arithemtic (IA) or linear programming
    (LP) to compute the bounds. Note that LP produces tighter bounds, but loses
    the gradient information (The output bounds will not carry gradient).
    """
    assert (isinstance(input_lo, torch.Tensor))
    assert (isinstance(input_up, torch.Tensor))
    dtype = input_lo.dtype
    if isinstance(layer, torch.nn.ReLU):
        # ReLU is a monotonic increasing function.
        output_lo = layer(input_lo)
        output_up = layer(input_up)
    elif isinstance(layer, torch.nn.LeakyReLU):
        lo = layer(input_lo)
        up = layer(input_up)
        if layer.negative_slope < 0:
            output_lo = torch.min(lo, up)
            output_lo[torch.logical_and(input_lo < 0, input_up > 0)] = 0
            output_up = torch.max(lo, up)
        else:
            output_lo = lo
            output_up = up
    elif isinstance(layer, torch.nn.Linear):
        bias = torch.zeros((layer.out_features,), dtype=dtype) if\
            layer.bias is None else layer.bias.clone()
        output_lo, output_up = compute_range_by_IA(layer.weight, bias,
                                                   input_lo, input_up)
    else:
        raise Exception("progagate_bounds(): unknown layer type.")
    return output_lo, output_up
