import torch
import numpy as np
import gurobipy


def compute_range_by_lp(
    A: np.ndarray, b: np.ndarray, x_lb: np.ndarray, x_ub: np.ndarray,
        C: np.ndarray, d: np.ndarray) -> (np.ndarray, np.ndarray):
    """
    Given x_lb <= x <= x_ub and C * x <= d, compute the range of y = A * x + b
    through linear programming
    @param x_lb Either np.ndarray or None
    @param x_ub Either np.ndarray or None
    @param C Either np.ndarray or None
    @param d Either np.ndarray or None
    """
    assert(isinstance(A, np.ndarray))
    assert(isinstance(b, np.ndarray))
    assert(isinstance(x_lb, np.ndarray) or x_lb is None)
    assert(isinstance(x_ub, np.ndarray) or x_ub is None)
    assert(isinstance(C, np.ndarray) or C is None)
    assert(isinstance(d, np.ndarray) or d is None)

    y_dim = A.shape[0]
    x_dim = A.shape[1]
    if x_lb is None:
        x_lb = np.full((x_dim,), -np.inf)
    if x_ub is None:
        x_ub = np.full((x_dim,), np.inf)
    y_lb = np.empty(y_dim)
    y_ub = np.empty(y_dim)
    model = gurobipy.Model()
    x = model.addMVar(x_dim, lb=x_lb, ub=x_ub)
    if C is not None:
        model.addMConstrs(C, x, gurobipy.GRB.LESS_EQUAL, d)
    for i in range(y_dim):
        # First find the upper bound.
        model.setMObjective(Q=None, c=A[i], constant=b[i], xQ_L=None,
                            xQ_R=None, xc=x, sense=gurobipy.GRB.MAXIMIZE)
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
        model.setMObjective(Q=None, c=A[i], constant=b[i], xQ_L=None,
                            xQ_R=None, xc=x, sense=gurobipy.GRB.MINIMIZE)
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
    assert(isinstance(A, torch.Tensor))
    assert(isinstance(b, torch.Tensor))
    output_dim = A.shape[0]
    x_dim = A.shape[1]
    assert(b.shape == (output_dim,))
    assert(isinstance(x_lb, torch.Tensor))
    assert(isinstance(x_ub, torch.Tensor))
    assert(x_lb.shape == (x_dim,))
    assert(x_ub.shape == (x_dim,))
    output_lb = b.clone()
    output_ub = b.clone()
    for j in range(x_dim):
        for i in range(output_dim):
            if A[i, j] < 0:
                output_lb[i] += A[i, j] * x_ub[j]
                output_ub[i] += A[i, j] * x_lb[j]
            else:
                output_lb[i] += A[i, j] * x_lb[j]
                output_ub[i] += A[i, j] * x_ub[j]
    return output_lb, output_ub


def propagate_bounds_IA(layer, input_lo, input_up):
    """
    Given the bound of the layer's input, find the bound of the output through
    Interval Arithmetics (IA).
    """
    assert(isinstance(input_lo, torch.Tensor))
    assert(isinstance(input_up, torch.Tensor))
    dtype = input_lo.dtype
    if isinstance(layer, torch.nn.ReLU):
        # ReLU is a monotonic increasing function.
        output_lo = layer(input_lo)
        output_up = layer(input_up)
    elif isinstance(layer, torch.nn.LeakyReLU):
        assert(layer.negative_slope >= 0)
        # Leaky ReLU is a monotonic increasing function
        output_lo = layer(input_lo)
        output_up = layer(input_up)
    elif isinstance(layer, torch.nn.Linear):
        if layer.bias is None:
            output_lo = torch.zeros((layer.out_features,), dtype=dtype)
            output_up = torch.zeros((layer.out_features,), dtype=dtype)
        else:
            output_lo = layer.bias.clone()
            output_up = layer.bias.clone()
        for j in range(layer.in_features):
            for i in range(layer.out_features):
                if layer.weight[i, j] < 0:
                    output_lo[i] += layer.weight[i, j] * input_up[j]
                    output_up[i] += layer.weight[i, j] * input_lo[j]
                else:
                    output_lo[i] += layer.weight[i, j] * input_lo[j]
                    output_up[i] += layer.weight[i, j] * input_up[j]
    else:
        raise Exception("progagate_bounds_IA(): unknown layer type.")
    return output_lo, output_up
