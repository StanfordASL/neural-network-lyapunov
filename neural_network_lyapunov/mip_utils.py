import torch
import numpy as np
import gurobipy
import enum
import warnings
import itertools


def strengthen_relu_mip_w_indices(c: float, w: torch.Tensor, b: torch.Tensor,
                                  lo: torch.Tensor, up: torch.Tensor,
                                  indices: set):
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
    y <= x_coeff * x + binary_coeff * β + constant
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
    constant = b * c
    nx = w.shape[0]
    not_indices = set(range(nx)) - indices
    x_coeff = torch.zeros((nx, ), dtype=dtype)
    binary_coeff = b * (1 - c)
    for i in indices:
        if w[i] >= 0:
            lo_bar_i = lo[i]
        else:
            lo_bar_i = up[i]
        x_coeff[i] += w[i]
        constant += -(1 - c) * w[i] * lo_bar_i
        binary_coeff += (1 - c) * w[i] * lo_bar_i
    for i in not_indices:
        if w[i] >= 0:
            up_bar_i = up[i]
        else:
            up_bar_i = lo[i]
        x_coeff[i] += c * w[i]
        binary_coeff += (1 - c) * w[i] * up_bar_i
    return (x_coeff, binary_coeff, constant)


def find_index_set_to_strengthen(w: torch.Tensor, lo: torch.Tensor,
                                 up: torch.Tensor, xhat, beta_hat):
    """
    Given a point (xhat, beta_hat, y_hat), find the index set ℑ that best
    separates the point from the convex hull of integral solutions.
    This index set is defined as
    ℑ = {i | wᵢx̂ᵢ ≤ (1−β̂)wᵢL̅ᵢ +β̂wᵢU̅ᵢ}
    For more details, refer to doc/ideal_formulation.tex
    Notice that using this index set ℑ, the constrained computed from
    strengthen_relu_mip_w_indices() might not separate the point.
    """
    indices = set()
    nx = w.shape[0]
    for i in range(nx):
        if w[i] >= 0 and xhat[i] <= (1 - beta_hat) * lo[i] + beta_hat * up[i]:
            indices.add(i)
        elif w[i] < 0 and xhat[i] > (1 - beta_hat) * up[i] + beta_hat * lo[i]:
            indices.add(i)
    return indices


def strengthen_relu_mip_given_pts(c: float, w: torch.Tensor, b: torch.Tensor,
                                  lo: torch.Tensor, up: torch.Tensor,
                                  linear_inputs: list, relu_outputs: list,
                                  relu_activations: list):
    """
    Strengthen the big-M formulation of the leaky ReLU function
    y = max(c * (w'x+b), w'x+b)
    with some facets in the ideal formulation.
    This approach is described in Strong mixed integer programming formulations
    for trained neural networks.
    Given a list of (x̂, ŷ, β̂) (namely the linear layer input, the relu
    output, and activation), find the most violated constraint in the ideal
    formulation, and add that constraint if it is violated.
    """
    assert (isinstance(linear_inputs, list))
    assert (isinstance(relu_outputs, list))
    assert (isinstance(relu_activations, list))
    assert (len(linear_inputs) == len(relu_outputs))
    assert (len(linear_inputs) == len(relu_activations))
    x_coeffs = []
    binary_coeffs = []
    constants = []
    for (xhat, yhat, beta_hat) in zip(linear_inputs, relu_outputs,
                                      relu_activations):
        indices = find_index_set_to_strengthen(w, lo, up, xhat, beta_hat)
        x_coeff, binary_coeff, constant = strengthen_relu_mip_w_indices(
            c, w, b, lo, up, indices)
        assert (x_coeff.shape == (w.shape[0], ))
        if yhat > x_coeff @ xhat + binary_coeff * beta_hat + constant:
            # This constraint is violated.
            x_coeffs.append(x_coeff.reshape((1, -1)))
            binary_coeffs.append(binary_coeff)
            constants.append(constant)
    if len(x_coeffs) > 0:
        return torch.cat(
            x_coeffs,
            dim=0), torch.stack(binary_coeffs), torch.stack(constants)
    else:
        return None, None, None


def _get_linear_input_vertices(lo, up, w, b, relu_input_lo, relu_input_up):
    """
    For the region
    lo <= x <= up
    relu_input_lo <= w*x+b <= relu_input_up
    return all the vertices of the box lo <= x <= up if the vertex also
    satisfies relu_input-lo <= w*x+b <= relu_input-up
    """
    assert isinstance(lo, torch.Tensor)
    assert isinstance(up, torch.Tensor)
    assert (len(lo.shape) == 1)
    assert (lo.shape == up.shape)
    assert (lo.shape == w.shape)
    assert isinstance(relu_input_lo, torch.Tensor)
    assert isinstance(relu_input_up, torch.Tensor)
    linear_input_bounds = [(lo[i], up[i]) for i in range(lo.shape[0])]
    box_vertices = itertools.product(*linear_input_bounds)
    relu_input_lo_ia, relu_input_up_ia = compute_range_by_IA(
        w.reshape((1, -1)), b.reshape((-1, )), lo, up)
    vertices = []
    if relu_input_lo == relu_input_lo_ia[
            0] and relu_input_up == relu_input_up_ia[0]:
        for vertex in box_vertices:
            vertices.append(torch.stack(vertex))
        return vertices
    assert (relu_input_lo >= relu_input_lo_ia[0])
    assert (relu_input_up <= relu_input_up_ia[0])
    vertices = []
    for vertex in box_vertices:
        relu_input_vertex = w @ torch.stack(vertex) + b
        if relu_input_lo <= relu_input_vertex <= relu_input_up:
            vertices.append(torch.stack(vertex))
    return vertices


def _compute_beta_range(c: float, w: torch.Tensor, b: torch.Tensor, x_coeffs,
                        binary_coeffs, constants, xhat: torch.Tensor):
    """
    Compute the range of beta_hat, such that
    max(c(w'*xhat+b), w'*xhat+b) <=
        min(x_coeffs*xhat + binary_coeffs*beta_hat+constants).
    Also beta_hat is in the range [0, 1].
    returns beta_hat_lo, beta_hat_up
    """
    dtype = w.dtype
    lhs = torch.max(w @ xhat + b, c * (w @ xhat + b))
    beta_lo = [torch.tensor(0, dtype=dtype)]
    beta_up = [torch.tensor(1, dtype=dtype)]
    for i in range(len(constants)):
        bound = lhs - x_coeffs[i] @ xhat - constants[i]
        if binary_coeffs[i] > 0:
            beta_lo.append(bound / binary_coeffs[i])
        elif binary_coeffs[i] < 0:
            beta_up.append(bound / binary_coeffs[i])
    return torch.max(torch.stack(beta_lo)), torch.min(torch.stack(beta_up))


def _max_y_given_linear_input(c: float, w: torch.Tensor, b: torch.Tensor,
                              x_coeffs, beta_coeffs, constants,
                              x_hat) -> (float, np.ndarray):
    """
    For the constraint
    c(w'*x_hat+b) <= y
    w'*x_hat+b <= y
    y <= x_coeffs * x_hat + beta_coeffs * beta + constants
    0 <= beta <= 1
    Find the maximal value of y, together with the value beta that gives the
    maximal value of y.
    returns (y_max, beta_val) Notice that y_max and beta_val aren't torch
    tensors, hence we cannot do automatic differentiation on these two.
    """
    prog = gurobipy.Model()
    y_lower = torch.max(c * (w @ x_hat + b), w @ x_hat + b)
    y = prog.addVar(lb=y_lower.item())
    beta = prog.addVar(lb=0., ub=1.)
    prog.setObjective(gurobipy.LinExpr([1.], [y]), sense=gurobipy.GRB.MAXIMIZE)
    for i in range(len(x_coeffs)):
        prog.addLConstr(gurobipy.LinExpr([1., -beta_coeffs[i].item()],
                                         [y, beta]),
                        sense=gurobipy.GRB.LESS_EQUAL,
                        rhs=(x_coeffs[i] @ x_hat + constants[i]).item())
    prog.setParam(gurobipy.GRB.Param.OutputFlag, False)
    prog.optimize()
    assert (prog.status == gurobipy.GRB.Status.OPTIMAL)
    beta_val = beta.x
    y_max = y.x
    return (y_max, beta_val)


def strengthen_relu_mip(c: float, w: torch.Tensor, b: torch.Tensor,
                        lo: torch.Tensor, up: torch.Tensor, relu_input_lo,
                        relu_input_up, selective: bool):
    """
    For the (leaky) ReLU unit y = max(c*(wᵀx+b), wᵀx+b), strengthen its big-M
    formulation, by adding the constraint
    y <= bc + b(1-c)β + ∑ i∈ℑ (wᵢxᵢ−(1−c)(1−β)wᵢL̅ᵢ) + ∑i∉ℑ(cwᵢxᵢ+(1−c)βwᵢU̅ᵢ)
    if this additional constraint tightens the existing mixed-integer linear
    constraints. ℑ is a subset of {1, 2, ..., n}, where n is the dimension of x

    We start with just the constraint
    y >= c*(wᵀx+b)
    y >= wᵀx+b
    y <= c(wᵀx+b)+(1−c)m⁺β
    y <= wᵀx+b−(1−c)m⁻(1−β)
    L <= x <= U
    0 <= β <= 1

    we loop through each vertex of the box L <= x <= U, compute the bound of β
    given the existing constraints (by calling _compute_beta_range()), and then
    find the corresponding index set through find_index_set_to_strengthen(). If
    the generated constraint reduces the upper bound of y at that x_hat and
    beta_hat, then we add the constraint.
    Refer to the documentation in doc/ideal_formulation.tex for more details.
    @param c The negative slope of the leaky ReLU unit.
    @param w The linear coefficient of the linear layer, a 1D vector.
    @param b The bias of the linear layer, A 0-D tensor.
    @param lo The lower bound of the linear input, L in the documentation
    above.
    @param up The upper bound of the linear input, U in the documentation
    above.
    @param relu_input_lo The lower bound of the ReLU input, m⁻ in the
    documentation above.
    @param relu_input_up The upper bound of the ReLU input, m⁺ in the
    documentation above.
    @param selective If set to false, then add all 2ⁿ-2 number of additional
    linear constraints, each constraint correspond to an index set ℑ, that ℑ
    is a subset of {1, 2, ..., n} except the empty set and the whole set. If
    set to False, then we only add the linear constraint by choosing the most
    violated additional constraint, evaluated at certain point of (x̂, ŷ, β̂).
    @retun x_coeffs, binary_coeffs, constants. The strengthened constraints in
    the form of
    y <= x_coeffs * x + binary_coeffs * beta + constants
    """
    assert (relu_input_lo < 0)
    assert (relu_input_up > 0)
    assert (isinstance(lo, torch.Tensor))
    assert (isinstance(up, torch.Tensor))
    assert (len(lo.shape) == 1)
    assert (lo.shape == up.shape)
    assert (isinstance(relu_input_lo, torch.Tensor))
    assert (isinstance(relu_input_up, torch.Tensor))
    assert (len(relu_input_lo.shape) == 0)
    assert (len(relu_input_up.shape) == 0)
    # Before strengthening, we already have two constraint to bound the upper
    # value of y as
    # y <= c(wᵀx+b)+(1−c)m⁺β
    # y <= wᵀx+b−(1−c)m⁻(1−β)
    x_coeffs_exist = [c * w, w]
    binary_coeffs_exist = [(1 - c) * relu_input_up, (1 - c) * relu_input_lo]
    constants_exist = [c * b, b - (1 - c) * relu_input_lo]
    dtype = w.dtype
    x_coeffs = []
    binary_coeffs = []
    constants = []
    if not selective:
        nx = w.shape[0]
        for candidate_index in itertools.chain.from_iterable(
                itertools.combinations(list(range(nx)), r)
                for r in range(1, nx)):
            x_coeff, binary_coeff, constant = strengthen_relu_mip_w_indices(
                c, w, b, lo, up, set(candidate_index))
            x_coeffs.append(x_coeff)
            binary_coeffs.append(binary_coeff)
            constants.append(constant)
    else:
        for x_hat in _get_linear_input_vertices(lo, up, w, b, relu_input_lo,
                                                relu_input_up):
            beta_lo, beta_up = _compute_beta_range(
                c, w, b, x_coeffs_exist + x_coeffs,
                binary_coeffs_exist + binary_coeffs,
                constants_exist + constants, x_hat)
            for beta_hat in (beta_lo, beta_up):
                indices = find_index_set_to_strengthen(w, lo, up, x_hat,
                                                       beta_hat)
                x_coeff, binary_coeff, constant =\
                    strengthen_relu_mip_w_indices(
                        c, w, b, lo, up, indices)
                # Now evaluate the right-hand side at x_hat, beta_hat
                y_upper_bound = torch.min(
                    torch.stack([
                        x_coeffs_exist[i] @ x_hat +
                        binary_coeffs_exist[i] * beta_hat + constants_exist[i]
                        for i in range(len(x_coeffs_exist))
                    ] + [
                        x_coeffs[i] @ x_hat + binary_coeffs[i] * beta_hat +
                        constants[i] for i in range(len(x_coeffs))
                    ]))
                y_upper_bound_new = x_coeff @ x_hat + binary_coeff * beta_hat\
                    + constant
                if y_upper_bound_new < y_upper_bound - 1E-6:
                    x_coeffs.append(x_coeff)
                    binary_coeffs.append(binary_coeff)
                    constants.append(constant)
    if len(x_coeffs) > 0:
        return torch.cat(
            [v.reshape((1, -1)) for v in x_coeffs],
            dim=0), torch.stack(binary_coeffs), torch.stack(constants)
    else:
        return torch.empty((0, w.shape[0]), dtype=dtype), torch.empty(
            (0, ), dtype=dtype), torch.empty((0, ), dtype=dtype)


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
    # First propagate the bounds using IA, then propagate the bounds using MIP
    # to tighten the bounds (but still keep the IA bounds if these bounds might
    # be active). Note that by computing the bounds using MIP we lose the
    # gradient of these bounds.
    IA_MIP = 4


def propagate_bounds(layer, input_lo, input_up):
    """
    Given the bound of the layer's input, find the bound of the output.
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
