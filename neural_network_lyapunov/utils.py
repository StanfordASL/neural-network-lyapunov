import numpy as np
import torch
import cvxpy as cp
import gurobipy
import neural_network_lyapunov.gurobi_torch_mip as gurobi_torch_mip
import scipy.integrate


def update_progress(progress):
    bar_length = 40
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
    if progress < 0:
        progress = 0
    if progress >= 1:
        progress = 1

    block = int(round(bar_length * progress))

    text = "Progress: [{0}] {1:.1f}%".format(
        "#" * block + "-" * (bar_length - block), progress * 100)
    print(text)


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


def check_shape_and_type(A, shape_expected, dtype_expected):
    assert (A.shape == shape_expected)
    assert (A.dtype == dtype_expected)


def replace_binary_continuous_product(x_lo, x_up, dtype=torch.float64):
    """
    We replace the product between a binary variable α and a continuous
    variable x by a slack variable s, such that s = α * x. To impose this
    constraint on s, assuming that x is bounded between [xₗₒ, xᵤₚ], we
    introduce the following linear constraints
    s ≥ α*xₗₒ
    s ≤ α*xᵤₚ
    x - s + (α - 1)*xᵤₚ ≤ 0
    x - s + (α - 1)*xₗₒ ≥ 0
    We write these constraints concisely as
    Aₓ*x + Aₛ*s + A_alpha*α ≤ rhs
    @param x_lo The lower bound of x.
    @param x_up The upper bound of x.
    @param (A_x, A_s, A_alpha, rhs) A_x, A_s, A_alpha, rhs are all arrays of
    length 4.
    """
    if isinstance(x_lo, float):
        x_lo = torch.tensor(x_lo, dtype=dtype)
    if isinstance(x_up, float):
        x_up = torch.tensor(x_up, dtype=dtype)
    assert (isinstance(x_lo, torch.Tensor))
    assert (x_lo <= x_up)
    A_x = torch.tensor([0, 0, 1, -1], dtype=dtype)
    A_s = torch.tensor([-1, 1, -1, 1], dtype=dtype)
    A_alpha = torch.stack((x_lo, -x_up, x_up, -x_lo))
    rhs = torch.zeros(4, dtype=dtype)
    rhs = torch.stack(
        (torch.tensor(0, dtype=dtype), torch.tensor(0,
                                                    dtype=dtype), x_up, -x_lo))
    return (A_x, A_s, A_alpha, rhs)


def max_as_mixed_integer_constraint(
        x_lo: torch.Tensor,
        x_up: torch.Tensor) -> gurobi_torch_mip.MixedIntegerConstraintsReturn:
    """
    Formulate y=max(x) as mixed-integer constraints on x.
    y >= xᵢ
    y <= xᵢ + (1-αᵢ)(max(x_up) - x_lo[i])
    ∑ᵢ αᵢ = 1
    The slack variable is y, which is also the output
    """
    assert (isinstance(x_lo, torch.Tensor))
    assert (isinstance(x_up, torch.Tensor))
    nx = x_lo.shape[0]
    assert (x_lo.shape == (nx, ))
    assert (x_up.shape == (nx, ))
    assert (torch.all(x_up >= x_lo))
    ret = gurobi_torch_mip.MixedIntegerConstraintsReturn()
    dtype = x_lo.dtype
    ret.Aout_slack = torch.tensor([[1]], dtype=dtype)
    ret.Ain_input = torch.cat(
        (torch.eye(nx, dtype=dtype), -torch.eye(nx, dtype=dtype)), dim=0)
    ret.Ain_slack = torch.cat((-torch.ones(
        (nx, 1), dtype=dtype), torch.ones((nx, 1), dtype=dtype)),
                              dim=0)
    max_x_up = torch.max(x_up)
    ret.Ain_binary = torch.cat((torch.zeros(
        (nx, nx), dtype=dtype), torch.diag(max_x_up - x_lo)),
                               dim=0)
    ret.rhs_in = torch.cat((torch.zeros((nx, ), dtype=dtype), max_x_up - x_lo))
    ret.Aeq_binary = torch.ones((1, nx), dtype=dtype)
    ret.rhs_eq = torch.tensor([1], dtype=dtype)
    # If a the upper bound of x[i] is less than the lower bound of another
    # variable, then it can't be the maximal.
    non_maximal_idx = torch.nonzero(
        torch.any(x_up.unsqueeze(1).repeat(1, nx) -
                  x_lo.unsqueeze(1).T.repeat(nx, 1) < 0,
                  dim=1)).squeeze().tolist()
    if (len(non_maximal_idx) > 0):
        Aeq_binary_non_maximal = torch.zeros((len(non_maximal_idx), nx),
                                             dtype=dtype)
        for (i, idx) in enumerate(non_maximal_idx):
            Aeq_binary_non_maximal[i, idx] = 1
        ret.Aeq_binary = torch.cat((ret.Aeq_binary, Aeq_binary_non_maximal),
                                   dim=0)
        ret.rhs_eq = torch.cat(
            (ret.rhs_eq, torch.zeros((len(non_maximal_idx), ), dtype=dtype)))
        ret.binary_lo = torch.zeros((nx, ), dtype=dtype)
        ret.binary_up = torch.ones((nx, ), dtype=dtype)
        ret.binary_up[non_maximal_idx] = 0
    return ret


def leaky_relu_gradient_times_x(x_lo,
                                x_up,
                                negative_slope,
                                dtype=torch.float64):
    """
    Write the function
    y = x if α = 1
    y = c*x if α = 0
    as mixed-integer linear constraints on x, y and α. Note that y could be
    regarded as the gradient of a leaky relu unit (with negative slope = c)
    times x.
    The mixed-integer linear constraints are
    A_x * x + A_y * y + A_alpha * alpha <= rhs
    @param x_lo The lower bound of x.
    @param x_up The upper bound of x.
    """
    if isinstance(x_lo, float):
        x_lo = torch.tensor(x_lo, dtype=dtype)
    if isinstance(x_up, float):
        x_up = torch.tensor(x_up, dtype=dtype)
    assert (isinstance(x_lo, torch.Tensor))
    assert (x_up >= x_lo)
    dtype = x_up.dtype
    A_x = torch.tensor([-1, 1, negative_slope, -negative_slope], dtype=dtype)
    A_y = torch.tensor([1, -1, -1, 1], dtype=dtype)
    A_alpha = torch.stack(
        ((negative_slope - 1.) * x_lo, (1. - negative_slope) * x_up,
         (1. - negative_slope) * x_lo, (negative_slope - 1.) * x_up))
    rhs = torch.stack(
        ((negative_slope - 1.) * x_lo, (1. - negative_slope) * x_up,
         torch.tensor(0, dtype=dtype), torch.tensor(0, dtype=dtype)))
    if negative_slope < 1.:
        return (A_x, A_y, A_alpha, rhs)
    else:
        return (-A_x, -A_y, -A_alpha, -rhs)


def absolute_value_as_mixed_integer_constraint(
    x_lo: torch.Tensor,
    x_up: torch.Tensor,
    binary_for_zero_input=False
) -> gurobi_torch_mip.MixedIntegerConstraintsReturn:
    """
    For a variable x in the interval [x_lo, x_up], we denote the absolute
    value |x| as s, and returns the mixed-integer constraints on x, s and the
    binary variables.

    Case 1. When x_lo < 0 < x_up.
    If binary_for_zero_input=False, we introduce a binary variable
    α, such that
    α = 1 => x >= 0
    α = 0 => x <= 0
    then s, x and alpha should satisfy the following mixed-integer constraint
    x - s <= 0
    -x - s <= 0
    -x + s - 2 * x_lo*α <= -2 * x_lo
    x + s - 2 * x_up *α <= 0

    If binary_for_zero_input=True, we use 3 binary variables α[0], α[1], α[2],
    such that
    α[0] = 1 => x <= 0
    α[1] = 1 => x = 0
    α[2] = 1 => x >= 0
    then s, x, alpha should satisfy the following mixed-integer constraint
    s >= x
    s >= -x
    -x + s + 2 * x_lo*α[0] <= 0
    x + s - 2 * x_up *α[2] <= 0
    α[0] + α[1] + α[2] = 1

    case 2. If (x_lo >= 0 and binary_for_zero_input=False) or
        (x_lo > 0 and binary_for_zero_input=True)
    then we impose the linear equality constraint
    s=x
    α=1 (if binary_for_zero_input=False)
    α[0] = α[1] = 0, α[2] = 1 (if binary_for_zero_input=True)

    case 3. If x_lo = 0 and binary_for_zero_input=True
    s=x
    x + s - 2 * x_up * α[2] <= 0
    α[0] = 0
    α[1] + α[2] = 1

    case 4. If (x_up <= 0 and binary_for_zero_input=False) or
        (x_up < 0 and binary_for_zero_input=True)
    then we impose the linear equality constraint
    s=-x
    α=0 (if binary_for_zero_input=False)
    α[0] = 1,  α[1] = α[2] = 0 (if binary_for_zero_input=True)

    case 5. If x_up = 0 and binary_for_zero_input=True
    s = -x
    -x + s + 2 * x_lo*α[0] <= 0
    α[2] = 0
    α[0] + α[1] = 1
    """
    ret = gurobi_torch_mip.MixedIntegerConstraintsReturn()
    if isinstance(x_lo, float):
        x_lo = torch.tensor(x_lo, dtype=torch.float64)
    if isinstance(x_up, float):
        x_up = torch.tensor(x_up, dtype=torch.float64)

    assert (isinstance(x_lo, torch.Tensor))
    assert (isinstance(x_up, torch.Tensor))
    assert (x_lo <= x_up)
    dtype = x_lo.dtype
    ret.Aout_slack = torch.tensor([[1]], dtype=dtype)
    if x_lo < 0 and x_up > 0:
        # case 1.
        # s >= x
        # s >= -x
        ret.Ain_input = torch.tensor([[1], [-1], [-1], [1]], dtype=dtype)
        ret.Ain_slack = torch.tensor([[-1], [-1], [1], [1]], dtype=dtype)
        if binary_for_zero_input:
            # -x + s + 2 * x_lo*α[0] <= 0
            # x + s - 2 * x_up *α[2] <= 0
            # α[0] + α[1] + α[2] = 1
            ret.Ain_binary = torch.zeros((4, 3), dtype=dtype)
            ret.Ain_binary[2, 0] = 2 * x_lo
            ret.Ain_binary[3, 2] = -2 * x_up
            ret.rhs_in = torch.zeros(4, dtype=dtype)
            ret.Aeq_input = torch.tensor([[0]], dtype=dtype)
            ret.Aeq_slack = torch.tensor([[0]], dtype=dtype)
            ret.Aeq_binary = torch.tensor([[1, 1, 1]], dtype=dtype)
            ret.rhs_eq = torch.tensor([1], dtype=dtype)
        else:
            ret.Ain_binary = torch.stack(
                (torch.tensor(0, dtype=dtype), torch.tensor(0, dtype=dtype),
                 -2 * x_lo, -2 * x_up)).reshape((-1, 1))
            ret.rhs_in = torch.stack(
                (torch.tensor(0, dtype=dtype), torch.tensor(0, dtype=dtype),
                 -2 * x_lo, torch.tensor(0, dtype=dtype)))
    elif x_lo >= 0 and not binary_for_zero_input:
        # x >= x_lo
        # x <= x_up
        # s = x, α=1
        ret.Ain_input = torch.tensor([[-1], [1]], dtype=dtype)
        ret.Ain_slack = torch.tensor([[0], [0]], dtype=dtype)
        ret.Ain_binary = torch.tensor([[0], [0]], dtype=dtype)
        ret.rhs_in = torch.stack((-x_lo, x_up))
        ret.Aeq_input = torch.tensor([[1], [0]], dtype=dtype)
        ret.Aeq_slack = torch.tensor([[-1], [0]], dtype=dtype)
        ret.Aeq_binary = torch.tensor([[0], [1]], dtype=dtype)
        ret.rhs_eq = torch.tensor([0, 1], dtype=dtype)
        ret.binary_lo = torch.tensor([1], dtype=dtype)
        ret.binary_up = torch.tensor([1], dtype=dtype)
    elif x_lo > 0 and binary_for_zero_input:
        # x >= x_lo
        # x <= x_up
        # s = x, α[0] = α[1] = 0, α[2] = 1
        ret.Ain_input = torch.tensor([[-1], [1]], dtype=dtype)
        ret.Ain_slack = torch.tensor([[0], [0]], dtype=dtype)
        ret.Ain_binary = torch.zeros((2, 3), dtype=dtype)
        ret.rhs_in = torch.stack((-x_lo, x_up))
        ret.Aeq_input = torch.tensor([[1], [0], [0], [0]], dtype=dtype)
        ret.Aeq_slack = torch.tensor([[-1], [0], [0], [0]], dtype=dtype)
        ret.Aeq_binary = torch.vstack((torch.zeros(
            (1, 3), dtype=dtype), torch.eye(3, dtype=dtype)))
        ret.rhs_eq = torch.tensor([0, 0, 0, 1], dtype=dtype)
        ret.binary_lo = torch.tensor([0, 0, 1], dtype=dtype)
        ret.binary_up = torch.tensor([0, 0, 1], dtype=dtype)
    elif x_lo == 0 and binary_for_zero_input:
        # s=x
        # x >= x_lo
        # x <= x_up
        # x + s - 2 * x_up * α[2] <= 0
        # α[0] = 0
        # α[1] + α[2] = 1
        ret.Ain_input = torch.tensor([[-1], [1], [1]], dtype=dtype)
        ret.Ain_salck = torch.tensor([[0], [0], [1]], dtype=dtype)
        ret.Ain_binary = torch.zeros((3, 3), dtype=dtype)
        ret.Ain_binary[2, 2] = -2 * x_up
        ret.rhs_in = torch.stack((-x_lo, x_up, torch.tensor(0, dtype=dtype)))
        ret.Aeq_input = torch.tensor([[1], [0], [0]], dtype=dtype)
        ret.Aeq_slack = torch.tensor([[-1], [0], [0]], dtype=dtype)
        ret.Aeq_binary = torch.tensor([[0, 0, 0], [1, 0, 0], [0, 1, 1]],
                                      dtype=dtype)
        ret.rhs_eq = torch.tensor([0, 0, 1], dtype=dtype)
        ret.binary_lo = torch.tensor([0, 0, 0], dtype=dtype)
        ret.binary_up = torch.tensor([0, 1, 1], dtype=dtype)
    elif x_up <= 0 and not binary_for_zero_input:
        # x >= x_lo
        # x <= x_up
        # s = -x
        # α = 0
        ret.Ain_input = torch.tensor([[-1], [1]], dtype=dtype)
        ret.Ain_slack = torch.tensor([[0], [0]], dtype=dtype)
        ret.Ain_binary = torch.tensor([[0], [0]], dtype=dtype)
        ret.rhs_in = torch.stack((-x_lo, x_up))
        ret.Aeq_input = torch.tensor([[1], [0]], dtype=dtype)
        ret.Aeq_slack = torch.tensor([[1], [0]], dtype=dtype)
        ret.Aeq_binary = torch.tensor([[0], [1]], dtype=dtype)
        ret.rhs_eq = torch.tensor([0, 0], dtype=dtype)
        ret.binary_lo = torch.tensor([0], dtype=dtype)
        ret.binary_up = torch.tensor([0], dtype=dtype)
    elif x_up < 0 and binary_for_zero_input:
        # x >= x_lo
        # x <= x_up
        # s = -x
        # α = [1, 0, 0]
        ret.Ain_input = torch.tensor([[-1], [1]], dtype=dtype)
        ret.Ain_slack = torch.tensor([[0], [0]], dtype=dtype)
        ret.Ain_binary = torch.zeros((2, 3), dtype=dtype)
        ret.rhs_in = torch.stack((-x_lo, x_up))
        ret.Aeq_input = torch.tensor([[1], [0], [0], [0]], dtype=dtype)
        ret.Aeq_slack = torch.tensor([[1], [0], [0], [0]], dtype=dtype)
        ret.Aeq_binary = torch.vstack((torch.zeros(
            (1, 3), dtype=dtype), torch.eye(3, dtype=dtype)))
        ret.rhs_eq = torch.tensor([0, 1, 0, 0], dtype=dtype)
        ret.binary_lo = torch.tensor([1, 0, 0], dtype=dtype)
        ret.binary_up = torch.tensor([1, 0, 0], dtype=dtype)
    elif x_up == 0 and binary_for_zero_input:
        # x >= x_lo
        # x <= x_up
        # s = -x
        # -x + s + 2 * x_lo*α[0] <= 0
        # α[2] = 0
        # α[0] + α[1] = 1
        ret.Ain_input = torch.tensor([[-1], [1], [-1]], dtype=dtype)
        ret.Ain_slack = torch.tensor([[0], [0], [1]], dtype=dtype)
        ret.Ain_binary = torch.zeros((3, 3), dtype=dtype)
        ret.Ain_binary[2, 0] = 2 * x_lo
        ret.rhs_in = torch.stack((-x_lo, x_up, torch.tensor(0, dtype=dtype)))
        ret.Aeq_input = torch.tensor([[1], [0], [0]], dtype=dtype)
        ret.Aeq_slack = torch.tensor([[1], [0], [0]], dtype=dtype)
        ret.Aeq_binary = torch.zeros((3, 3), dtype=dtype)
        ret.Aeq_binary[1, 2] = 1.
        ret.Aeq_binary[2, 0] = 1.
        ret.Aeq_binary[2, 1] = 1.
        ret.rhs_eq = torch.tensor([0, 0, 1], dtype=dtype)
        ret.binary_lo = torch.tensor([0, 0, 0], dtype=dtype)
        ret.binary_up = torch.tensor([1, 1, 0], dtype=dtype)
    return ret


def replace_relu_with_mixed_integer_constraint(x_lo,
                                               x_up,
                                               dtype=torch.float64):
    """
    For a ReLU activation unit y = max(0, x), we can replace this function with
    mixed-integer linear constraint on x, y and β, where β is the binary
    variable indicating whether the ReLU unit is activated or not.

    When the bound of x is [xₗₒ, xᵤₚ], and xₗₒ < 0 <  xᵤₚ, we use the "convex
    hull" trick to impose the constraint y = max(0, x) as mixed-integer linear
    constraints. Consider the point (x, y, β). The feasible region of this
    point are two line segment (x, 0, 0) for xₗₒ <= x < 0 and (x, x, 1) for
    0 <= x <= xᵤₚ. The convex hull of these two line segments is a tetrahedron,
    described by linear inequality constraints
    y ≥ 0
    y ≥ x
    y ≤ xᵤₚβ
    x - y + xₗₒβ ≥ xₗₒ

    We return these four linear constraints as
    A_x * x + A_y * y + A_beta * β ≤ rhs
    @param x_lo The lower bound of x. @pre x_lo is negative.
    @param x_up The upper bound of x. @pre x_up is positive.
    @return (A_x, A_y, A_beta, rhs). A_x, A_y, A_beta, rhs are all 4 x 1 column
    vectors.
    """
    assert (x_lo < 0)
    assert (x_up > 0)
    A_x = torch.tensor([0, 1, 0, -1], dtype=dtype)
    A_y = torch.tensor([-1, -1, 1, 1], dtype=dtype)
    A_beta = torch.zeros(4, dtype=dtype)
    A_beta[2] = -x_up
    A_beta[3] = -x_lo
    rhs = torch.zeros(4, dtype=dtype)
    rhs[3] = -x_lo
    return (A_x, A_y, A_beta, rhs)


def replace_leaky_relu_mixed_integer_constraint(negative_slope,
                                                x_lo,
                                                x_up,
                                                dtype=torch.float64):
    """
    For input x ∈ [x_lo, x_up] (and x_lo < 0 < x_up), the leaky relu output
    y satisfies
    y = x if x >= 0
    y = a*x if x <= 0
    where a is the negative slope. We can use a binary variable β to
    indicate whether the ReLU unit is active or not. Namely
    β = 1 => x >= 0
    β = 0 => x <= 0
    We can write the relationship between (x, y, β) as mixed-integer linear
    constraints
    if a <=1:
    y >= x
    y >= a*x
    -a*x + y + (a-1)*x_up * β <= 0
    -x + y + (a-1)*x_lo * β <= (a-1)*x_lo
    if a >= 1:
    y <= x
    y <= a*x
    -a*x + y + (a-1)*x_up * β >= 0
    -x + y + (a-1)*x_lo * β >= (a-1)*x_lo
    We write these constraints concisely as
    A_x * x + A_y * y + A_beta * β <= rhs
    @param negative_slope The slope in the negative domain of leaky relu. This
    number has to be smaller than 1
    @param x_lo The lower bound of input x.
    @param x_up The upper bound of input x.
    @return (A_x, A_y, A_beta, rhs)
    """
    assert (x_lo < 0)
    assert (x_up > 0)
    A_x = torch.tensor([1., negative_slope, -negative_slope, -1], dtype=dtype)
    A_y = torch.tensor([-1., -1., 1., 1.], dtype=dtype)
    A_beta = torch.zeros(4, dtype=dtype)
    A_beta[2] = (negative_slope - 1) * x_up
    A_beta[3] = (negative_slope - 1) * x_lo
    rhs = torch.zeros(4, dtype=dtype)
    rhs[3] = (negative_slope - 1) * x_lo
    if negative_slope <= 1:
        return (A_x, A_y, A_beta, rhs)
    else:
        return (-A_x, -A_y, -A_beta, -rhs)


def add_saturation_as_mixed_integer_constraint(mip, input_var, output_var,
                                               lower_limit, upper_limit,
                                               input_lower_bound,
                                               input_upper_bound,
                                               binary_var_type):
    """
    For a saturation block
    y = upper_limit if x >= upper_limit
    y = x if lower_limit <= x <= upper_limit
    y = lower_limit if x <= lower_limit
    We can write this piecewise linear relationship using mixed-integer linear
    constraints (provided that the input x is bounded
    input_lower_boun <= x <= input_upper_bound).
    Depending on the input bounds, we might need to introduce two binary
    variables b_lower and b_upper, such
    that
    b_lower = 1 => x <= lower_limit
    b_upper = 1 => x >= upper_limit
    @note Depending on the input bounds, sometimes we don't need to introduce
    the binary variables. For example, if the input bounds are all less than
    lower_limit, then we know that the output is always lower_limit, hence the
    output function is not piecewise linear, and we don't need binary
    variables.
    @param binary_var_type The variable type for binary variable. Could be
    gurobipy.GRB.BINARY, gurobipy.GRB.CONTINUOUS or
    gurobi_torch_mip.BINARYRELAX.
    @return binary_variables If no binary variable is introduced, then return
    an empty list. If the input bounds cover only the lower limit, then we
    return [b_lower]; If the input bounds cover only the upper limit, then we
    return [b_upper]; If the input bounds cover both saturation limits, then we
    return [b_lower, b_upper].
    @note that I don't always add the constraint
    input_lower_bound <= x <= input_upper_bound in this function. Specifically
    when the output is a linear function of the input (no saturation could
    happen), we don't add the constraint
    input_lower_bound <= x <= input_upper_bound.
    """
    assert (isinstance(mip, gurobi_torch_mip.GurobiTorchMIP))
    assert (isinstance(input_var, gurobipy.Var))
    assert (isinstance(output_var, gurobipy.Var))
    if input_upper_bound <= lower_limit:
        # The input x will always be <= lower_limit, the output will always be
        # lower_limit.
        mip.addLConstr([torch.ones([1], dtype=torch.float64)], [[output_var]],
                       sense=gurobipy.GRB.EQUAL,
                       rhs=lower_limit)
        return []
    elif input_lower_bound >= upper_limit:
        # The input x will always be >= upper_limit, the output will always be
        # upper_limit.
        mip.addLConstr([torch.ones([1], dtype=torch.float64)], [[output_var]],
                       sense=gurobipy.GRB.EQUAL,
                       rhs=upper_limit)
        return []
    elif input_lower_bound >= lower_limit and input_upper_bound <= upper_limit:
        # The input is never saturated, the output equals to the input.
        mip.addLConstr([torch.tensor([1, -1], dtype=torch.float64)],
                       [[input_var, output_var]],
                       sense=gurobipy.GRB.EQUAL,
                       rhs=0.)
        return []
    elif input_lower_bound < lower_limit and input_upper_bound <= upper_limit:
        # The input can saturate the lower limit. We need a binary variable to
        # determine whether the lower saturation happens.
        # Namely y - lower_limit = relu(x - lower_limit)
        A_x, A_y, A_beta, rhs = replace_relu_with_mixed_integer_constraint(
            input_lower_bound - lower_limit, input_upper_bound - lower_limit)
        # beta=1 implies that the lower limit is active, the output is
        # lower_limit.
        # A_x*(x - lower_limit) + A_y*(y-lower_limit) + A_beta*(1-beta) <= rhs
        # Equivalently
        # A_x*x + A_y*y - A_beta*beta <= rhs - A_beta + (A_x+A_y) * lower_limit
        beta = mip.addVars(1,
                           lb=0.,
                           ub=1.,
                           vtype=binary_var_type,
                           name="saturation_lower")
        mip.addMConstrs([
            A_x.reshape((-1, 1)),
            A_y.reshape((-1, 1)), -A_beta.reshape((-1, 1))
        ], [[input_var], [output_var], beta],
                        sense=gurobipy.GRB.LESS_EQUAL,
                        b=rhs - A_beta + (A_x + A_y) * lower_limit)
        return beta
    elif input_lower_bound >= lower_limit and input_upper_bound > upper_limit:
        # The input can saturate the upper limit. We need a binary variable to
        # determine whether the upper limit saturation happens.
        # Namely upper_limit - y = relu(upper_limit - x)
        A_x, A_y, A_beta, rhs = replace_relu_with_mixed_integer_constraint(
            upper_limit - input_upper_bound, upper_limit - input_lower_bound)
        # beta=1 implies the upper limit is active, the output is upper_limit.
        # A_x*(upper_limit-x)+A_y*(upper_limit-y)+A_beta*(1-beta)<=rhs
        # Equilvalently
        # -A_x*x -A_y*y - A_beta*beta <= rhs-A_beta - (A_x+A_y)*upper_limit
        beta = mip.addVars(1,
                           lb=0.,
                           ub=1.,
                           vtype=binary_var_type,
                           name="saturation_upper")
        mip.addMConstrs([
            -A_x.reshape((-1, 1)), -A_y.reshape((-1, 1)), -A_beta.reshape(
                (-1, 1))
        ], [[input_var], [output_var], beta],
                        sense=gurobipy.GRB.LESS_EQUAL,
                        b=rhs - A_beta - (A_x + A_y) * upper_limit)
        return beta
    else:
        # input_lower_bound < lower_limit < upper_limit < input_upper_bound. We
        # need two binary variables to determine which linear segment the
        # output lives in.

        # We introduce a slack continuous variable z
        # z - lower_limit = relu(x - lower_limit)
        # upper_limit - y = relu(upper_limit - z)
        z = mip.addVars(1,
                        lb=-gurobipy.GRB.INFINITY,
                        vtype=gurobipy.GRB.CONTINUOUS,
                        name="saturation_slack")
        # beta[0] is active when the lower limit is saturated.
        # beta[1] is active when the upper limit is saturated.
        beta = mip.addVars(2,
                           lb=0.,
                           ub=1.,
                           vtype=binary_var_type,
                           name="saturation_binary")
        # The two binary variables cannot be both active.
        mip.addLConstr([torch.tensor([1, 1], dtype=torch.float64)], [beta],
                       rhs=1.,
                       sense=gurobipy.GRB.LESS_EQUAL)
        # Now add the first constraint z - lower_limit = relu(x - lower_limit)
        A_x1, A_z1, A_beta1, rhs1 = replace_relu_with_mixed_integer_constraint(
            input_lower_bound - lower_limit, input_upper_bound - lower_limit)
        mip.addMConstrs([
            A_x1.reshape((-1, 1)),
            A_z1.reshape((-1, 1)), -A_beta1.reshape((-1, 1))
        ], [[input_var], z, [beta[0]]],
                        sense=gurobipy.GRB.LESS_EQUAL,
                        b=rhs1 - A_beta1 + (A_x1 + A_z1) * lower_limit)
        # Now add the second constraint upper_limit - y = relu(upper_limit - y)
        A_z2, A_y2, A_beta2, rhs2 = replace_relu_with_mixed_integer_constraint(
            upper_limit - input_upper_bound, upper_limit - input_lower_bound)
        mip.addMConstrs([
            -A_z2.reshape((-1, 1)), -A_y2.reshape((-1, 1)), -A_beta2.reshape(
                (-1, 1))
        ], [z, [output_var], [beta[1]]],
                        sense=gurobipy.GRB.LESS_EQUAL,
                        b=rhs2 - A_beta2 - (A_z2 + A_y2) * upper_limit)
        return beta


def compare_numpy_matrices(actual, desired, rtol, atol):
    try:
        np.testing.assert_allclose(actual, desired, rtol=rtol, atol=atol)
        res = True
    except AssertionError as err:
        res = False
        print(err)
    return res


def compute_numerical_gradient(fun, *args, **kwargs):
    """
    Compute the gradient of a function through numerical differentiation.
    @param fun The function whose gradient is evaluated. fun must takes in
    @p *argv, and returns a numpy array.
    @param *args The input to function @p fun.
    @param *kargs The options. The supported options are
                  dx  The perturbation of each input. Must be a scalar.
    @return The gradient of fun w.r.t each argument in *args. If @p fun takes
    multiple inputs (for example, fun(x, y)), then the return is a
    list of the same length as *argv, grad[i] is the gradient of fun w.r.t
    argv[i]. Namely grad[0] is ∂f/∂x, and grad[1] is ∂f/∂y. If @p fun only
    takes a single input (for example, fun(x)), then grad is a numpy
    array/matrix, namely it is ∂f/∂x.
    """
    dx = kwargs["dx"] if "dx" in kwargs else 1e-7
    assert (isinstance(dx, float))
    grad = [None] * len(args)
    perturbed_args = [np.copy(arg) for arg in args]
    fun_type_checked = False
    for arg_index, perturbed_arg in enumerate(perturbed_args):
        assert (isinstance(perturbed_arg, np.ndarray))
        assert (len(perturbed_arg.shape) == 1)
        for i in range(np.size(perturbed_arg)):
            val = perturbed_arg[i]
            perturbed_arg[i] += dx
            fun_plus = fun(*perturbed_args)
            if not fun_type_checked:
                assert (isinstance(fun_plus, np.ndarray)
                        or isinstance(fun_plus, float))
                if (isinstance(fun_plus, np.ndarray)):
                    assert (len(fun_plus.shape) == 1)
            perturbed_arg[i] -= 2 * dx
            fun_minus = fun(*perturbed_args)
            perturbed_arg[i] = val
            if (grad[arg_index] is None):
                grad[arg_index] =\
                    np.empty((np.size(fun_plus), np.size(perturbed_arg)))\
                    if isinstance(fun_plus, np.ndarray)\
                    else np.empty(np.size(perturbed_arg))
            if (isinstance(fun_plus, np.ndarray)):
                grad[arg_index][:, i] = (fun_plus - fun_minus) / (2 * dx)
            else:
                grad[arg_index][i] = (fun_plus - fun_minus) / (2 * dx)

    if (len(args) == 1):
        return grad[0]
    return grad


def torch_to_numpy(torch_array_list, squeeze=True):
    """
    Takes in a list of pytorch arrays and
    returns a list of numpy arrays. Squeezes out any
    extra dimensions as well

    @param squeeze: whether or not to squeeze extra dimesions
    @param torch_array_list: A list of torch arrays
    @return A list of numpy arrays corresponding to the torch arrays
    """
    numpy_array_list = []
    for A in torch_array_list:
        if isinstance(A, torch.Tensor):
            if squeeze:
                numpy_array_list.append(A.detach().numpy().squeeze())
            else:
                numpy_array_list.append(A.detach().numpy())
        else:
            numpy_array_list.append(A)
    return numpy_array_list


def train_model(model,
                inputs,
                labels,
                batch_size=100,
                num_epoch=1000,
                learning_rate=1e-3,
                print_loss=False):
    """
    trains a pytorch model with an L2 loss function using the
    Adam training algorithm

    @param model the pytorch model to train
    @param input the training data
    @param label the label corresponding to the training data
    @param num_epoch the number of epochs
    """
    loss_fn = torch.nn.MSELoss(reduction="sum")
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    device = next(model.parameters()).device

    data_set = torch.utils.data.TensorDataset(inputs, labels)
    data_loader = torch.utils.data.DataLoader(data_set,
                                              batch_size=batch_size,
                                              shuffle=True)

    for epoch in range(num_epoch):
        for batch_data, batch_label in data_loader:
            batch_data, batch_label = batch_data.to(device), batch_label.to(
                device)
            y_pred = model(batch_data)
            loss = loss_fn(y_pred, batch_label) / batch_size
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if epoch % 10 == 0 and print_loss:
                print(loss)

    return model


def is_polyhedron_bounded(P):
    """
    Returns true if the polyhedron P*x<=q is bounded.
    Assuming that P*x<=q is non-empty, then P*x <= q being bounded is
    equivalent to 0 being the only solution to P*x<=0.
    Equivalently I can check the following conditions:
    For all i = 1, ..., n where n is the number of columns in P, both
    min 0
    s.t P * x <= 0
        x[i] = 1
    and
    min 0
    s.t P * x <= 0
        x[i] = -1
    are infeasible.
    """
    assert (isinstance(P, torch.Tensor))
    P_np = P.detach().numpy()
    x_bar = cp.Variable(P.shape[1])
    objective = cp.Maximize(0)
    con1 = P_np @ x_bar <= np.zeros(P.shape[0])
    for i in range(P.shape[1]):
        prob = cp.Problem(objective, [con1, x_bar[i] == 1.])
        prob.solve(solver="GUROBI")
        if (prob.status != 'infeasible'):
            return False
        prob = cp.Problem(objective, [con1, x_bar[i] == -1.])
        prob.solve(solver="GUROBI")
        if (prob.status != 'infeasible'):
            return False
    return True


def get_simple_trajopt_cost(x_dim, u_dim, alpha_dim, dtype):
    """
    Returns a set of tensors that represent the a simple cost for a
    trajectory optimization problem. This is useful to write tests for
    example

    @return Q, R, Z, q, r, z, Qt, Rt, Zt, qt, rt, zt
    where
    min ∑(.5 xᵀ[n] Q x[n] + .5 uᵀ[n] R u[n] + .5 αᵀ[n] Z α[n] + qᵀx[n]
          + rᵀu[n] + zᵀα[n])
            + .5 xᵀ[N] Qt x[N] + .5 uᵀ[N] Rt u[N] + .5 αᵀ[N] Zt α[N]
            + qtᵀx[N] + rtᵀu[N] + ztᵀα[N]
    """

    Q = torch.eye(x_dim, dtype=dtype) * 0.1
    q = torch.ones(x_dim, dtype=dtype) * 0.2
    R = torch.eye(u_dim, dtype=dtype) * 1.3
    r = torch.ones(u_dim, dtype=dtype) * 0.4
    Z = torch.eye(alpha_dim, dtype=dtype) * 0.5
    z = torch.ones(alpha_dim, dtype=dtype) * 0.6

    Qt = torch.eye(x_dim, dtype=dtype) * 0.7
    qt = torch.ones(x_dim, dtype=dtype) * 0.8
    Rt = torch.eye(u_dim, dtype=dtype) * 1.9
    rt = torch.ones(u_dim, dtype=dtype) * 0.11
    Zt = torch.eye(alpha_dim, dtype=dtype) * 0.12
    zt = torch.ones(alpha_dim, dtype=dtype) * 0.13

    return (Q, R, Z, q, r, z, Qt, Rt, Zt, qt, rt, zt)


def compute_bounds_from_polytope(P, q, i):
    """
    Compute the bounds on x(j) subject to the polytopic constraint P * x <= q.
    We obtain these bounds by solving the following two LPs
    max x(j)
    s.t P * x <= q
    and
    min x(j)
    s.t P * x <= q
    @param P The constraint of the polytope.
    @param q The rhs constraint of the polytope
    @param i We want to find the bounds of x(i) subject to P * x <= q.
    @return (xi_lo, xi_up) xi_lo is the lower bound of x(i), xi_up is the upper
    bound of x(i)
    """
    if isinstance(P, torch.Tensor):
        P_np = P.detach().numpy()
    elif (isinstance(P, np.ndarray)):
        P_np = P
    else:
        raise Exception("Unknown P")
    if isinstance(q, torch.Tensor):
        q_np = q.detach().numpy()
    elif (isinstance(q, np.ndarray)):
        q_np = q
    else:
        raise Exception("Unknown q")
    model = gurobipy.Model()
    x_vars = model.addVars(P.shape[1],
                           lb=-np.inf,
                           vtype=gurobipy.GRB.CONTINUOUS)
    x = [x_vars[i] for i in range(P.shape[1])]

    for j in range(P.shape[0]):
        model.addLConstr(gurobipy.LinExpr(P_np[j].tolist(), x),
                         sense=gurobipy.GRB.LESS_EQUAL,
                         rhs=q_np[j])
    model.setObjective(gurobipy.LinExpr(1., x[i]), gurobipy.GRB.MAXIMIZE)
    model.setParam(gurobipy.GRB.Param.OutputFlag, 0)
    model.setParam(gurobipy.GRB.Param.DualReductions, 0)
    model.optimize()
    if model.status == gurobipy.GRB.OPTIMAL:
        xi_up = model.ObjVal
    elif model.status == gurobipy.GRB.UNBOUNDED:
        xi_up = np.inf
    elif model.status == gurobipy.GRB.INFEASIBLE:
        xi_up = -np.inf
    else:
        raise Exception("compute_bounds_from_polytope: unknown gurobi status.")
    model.setObjective(gurobipy.LinExpr(1., x[i]), gurobipy.GRB.MINIMIZE)
    model.optimize()
    if model.status == gurobipy.GRB.OPTIMAL:
        xi_lo = model.ObjVal
    elif model.status == gurobipy.GRB.UNBOUNDED:
        xi_lo = -np.inf
    elif model.status == gurobipy.GRB.INFEASIBLE:
        xi_lo = np.inf
    else:
        raise Exception("compute_bounds_from_polytope: unknown gurobi status.")
    return (xi_lo, xi_up)


def linear_program_cost(c, d, A_in, b_in, A_eq, b_eq):
    """
    For a linear programming problem
    max cᵀx + d
    s.t A_in * x <= b_in
        A_eq * x = b_eq
    We can write its optimal cost as a function of c, d, A_in, b_in, A_eq, b_eq
    cost(c, d, A_in, b_in, A_eq, b_eq).
    Notice at the optimal, we can select the active rows of the linear
    constraints as A_act * x = b_act, hence the cost is
    cᵀ * A_act⁻¹ * b_act + d
    @return The optimal cost as a function of the input.
    """
    x_dim = A_in.shape[1]
    check_shape_and_type(c, (x_dim, ), torch.float64)
    check_shape_and_type(d, (), torch.float64)
    num_in = A_in.shape[0]
    check_shape_and_type(A_in, (num_in, x_dim), torch.float64)
    check_shape_and_type(b_in, (num_in, ), torch.float64)
    num_eq = A_eq.shape[0]
    check_shape_and_type(A_eq, (num_eq, x_dim), torch.float64)
    check_shape_and_type(b_eq, (num_eq, ), torch.float64)

    model = gurobipy.Model()
    x_vars = model.addVars(x_dim, lb=-np.inf, vtype=gurobipy.GRB.CONTINUOUS)
    x = [x_vars[i] for i in range(x_dim)]

    for i in range(num_in):
        model.addLConstr(gurobipy.LinExpr(A_in[i].tolist(), x),
                         sense=gurobipy.GRB.LESS_EQUAL,
                         rhs=b_in[i])
    for i in range(num_eq):
        model.addLConstr(gurobipy.LinExpr(A_eq[i].tolist(), x),
                         sense=gurobipy.GRB.EQUAL,
                         rhs=b_eq[i])
    model.setObjective(gurobipy.LinExpr(c, x) + d, gurobipy.GRB.MAXIMIZE)
    model.setParam(gurobipy.GRB.Param.OutputFlag, 0)
    model.optimize()
    if (model.status != gurobipy.GRB.Status.OPTIMAL):
        return None
    # Now pick the active constraint
    x_sol = np.array([var.x for var in x])
    lhs_in = A_in.detach().numpy() @ x_sol
    active_in_flag = b_in.detach().numpy() - lhs_in < 1e-5
    num_act = np.sum(active_in_flag) + num_eq
    A_act = torch.empty((num_act, x_dim), dtype=torch.float64)
    b_act = torch.empty((num_act, ), dtype=torch.float64)
    A_act[:num_eq, :] = A_eq
    b_act[:num_eq] = b_eq
    active_in_indices = np.nonzero(active_in_flag)
    for i in range(num_act - num_eq):
        A_act[i + num_eq] = A_in[active_in_indices[i]]
        b_act[i + num_eq] = b_in[active_in_indices[i]]
    return c @ torch.inverse(A_act) @ b_act + d


def leaky_relu_interval(negative_slope, x_lo, x_up):
    """
    Given a Leaky ReLU unit and an input interval [x_lo, x_up], return the
    interval of the output
    @param negative_slope The negative slope of the leaky ReLU unit.
    @param x_lo The lower bound of the interval.
    @param x_up The upper bound of the interval.
    @return (output_lo, output_up) The output is in the interval
    [output_lo, output_up]
    """
    assert (x_up > x_lo)
    assert (type(x_lo) == type(x_up))
    if (negative_slope >= 0):
        if x_lo >= 0:
            if isinstance(x_lo, torch.Tensor):
                return (x_lo.clone(), x_up.clone())
            else:
                return (x_lo, x_up)
        elif x_up <= 0:
            return (negative_slope * x_lo, negative_slope * x_up)
        else:
            if isinstance(x_up, torch.Tensor):
                return (negative_slope * x_lo, x_up.clone())
            else:
                return (negative_slope * x_lo, x_up)
    else:
        if x_lo >= 0:
            if isinstance(x_lo, torch.Tensor):
                return (x_lo.clone(), x_up.clone())
            else:
                return (x_lo, x_up)
        elif x_up <= 0:
            return (negative_slope * x_up, negative_slope * x_lo)
        else:
            if isinstance(x_lo, torch.Tensor):
                return (0., torch.max(negative_slope * x_lo, x_up))
            else:
                return (0., np.maximum(negative_slope * x_lo, x_up))


def project_to_polyhedron(A, b, x):
    # project x to the polyhedron {y | A * y <= b}.
    if torch.all(A @ x <= b):
        return x
    else:
        # Find the closest point in A*y<= b to x
        y = cp.Variable(x.shape[0])
        objective = cp.Minimize(cp.sum_squares(x.detach().numpy() - y))
        con = A.detach().numpy() @ y <= b.detach().numpy()
        prob = cp.Problem(objective, [con])
        prob.solve(solver="GUROBI")
        return torch.from_numpy(y.value).type(x.dtype)


def setup_relu(relu_layer_width: tuple,
               params=None,
               negative_slope: float = 0.01,
               bias: bool = True,
               dtype=torch.float64):
    """
    Setup a relu network.
    @param negative_slope The negative slope of the leaky relu units.
    @param bias whether the linear layer has bias or not.
    """
    assert (isinstance(relu_layer_width, tuple))
    if params is not None:
        assert (isinstance(params, torch.Tensor))

    def set_param(linear, param_count):
        linear.weight.data = params[param_count:param_count +
                                    linear.in_features *
                                    linear.out_features].clone().reshape(
                                        (linear.out_features,
                                         linear.in_features))
        param_count += linear.in_features * linear.out_features
        if bias:
            linear.bias.data = params[param_count:param_count +
                                      linear.out_features].clone()
            param_count += linear.out_features
        return param_count

    linear_layers = [None] * (len(relu_layer_width) - 1)
    param_count = 0
    for i in range(len(linear_layers)):
        next_layer_width = relu_layer_width[i + 1]
        linear_layers[i] = torch.nn.Linear(relu_layer_width[i],
                                           next_layer_width,
                                           bias=bias).type(dtype)
        if params is None:
            pass
        else:
            param_count = set_param(linear_layers[i], param_count)
    layers = [None] * (len(linear_layers) * 2 - 1)
    for i in range(len(linear_layers) - 1):
        layers[2 * i] = linear_layers[i]
        layers[2 * i + 1] = torch.nn.LeakyReLU(negative_slope)
    layers[-1] = linear_layers[-1]
    relu = torch.nn.Sequential(*layers)
    return relu


def update_relu_params(relu, params: torch.Tensor):
    """
    Sets the weights and bias of the ReLU network to @p params.
    """
    params_count = 0
    for layer in relu:
        if isinstance(layer, torch.nn.Linear):
            layer.weight.data = params[params_count:params_count +
                                       layer.in_features *
                                       layer.out_features].reshape(
                                           (layer.out_features,
                                            layer.in_features))
            params_count += layer.in_features * layer.out_features
            if layer.bias is not None:
                layer.bias.data = params[params_count:params_count +
                                         layer.out_features]
                params_count += layer.out_features


def extract_relu_parameters(relu):
    """
    For a feedforward network with (leaky) relu activation units, extract the
    weights and bias into one tensor.
    """
    weights_biases = []
    for layer in relu:
        if isinstance(layer, torch.nn.Linear):
            weights_biases.append(layer.weight.data.reshape((-1)))
            if layer.bias is not None:
                weights_biases.append(layer.bias.data.reshape((-1)))
    return torch.cat(weights_biases)


def extract_relu_parameters_grad(relu):
    """
    For a feedforward network with (leaky) relu activation units, extract the
    weights and bias gradient into one tensor.
    """
    weights_biases_grad = []
    for layer in relu:
        if isinstance(layer, torch.nn.Linear):
            if layer.weight.grad is None:
                weights_biases_grad.append(
                    torch.zeros_like(layer.weight).reshape((-1)))
            else:
                weights_biases_grad.append(layer.weight.grad.reshape((-1)))
            if layer.bias is not None:
                if layer.bias.grad is None:
                    weights_biases_grad.append(
                        torch.zeros_like(layer.bias).reshape((-1)))
                else:
                    weights_biases_grad.append(layer.bias.grad.reshape((-1)))
    return torch.cat(weights_biases_grad)


def extract_relu_structure(relu_network):
    """
    Get the linear_layer_width, negative_slope and bias flag.
    """
    linear_layer_width = []
    negative_slope = None
    bias = None
    for layer in relu_network:
        if isinstance(layer, torch.nn.Linear):
            if len(linear_layer_width) == 0:
                # first layer
                linear_layer_width.extend(
                    [layer.in_features, layer.out_features])
            else:
                linear_layer_width.append(layer.out_features)
            if layer.bias is not None:
                assert (bias is None or bias)
                bias = True
            else:
                assert (bias is None or not bias)
                bias = False
        elif isinstance(layer, torch.nn.ReLU):
            if negative_slope is None:
                negative_slope = 0.
            else:
                assert (negative_slope == 0.)
        elif isinstance(layer, torch.nn.LeakyReLU):
            if negative_slope is None:
                negative_slope = layer.negative_slope
            else:
                assert (negative_slope == layer.negative_slope)
        else:
            raise Exception("extract_relu_structure(): unknown layer.")
    return tuple(linear_layer_width), negative_slope, bias


def get_meshgrid_samples(lower, upper, mesh_size: tuple, dtype) ->\
        torch.Tensor:
    """
    Often we want to get the mesh samples in a box lower <= x <= upper.
    This returns a torch tensor of size (prod(mesh_size), sample_dim), where
    each row is a sample in the meshgrid.
    """
    sample_dim = len(mesh_size)
    assert (len(upper) == sample_dim)
    assert (len(lower) == sample_dim)
    assert (len(mesh_size) == sample_dim)
    meshes = []
    for i in range(sample_dim):
        meshes.append(
            torch.linspace(lower[i], upper[i], mesh_size[i], dtype=dtype))
    mesh_tensors = torch.meshgrid(*meshes)
    return torch.cat(
        [mesh_tensors[i].reshape((-1, 1)) for i in range(sample_dim)], dim=1)


def save_second_order_forward_model(forward_relu, q_equilibrium, u_equilibrium,
                                    dt, file_path):
    linear_layer_width, negative_slope, bias = extract_relu_structure(
        forward_relu)
    torch.save(
        {
            "linear_layer_width": linear_layer_width,
            "state_dict": forward_relu.state_dict(),
            "negative_slope": negative_slope,
            "bias": bias,
            "q_equilibrium": q_equilibrium,
            "u_equilibrium": u_equilibrium,
            "dt": dt
        }, file_path)


def save_lyapunov_model(lyapunov_relu, V_lambda, lyapunov_positivity_epsilon,
                        lyapunov_derivative_epsilon, eps_type, R_options,
                        file_path):
    linear_layer_width, negative_slope, bias = extract_relu_structure(
        lyapunov_relu)
    saved_params = {
        "linear_layer_width": linear_layer_width,
        "state_dict": lyapunov_relu.state_dict(),
        "negative_slope": negative_slope,
        "V_lambda": V_lambda,
        "lyapunov_positivity_epsilon": lyapunov_positivity_epsilon,
        "lyapunov_derivative_epsilon": lyapunov_derivative_epsilon,
        "eps_type": eps_type,
        "bias": bias,
        "R": R_options.R(),
        "fixed_R": R_options.fixed_R
    }
    R_params = R_options.extract_params()
    saved_params.update(R_params)
    torch.save(saved_params, file_path)


def save_controller_model(controller_relu, x_lo, x_up, u_lo, u_up, file_path):
    linear_layer_width, negative_slope, bias = extract_relu_structure(
        controller_relu)
    torch.save(
        {
            "linear_layer_width": linear_layer_width,
            "state_dict": controller_relu.state_dict(),
            "negative_slope": negative_slope,
            "x_lo": x_lo,
            "x_up": x_up,
            "u_lo": u_lo,
            "u_up": u_up,
            "bias": bias
        }, file_path)


def save_control_barrier_function(barrier_relu, x_star, c, epsilon, x_lo, x_up,
                                  u_lo, u_up, inf_norm_term, save_path):
    linear_layer_width, negative_slope, bias = extract_relu_structure(
        barrier_relu)
    inf_norm_data = None if inf_norm_term is None else {
        "R": inf_norm_term.R,
        "p": inf_norm_term.p
    }
    torch.save(
        {
            "linear_layer_width": linear_layer_width,
            "negative_slope": negative_slope,
            "bias": bias,
            "state_dict": barrier_relu.state_dict(),
            "x_star": x_star,
            "c": c,
            "epsilon": epsilon,
            "x_lo": x_lo,
            "x_up": x_up,
            "u_lo": u_lo,
            "u_up": u_up,
            "inf_norm_term": inf_norm_data
        }, save_path)


def get_gurobi_terminate_if_callback(threshold=0.):
    """
    helper function that returns a callback that terminates gurobi as
    soon as a counterexample is found. A counterexamples happens when
    the objective > threshold
    @param threshold float terminate if the objective becomes more than
    threshold
    """
    def gurobi_terminate_if(model, where):
        """
        callback
        @param model, where see Gurobi callback documentation
        """
        if where == gurobipy.GRB.Callback.MIPNODE:
            solcnt = model.cbGet(gurobipy.GRB.Callback.MIPNODE_SOLCNT)
            if solcnt > 0:
                status = model.cbGet(gurobipy.GRB.Callback.MIPNODE_STATUS)
                if status == gurobipy.GRB.Status.OPTIMAL:
                    objbst = model.cbGet(gurobipy.GRB.Callback.MIPNODE_OBJBST)
                    if objbst > threshold:
                        model.terminate()

    return gurobi_terminate_if


def network_zero_grad(network):
    """
    Set the gradient of all parameters in the network to zero.
    """
    for layer in network:
        if isinstance(layer, torch.nn.Linear):
            if layer.weight.grad is not None:
                layer.weight.grad.data.zero_()
            if layer.bias is not None and layer.bias.grad is not None:
                layer.bias.grad.data.zero_()
        elif isinstance(layer, torch.nn.ReLU) or isinstance(
                layer, torch.nn.LeakyReLU):
            pass
        else:
            raise Exception("network_zero_grad: unsupported layer.")


class SigmoidAnneal:
    def __init__(self, dtype, lo, up, center_step, steps_lo_to_up):
        """
        provides a sigmoid function that can be used to do weight scheduling
        for training
        @dtype torch data type
        @param lo float lower value for the sigmoid
        @param up float upper value for the sigmoid
        @param center_step int step where the sigmoid will be halfway up
        @param steps_lo_to_up in width of the sigmoid
        """
        self.dtype = dtype
        self.lo = lo
        self.up = up
        self.center_step = center_step
        self.steps_lo_to_up = steps_lo_to_up
        self.sigmoid = torch.nn.Sigmoid()

    def __call__(self, step):
        """
        @param step int step number
        @return value of the sigmoid at that point
        """
        return self.lo + (self.up - self.lo) * self.sigmoid(
            torch.tensor(
                float(step - self.center_step) / float(self.steps_lo_to_up),
                dtype=self.dtype))


def step_system(system, x_start, steps):
    """
    Step forward a closed loop system for N steps. Returns the whole path.
    """
    path = [x_start]
    with torch.no_grad():
        for i in range(steps):
            path.append(system.step_forward(path[-1]))
    return path


def simulate_plant_with_controller(plant, controller_relu, t_span,
                                   x_equilibrium, u_equilibrium, u_lo, u_up,
                                   x0):
    """
    Simulate a continuous time system with a controller. The controller is
    computed as u = saturate(ϕ(x) − ϕ(x*) + u*)
    """
    def dyn(t, x):
        with torch.no_grad():
            x_torch = torch.from_numpy(x)
            u_torch = controller_relu(x_torch)\
                - controller_relu(x_equilibrium) + u_equilibrium
            u = torch.max(torch.min(u_torch, u_up), u_lo).detach().numpy()
        return plant.dynamics(x, u)

    result = scipy.integrate.solve_ivp(dyn,
                                       t_span,
                                       x0,
                                       t_eval=np.arange(start=t_span[0],
                                                        stop=t_span[1],
                                                        step=0.01))
    return result


def train_approximator(dataset,
                       model,
                       output_fun,
                       batch_size,
                       num_epochs,
                       lr,
                       additional_variable=None,
                       output_fun_args=dict(),
                       verbose=True):
    """
    @param additional_variable A list of torch tensors (with
    requires_grad=True), such that we will optimize the model together with
    additional_variable.
    @param output_fun_args A dictionnary of additional arguments to pass to
    output_fun
    """
    train_set_size = int(len(dataset) * 0.8)
    test_set_size = len(dataset) - train_set_size
    train_set, test_set = torch.utils.data.random_split(
        dataset, [train_set_size, test_set_size])
    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=batch_size,
                                               shuffle=True)

    variables = model.parameters() if additional_variable is None else\
        list(model.parameters()) + additional_variable
    optimizer = torch.optim.Adam(variables, lr=lr)
    loss = torch.nn.MSELoss()

    model_params = []
    for epoch in range(num_epochs):
        running_loss = 0.
        for i, data in enumerate(train_loader, 0):
            input_samples, target = data
            optimizer.zero_grad()

            output_samples = output_fun(model, input_samples,
                                        **output_fun_args)
            batch_loss = loss(output_samples, target)
            batch_loss.backward()
            optimizer.step()

            running_loss += batch_loss.item()
        test_input_samples, test_target = test_set[:]
        test_output_samples = output_fun(model, test_input_samples,
                                         **output_fun_args)
        test_loss = loss(test_output_samples, test_target)

        if verbose:
            print(f"epoch {epoch} training loss " +
                  f"{running_loss/len(train_loader)}," +
                  f" test loss {test_loss}")
        model_params.append(extract_relu_parameters(model))
    pass


def uniform_sample_in_box(lo: torch.Tensor, hi: torch.Tensor,
                          num_samples) -> torch.Tensor:
    """
    Take uniform samples in the box lo <= x <= hi.
    @return samples A num_samples x x_dim tensor.
    """
    x_dim = lo.numel()
    assert (hi.shape == (x_dim, ))
    samples = torch.rand(num_samples, x_dim, dtype=torch.float64)
    samples = samples @ torch.diag(hi - lo)
    samples += torch.reshape(lo, (1, x_dim))
    return samples


def uniform_sample_on_box_boundary(lo: torch.Tensor, hi: torch.Tensor,
                                   num_samples) -> torch.Tensor:
    """
    Uniformly samples on the boundary of the box lo <= x <= hi
    """
    samples_in_box = uniform_sample_in_box(lo, hi, num_samples)
    x_dim = lo.numel()
    boundary_face_rand = torch.rand((num_samples, )) * 2 - 1
    for i in range(num_samples):
        boundary_face_index = int(
            (torch.abs(boundary_face_rand[i]) * x_dim).item())
        samples_in_box[i, boundary_face_index] = lo[
            boundary_face_index] if boundary_face_rand[i] < 0 else hi[
                boundary_face_index]
    return samples_in_box


def relu_network_gradient(relu_network,
                          x: torch.Tensor,
                          *,
                          zero_tol: float = 0.) -> torch.Tensor:
    """
    For a fully-connected neural network ϕ(x) with (leaky) relu units,
    compute the gradient ∂ϕ/∂x.
    Notice that since (leaky) ReLU unit is non-differentiable at 0, we
    consider both the left and right gradient at 0. This function computes all
    the possible ∂ϕ/∂x, where the (leaky) ReLU units takes both the left and
    the right gradient.

    Args:
      relu_network: A fully connected neural network with (leaky) ReLU units.
      x: The network input.
      zero_tol: When the absolute value of the ReLU unit input is less than
      zero_tol, we consider both the left and right derivative of the ReLU
      unit.
    Return:
      dphi_dx: A tensor of shape (num_possible_gradients, phi_dim, x_dim)
      where phi_dim is the dimension of the network output ϕ(x).
    """
    assert (x.shape == (relu_network[0].in_features, ))
    assert (zero_tol >= 0)
    dphi_dx = torch.eye(relu_network[0].in_features,
                        dtype=x.dtype).unsqueeze(0)
    layer_input = x
    for layer in relu_network:
        if (isinstance(layer, torch.nn.Linear)):
            dphi_dx = layer.weight.unsqueeze(0) @ dphi_dx
        else:
            if isinstance(layer, torch.nn.ReLU):
                c = 0
            elif isinstance(layer, torch.nn.LeakyReLU):
                c = layer.negative_slope
            else:
                raise Exception(
                    "relu_network_gradient(): We only accept linear layer, " +
                    "relu layer or leaky ReLU layer")
            for i in range(layer_input.shape[0]):
                if layer_input[i] > zero_tol:
                    pass
                elif layer_input[i] < -zero_tol:
                    dphi_dx[:, i, :] *= c
                else:
                    # (leaky) ReLU unit has input 0. We need to consider both
                    # the left and the right gradient.
                    dphi_dx = torch.cat((dphi_dx, dphi_dx), dim=0)
                    dphi_dx[int(dphi_dx.shape[0] / 2):, i, :] *= c
        # Propagate the layer value.
        layer_input = layer(layer_input)

    return dphi_dx


def l1_gradient(x: torch.Tensor,
                *,
                zero_tol: float = 0.,
                subgradient_samples: np.ndarray = None) -> torch.Tensor:
    """
    Compute all the possible gradient of the 1-norm |x|₁
    Notice that when x(i)=0, the 1-norm is non-differentiable. We consider
    both the left and right gradient, and also the sampled subgradient

    Args:
      zero_tol: When abs(x(i)) <= zero_tol, we consider both the left and right
      gradient.
      subgradient_samples: an array of sampled subgradient (in the range
      (-1, 1)) that will be included in grad.

    Return:
      grad: A torch tensor of shape (num_possible_gradient, x_dim), where
            num_possible_gradient is
            power(2+subgradient_samples.size, number of x(i)=0).
    """
    assert (len(x.shape) == 1)
    assert (zero_tol >= 0)
    if subgradient_samples is not None:
        assert (isinstance(subgradient_samples, np.ndarray))
        assert (np.all(subgradient_samples > -1)
                and np.all(subgradient_samples < 1))

    if not torch.any(torch.abs(x) <= zero_tol):
        return torch.sign(x).reshape((1, -1))
    elif torch.sum(torch.abs(x) < zero_tol) == 1:
        s = torch.sign(x)
        s_plus = s.clone()
        s_plus[torch.abs(s_plus) <= zero_tol] = 1
        s_minus = s.clone()
        s_minus[torch.abs(s_minus) <= zero_tol] = -1
        if subgradient_samples is None:
            return torch.vstack((s_plus, s_minus))
        else:
            s_list = [s_plus, s_minus]
            for i in range(subgradient_samples.size):
                s_subgradient = s.clone()
                s_subgradient[torch.abs(s_subgradient) <=
                              zero_tol] = subgradient_samples[i]
                s_list.append(s_subgradient)
            return torch.vstack(s_list)
    else:
        # Denote the first index of x[i] == 0 as k
        # Get the gradient of 1-norm(x[:k])
        first_zero_index = (torch.abs(x) <= zero_tol).nonzero(
            as_tuple=True)[0][0]
        grad_before = torch.sign(x[:first_zero_index])
        # The gradient w.r.t x[k] is 1 and -1 (plus subgradient_samples)
        # Also compute the gradient w.r.t x[k+1:]
        grad_after = l1_gradient(x[first_zero_index + 1:],
                                 zero_tol=zero_tol,
                                 subgradient_samples=subgradient_samples)
        if subgradient_samples is None:
            grad = torch.hstack(
                (grad_before.repeat((2 * grad_after.shape[0], 1)),
                 torch.vstack((torch.ones(
                     (grad_after.shape[0], 1), dtype=x.dtype), -torch.ones(
                         (grad_after.shape[0], 1), dtype=x.dtype))),
                 torch.vstack((grad_after, grad_after))))
        else:
            all_subgradient = torch.from_numpy(
                np.concatenate((np.array([1., -1.]), subgradient_samples)))
            grad = torch.hstack((grad_before.repeat(
                ((2 + subgradient_samples.size) * grad_after.shape[0],
                 1)), (all_subgradient.repeat(
                     (grad_after.shape[0], 1)).T).reshape((-1, 1)),
                                 grad_after.repeat(
                                     (2 + subgradient_samples.size, 1))))
        return grad


def l_infinity_gradient(x, *, max_tol=0.) -> torch.Tensor:
    """
    Compute the gradient of the infinity-norm |x|∞
    Args:
      max_tol: If |x[i]| is within max_tol to |x|∞, then we consider the
      gradient w.r.t x[i].

    Return:
      gradient: A (num_possible_gradient x x_dim) size tensor. gradient[i] is
      the i'th possible gradient.
    """
    assert (isinstance(x, torch.Tensor))
    x_dim = x.shape[0]
    assert (x.shape == (x_dim, ))
    inf_norm = torch.norm(x, p=float("inf"))
    if inf_norm <= max_tol:
        return torch.cat((torch.eye(
            x_dim, dtype=x.dtype), -torch.eye(x_dim, dtype=x.dtype)),
                         dim=0)
    grad = torch.where(torch.abs(x - inf_norm) <= max_tol, 1, 0) + \
        torch.where(torch.abs(x + inf_norm) <= max_tol, -1, 0)
    nonzero_grad_indices = torch.nonzero(grad).squeeze(1).tolist()
    if len(nonzero_grad_indices) == 1:
        return grad.unsqueeze(0).type(x.dtype)
    all_grad = torch.zeros((len(nonzero_grad_indices), x_dim), dtype=x.dtype)
    for (i, grad_index) in enumerate(nonzero_grad_indices):
        all_grad[i, grad_index] = grad[grad_index]
    return all_grad


def box_boundary(x_lo, x_up) -> gurobi_torch_mip.MixedIntegerConstraintsReturn:
    """
    Given a box region x_lo <= x <= x_up, return the mixed-integer constraint
    that x is on the boundary of this box region.
    For each x[i], we introduce  binary variable b1[i], b2[i], with the
    constraint
    x[i] >= x_lo[i] + (x_up[i] - x_lo[i]) * b1[i]
    x[i] <= x_up[i] - (x_up[i] - x_lo[i]) * b2[i]
    ∑ᵢ b1[i] + ∑ᵢb2[i] = 1
    """
    mixed_integer_cnstr = gurobi_torch_mip.MixedIntegerConstraintsReturn()
    nx = x_lo.shape[0]
    assert (x_lo.shape == (nx, ))
    assert (x_up.shape == (nx, ))
    dtype = x_lo.dtype
    # The constraint is
    # -x[i] + (x_up[i] - x_lo[i])*b1[i] <= -x_lo[i]
    # x[i] + (x_up[i] - x_lo[i])*b2[i] <= x_up[i]
    mixed_integer_cnstr.Ain_input = torch.cat(
        (-torch.eye(nx, dtype=dtype), torch.eye(nx, dtype=dtype)), dim=0)
    mixed_integer_cnstr.Ain_binary = torch.block_diag(torch.diag(x_up - x_lo),
                                                      torch.diag(x_up - x_lo))
    mixed_integer_cnstr.rhs_in = torch.cat((-x_lo, x_up))
    # ∑ᵢ b1[i] + ∑ᵢb2[i] = 1
    mixed_integer_cnstr.Aeq_binary = torch.ones((1, 2 * nx), dtype=dtype)
    mixed_integer_cnstr.rhs_eq = torch.tensor([1], dtype=dtype)
    return mixed_integer_cnstr


def minkowski_sum(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Given two tensors x, y, such that x.shape[i] = y.shape[i], i>=1. We want
    to return the Minkowski sum of x[i] + y[j] for each pair (i, j).

    Return:
      sum: sum[i * y.shape[0] + j] = x[i] + y[j]
    """
    assert (len(x.shape) == len(y.shape))
    assert (x.shape[1:] == y.shape[1:])
    nx = x.shape[0]
    ny = y.shape[0]
    result = y.repeat(*([
        nx,
    ] + [1] * (len(x.shape) - 1))) + x.unsqueeze(1).repeat(
        *([1, ny] + [1] *
          (len(x.shape) - 1))).reshape([nx * ny] + list(x.shape[1:]))
    return result
