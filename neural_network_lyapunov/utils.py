from IPython.display import clear_output
import numpy as np
import torch
import cvxpy as cp
import gurobipy
import neural_network_lyapunov.gurobi_torch_mip as gurobi_torch_mip


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

    clear_output(wait=True)
    text = "Progress: [{0}] {1:.1f}%".format("#" * block + "-" *
                                             (bar_length - block),
                                             progress * 100)
    print(text)


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


def check_shape_and_type(A, shape_expected, dtype_expected):
    assert(A.shape == shape_expected)
    assert(A.dtype == dtype_expected)


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
    assert(isinstance(x_lo, torch.Tensor))
    assert(x_lo <= x_up)
    A_x = torch.tensor([0, 0, 1, -1], dtype=dtype)
    A_s = torch.tensor([-1, 1, -1, 1], dtype=dtype)
    A_alpha = torch.stack((x_lo, -x_up, x_up, -x_lo))
    rhs = torch.zeros(4, dtype=dtype)
    rhs = torch.stack((
        torch.tensor(0, dtype=dtype), torch.tensor(0, dtype=dtype), x_up,
        -x_lo))
    return (A_x, A_s, A_alpha, rhs)


def leaky_relu_gradient_times_x(
        x_lo, x_up, negative_slope, dtype=torch.float64):
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
    assert(isinstance(x_lo, torch.Tensor))
    assert(x_up >= x_lo)
    dtype = x_up.dtype
    A_x = torch.tensor([-1, 1, negative_slope, -negative_slope], dtype=dtype)
    A_y = torch.tensor([1, -1, -1, 1], dtype=dtype)
    A_alpha = torch.stack((
        (negative_slope-1.) * x_lo, (1. - negative_slope) * x_up,
        (1. - negative_slope) * x_lo, (negative_slope - 1.) * x_up))
    rhs = torch.stack((
        (negative_slope - 1.) * x_lo, (1. - negative_slope) * x_up,
        torch.tensor(0, dtype=dtype), torch.tensor(0, dtype=dtype)))
    if negative_slope < 1.:
        return (A_x, A_y, A_alpha, rhs)
    else:
        return (-A_x, -A_y, -A_alpha, -rhs)


def replace_absolute_value_with_mixed_integer_constraint(
        x_lo, x_up, dtype=torch.float64):
    """
    For a variable x in the interval [x_lo, x_up], where x_lo < 0 < x_up,
    if we denote the absolute value |x| as s, and introduce a binary variable
    alpha, such that
    alpha = 1 => x >= 0
    alpha = 0 => x <= 0
    then s, x and alpha shouls satisfy the following mixed-integer constraint
    s >= x
    s >= -x
    -x + s - 2 * x_lo*alpha <= -2 * x_lo
    x + s - 2 * x_up * alpha <= 0
    We write this constraint in the conside form as
    Ain_x * x + Ain_s * s + Ain_alpha * alpha <= rhs_in
    @return (Ain_x, Ain_s, Ain_alpha, rhs_in)
    """
    if isinstance(x_lo, float):
        x_lo = torch.tensor(x_lo, dtype=dtype)
    if isinstance(x_up, float):
        x_up = torch.tensor(x_up, dtype=dtype)
    assert(isinstance(x_lo, torch.Tensor))
    assert(isinstance(x_up, torch.Tensor))
    assert(x_lo < 0)
    assert(x_up > 0)
    Ain_x = torch.tensor([1, -1, -1, 1], dtype=dtype)
    Ain_s = torch.tensor([-1, -1, 1, 1], dtype=dtype)
    Ain_alpha = torch.stack((
        torch.tensor(0, dtype=dtype), torch.tensor(0, dtype=dtype), -2*x_lo,
        -2*x_up))
    rhs_in = torch.stack((
        torch.tensor(0, dtype=dtype), torch.tensor(0, dtype=dtype), -2*x_lo,
        torch.tensor(0, dtype=dtype)))
    return (Ain_x, Ain_s, Ain_alpha, rhs_in)


def replace_relu_with_mixed_integer_constraint(x_lo, x_up,
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
    assert(x_lo < 0)
    assert(x_up > 0)
    A_x = torch.tensor([0, 1, 0, -1], dtype=dtype)
    A_y = torch.tensor([-1, -1, 1, 1], dtype=dtype)
    A_beta = torch.zeros(4, dtype=dtype)
    A_beta[2] = -x_up
    A_beta[3] = -x_lo
    rhs = torch.zeros(4, dtype=dtype)
    rhs[3] = -x_lo
    return (A_x, A_y, A_beta, rhs)


def replace_leaky_relu_mixed_integer_constraint(
        negative_slope, x_lo, x_up, dtype=torch.float64):
    """
    For input x ∈ [x_lo, x_up] (and x_lo < 0 < x_up), the leaky relu output
    y satisfies
    y = x if x >= 0
    y = a*x if x <= 0
    where a is the negative slope. We can use a binary variable β to
    indicate whether the ReLU unit is active or not. Namely
    β = 1 => x >= 0
    β = 0 => x <= 0
    We can writ the relationship between (x, y, β) as mixed-integer linear
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
    assert(x_lo < 0)
    assert(x_up > 0)
    A_x = torch.tensor([1., negative_slope, -negative_slope, -1], dtype=dtype)
    A_y = torch.tensor([-1., -1., 1., 1.], dtype=dtype)
    A_beta = torch.zeros(4, dtype=dtype)
    A_beta[2] = (negative_slope-1) * x_up
    A_beta[3] = (negative_slope-1) * x_lo
    rhs = torch.zeros(4, dtype=dtype)
    rhs[3] = (negative_slope-1) * x_lo
    if negative_slope <= 1:
        return (A_x, A_y, A_beta, rhs)
    else:
        return (-A_x, -A_y, -A_beta, -rhs)


def add_saturation_as_mixed_integer_constraint(
    mip, input_var, output_var, lower_limit, upper_limit, input_lower_bound,
        input_upper_bound):
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
    assert(isinstance(mip, gurobi_torch_mip.GurobiTorchMIP))
    assert(isinstance(input_var, gurobipy.Var))
    assert(isinstance(output_var, gurobipy.Var))
    if input_upper_bound <= lower_limit:
        # The input x will always be <= lower_limit, the output will always be
        # lower_limit.
        mip.addLConstr(
            [torch.ones([1], dtype=torch.float64)], [[output_var]],
            sense=gurobipy.GRB.EQUAL, rhs=lower_limit)
        return []
    elif input_lower_bound >= upper_limit:
        # The input x will always be >= upper_limit, the output will always be
        # upper_limit.
        mip.addLConstr(
            [torch.ones([1], dtype=torch.float64)], [[output_var]],
            sense=gurobipy.GRB.EQUAL, rhs=upper_limit)
        return []
    elif input_lower_bound >= lower_limit and input_upper_bound <= upper_limit:
        # The input is never saturated, the output equals to the input.
        mip.addLConstr(
            [torch.tensor([1, -1], dtype=torch.float64)],
            [[input_var, output_var]], sense=gurobipy.GRB.EQUAL, rhs=0.)
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
        beta = mip.addVars(
            1, lb=-gurobipy.GRB.INFINITY, vtype=gurobipy.GRB.BINARY,
            name="saturation_lower")
        mip.addMConstrs(
            [A_x.reshape((-1, 1)), A_y.reshape((-1, 1)),
             -A_beta.reshape((-1, 1))], [[input_var], [output_var], beta],
            sense=gurobipy.GRB.LESS_EQUAL,
            b=rhs - A_beta + (A_x+A_y) * lower_limit)
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
        beta = mip.addVars(
            1, lb=-gurobipy.GRB.INFINITY, vtype=gurobipy.GRB.BINARY,
            name="saturation_upper")
        mip.addMConstrs(
            [-A_x.reshape((-1, 1)), -A_y.reshape((-1, 1)),
             -A_beta.reshape((-1, 1))], [[input_var], [output_var], beta],
            sense=gurobipy.GRB.LESS_EQUAL,
            b=rhs-A_beta-(A_x+A_y)*upper_limit)
        return beta
    else:
        # input_lower_bound < lower_limit < upper_limit < input_upper_bound. We
        # need two binary variables to determine which linear segment the
        # output lives in.

        # We introduce a slack continuous variable z
        # z - lower_limit = relu(x - lower_limit)
        # upper_limit - y = relu(upper_limit - z)
        z = mip.addVars(
            1, lb=-gurobipy.GRB.INFINITY, vtype=gurobipy.GRB.CONTINUOUS,
            name="saturation_slack")
        # beta[0] is active when the lower limit is saturated.
        # beta[1] is active when the upper limit is saturated.
        beta = mip.addVars(
            2, lb=-gurobipy.GRB.INFINITY, vtype=gurobipy.GRB.BINARY,
            name="saturation_binary")
        # The two binary variables cannot be both active.
        mip.addLConstr(
            [torch.tensor([1, 1], dtype=torch.float64)], [beta], rhs=1.,
            sense=gurobipy.GRB.LESS_EQUAL)
        # Now add the first constraint z - lower_limit = relu(x - lower_limit)
        A_x1, A_z1, A_beta1, rhs1 = replace_relu_with_mixed_integer_constraint(
            input_lower_bound - lower_limit, input_upper_bound - lower_limit)
        mip.addMConstrs(
            [A_x1.reshape((-1, 1)), A_z1.reshape((-1, 1)),
             -A_beta1.reshape((-1, 1))], [[input_var], z, [beta[0]]],
            sense=gurobipy.GRB.LESS_EQUAL,
            b=rhs1 - A_beta1 + (A_x1+A_z1) * lower_limit)
        # Now add the second constraint upper_limit - y = relu(upper_limit - y)
        A_z2, A_y2, A_beta2, rhs2 = replace_relu_with_mixed_integer_constraint(
            upper_limit - input_upper_bound, upper_limit - input_lower_bound)
        mip.addMConstrs(
            [-A_z2.reshape((-1, 1)), -A_y2.reshape((-1, 1)),
             -A_beta2.reshape((-1, 1))], [z, [output_var], [beta[1]]],
            sense=gurobipy.GRB.LESS_EQUAL,
            b=rhs2-A_beta2-(A_z2+A_y2)*upper_limit)
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
    assert(isinstance(dx, float))
    grad = [None] * len(args)
    perturbed_args = [np.copy(arg) for arg in args]
    fun_type_checked = False
    for arg_index, perturbed_arg in enumerate(perturbed_args):
        assert(isinstance(perturbed_arg, np.ndarray))
        assert(len(perturbed_arg.shape) == 1)
        for i in range(np.size(perturbed_arg)):
            val = perturbed_arg[i]
            perturbed_arg[i] += dx
            fun_plus = fun(*perturbed_args)
            if not fun_type_checked:
                assert(isinstance(fun_plus, np.ndarray) or
                       isinstance(fun_plus, float))
                if (isinstance(fun_plus, np.ndarray)):
                    assert(len(fun_plus.shape) == 1)
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


def train_model(model, inputs, labels, batch_size=100,
                num_epoch=1000, learning_rate=1e-3, print_loss=False):
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
    data_loader = torch.utils.data.DataLoader(
        data_set, batch_size=batch_size, shuffle=True)

    for epoch in range(num_epoch):
        for batch_data, batch_label in data_loader:
            batch_data, batch_label = batch_data.to(
                device), batch_label.to(device)
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
    assert(isinstance(P, torch.Tensor))
    P_np = P.detach().numpy()
    x_bar = cp.Variable(P.shape[1])
    objective = cp.Maximize(0)
    con1 = P_np @ x_bar <= np.zeros(P.shape[0])
    for i in range(P.shape[1]):
        prob = cp.Problem(objective, [con1, x_bar[i] == 1.])
        prob.solve()
        if (prob.status != 'infeasible'):
            return False
        prob = cp.Problem(objective, [con1, x_bar[i] == -1.])
        prob.solve()
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

    return(Q, R, Z, q, r, z, Qt, Rt, Zt, qt, rt, zt)


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
    elif(isinstance(P, np.ndarray)):
        P_np = P
    else:
        raise Exception("Unknown P")
    if isinstance(q, torch.Tensor):
        q_np = q.detach().numpy()
    elif(isinstance(q, np.ndarray)):
        q_np = q
    else:
        raise Exception("Unknown q")
    model = gurobipy.Model()
    x_vars = model.addVars(
        P.shape[1], lb=-np.inf, vtype=gurobipy.GRB.CONTINUOUS)
    x = [x_vars[i] for i in range(P.shape[1])]

    for j in range(P.shape[0]):
        model.addLConstr(
            gurobipy.LinExpr(P_np[j].tolist(), x),
            sense=gurobipy.GRB.LESS_EQUAL, rhs=q_np[j])
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
    check_shape_and_type(b_in, (num_in,), torch.float64)
    num_eq = A_eq.shape[0]
    check_shape_and_type(A_eq, (num_eq, x_dim), torch.float64)
    check_shape_and_type(b_eq, (num_eq,), torch.float64)

    model = gurobipy.Model()
    x_vars = model.addVars(x_dim, lb=-np.inf, vtype=gurobipy.GRB.CONTINUOUS)
    x = [x_vars[i] for i in range(x_dim)]

    for i in range(num_in):
        model.addLConstr(
            gurobipy.LinExpr(A_in[i].tolist(), x),
            sense=gurobipy.GRB.LESS_EQUAL, rhs=b_in[i])
    for i in range(num_eq):
        model.addLConstr(
            gurobipy.LinExpr(A_eq[i].tolist(), x),
            sense=gurobipy.GRB.EQUAL, rhs=b_eq[i])
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
    b_act = torch.empty((num_act,), dtype=torch.float64)
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
    assert(x_up > x_lo)
    assert(type(x_lo) == type(x_up))
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
        prob.solve()
        return torch.from_numpy(y.value).type(x.dtype)


def setup_relu(
    relu_layer_width: tuple, params=None, negative_slope: float = 0.01,
        bias: bool = True, dtype=torch.float64):
    """
    Setup a relu network.
    @param negative_slope The negative slope of the leaky relu units.
    @param bias whether the linear layer has bias or not.
    """
    assert(isinstance(relu_layer_width, tuple))
    if params is not None:
        assert(isinstance(params, torch.Tensor))

    def set_param(linear, param_count):
        linear.weight.data = params[
            param_count: param_count +
            linear.in_features * linear.out_features].clone().reshape((
                linear.out_features, linear.in_features))
        param_count += linear.in_features * linear.out_features
        if bias:
            linear.bias.data = params[
                param_count: param_count + linear.out_features].clone()
            param_count += linear.out_features
        return param_count

    linear_layers = [None] * (len(relu_layer_width) - 1)
    param_count = 0
    for i in range(len(linear_layers)):
        next_layer_width = relu_layer_width[i+1]
        linear_layers[i] = torch.nn.Linear(
            relu_layer_width[i], next_layer_width, bias=bias).type(dtype)
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
            layer.weight.data = params[
                params_count:
                params_count + layer.in_features * layer.out_features].reshape(
                    (layer.out_features, layer.in_features))
            params_count += layer.in_features * layer.out_features
            if layer.bias is not None:
                layer.bias.data = params[
                    params_count: params_count + layer.out_features]
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
                assert(bias is None or bias)
                bias = True
            else:
                assert(bias is None or not bias)
                bias = False
        elif isinstance(layer, torch.nn.ReLU):
            if negative_slope is None:
                negative_slope = 0.
            else:
                assert(negative_slope == 0.)
        elif isinstance(layer, torch.nn.LeakyReLU):
            if negative_slope is None:
                negative_slope = layer.negative_slope
            else:
                assert(negative_slope == layer.negative_slope)
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
    assert(len(upper) == sample_dim)
    assert(len(lower) == sample_dim)
    assert(len(mesh_size) == sample_dim)
    meshes = []
    for i in range(sample_dim):
        meshes.append(torch.linspace(
            lower[i], upper[i], mesh_size[i], dtype=dtype))
    mesh_tensors = torch.meshgrid(*meshes)
    return torch.cat([
        mesh_tensors[i].reshape((-1, 1)) for i in range(sample_dim)], dim=1)


def save_second_order_forward_model(
        forward_relu, q_equilibrium, u_equilibrium, dt, file_path):
    linear_layer_width, negative_slope, bias = extract_relu_structure(
        forward_relu)
    torch.save({"linear_layer_width": linear_layer_width,
                "state_dict": forward_relu.state_dict(),
                "negative_slope": negative_slope,
                "bias": bias, "q_equilibrium": q_equilibrium,
                "u_equilibrium": u_equilibrium, "dt": dt}, file_path)


def save_lyapunov_model(
    lyapunov_relu, V_lambda, lyapunov_positivity_epsilon,
        lyapunov_derivative_epsilon, file_path):
    linear_layer_width, negative_slope, bias = extract_relu_structure(
        lyapunov_relu)
    torch.save({"linear_layer_width": linear_layer_width,
                "state_dict": lyapunov_relu.state_dict(),
                "negative_slope": negative_slope, "V_lambda": V_lambda,
                "lyapunov_positivity_epsilon": lyapunov_positivity_epsilon,
                "lyapunov_derivative_epsilon": lyapunov_derivative_epsilon,
                "bias": bias}, file_path)


def save_controller_model(controller_relu, x_lo, x_up, u_lo, u_up, file_path):
    linear_layer_width, negative_slope, bias = extract_relu_structure(
        controller_relu)
    torch.save({"linear_layer_width": linear_layer_width,
                "state_dict": controller_relu.state_dict(),
                "negative_slope": negative_slope, "x_lo": x_lo, "x_up": x_up,
                "u_lo": u_lo, "u_up": u_up, "bias": bias}, file_path)


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
        return self.lo + (self.up - self.lo) * self.sigmoid(torch.tensor(
            float(step - self.center_step) / float(self.steps_lo_to_up),
            dtype=self.dtype))


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
