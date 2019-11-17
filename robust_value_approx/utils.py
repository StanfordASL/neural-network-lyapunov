from IPython.display import clear_output
import numpy as np
import torch
import cvxpy as cp
import gurobipy


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
    assert(x_lo <= x_up)
    A_x = torch.tensor([0, 0, 1, -1], dtype=dtype)
    A_s = torch.tensor([-1, 1, -1, 1], dtype=dtype)
    A_alpha = torch.tensor([x_lo, -x_up, x_up, -x_lo], dtype=dtype)
    rhs = torch.tensor([0, 0, x_up, -x_lo], dtype=dtype)
    return (A_x, A_s, A_alpha, rhs)


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
    A_x = torch.tensor([[0], [1], [0], [-1]], dtype=dtype)
    A_y = torch.tensor([[-1], [-1], [1], [1]], dtype=dtype)
    A_alpha = torch.tensor([[0], [0], [-x_up], [-x_lo]], dtype=dtype)
    rhs = torch.tensor([[0], [0], [0], [-x_lo]], dtype=dtype)
    return (A_x, A_y, A_alpha, rhs)


def compare_numpy_matrices(actual, desired, rtol, atol):
    try:
        np.testing.assert_allclose(actual, desired, rtol=rtol, atol=atol)
        res = True
    except AssertionError as err:
        res = False
        print(err)
    return res


def compute_numerical_gradient(fun, x, dx=1e-7):
    grad = None
    for i in range(x.size):
        x_plus = x.copy()
        x_plus[i] += dx
        x_minus = x.copy()
        x_minus[i] -= dx
        y_plus = fun(x_plus)
        y_minus = fun(x_minus)
        if (grad is None):
            grad = np.zeros((y_plus.size, x.size))
        grad[:, i:i + 1] = (y_plus - y_minus).reshape((y_plus.size, 1))\
            / (2 * dx)
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
                num_epoch=1000, learning_rate=1e-3):
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
            # if epoch % 10 == 0:
            #     print(loss)

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
    x = cp.Variable(P.shape[1])
    con = [P_np @ x <= q_np]
    prob = cp.Problem(cp.Maximize(x[i]), con)
    xi_up = prob.solve()
    prob = cp.Problem(cp.Minimize(x[i]), con)
    xi_lo = prob.solve()
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


def gurobi_milp_to_standard_form(milp):
    """
    For an MILP in stored in a gurobi model, we extract the model to a standard
    form
        max cᵣᵀ * r + c_zetaᵀ * ζ + c_constant
        s.t Ain_r * r + Ain_zeta * ζ <= rhs_in
            Aeq_r * r + Aeq_zeta * ζ = rhs_eq
    where r contains all the continuous variables, and ζ contains all the
    binary variables.
    @param milp A gurobi model instance, has to be an MILP.
    @return r, zeta, c_r, c_zeta, c_constant, Ain_r, Ain_zeta, rhs_in, Aeq_r,
    Aeq_zeta, rhs_eq
    """
    assert(isinstance(milp, gurobipy.Model))
    assert(milp.getAttr(gurobipy.GRB.Attr.NumSOS) == 0)
    assert(milp.getAttr(gurobipy.GRB.Attr.NumQConstrs) == 0)
    assert(milp.getAttr(gurobipy.GRB.Attr.NumGenConstrs) == 0)
    assert(milp.getAttr(gurobipy.GRB.Attr.NumPWLObjVars) == 0)
    num_bin_vars = milp.getAttr(gurobipy.GRB.Attr.NumBinVars)
    assert(milp.getAttr(gurobipy.GRB.Attr.NumIntVars) == num_bin_vars)
    assert(milp.getAttr(gurobipy.GRB.Attr.IsMIP))
    assert(not milp.getAttr(gurobipy.GRB.Attr.IsQP))
    assert(not milp.getAttr(gurobipy.GRB.Attr.IsQCP))
    assert(not milp.getAttr(gurobipy.GRB.Attr.IsMultiObj))
    num_cont_vars = milp.getAttr(gurobipy.GRB.Attr.NumVars) - num_bin_vars

    # Load all variables
    r = [None] * num_cont_vars
    zeta = [None] * num_bin_vars
    bin_var_count = 0
    cont_var_count = 0
    num_cont_var_bounds = 0
    for var in milp.getVars():
        if (var.vtype == gurobipy.GRB.CONTINUOUS):
            r[cont_var_count] = var
            cont_var_count += 1
            if var.lb != -gurobipy.GRB.INFINITY:
                num_cont_var_bounds += 1
            if var.ub != gurobipy.GRB.INFINITY:
                num_cont_var_bounds += 1
        elif (var.vtype == gurobipy.GRB.BINARY):
            zeta[bin_var_count] = var
            bin_var_count += 1
        else:
            raise Exception("Only allow continuous or binary variables.")

    # Parse all constraints into standard form

