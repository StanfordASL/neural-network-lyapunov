from IPython.display import clear_output
import numpy as np
import torch


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
    @param (A_x, A_s, A_alpha, rhs) A_x, A_s, A_alpha, rhs are all 4 x 1 column
    vectors.
    """
    assert(x_lo <= x_up)
    A_x = torch.tensor([[0], [0], [1], [-1]], dtype=dtype)
    A_s = torch.tensor([[-1], [1], [-1], [1]], dtype=dtype)
    A_alpha = torch.tensor([[x_lo], [-x_up], [x_up], [-x_lo]], dtype=dtype)
    rhs = torch.tensor([[0], [0], [x_up], [-x_lo]], dtype=dtype)
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
        grad[:, i:i+1] = (y_plus - y_minus).reshape((y_plus.size, 1))\
            / (2 * dx)
    return grad
