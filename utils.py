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
