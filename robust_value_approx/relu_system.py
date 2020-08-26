import torch
import numpy as np
import cvxpy as cp
from scipy.integrate import solve_ivp

import robust_value_approx.relu_to_optimization as relu_to_optimization
import robust_value_approx.utils as utils
from robust_value_approx.utils import (
    check_shape_and_type,
    replace_binary_continuous_product,
    is_polyhedron_bounded,
)


class ReLUSystem:
    """
    This system models a discrete time autonomous hybrid linear system 
    (piecewise affine system) using a feedforward neural network with ReLU 
    activations
    x[n+1] = relu(x[n])
    """

    def __init__(self, x_dim, dtype, x_lo, x_up, relu_model):
        """
        @param x_dim The dimension of x.
        @param dtype The torch datatype
        """
        self.dtype = dtype
        self.x_dim = x_dim
        self.x_lo = x_lo
        self.x_up = x_up
        self.relu_free_pattern = relu_to_optimization.ReLUFreePattern(
            relu_model, dtype)  

    @property
    def x_lo_all(self):
        return self.x_lo.detach().numpy()

    @property
    def x_up_all(self):
        return self.x_up.detach().numpy()

    def mixed_integer_constraints(self, relu_model):
        """
        @return ()
        @note 1. We do not impose the constraint that one and only one mode
                 is active. The user should impose this constraint separately.
        """
        (Ain_x, Ain_s, Ain_gamma, rhs_in, Aeq_x, Aeq_s, Aeq_gamma, rhs_eq, 
         Aout_s, Cout, z_pre_relu_lo, z_pre_relu_up, z_post_relu_lo,
         z_post_relu_up) = self.relu_free_pattern.output_constraint(
            relu_model, self.x_lo, self.x_up)

        gamma_dim = Ain_gamma.shape[1]
        Aout_gamma = torch.zeros((self.x_dim, gamma_dim), dtype=self.dtype)

        return (Aout_s, Aout_gamma, Cout,
            Ain_x, Ain_s, Ain_gamma, rhs_in,
            Aeq_x, Aeq_s, Aeq_gamma, rhs_eq)

    class StepForwardException(Exception):
        pass

    def step_forward(self, x, relu_model):
        """
        Compute the one-step forward simulation x[n+1]
        @param x The starting state.
        @return x_next The next continuous state.
        """
        return relu_model(x)