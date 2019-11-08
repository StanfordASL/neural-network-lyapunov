# -*- coding: utf-8 -*-
import cvxpy as cp
import numpy as np
import torch

import robust_value_approx.relu_to_optimization as relu_to_optimization
import robust_value_approx.hybrid_linear_system as hybrid_linear_system

class LyapunovDiscreteTimeHybridSystem:
    """
    For a discrete time hybrid linear system
    x[n+1] = Aᵢ*x[n] + Bᵢ*u[n] + cᵢ
    if Pᵢ * [x[n]; u[n]] <= qᵢ
    i = 1, ..., K.
    we want to learn a ReLU network as the Lyapunov function for the system.
    The condition for the Lyapunov function is that
    V[x[n+1]] <= V[x[n]] ∀x[n]
    We will first formulate this condition as the optimal cost of a certain
    mixed-integer linear program (MILP) being non-positive. The optimal cost
    is the loss function of our neural network. We will compute the gradient of
    this loss function w.r.t to network weights/bias, and then call gradient
    based optimization (SGD/Adam) to reduce the loss.
    """

    def __init__(self, system):
        """
        @param system A AutonomousHybridLinearSystem instance.
        """
        assert(isinstance(
            system, hybrid_linear_system.AutonomousHybridLinearSystem))
        self.system = system

    def lyapunov_as_milp(self, relu_model):
        """
        Formulate the Lyapunov condition V[x[n+1]] <= V[x[n]] ∀x[n] as the
        maximal cost of an MILP is no larger than 0. This function returns the
        MILP formulation.
        max cᵣᵀ * r + c_zetaᵀ * ζ + c_constant
        s.t Ain_r * r + Ain_zeta * ζ <= rhs_in
            Aeq_r * r + Aeq_zeta * ζ = rhs_eq
        where r includes all continuous variables, and ζ includes all binary
        variables.
        @param relu_model A pytorch ReLU network. Notice that we want the last
        layer to be a ReLU activation layer, so as to guarantee the Lyapunov
        function to be non-negative.
        """

        relu_free_pattern = relu_to_optimization.ReLUFreePattern(relu_model)

        continuous_variable_count = 0
        binary_variable_count = 0

        def add_continuous_var(num_var):
            var_indices = range(continuous_variable_count,
                                continuous_variable_count + num_var)
            continuous_variable_count += num_var
            return var_indices

        def add_binary_var(num_var):
            var_indices = range(binary_variable_count,
                                binary_variable_count + num_var)
            binary_variable_count += num_var
            return var_indices

        # xn_index is the index of x[n] in r
        xn_index = add_continuous_var(self.system.x_dim)
        # beta_xn_index is the index of beta (the activation variables) for
        # x[n] being the network input.
        beta_xn_index = add_binary_var(relu_free_pattern.num_relu_units)
        # s_index is the index of s (the slack variables for the hybrid
        # system) for x[n]
        s_index = add_continuous_var(
            self.system.x_dim * self.system.num_modes)
        # gamma_index is the index of the binary variables for the hybrid mode
        # of x[n]
        gamma_index = add_binary_var(self.system.num_modes)
