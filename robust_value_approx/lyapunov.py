# -*- coding: utf-8 -*-
import gurobipy
import numpy as np
import torch

import robust_value_approx.relu_to_optimization as relu_to_optimization
import robust_value_approx.hybrid_linear_system as hybrid_linear_system


class LyapunovDiscreteTimeHybridSystem:
    """
    For a discrete time autonomous hybrid linear system
    x[n+1] = Aᵢ*x[n] + cᵢ
    if Pᵢ * x[n] <= qᵢ
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
        return (milp, x, x_next, s, gamma, z, z_next, beta, beta_next)
        TODO(hongkai.dai): return Ain_r, Ain_zeta, Aeq_r, Aeq_zeta, rhs_in,
        rhs_eq, r, zeta instead
        """

        relu_free_pattern = relu_to_optimization.ReLUFreePattern(
            relu_model, self.system.dtype)

        milp = gurobipy.Model()

        # x is the variable x[n]
        def addVars(num_vars, lb, vtype, name):
            var = milp.addVars(num_vars, lb=lb, vtype=vtype, name=name)
            return [var[i] for i in range(num_vars)]

        x = addVars(
            self.system.x_dim, lb=-gurobipy.GRB.INFINITY,
            vtype=gurobipy.GRB.CONTINUOUS, name="x[n]")
        # x_next is the variable x[n+1]
        x_next = addVars(
            self.system.x_dim, lb=-gurobipy.GRB.INFINITY,
            vtype=gurobipy.GRB.CONTINUOUS, name="x[n+1]")
        # s is the slack variable to convert hybrid linear system to
        # mixed-integer linear constraint.
        s = addVars(
            self.system.x_dim * self.system.num_modes,
            lb=-gurobipy.GRB.INFINITY, vtype=gurobipy.GRB.CONTINUOUS, name="s")
        # gamma is the binary variable determining the hybrid mode of x[n]
        gamma = addVars(
            self.system.num_modes, lb=0, vtype=gurobipy.GRB.BINARY,
            name="gamma")
        # Now add the milp constraint to formulate the hybrid linear system.
        (Aeq_s1, Aeq_gamma1, Ain_x1, Ain_s1, Ain_gamma1, rhs_in1) =\
            self.system.mixed_integer_constraints()
        # Add the constraint x[n+1] = Aeq_s1 * s + Aeq_gamma1 * gamma
        for i in range(self.system.x_dim):
            lin_expr = x_next[i] - gurobipy.LinExpr(Aeq_s1[i].tolist(), s) -\
                gurobipy.LinExpr(Aeq_gamma1[i].tolist(), gamma)
            milp.addLConstr(lin_expr, sense=gurobipy.GRB.EQUAL, rhs=0.)
        # Now add the constraint
        # Ain_x1 * x + Ain_s1 * s + Ain_gamma1 * gamma <= rhs_in1
        for i in range(Ain_x1.shape[0]):
            lin_expr = gurobipy.LinExpr(Ain_x1[i].tolist(), x) +\
                gurobipy.LinExpr(Ain_s1[i].tolist(), s) +\
                gurobipy.LinExpr(Ain_gamma1[i].tolist(), gamma)
            milp.addLConstr(lin_expr, sense=gurobipy.GRB.LESS_EQUAL,
                            rhs=rhs_in1[i])
        # Now add the constraint that sum gamma = 1
        milp.addLConstr(
            gurobipy.LinExpr(np.ones((self.system.num_modes, 1)), gamma),
            sense=gurobipy.GRB.EQUAL, rhs=1.)

        # Add the mixed-integer constraint that formulates the output of
        # ReLU(x[n]).
        # z is the slack variable to write the output of ReLU(x[n]) with mixed
        # integer linear constraints.
        (Ain_x2, Ain_z, Ain_beta, rhs_in2, Aeq_x2, Aeq_z, Aeq_beta, rhs_eq2,
         a_out, b_out, z_lo, z_up) = relu_free_pattern.output_constraint(
             relu_model, torch.from_numpy(self.system.x_lo_all),
             torch.from_numpy(self.system.x_up_all))
        z = addVars(Ain_z.shape[1], lb=0, vtype=gurobipy.GRB.CONTINUOUS,
                    name="z[n]")
        beta = addVars(Ain_beta.shape[1], lb=0., vtype=gurobipy.GRB.BINARY,
                       name="beta[n]")

        for i in range(Ain_x2.shape[0]):
            milp.addLConstr(
                gurobipy.LinExpr(Ain_x2[i].tolist(), x) +
                gurobipy.LinExpr(Ain_z[i].tolist(), z) +
                gurobipy.LinExpr(Ain_beta[i].tolist(), beta),
                sense=gurobipy.GRB.LESS_EQUAL, rhs=rhs_in2[i],
                name="milp_relu_xn[" + str(i) + "]")

        for i in range(Aeq_x2.shape[0]):
            milp.addLConstr(
                gurobipy.LinExpr(Aeq_x2[i].tolist(), x) +
                gurobipy.LinExpr(Aeq_z[i].tolist(), z) +
                gurobipy.LinExpr(Aeq_beta[i].tolist(), beta),
                sense=gurobipy.GRB.EQUAL, rhs=rhs_eq2[i],
                name="milp_relu_xn[" + str(i) + "]")

        # Now write the ReLU output ReLU(x[n+1]) as mixed integer linear
        # constraints
        z_next = addVars(Ain_z.shape[1], lb=0, vtype=gurobipy.GRB.CONTINUOUS,
                         name="z[n+1]")
        beta_next = addVars(Ain_beta.shape[1], lb=0.,
                            vtype=gurobipy.GRB.BINARY, name="beta[n+1]")
        milp.update()

        for i in range(Ain_x2.shape[0]):
            milp.addLConstr(
                gurobipy.LinExpr(Ain_x2[i].tolist(), x_next) +
                gurobipy.LinExpr(Ain_z[i].tolist(), z_next) +
                gurobipy.LinExpr(Ain_beta[i].tolist(), beta_next),
                sense=gurobipy.GRB.LESS_EQUAL, rhs=rhs_in2[i],
                name="milp_relu_xnext[" + str(i) + "]")
        for i in range(Aeq_x2.shape[0]):
            milp.addLConstr(
                gurobipy.LinExpr(Aeq_x2[i].tolist(), x_next) +
                gurobipy.LinExpr(Aeq_z[i].tolist(), z_next) +
                gurobipy.LinExpr(Aeq_beta[i].tolist(), beta_next),
                sense=gurobipy.GRB.EQUAL, rhs=rhs_eq2[i],
                name="milp_relu_xnext[" + str(i) + "]")

        # The cost function is max ReLU(x[n+1]) - ReLU(x[n])
        milp.setObjective(
            gurobipy.LinExpr(a_out.tolist(), z_next) -
            gurobipy.LinExpr(a_out.tolist(), z), gurobipy.GRB.MAXIMIZE)

        return (milp, x, x_next, s, gamma, z, z_next, beta, beta_next)
