# -*- coding: utf-8 -*-
import cvxpy as cp
import numpy as np
import torch

import robust_value_approx.relu_to_optimization as relu_to_optimization


class ControlLyapunovFixedActivationPattern:
    """
    For a fixed activation pattern on a ReLU network, verify that the output of
    the network is a valid control Lyapunov function. i.e., there exists a
    control action u, such that the value function goes downhill.

    Mathematically, if we denote the ReLU output as η(x), and the dynamics as
    ẋ = Ax + Bu + d, then the control Lyapunov condition is
    ∃ u ∈ U, s.t ∂η/∂x * Ax + ∂η/∂x * Bu + ∂η/∂x * d ≤ 0
    Assuming that the admissible set of control action U is a polytope, with
    vertices uᵢ*, and the ReLU network output along this activation pattern is
    η(x) = gᵀx+h, while P*x ≤ q, then the control Lyapunov condition that the
    optimal cost of the following problem is no larger than 0.
    max gᵀAx + min_i gᵀBuᵢ + gᵀd
    s.t Px ≤ q
    """

    def __init__(self, g, P, q, A, B, d, u_vertices):
        """
        @param g The gradient of the ReLU output along this pattern.
        @param P P*x <= q is the constraint that x activates this pattern.
        @param q P*x <= q is the constraint that x activates this pattern.
        @param A The dynamics is ẋ = Ax + Bu + d
        @param B The dynamics is ẋ = Ax + Bu + d
        @param d The dynamics is ẋ = Ax + Bu + d
        @param u_vertices u_vertices[:,i] is the i'th vertex of the control
        action polytope U.
        """
        xdim = g.size
        self.x = cp.Variable((xdim, 1))
        self.dtype = g.dtype
        assert(P.dtype == self.dtype)
        assert(q.dtype == self.dtype)
        assert(A.dtype == self.dtype)
        assert(B.dtype == self.dtype)
        assert(d.dtype == self.dtype)
        assert(u_vertices.dtype == self.dtype)
        assert(g.shape[0] == xdim)
        assert(g.shape[1] == 1)
        assert(P.shape[1] == xdim)
        assert(P.shape[0] == q.shape[0])
        assert(q.shape[1] == 1)
        assert(A.shape[0] == A.shape[1])
        assert(A.shape[0] == xdim)
        assert(B.shape[0] == A.shape[0])
        assert(B.shape[1] == u_vertices.shape[0])
        assert(d.shape[0] == xdim)
        assert(d.shape[1] == 1)
        cost_constant = np.min(g.T.dot(B * u_vertices)) + g.T.dot(d)

        self.objective = g.T * (self.x) + cost_constant
        self.constraints = [P * self.x <= q]

    def construct_program(self) -> cp.Problem:
        prob = cp.Problem(cp.Maximize(self.objective), self.constraints)
        return prob


class ControlLyapunovFreeActivationPattern:
    """
    Given a candidate Lyapunov function (value function) approximated by a ReLU
    network η(x), we want to verify that the network satisfies the control
    Lyapunov condition
    ∀ x, ∃ u ∈ U, s.t η(f(x, u)) - η(x) ≤ 0 (discrete time system)
    ∀ x, ∃ u ∈ U, s.t ∂η/∂x * f(x,u) ≤ 0 (continuous time system)
    where f(x, u) is the state dynamics.
    """

    def __init__(self, model, dtype):
        """
        @param model A ReLU network.
        @param dtype The data type for the pytorch tensor. Please use the same
        datatype as the tensors in @p model.
        """
        self.model = model
        self.dtype = dtype
        self.relu_free_pattern = relu_to_optimization.\
            ReLUFreePattern(self.model, self.dtype)

    def generate_program_verify_continuous_affine_system(self, A_dyn, B_dyn,
                                                         d_dyn, u_vertices,
                                                         x_lo, x_up):
        """
        For a continuous affine system
        ẋ = Ax + Bu + d
        with the admissible set of control action being ConvexHull(u_vertices),
        generate the constraint to verify that ReLU network satisfies the
        control Lyapunov condition
        ∀ x, ∃ u ∈ ConvexHull(u), s.t ∂η /∂ x (Ax + Bu + d) ≤ 0
        Since the gradient of the ReLU network can be written as αᵀM, subject
        to B1*α+B2*β ≤ e (for details, please refer to
        ReLUFreePattern.output_gradient()), the control Lyapunov condition is
        equivalent to
        maxₓ min_u∈U αᵀMAx + αᵀMBu + αᵀMd
        s.t B1*α+B2*β ≤ e

        If the maximal of this optimization problem is no larger than 0, then
        we prove that the network satisfies the control Lyapunov condition.

        In order to get rid of the inner minimization, we introduce a new slack
        variable t, and change the optimization problem to
        max αᵀMAx + t + αᵀMd
        s.t t ≤ αᵀMBuᵢ ∀ i
            B1*α+B2*β ≤ e
        where uᵢ is the i'th vertex of the the set of admissible control
        actions.
        Note that in the objective, we still have αᵀMAx where the decision
        variable α and x multiply together. To resolve this, we introduce
        new decision variables s(i, j) = α(i)x(j), and the cost function
        becomes a linear function of s. To enforce s(i, j) = α(i)x(j), we
        also impose
        the following constraint
        xₗₒ(j)α(i) - s(i, j) ≤ 0
        s(i, j) - xᵤₚ(j)α(i) ≤ 0
        -x(j) + s(i,j) - xₗₒ(j)α(i) ≤ -xₗₒ(j)
        x(j) - s(i, j) + xᵤₚ(j)α(i) ≤ xᵤₚ(j)
        where [xₗₒ, xᵤₚ] is the bound of x.
        We write the whole problem concisely in the following form
        max c₁ᵀs + t + c₂ᵀα
        s.t Ain1*x + Ain2*s + Ain3*t + Ain4*α + Ain5*β ≤ rhs
        @note that this function does NOT impose the constraint that β is
        the right ReLU activation pattern for the input x. This constraint
        is imposed separately in ReLUFreePattern.output_constraint() function.

        @param A_dyn The A matrix in ẋ = Ax + Bu + d
        @param B_dyn The B matrix in ẋ = Ax + Bu + d
        @param d_dyn The d column vector in ẋ = Ax + Bu + d
        @param u_vertices A num_u x num_vertices matrix. u_vertices[:,i] is the
        i'th vertex of the admissible set of control actions.
        @param x_lo A 1-D array, the lower bound of the network input x.
        @param x_up A 1-D array, the upper bound of the network input x.
        @return (c1, c2, Ain1, Ain2, Ain3, Ain4, Ain5, rhs) c1, c2, rhs are
        column vectors. Ain1, Ain2, Ain3, Ain4, Ain5 are matrices.
        """
        x_size = self.relu_free_pattern.x_size
        assert(A_dyn.dtype == self.dtype)
        assert(B_dyn.dtype == self.dtype)
        assert(d_dyn.dtype == self.dtype)
        assert(u_vertices.dtype == self.dtype)
        assert(x_lo.dtype == self.dtype)
        assert(x_up.dtype == self.dtype)
        assert(A_dyn.shape[0] == x_size)
        assert(A_dyn.shape[1] == x_size)
        num_u = B_dyn.shape[1]
        assert(d_dyn.shape[0] == x_size)
        assert(d_dyn.shape[1] == 1)
        assert(u_vertices.shape[0] == num_u)
        num_u_vertices = u_vertices.shape[1]
        assert(x_lo.shape == (x_size,))
        assert(x_up.shape == (x_size,))
        assert(torch.all(torch.le(x_lo, x_up)))

        (M, B1, B2, e) = self.relu_free_pattern.output_gradient(self.model)
        num_alpha = M.shape[0]
        num_s = num_alpha * x_size
        MA = M @ A_dyn
        MB = M @ B_dyn
        num_ineq = 4 * num_s + num_u_vertices + B1.shape[0]
        # We will store Ain1, Ain2, Ain3, Ain4, Ain5 in sparse format, with
        # row/column indices and nonzero values.
        Ain1_indices = torch.empty((2, 2 * num_s), dtype=torch.int64)
        Ain1_val = torch.empty(Ain1_indices.shape[1], dtype=self.dtype)
        Ain2_indices = torch.empty((2, 4 * num_s), dtype=torch.int64)
        Ain2_val = torch.empty(Ain2_indices.shape[1], dtype=self.dtype)
        Ain3_indices = torch.empty((2, num_u_vertices), dtype=torch.int64)
        Ain3_val = torch.empty(Ain3_indices.shape[1], dtype=self.dtype)
        Ain4_indices = torch.empty((2, 4 * num_s + num_alpha * num_u_vertices
                                    + B1.numel()), dtype=torch.int64)
        Ain4_val = torch.empty(Ain4_indices.shape[1], dtype=self.dtype)
        Ain5_indices = torch.empty((2, B2.numel()), dtype=torch.int64)
        Ain5_val = torch.empty(Ain5_indices.shape[1], dtype=self.dtype)
        rhs = torch.empty((num_ineq, 1), dtype=self.dtype)
        # First enforce the constraint s(i, j) = α(i)x(j). We impose
        # the following constraint
        # xₗₒ(j)α(i) - s(i, j) ≤ 0
        # s(i, j) - xᵤₚ(j)α(i) ≤ 0
        # -x(j) + s(i,j) - xₗₒ(j)α(i) ≤ -xₗₒ(j)
        # x(j) - s(i, j) + xᵤₚ(j)α(i) ≤ xᵤₚ(j)
        ineq_count = 0
        Ain_indices = [[], Ain1_indices, Ain2_indices, Ain3_indices,
                       Ain4_indices, Ain5_indices]
        Ain_val = [[], Ain1_val, Ain2_val, Ain3_val, Ain4_val, Ain5_val]
        Ain_entry_count = [[], 0, 0, 0, 0, 0]

        def compute_s_index(s_row, s_col):
            return s_row * x_size + s_col

        def insert_Ain_entry(Ain_index, row_index, col_index, val):
            Ain_indices[Ain_index][0][Ain_entry_count[Ain_index]] = row_index
            Ain_indices[Ain_index][1][Ain_entry_count[Ain_index]] = col_index
            Ain_val[Ain_index][Ain_entry_count[Ain_index]] = val
            Ain_entry_count[Ain_index] += 1

        def insert_Ain1_entry(row_index, col_index, val):
            insert_Ain_entry(1, row_index, col_index, val)

        def insert_Ain2_entry(row_index, col_index, val):
            insert_Ain_entry(2, row_index, col_index, val)

        def insert_Ain3_entry(row_index, col_index, val):
            insert_Ain_entry(3, row_index, col_index, val)

        def insert_Ain4_entry(row_index, col_index, val):
            insert_Ain_entry(4, row_index, col_index, val)

        def insert_Ain5_entry(row_index, col_index, val):
            insert_Ain_entry(5, row_index, col_index, val)

        for i in range(num_alpha):
            for j in range(x_size):
                s_index = compute_s_index(i, j)
                # xₗₒ(j)α(i) - s(i, j) ≤ 0
                insert_Ain2_entry(ineq_count, s_index, -1.)
                insert_Ain4_entry(ineq_count, i, x_lo[j])
                rhs[ineq_count][0] = 0.
                ineq_count += 1
                # s(i, j) - xᵤₚ(j)α(i) ≤ 0
                insert_Ain2_entry(ineq_count, s_index, 1.)
                insert_Ain4_entry(ineq_count, i, -x_up[j])
                rhs[ineq_count][0] = 0.
                ineq_count += 1
                # -x(j) + s(i,j) - xₗₒ(j)α(i) ≤ -xₗₒ(j)
                insert_Ain1_entry(ineq_count, j, -1.)
                insert_Ain2_entry(ineq_count, s_index, 1.)
                insert_Ain4_entry(ineq_count, i, -x_lo[j])
                rhs[ineq_count][0] = -x_lo[j]
                ineq_count += 1
                # x(j) - s(i, j) + xᵤₚ(j)α(i) ≤ xᵤₚ(j)
                insert_Ain1_entry(ineq_count, j, 1.)
                insert_Ain2_entry(ineq_count, s_index, -1.)
                insert_Ain4_entry(ineq_count, i, x_up[j])
                rhs[ineq_count][0] = x_up[j]
                ineq_count += 1

        # Now add the constraint t ≤ αᵀMBuᵢ ∀ i
        for i in range(num_u_vertices):
            insert_Ain3_entry(ineq_count, 0, 1.)
            MBui = MB @ u_vertices[:, i].reshape((-1, 1))
            for j in range(num_alpha):
                insert_Ain4_entry(ineq_count, j, -MBui[j][0])
            rhs[ineq_count][0] = 0.
            ineq_count += 1

        # Now add the constraint B1*α+B2*β ≤ e
        for i in range(B1.shape[0]):
            for j in range(num_alpha):
                if B1[i][j] != 0:
                    insert_Ain4_entry(ineq_count, j, B1[i][j])
            for j in range(B2.shape[1]):
                if B2[i][j] != 0:
                    insert_Ain5_entry(ineq_count, j, B2[i][j])
            rhs[ineq_count][0] = e[i][0]
            ineq_count += 1
        Ain1_entry_count = Ain_entry_count[1]
        Ain2_entry_count = Ain_entry_count[2]
        Ain3_entry_count = Ain_entry_count[3]
        Ain4_entry_count = Ain_entry_count[4]
        Ain5_entry_count = Ain_entry_count[5]

        Ain1 = torch.sparse.DoubleTensor(Ain1_indices[:, :Ain1_entry_count],
                                         Ain1_val[:Ain1_entry_count],
                                         torch.Size((ineq_count, x_size)))
        Ain2 = torch.sparse.DoubleTensor(Ain2_indices[:, :Ain2_entry_count],
                                         Ain2_val[:Ain2_entry_count],
                                         torch.Size((ineq_count, num_s)))
        Ain3 = torch.sparse.DoubleTensor(Ain3_indices[:, :Ain3_entry_count],
                                         Ain3_val[:Ain3_entry_count],
                                         torch.Size((ineq_count, 1)))
        Ain4 = torch.sparse.DoubleTensor(Ain4_indices[:, :Ain4_entry_count],
                                         Ain4_val[:Ain4_entry_count],
                                         torch.Size((ineq_count, num_alpha)))
        Ain5 = torch.sparse.DoubleTensor(Ain5_indices[:, :Ain5_entry_count],
                                         Ain5_val[:Ain5_entry_count],
                                         torch.Size((ineq_count, B2.shape[1])))

        # Now compute c1 and c2.
        c1 = torch.empty((num_s, 1), dtype=self.dtype)
        for i in range(num_alpha):
            for j in range(x_size):
                c1[compute_s_index(i, j)][0] = MA[i][j]
        c2 = M @ d_dyn

        return (c1, c2, Ain1, Ain2, Ain3, Ain4, Ain5, rhs)
