import torch
import numpy as np

import robust_value_approx.utils as utils
from robust_value_approx.utils import (
    check_shape_and_type,
    replace_binary_continuous_product,
    is_polyhedron_bounded,
)


class HybridLinearSystem:
    """
    This system models the hybrid linear system
    x[n+1] = Aᵢ*x[n] + Bᵢ*u[n] + cᵢ
    if Pᵢ * [x[n]; u[n]] <= qᵢ
    i = 1, ..., K.
    Namely there are K different modes, each mode constrains the state/control
    of the dynamical system to be within a polytope. Inside each mode, the
    discrete time dynamics is affine.
    Note that the polytope Pᵢ * [x[n]; u[n]] <= qᵢ has to be bounded.
    """

    def __init__(self, x_dim, u_dim, dtype):
        """
        @param x_dim The dimension of x.
        @param u_dim The dimension of u.
        @param dtype The torch datatype of A, B, c, P, q.
        """
        self.dtype = dtype

        self.x_dim = x_dim
        self.u_dim = u_dim
        self.A = []
        self.B = []
        self.c = []
        self.P = []
        self.q = []
        # self.x_lo[i] (self.u_lo[i]) stores the lower bound of x (u) in mode i
        # (inferred from Pᵢ[x; u] ≤ qᵢ)
        self.x_lo = []
        self.u_lo = []
        # self.x_up[i] (self.u_up[i]) stores the upper bound of x (u) in mode i
        # (inferred from Pᵢ[x; u] ≤ qᵢ)
        self.x_up = []
        self.u_up = []
        self.num_modes = 0

    def add_mode(self, Ai, Bi, ci, Pi, qi, check_polyhedron_bounded=False):
        """
        Add a new mode
        x[n+1] = Aᵢ*x[n] + Bᵢ*u[n] + cᵢ
        if Pᵢ * [x[n]; u[n]] <= qᵢ
        @param Ai A x_dim * x_dim torch matrix.
        @param Bi A x_dim * u_dim torch matrix.
        @param ci A x_dim * 1 torch column vector.
        @param Pi A num_constraint * (x_dim + u_dim) torch matrix.
        @param qi A num_constraint * 1 torch column vector.
        @param check_polyhedron_bounded Set to True if you want to check that
        the polytope Pᵢ * [x[n]; u[n]] <= qᵢ is bounded. Default to False.
        @note that the polytope Pᵢ * [x[n]; u[n]] <= qᵢ has to be bounded.
        """
        check_shape_and_type(Ai, (self.x_dim, self.x_dim), self.dtype)
        check_shape_and_type(Bi, (self.x_dim, self.u_dim), self.dtype)
        check_shape_and_type(ci, (self.x_dim,), self.dtype)
        num_constraint = Pi.shape[0]
        check_shape_and_type(Pi, (num_constraint, self.x_dim + self.u_dim),
                             self.dtype)
        check_shape_and_type(qi, (num_constraint,), self.dtype)
        if (check_polyhedron_bounded):
            assert(is_polyhedron_bounded(Pi))
        self.A.append(Ai)
        self.B.append(Bi)
        self.c.append(ci)
        self.P.append(Pi)
        self.q.append(qi)
        x_lo = np.empty(self.x_dim)
        x_up = np.empty(self.x_dim)
        u_lo = np.empty(self.u_dim)
        u_up = np.empty(self.u_dim)
        for j in range(self.x_dim):
            (x_lo[j], x_up[j]) = utils.compute_bounds_from_polytope(Pi, qi, j)
        for j in range(self.u_dim):
            (u_lo[j], u_up[j]) = utils.compute_bounds_from_polytope(
                Pi, qi, self.x_dim + j)
        self.x_lo.append(x_lo)
        self.x_up.append(x_up)
        self.u_lo.append(u_lo)
        self.u_up.append(u_up)
        self.num_modes += 1

    def mixed_integer_constraints(self, x_lo, x_up, u_lo, u_up):
        """
        We can rewrite the hybrid dynamics as mixed integer linear constraints.
        We denote αᵢ = 1 if the system is in mode i.
        x[n+1] = ∑ᵢαᵢAᵢx[n] + αᵢBᵢu[n]+αᵢcᵢ (1)
        Pᵢ * αᵢ*[x[n];u[n]] ≤ qᵢαᵢ          (2)
        Note that there we still have the product αᵢ*x[n], αᵢ*u[n]. To get
        rid of the product, we introduce slack variable s[n], t[n], defined
        as sᵢ[n] = αᵢ*x[n], tᵢ[n] = αᵢ*u[n]. The condition sᵢ[n] = αᵢ*x[n],
        tᵢ[n] = αᵢ*u[n] can be enforced through linear constraints, @see
        replace_binary_continuous_product() function.
        The condition (1) and (2) can be written as the mixed-integer linear
        constraints
        x[n+1] = Aeq_slack * slack + Aeq_alpha * α
        Ain_x*x[n] + Ain_u*u[n] + Ain_slack * slack + Ain_alpha*α ≤ rhs_in
        where slack = [s₁[n];s₂[n];...s_K[n];t₁[n];t₂[n];...;t_K[n]]
        α = [α₁;α₂;...;α_K].
        @param x_lo The lower bound of x[n], a column vector. If x_lo = None,
        then we will use the lower bounds on x inferred from
        Pᵢ * [x[n]; u[n]] <= qᵢ
        @param x_up The upper bound of x[n], a column vector. If x_up = None,
        then we will use the upper bounds on x inferred from
        Pᵢ * [x[n]; u[n]] <= qᵢ
        @param u_lo The lower bound of u[n], a column vector. If u_lo = None,
        then we will use the lower bounds on u inferred from
        Pᵢ * [x[n]; u[n]] <= qᵢ
        @param u_up The upper bound of u[n], a column vector. If u_up = None,
        then we will use the upper bounds on u inferred from
        Pᵢ * [x[n]; u[n]] <= qᵢ
        @return (Aeq_slack, Aeq_alpha, Ain_x, Ain_u, Ain_slack, Ain_alpha,
        rhs_in)
        @note 1. This function doesn't require the polytope
                 Pᵢ * [x[n]; u[n]] <= qᵢ to be mutually exclusive.
              2. We do not impose the constraint that one and only one mode
                 is active. The user should impose this constraint separately.
        """
        def check_and_to_numpy(array, shape, dtype):
            if (isinstance(array, torch.Tensor)):
                check_shape_and_type(array, shape, dtype)
                return array.detach().numpy()
            elif (isinstance(array, np.ndarray)):
                assert(array.shape == shape)
                return array
        x_lo_np = check_and_to_numpy(x_lo, (self.x_dim,), self.dtype)
        x_up_np = check_and_to_numpy(x_up, (self.x_dim,), self.dtype)
        u_lo_np = check_and_to_numpy(u_lo, (self.u_dim,), self.dtype)
        u_up_np = check_and_to_numpy(u_up, (self.u_dim,), self.dtype)
        x_lo_all = np.amin(np.stack(self.x_lo, axis=1), axis=1)
        x_up_all = np.amax(np.stack(self.x_up, axis=1), axis=1)
        u_lo_all = np.amin(np.stack(self.u_lo, axis=1), axis=1)
        u_up_all = np.amax(np.stack(self.u_up, axis=1), axis=1)
        if x_lo is not None:
            x_lo_all = np.maximum(x_lo_all, x_lo_np)
        if x_up is not None:
            x_up_all = np.minimum(x_up_all, x_up_np)
        if u_lo is not None:
            u_lo_all = np.maximum(u_lo_all, u_lo_np)
        if u_up is not None:
            u_up_all = np.minimum(u_up_all, u_up_np)
        assert(np.all(x_lo_all <= x_up_all))
        assert(np.all(u_lo_all <= u_up_all))
        Aeq_slack = torch.cat((torch.cat(self.A, dim=1),
                               torch.cat(self.B, dim=1)), dim=1)
        Aeq_alpha = torch.cat([c.reshape((-1, 1)) for c in self.c], dim=1)

        num_slack = (self.x_dim + self.u_dim) * self.num_modes
        num_ineq = np.sum(np.array([Pi.shape[0]
                                    for Pi in self.P])) + num_slack * 4
        Ain_x = torch.zeros(num_ineq, self.x_dim, dtype=self.dtype)
        Ain_u = torch.zeros(num_ineq, self.u_dim, dtype=self.dtype)
        Ain_slack = torch.zeros(num_ineq, num_slack, dtype=self.dtype)
        Ain_alpha = torch.zeros(num_ineq, self.num_modes, dtype=self.dtype)
        rhs_in = torch.zeros(num_ineq, dtype=self.dtype)

        ineq_count = 0

        # We first add the constraint sᵢ[n] = αᵢ*x[n], tᵢ[n] = αᵢ*u[n]

        def s_index(i, j):
            # The index of sᵢ[n][j, 0] in the slack variable.
            return i * self.x_dim + j

        def t_index(i, j):
            # The index of tᵢ[n][j, 0] in the slack variable.
            return self.num_modes * self.x_dim + i * self.u_dim + j
        for i in range(self.num_modes):
            for j in range(self.x_dim):
                (Ain_x[ineq_count: ineq_count + 4, j],
                 Ain_slack[ineq_count: ineq_count + 4, s_index(i, j)],
                 Ain_alpha[ineq_count:ineq_count + 4, i],
                 rhs_in[ineq_count:ineq_count + 4]) =\
                    replace_binary_continuous_product(x_lo_all[j], x_up_all[j],
                                                      self.dtype)
                ineq_count += 4
            for j in range(self.u_dim):
                (Ain_u[ineq_count: ineq_count + 4, j],
                 Ain_slack[ineq_count:ineq_count+4, t_index(i, j)],
                 Ain_alpha[ineq_count:ineq_count+4, i],
                 rhs_in[ineq_count:ineq_count+4]) =\
                    replace_binary_continuous_product(u_lo_all[j], u_up_all[j],
                                                      self.dtype)
                ineq_count += 4

        # Add the constraint Pᵢ * αᵢ*[x[n];u[n]] ≤ qᵢαᵢ
        # Namely Pᵢ * slack ≤ qᵢαᵢ
        for i in range(self.num_modes):
            Ain_slack[ineq_count: ineq_count+self.P[i].shape[0],
                      i*self.x_dim: (i+1) * self.x_dim] =\
                self.P[i][:, :self.x_dim].clone()
            Ain_slack[ineq_count:ineq_count+self.P[i].shape[0],
                      self.num_modes * self.x_dim + i * self.u_dim:
                      self.num_modes * self.x_dim + (i+1) * self.u_dim] =\
                self.P[i][:, self.x_dim:self.x_dim + self.u_dim].clone()
            Ain_alpha[ineq_count: ineq_count +
                      self.P[i].shape[0], i] = -self.q[i]
            ineq_count += self.P[i].shape[0]

        return (Aeq_slack, Aeq_alpha, Ain_x, Ain_u, Ain_slack, Ain_alpha,
                rhs_in)


class AutonomousHybridLinearSystem:
    """
    This system models the autonomous hybrid linear system (switch linear
    system)
    ẋ = Aᵢx+gᵢ
    if Pᵢx ≤ qᵢ
    i = 1, ..., K
    Namely there are K different modes, each mode constraints the state to be
    within a polytope. Inside each mode, the continuous time dynamics is
    affine.
    Note that the polyhedron Pᵢx ≤ qᵢ has to be bounded.
    """

    def __init__(self, x_dim, dtype):
        """
        @param x_dim The dimension of x.
        @param dtype The torch datatype of A, g, P, q.
        """
        self.dtype = dtype

        self.x_dim = x_dim
        self.A = []
        self.g = []
        self.P = []
        self.q = []
        # self.x_lo[i] stores the lower bound of x in mode i (inferred from
        # Pᵢx ≤ qᵢ)
        self.x_lo = []
        # self.x_up[i] stores the upper bound of x in mode i (inferred from
        # Pᵢx ≤ qᵢ)
        self.x_up = []
        self.num_modes = 0

    def add_mode(self, Ai, gi, Pi, qi, check_polyhedron_bounded=False):
        """
        Add a new mode
        ẋ = Aᵢx+gᵢ
        if Pᵢx ≤ qᵢ
        @param Ai A x_dim * x_dim torch matrix.
        @param gi A x_dim torch array.
        @param Pi A num_constraint * x_dim torch matrix.
        @param qi A num_constraint torch array.
        @param check_polyhedron_bounded Set to True if you want to check that
        the polyhedron Pᵢ * x[n] <= qᵢ is bounded. Default to False.
        @note that the polyhedron Pᵢ * x[n] <= qᵢ has to be bounded.
        """
        check_shape_and_type(Ai, (self.x_dim, self.x_dim), self.dtype)
        check_shape_and_type(gi, (self.x_dim,), self.dtype)
        num_constraint = Pi.shape[0]
        check_shape_and_type(Pi, (num_constraint, self.x_dim), self.dtype)
        check_shape_and_type(qi, (num_constraint,), self.dtype)
        if (check_polyhedron_bounded):
            assert(is_polyhedron_bounded(Pi))
        self.A.append(Ai)
        self.g.append(gi)
        self.P.append(Pi)
        self.q.append(qi)
        x_lo = np.empty(self.x_dim)
        x_up = np.empty(self.x_dim)
        for j in range(self.x_dim):
            (x_lo[j], x_up[j]) = utils.compute_bounds_from_polytope(Pi, qi, j)
        self.x_lo.append(x_lo)
        self.x_up.append(x_up)
        self.num_modes += 1

    def mixed_integer_constraints(self, x_lo=None, x_up=None):
        """
        We can rewrite the hybrid dynamics as mixed integer linear constraints.
        We denote γᵢ = 1 if the system is in mode i.
        ẋ = ∑ᵢ γᵢAᵢx+γᵢgᵢ         (1)
        Pᵢγᵢx ≤ qᵢγᵢ              (2)
        Note that there we still have the product γᵢ*x[n]. To get rid of the
        product, we introduce slack variable s, defined as sᵢ = γᵢ*x. The
        condition sᵢ = γᵢ*x, can be enforced through linear constraints, @see
        replace_binary_continuous_product() function.
        The condition (1) and (2) can be written as the mixed-integer linear
        constraints
        ẋ = Aeq_s * s + Aeq_gamma * γ
        Ain_x*x + Ain_s * s + Ain_gamma * γ ≤ rhs_in
        where s = [s₁;s₂;...s_K], γ = [γ₁;γ₂;...;γ_K].
        @param x_lo The lower bound of x, a column vector. If you set this to
        None, then we will use the lower bound inferred from the polytopic
        constraint exists i, Pᵢx ≤ qᵢ
        @param x_up The upper bound of x, a column vector. If you set this to
        None, then we will use the upper bound inferred from the polytopic
        constraint exists i, Pᵢx ≤ qᵢ
        @return (Aeq_s, Aeq_gamma, Ain_x, Ain_s, Ain_gamma, rhs_in)
        @note 1. This function doesn't require the polytope
                 Pᵢ * x[n] <= qᵢ to be mutually exclusive.
              2. We do not impose the constraint that one and only one mode
                 is active. The user should impose this constraint separately.
        """
        if isinstance(x_lo, torch.Tensor):
            check_shape_and_type(x_lo, (self.x_dim,), self.dtype)
            x_lo_np = x_lo.detach().numpy()
        elif isinstance(x_lo, np.ndarray):
            x_lo_np = x_lo
        if isinstance(x_up, torch.Tensor):
            check_shape_and_type(x_up, (self.x_dim,), self.dtype)
            x_up_np = x_up.detach().numpy()
        elif isinstance(x_up, np.ndarray):
            x_up_np = x_up
        if x_lo is not None and x_up is not None:
            assert(np.all(x_lo_np <= x_up_np))
        # Find the minimum of x for all modes.
        x_lo_all = np.amin(np.stack(self.x_lo, axis=1), axis=1)
        # Find the maximum of x for all modes.
        x_up_all = np.amax(np.stack(self.x_up, axis=1), axis=1)
        if x_lo is not None:
            x_lo_all = np.maximum(x_lo_all, x_lo_np)
        if x_up is not None:
            x_up_all = np.minimum(x_up_all, x_up_np)
        Aeq_s = torch.cat(self.A, dim=1)
        Aeq_gamma = torch.cat([g.reshape((-1, 1)) for g in self.g], dim=1)

        num_s = self.x_dim * self.num_modes
        num_ineq = np.sum(np.array([Pi.shape[0]
                                    for Pi in self.P])) + num_s * 4
        Ain_x = torch.zeros(num_ineq, self.x_dim, dtype=self.dtype)
        Ain_s = torch.zeros(num_ineq, num_s, dtype=self.dtype)
        Ain_gamma = torch.zeros(num_ineq, self.num_modes, dtype=self.dtype)
        rhs_in = torch.zeros(num_ineq, dtype=self.dtype)

        ineq_count = 0

        # We first add the constraint sᵢ = γᵢ*x.

        def s_index(i, j):
            # The index of sᵢ[j] in the slack variable.
            return i * self.x_dim + j

        for i in range(self.num_modes):
            for j in range(self.x_dim):
                (Ain_x[ineq_count: ineq_count + 4, j],
                 Ain_s[ineq_count: ineq_count + 4, s_index(i, j)],
                 Ain_gamma[ineq_count:ineq_count + 4, i],
                 rhs_in[ineq_count:ineq_count + 4]) =\
                    replace_binary_continuous_product(x_lo_all[j], x_up_all[j],
                                                      self.dtype)
                ineq_count += 4

        # Add the constraint Pᵢγᵢx ≤ qᵢγᵢ
        # Namely Pᵢ * s ≤ qᵢγᵢ
        for i in range(self.num_modes):
            Ain_s[ineq_count: ineq_count+self.P[i].shape[0],
                  i*self.x_dim: (i+1) * self.x_dim] =\
                self.P[i][:, :self.x_dim].clone()
            Ain_gamma[ineq_count: ineq_count +
                      self.P[i].shape[0], i] = -self.q[i]
            ineq_count += self.P[i].shape[0]

        return (Aeq_s, Aeq_gamma, Ain_x, Ain_s, Ain_gamma, rhs_in)
