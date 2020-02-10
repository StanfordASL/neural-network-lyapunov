import torch
import numpy as np
import cvxpy as cp
from scipy.integrate import solve_ivp

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
    in discrete time, or
    ẋ = Aᵢx + Bᵢu + cᵢ
    if Pᵢ * [x;y] ≤ qᵢ
    i = 1, ..., K.
    in continuous time.
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

    def mixed_integer_constraints(
            self, x_lo=None, x_up=None, u_lo=None, u_up=None):
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

    def mode(self, x_start, u_start):
        """
        Returns the mode of x_start, u_start, namely
        self.P[mode] * (x_start, u_start) <= self.q[mode].
        If x_start, u_start is on the boundary of the neighbouring modes,
        return the mode with the smaller index.
        @param x_start The state.
        @param u_start The control
        @return mode If (x_start, u_start) doesn't belong to any mode, then
        returns None.
        """
        assert(isinstance(x_start, torch.Tensor))
        assert(isinstance(u_start, torch.Tensor))
        assert(x_start.shape == (self.x_dim,))
        assert(u_start.shape == (self.u_dim,))
        for j in range(self.num_modes):
            if (torch.all(self.P[j] @ torch.cat((x_start, u_start)) <=
                          self.q[j])):
                return j
        return None

    def step_forward(self, x_start, u_start):
        """
        Computes the next state and the currently active mode
        @param x_start A tensor representing the starting state
        @param u_start A tensor representing the control action over that
        interval
        TODO(blandry) @param num_steps to allow variables number of
        integration steps
        @return x_i A tensor with the next state, or None if no mode is active
        @return mode An integer correspoding to the mode that was active on
        that step, or None if no mode is active
        """
        assert(type(x_start) == torch.Tensor)
        assert(type(u_start) == torch.Tensor)
        mode = self.mode(x_start, u_start)
        if mode is None:
            return (None, None)
        return (self.A[mode] @ x_start + self.B[mode] @ u_start +
                self.c[mode], mode)


class AutonomousHybridLinearSystem:
    """
    This system models the autonomous hybrid linear system (piecewise affine
    system)
    ẋ = Aᵢx+gᵢ if Pᵢx ≤ qᵢ
    in continuous time, or
    x[n+1] = Aᵢx[n]+gᵢ if Pᵢx[n] ≤ qᵢ
    in discrete time. i = 1, ..., K
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
        # x_lo_all[i] is the lower bound of x across all modes.
        self.x_lo_all = np.full((x_dim,), np.inf)
        # x_up_all[i] is the upper bound of x across all modes.
        self.x_up_all = np.full((x_dim,), -np.inf)
        self.num_modes = 0

        # The lower and upper bounds on Ai * x if Pi * x <= qi.
        self.Ai_times_x_lower = []
        self.Ai_times_x_upper = []

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
        self.x_lo_all = np.minimum(self.x_lo_all, x_lo)
        self.x_up_all = np.maximum(self.x_up_all, x_up)
        self.num_modes += 1

        Ai_times_x_lower, Ai_times_x_upper = self.__compute_Ai_times_x_bounds(
            self.num_modes - 1)
        self.Ai_times_x_lower.append(Ai_times_x_lower)
        self.Ai_times_x_upper.append(Ai_times_x_upper)

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
        if x_lo is not None:
            x_lo_all = np.maximum(self.x_lo_all, x_lo_np)
        else:
            x_lo_all = self.x_lo_all
        if x_up is not None:
            x_up_all = np.minimum(self.x_up_all, x_up_np)
        else:
            x_up_all = self.x_up_all
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
                    replace_binary_continuous_product(
                        torch.tensor(x_lo_all[j], dtype=self.dtype),
                        torch.tensor(x_up_all[j], dtype=self.dtype),
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

    def mode(self, x):
        """
        Returns the mode of state x. Namely P[mode] * x <= q[mode].
        returns None if x is not in any mode.
        Notice that we choose the first mode that satisfies
        P[mode] * x <= q[mode]
        """
        assert(isinstance(x, torch.Tensor))
        assert(x.shape == (self.x_dim,))
        for i in range(self.num_modes):
            if torch.all(self.P[i] @ x <= self.q[i]):
                return i
        return None

    class StepForwardException(Exception):
        pass

    def step_forward(self, x, mode_x=None):
        """
        Compute the one-step forward simulation x[n+1] = A[i] * x[n] + g[i]
        where i is the mode of x.
        @param x The starting state.
        @param mode_x The mode in which x is in. If mode_x = None, then we will
        determine the mode of x.
        @return x_next The next continuous state.
        """
        assert(isinstance(x, torch.Tensor))
        assert(x.shape == (self.x_dim,))

        if mode_x is None:
            mode_x = self.mode(x)
        if mode_x is None:
            raise self.StepForwardException(
                "step_forward(): x is not in any mode.")
        return self.A[mode_x] @ x + self.g[mode_x]

    def possible_dx(self, x):
        """
        For state on the boundary of two modes, we regard that both modes are
        possible (because in numerical optimization we can't impose strict
        inequality constraint). So we return all Aᵢx+gᵢ if Pᵢx≤ qᵢ
        @param x The state
        @return next_states A list. If x is on the boundary of the modes, then
        return multiple possible next states, otherwise return a list of single
        next state. If x is not in any hybrid mode, then return an empty list.
        """
        assert(isinstance(x, torch.Tensor))
        assert(x.shape == (self.x_dim,))
        next_states = []
        for i in range(self.num_modes):
            if torch.all(self.P[i] @ x <= self.q[i]):
                next_states.append(self.A[i] @ x + self.g[i])
        return next_states

    def __compute_Ai_times_x_bounds(self, mode_index):
        lower = np.empty(self.x_dim)
        upper = np.empty(self.x_dim)
        for j in range(self.x_dim):
            x = cp.Variable(self.x_dim)
            con = [
                self.P[mode_index].detach().numpy() @ x <= self.q[mode_index]]
            prob = cp.Problem(
                cp.Maximize(self.A[mode_index][j].detach().numpy() @ x), con)
            prob.solve()
            upper[j] = prob.value
            prob = cp.Problem(
                cp.Minimize(self.A[mode_index][j].detach().numpy() @ x), con)
            prob.solve()
            lower[j] = prob.value
        return (lower, upper)

    def mode_derivative_bounds(self, mode_index):
        """
        Return the bounds on Aᵢx s.t Pᵢx ≤ qᵢ
        @param mode_index The mode index i
        @return (lower, upper) The lower and upper bounds on  Aᵢx s.t Pᵢx≤ qᵢ
        """
        assert(mode_index < self.num_modes and mode_index >= 0)
        lower = self.Ai_times_x_lower[mode_index]
        upper = self.Ai_times_x_upper[mode_index]
        return (lower, upper)


def compute_discrete_time_system_cost_to_go(
        system, x_start, num_steps, instantaneous_cost_fun, x_goal=None):
    """
    Compute the cost-to-go ∑ᵢ c(x[i]) starting from x_start for the
    discrete-time system. If the trajectory of x reaches x_goal, or after
    simulating for num_steps, we terminate the simulation.
    @param system An AutonomousHybridLinearSystem instance.
    @param x_start The starting state.
    @param num_steps The length of horizon for the cost-to-go.
    @param instantaneous_cost_fun A function evaluator that takes a state
    and evaluates the one-step cost c(x).
    @param x_goal If the trajectory reaches x_goal, then stop the simulation.
    @return total_cost, x_steps, costs total_cost is the cost-to-go from x0.
    x_steps[:,i] being the simulating of x after i steps.
    costs[i] is the cost-to-go starting from x_steps[:, i]
    """
    assert(isinstance(x_start, torch.Tensor))
    assert(x_start.shape == (system.x_dim,))
    if x_goal is not None:
        assert(isinstance(x_goal, torch.Tensor))
        assert(x_goal.shape == (system.x_dim,))
    total_cost = instantaneous_cost_fun(x_start)
    costs_from_start = [None] * (num_steps + 1)
    costs_from_start[0] = total_cost.clone()
    cost_steps = [None] * (num_steps + 1)
    cost_steps[0] = total_cost.clone()
    x_steps = [None] * (num_steps + 1)
    x_steps[0] = x_start.clone()
    terminal_step = num_steps
    for i in range(num_steps):
        if x_goal is not None and torch.norm(x_steps[i] - x_goal, p=2) < 1e-3:
            terminal_step = i
            break
        mode = None
        for j in range(system.num_modes):
            if (torch.all(system.P[j] @ x_steps[i] <= system.q[j])):
                mode = j
                x_steps[i+1] = system.A[j] @ x_steps[i] + system.g[j]
                break
        assert(mode is not None)
        cost_steps[i + 1] = instantaneous_cost_fun(x_steps[i + 1])
        total_cost += cost_steps[i+1]
        costs_from_start[i+1] = costs_from_start[i] + cost_steps[i+1]
    x_steps = torch.stack(x_steps[:terminal_step+1], axis=1)
    costs = torch.stack([
        total_cost - costs_from_start[i] + cost_steps[i] for i in
        range(terminal_step + 1)])
    return total_cost, x_steps, costs


def compute_continuous_time_system_cost_to_go(
        system, x0, T, instantaneous_cost, x_goal=None):
    """
    Compute the cost-to-go for a continuous time piecewise affine system.
    The cost-to-go is defined as
    V(x) = ∫ᵀ₀ cost(x(t))dt
    We will first compute the system trajectory x(t), subject to the initial
    value constraint x(0) = x0.
    @note If the trajectory x(t) reaches x_goal before t=T, we also stop the
    simulation. This is useful for computing the infinite horizon cost to go,
    when we want to stop the simulation when the trajectory converge.
    Mathematically, we solve the following initial value problem:
    Define a state as [x(t), V(t)] with the ODE
    ẋ = Aᵢx+gᵢ if Pᵢx≤ qᵢ
    V̇ = cost(x)
    We integrate this ODE from the initial value [x0, 0] until T.
    @param system An AutonomousHybridLinearSystem representing the continuous
    time piecewise affine system.
    @param T The terminal time. Must be a positive float.
    @param instantaneous_cost A callable. The function cost(x) that evaluates
    the instantaneous cost on x.
    @param x_goal If the trajectory x(t) reaches x_goal, then stop the
    simulation.
    @return (V, x_traj, cost_to_go_traj, sol) V is the total cost-to-go from
    x0, x_traj is the trajectory starting from x0. cost_to_go_traj[i] is the
    cost-to-go starting from x_traj[:, i], sol is the return from solve_ivp.
    Check
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html
    """
    assert(isinstance(system, AutonomousHybridLinearSystem))
    assert(isinstance(T, float))
    assert(T >= 0)
    assert(isinstance(x0, torch.Tensor))
    assert(x0.shape == (system.x_dim,))

    def ivp_fun(t, y):
        x = y[:system.x_dim]
        ydot = np.empty(system.x_dim + 1)
        with torch.no_grad():
            x_torch = torch.from_numpy(x)
            ydot[:system.x_dim] = system.step_forward(x_torch)
            ydot[-1] = instantaneous_cost(x_torch)
        return ydot

    events = []
    if x_goal is not None:
        x_goal_np = x_goal.detach().numpy()

        def reach_goal(t, x):
            return np.linalg.norm(x[:-1]-x_goal_np) - 1e-3
        reach_goal.terminal = True
        events = [reach_goal]

    sol = solve_ivp(
        ivp_fun, (0, T), np.hstack((x0, 0.)), rtol=1e-8, events=events)
    total_cost = torch.tensor(sol.y[-1, -1], dtype=system.dtype)
    x_traj = torch.tensor(sol.y[:-1, :], dtype=system.dtype)
    cost_to_go_traj = total_cost - torch.tensor(
        sol.y[-1, :], dtype=system.dtype)
    return (total_cost, x_traj, cost_to_go_traj, sol)


def generate_cost_to_go_samples(
        system, x0_samples, T, instantaneous_cost, discrete_time_flag,
        x_goal=None, pruner=None):
    """
    Generate the mapping from the initial state to the cost-to-go, by
    simulating the system for a given horizon.
    @param system An AutonomousHybridLinearSystem instance.
    @param x0_samples A list of pytorch tensors, x0_samples[i] is the i'th
    sample
    @param T The simulation horizon. T must be an int if
    discrete_time_flag=True, otherwise T is a float.
    @param instantaneous_cost A callable, evaluates the instantaneous cost for
    a state.
    @param discrete_time_flag A flag indicating whether @p system is
    discrete time or continuous time.
    @param x_goal If the trajectory starting from x0 reaches x_goal, then
    terminate simulating the trajectory. See
    compute_discrete_time_system_cost_to_go() and
    compute_continuous_time_system_cost_to_go() for more details.
    @pruner A callable that returns True or False. We might want to prune
    some state-value pairs. If pruner(state) returns True, then this
    state-value pair is not included.
    @return state_cost_pairs A list of tuples. state_cost_pairs[i] contains
    a tuple (state, cost). It only includes the states starting from which the
    trajectory always stays within the domain Pᵢ x≤ qᵢ for some mode i.
    """
    assert(isinstance(system, AutonomousHybridLinearSystem))
    assert(isinstance(x0_samples, list))
    if discrete_time_flag:
        assert(isinstance(T, int))
    else:
        assert(isinstance(T, float))
    if pruner is not None:
        assert(callable(pruner))
    state_cost_pairs = []
    for x0 in x0_samples:
        try:
            if discrete_time_flag:
                cost_x0, x_traj, cost_to_go_traj = \
                    compute_discrete_time_system_cost_to_go(
                        system, x0, T, instantaneous_cost, x_goal)
            else:
                cost_x0, x_traj, cost_to_go_traj, _ = \
                    compute_continuous_time_system_cost_to_go(
                        system, x0, T, instantaneous_cost, x_goal)
            for i in range(x_traj.shape[1]):
                if pruner is None or \
                        (pruner is not None and not pruner(x_traj[:, i])):
                    state_cost_pairs.append((x_traj[:, i], cost_to_go_traj[i]))
        except AutonomousHybridLinearSystem.StepForwardException:
            pass
    return state_cost_pairs


def partition_state_input_space(x_lo, x_up, u_lo, u_up,
                                num_breaks_x, num_breaks_u,
                                x_delta, u_delta):
    """
    Generate a grid over a state and input space. This is useful for
    approximating a nonlinear system with a piecewise affine system, with
    linear approximation in each cell of the partition.
    @param x_lo Tensor lower bound of the state space discretization
    @param x_up Tensor upper bound of the state space discretization
    @param u_lo Tensor lower bound of the input space discretization
    @param u_up Tensor upper bound of the input space discretization
    @param num_breaks_x Tensor of integer, number of discretization along
    each axis
    @param num_breaks_u Tensor of integer, number of discretization along
    each axis
    @param x_delta Tensor which says by how much to move the boundaries of each
    discretization as a fraction of the discretization size i.e. a value of
    [.1, .2] will increase the size of each cell by 10% of its current size
    in the positive and negative directions for the first state (.2 for the
    second state). Positive values means the cells will overlap, negative
    values mean the cells will have gaps between them.
    @param u_delta Tensor see x_delta
    @return state_x, state_u Tensors with the states/inputs around which each
    cell is centered (n X x_dim) and (n X u_dim)
    @return states_x_lo, states_x_up, states_u_lo, states_u_up, Tensors with
    the boundaries of each cell
    """
    assert(isinstance(x_lo, torch.Tensor))
    assert(isinstance(x_up, torch.Tensor))
    assert(isinstance(u_lo, torch.Tensor))
    assert(isinstance(u_up, torch.Tensor))
    assert(isinstance(num_breaks_x, torch.Tensor))
    assert(isinstance(num_breaks_u, torch.Tensor))
    assert(isinstance(x_delta, torch.Tensor))
    assert(isinstance(u_delta, torch.Tensor))
    assert(num_breaks_x.dtype == torch.int)
    assert(num_breaks_u.dtype == torch.int)
    dtype = x_lo.dtype
    x_dim = x_lo.shape[0]
    u_dim = u_lo.shape[0]
    grid_limits = []
    grid_samples = []
    grid_indices = []
    x_delta_scaled = x_delta * (x_up - x_lo) / num_breaks_x.type(dtype)
    u_delta_scaled = u_delta * (u_up - u_lo) / num_breaks_u.type(dtype)
    for i in range(x_dim):
        limits_ = np.linspace(x_lo[i], x_up[i], num_breaks_x[i] + 1)
        limits = [(limits_[k], limits_[k+1]) for k in range(num_breaks_x[i])]
        samples = [.5*(limits[k][0] + limits[k][1])
                   for k in range(num_breaks_x[i])]
        grid_limits.append(limits)
        grid_samples.append(samples)
        grid_indices.append(np.arange(num_breaks_x[i]))
    for i in range(u_dim):
        limits_ = np.linspace(u_lo[i], u_up[i], num_breaks_u[i] + 1)
        limits = [(limits_[k], limits_[k+1]) for k in range(num_breaks_u[i])]
        samples = [.5*(limits[k][0] + limits[k][1])
                   for k in range(num_breaks_u[i])]
        grid_limits.append(limits)
        grid_samples.append(samples)
        grid_indices.append(np.arange(num_breaks_u[i]))
    grid = np.meshgrid(*grid_indices)
    indices_cart_product = np.concatenate(
        [g.reshape(-1, 1) for g in grid], axis=1)
    states_x = torch.Tensor(0, x_dim).type(dtype)
    states_u = torch.Tensor(0, u_dim).type(dtype)
    states_x_lo = torch.Tensor(0, x_dim).type(dtype)
    states_x_up = torch.Tensor(0, x_dim).type(dtype)
    states_u_lo = torch.Tensor(0, u_dim).type(dtype)
    states_u_up = torch.Tensor(0, u_dim).type(dtype)
    for k in range(indices_cart_product.shape[0]):
        indices = indices_cart_product[k, :]
        sample = torch.Tensor(
            [grid_samples[i][indices[i]] for i in range(x_dim+u_dim)]).type(
            dtype)
        x_sample = sample[:x_dim]
        u_sample = sample[x_dim:x_dim+u_dim]
        sample_limits = torch.Tensor(
            [grid_limits[i][indices[i]] for i in range(x_dim+u_dim)]).type(
            dtype)
        sample_lo = sample_limits[:, 0]
        sample_up = sample_limits[:, 1]
        x_sample_lo = sample_lo[:x_dim] - x_delta_scaled
        x_sample_up = sample_up[:x_dim] + x_delta_scaled
        u_sample_lo = sample_lo[x_dim:x_dim+u_dim] - u_delta_scaled
        u_sample_up = sample_up[x_dim:x_dim+u_dim] + u_delta_scaled
        x_sample_lo = torch.min(torch.max(x_sample_lo, x_lo), x_up)
        x_sample_up = torch.min(torch.max(x_sample_up, x_lo), x_up)
        u_sample_lo = torch.min(torch.max(u_sample_lo, u_lo), u_up)
        u_sample_up = torch.min(torch.max(u_sample_up, u_lo), u_up)
        states_x = torch.cat((states_x, x_sample.unsqueeze(0)), axis=0)
        states_u = torch.cat((states_u, u_sample.unsqueeze(0)), axis=0)
        states_x_lo = torch.cat((states_x_lo, x_sample_lo.unsqueeze(0)),
                                axis=0)
        states_x_up = torch.cat((states_x_up, x_sample_up.unsqueeze(0)),
                                axis=0)
        states_u_lo = torch.cat((states_u_lo, u_sample_lo.unsqueeze(0)),
                                axis=0)
        states_u_up = torch.cat((states_u_up, u_sample_up.unsqueeze(0)),
                                axis=0)
    return(states_x, states_u,
           states_x_lo, states_x_up, states_u_lo, states_u_up)
