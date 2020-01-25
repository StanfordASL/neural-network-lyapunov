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

    def possible_next_states(self, x):
        """
        Optimization problem cannot impose strictly inequality constraint,
        hence for the state on the boundary of the hybrid modes, in
        optimization we need to think about multiple possible states.
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

    def mode_derivative_bounds(self, mode_index):
        """
        Return the bounds on Aᵢx s.t Pᵢx ≤ qᵢ
        @param mode_index The mode index i
        @return (lower, upper) The lower and upper bounds on  Aᵢx s.t Pᵢx≤ qᵢ
        """
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


def compute_discrete_time_system_cost_to_go(
        system, x_start, num_steps, instantaneous_cost_fun):
    """
    Compute the cost-to-go ∑ᵢ c(x[i]) starting from x_start for the
    discrete-time system.
    @param system An AutonomousHybridLinearSystem instance.
    @param x_start The starting state.
    @param num_steps The length of horizon for the cost-to-go.
    @param instantaneous_cost_fun A function evaluator that takes a state
    and evaluates the one-step cost c(x).
    """
    assert(isinstance(x_start, torch.Tensor))
    assert(x_start.shape == (system.x_dim,))
    total_cost = torch.tensor(0., dtype=system.dtype)
    x_i = x_start.clone()
    total_cost += instantaneous_cost_fun(x_i)
    for i in range(num_steps):
        mode = None
        for j in range(system.num_modes):
            if (torch.all(system.P[j] @ x_i <= system.q[j])):
                mode = j
                x_i = system.A[j] @ x_i + system.g[j]
                break
        assert(mode is not None)
        total_cost += instantaneous_cost_fun(x_i)
    return total_cost


def compute_continuous_time_system_cost_to_go(
        system, x0, T, instantaneous_cost):
    """
    Compute the cost-to-go for a continuous time piecewise affine system.
    The cost-to-go is defined as
    V(x) = ∫ᵀ₀ cost(x(t))dt
    We will first compute the system trajectory x(t), subjecto to the initial
    value constraint x(0) = x0.
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
    @return (V, sol) V is the total cost-to-go from x0., sol is the return from
    solve_ivp. Check
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

    sol = solve_ivp(ivp_fun, (0, T), np.hstack((x0, 0.)), rtol=1e-8)
    return (sol.y[-1, -1], sol)


def generate_cost_to_go_samples(
        system, x0_samples, T, instantaneous_cost, discrete_time_flag):
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
    state_cost_pairs = [None] * len(x0_samples)
    num_pairs = 0
    for x0 in x0_samples:
        try:
            if discrete_time_flag:
                cost_x0 = compute_discrete_time_system_cost_to_go(
                    system, x0, T, instantaneous_cost)
            else:
                cost_x0 = compute_continuous_time_system_cost_to_go(
                    system, x0, T, instantaneous_cost)
            state_cost_pairs[num_pairs] = (x0, cost_x0)
            num_pairs += 1
        except AutonomousHybridLinearSystem.StepForwardException:
            pass
    return state_cost_pairs[:num_pairs]
