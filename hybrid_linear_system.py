import torch
import numpy as np

from utils import check_shape_and_type, replace_binary_continuous_product


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
        self.num_modes = 0

    def add_mode(self, Ai, Bi, ci, Pi, qi):
        """
        Add a new mode
        x[n+1] = Aᵢ*x[n] + Bᵢ*u[n] + cᵢ
        if Pᵢ * [x[n]; u[n]] <= qᵢ
        @param Ai A x_dim * x_dim torch matrix.
        @param Bi A x_dim * u_dim torch matrix.
        @param ci A x_dim * 1 torch column vector.
        @param Pi A num_constraint * (x_dim + u_dim) torch matrix.
        @param qi A num_constraint * 1 torch column vector.
        @note that the polytope Pᵢ * [x[n]; u[n]] <= qᵢ has to be bounded.
        """
        check_shape_and_type(Ai, (self.x_dim, self.x_dim), self.dtype)
        check_shape_and_type(Bi, (self.x_dim, self.u_dim), self.dtype)
        check_shape_and_type(ci, (self.x_dim,), self.dtype)
        num_constraint = Pi.shape[0]
        check_shape_and_type(Pi, (num_constraint, self.x_dim + self.u_dim),
                             self.dtype)
        check_shape_and_type(qi, (num_constraint,), self.dtype)
        self.A.append(Ai)
        self.B.append(Bi)
        self.c.append(ci)
        self.P.append(Pi)
        self.q.append(qi)
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
        @param x_lo The lower bound of x[n], a column vector.
        @param x_up The upper bound of x[n], a column vector
        @param u_lo The lower bound of u[n], a column vector.
        @param u_up The upper bound of u[n], a column vector
        @return (Aeq_slack, Aeq_alpha, Ain_x, Ain_u, Ain_slack, Ain_alpha,
        rhs_in)
        @note 1. This function doesn't require the polytope
                 Pᵢ * [x[n]; u[n]] <= qᵢ to be mutually exclusive.
              2. We do not impose the constraint that one and only one mode
                 is active. The user should impose this constraint separately.
        """
        check_shape_and_type(x_lo, (self.x_dim,), self.dtype)
        check_shape_and_type(x_up, (self.x_dim,), self.dtype)
        check_shape_and_type(u_up, (self.u_dim,), self.dtype)
        check_shape_and_type(u_lo, (self.u_dim,), self.dtype)
        assert(torch.all(x_lo <= x_up))
        assert(torch.all(u_lo <= u_up))
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
                    replace_binary_continuous_product(x_lo[j], x_up[j],
                                                      self.dtype)
                ineq_count += 4
            for j in range(self.u_dim):
                (Ain_u[ineq_count: ineq_count + 4, j],
                 Ain_slack[ineq_count:ineq_count+4, t_index(i, j)],
                 Ain_alpha[ineq_count:ineq_count+4, i],
                 rhs_in[ineq_count:ineq_count+4]) =\
                    replace_binary_continuous_product(u_lo[j], u_up[j],
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
