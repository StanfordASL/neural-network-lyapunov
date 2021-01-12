import cvxpy as cp

import numpy as np


class SimplePWLLyapunov:
    """
    Some piecewise affine system has simple piecewise linear Lyapunov function,
    i.e., the Lyapunov function is linear in each hybrid mode. We can
    synthesize such simple piecewise linear Lyapunov function through linear
    programming.
    For a continuous time piecewise affine system
    ẋ = Aᵢx+gᵢ if x in ConvexHull(vᵢ¹, ..., vᵢᵐ) where vᵢʲ is the j'th vertex
    of the polytope.
    The Lyapunov condition requires
    V-ε₁|x-x*|₁ ≥ 0 ∀x     (1)
    V̇+ε₂V ≤ 0 ∀x           (2)
    Suppose the Lyapunov function in mode i is V(x)=cᵢᵀx+dᵢ
    Then the constraint (2) is equivalent to
    maxₓ V̇+εV = maxⱼ (cᵢᵀAᵢ+ε₂cᵢᵀ)vᵢʲ+cᵢᵀgᵢ+ε₂dᵢ <= 0 ∀ i      (3)
    For constraint (1), we consider the intersection of mode i with each box
    region x(j) - x*(j) >=0 or <= 0, and denote the vertices of the
    intersection region as v̅ᵢʲ, then the constraint (1) is
    (cᵢᵀ-ε₁*sign(v̅ᵢʲ-x*))v̅ᵢʲ+dᵢ+ε₁*sign(v̅ᵢʲ-x*)x*>=0           (4)
    Also if we denote the vertices on the boundary of mode i and mode j as
    uᵢ,ⱼ, then the continuity constraint of the Lyapunov function requires
    cᵢᵀuᵢ,ⱼ+dᵢ = cⱼᵀuᵢ,ⱼ+dⱼ                                  (5)
    In order to find such a Lyapunov function, we introduce the slack variable
    s1 and s2, and solve the following LP
    min (1-weight) * s1 + weight * s2
    s1 >= (cᵢᵀAᵢ+ε₂cᵢᵀ)vᵢʲ+cᵢᵀgᵢ+ε₂dᵢ ∀ i                         (6)
    s2 >= -(cᵢᵀ-ε₁*sign(v̅ᵢʲ-x*))v̅ᵢʲ - dᵢ - ε₁*sign(v̅ᵢʲ-x*)x*      (7)
    cᵢᵀuᵢ,ⱼ+dᵢ = cⱼᵀuᵢ,ⱼ+dⱼ                                     (8)
    cᵢᵀx* + dᵢ = 0                                              (9)
    s1 >= 0, s2 >= 0
    The decision variables are cᵢ, dᵢ, s1 and s2.
    s1 and s2 represents the violation of constraint (1) and (2). If this LP
    has optimal cost of 0, then we find a Lyapunov function. Otherwise, we
    find some function that minimizes the violation of Lyapunov violation.
    The parameter "weight" in the cost function adjust the relative weight
    between the penalty on violation constraint (1) and (2).
    """
    def __init__(self, x_dim, num_modes, lyapunov_positivity_epsilon,
                 lyapunov_derivative_epsilon, x_equilibrium, equilibrium_mode):
        """
        @param x_dim The dimension of x.
        @param num_modes The number of hybrid modes.
        @param lyapunov_positivity_epsilon ε₁ in the documentation above.
        @param lyapunov_derivative_epsilon ε₂ in the documentation above.
        @param x_equilibrium x* in the documentation above.
        @param equilibrium_mode The mode in which x* is in.
        """
        assert (isinstance(x_dim, int))
        assert (isinstance(num_modes, int))
        assert (isinstance(lyapunov_positivity_epsilon, float))
        assert (isinstance(lyapunov_derivative_epsilon, float))
        assert (isinstance(x_equilibrium, np.ndarray))
        assert (isinstance(equilibrium_mode, int))
        self.x_dim = x_dim
        self.num_modes = num_modes
        self.lyapunov_positivity_epsilon = lyapunov_positivity_epsilon
        self.lyapunov_derivative_epsilon = lyapunov_derivative_epsilon
        self.x_equilibrium = x_equilibrium
        self.c = cp.Variable((self.x_dim, self.num_modes))
        self.d = cp.Variable(self.num_modes)
        self.s1 = cp.Variable(1)
        self.s2 = cp.Variable(1)

        self.constraints = [
            self.s1[0] >= 0, self.s2[0] >= 0,
            self.c[:, equilibrium_mode] @ self.x_equilibrium +
            self.d[equilibrium_mode] == 0
        ]

    def add_lyapunov_derivative_in_mode(self, mode_index, mode_vertices, Ai,
                                        gi):
        """
        Add the constraint (6) in the documentation above.
        s1 >= (cᵢᵀAᵢ+ε₂cᵢᵀ)vᵢʲ+cᵢᵀgᵢ+ε₂dᵢ
        @param mode_vertices mode_vertices[j] is vᵢʲ.
        @param Ai The dynamics in mode i in Ai*x+gi
        @param gi The dynamics in mode i in Ai*x+gi
        """
        assert (isinstance(mode_index, int))
        assert (isinstance(mode_vertices, np.ndarray))
        assert (mode_vertices.shape[1] == self.x_dim)
        assert (isinstance(Ai, np.ndarray))
        assert (Ai.shape == (self.x_dim, self.x_dim))
        assert (isinstance(gi, np.ndarray))
        assert (gi.shape == (self.x_dim, ))
        self.constraints.extend([
            self.s1 >= self.c[:, mode_index] @ (
                (Ai + self.lyapunov_derivative_epsilon * np.eye(self.x_dim))
                @ mode_vertices[j] + gi) +
            self.lyapunov_derivative_epsilon * self.d[mode_index]
            for j in range(mode_vertices.shape[0])
        ])

    def add_lyapunov_positivity_in_mode(self, mode_index, mode_vertices,
                                        sign_v_minus_xstar):
        """
        Add constraint (7) in the documentation above.
        s2 >= -(cᵢᵀ-ε₁*sign(v̅ᵢʲ-x*))v̅ᵢʲ - dᵢ - ε₁*sign(v̅ᵢʲ-x*)x*
        @param mode_index i in the equation.
        @param mode_vertices The vertices of the intersection region of the
        mode domain and the box region sign(x - x*) = sign_v_minus_xstar.
        mode_vertices[j] is v̅ᵢʲ
        @param sign_v_minus_xstar An array defining the sign of the box,
        sign(v̅ᵢʲ-x*).
        """
        assert (isinstance(mode_index, int))
        assert (isinstance(mode_vertices, np.ndarray))
        assert (mode_vertices.shape[1] == self.x_dim)
        assert (isinstance(sign_v_minus_xstar, np.ndarray))
        assert (sign_v_minus_xstar.shape == (self.x_dim, ))
        for j in range(self.x_dim):
            if sign_v_minus_xstar[j] == 1:
                assert (np.all(mode_vertices[:, j] >= self.x_equilibrium[j]))
            elif sign_v_minus_xstar[j] == -1:
                assert (np.all(mode_vertices[:, j] <= self.x_equilibrium[j]))
            else:
                raise Exception(
                    f"add_lyapunov_positivity_in_mode: sign_v_minus_xstar[{j}]"
                    + "has to be either 1 or -1")
        self.constraints.extend([
            self.s2[0] >=
            -(self.c[:, mode_index] - self.lyapunov_positivity_epsilon *
              sign_v_minus_xstar) @ mode_vertices[j] - self.d[mode_index] -
            self.lyapunov_positivity_epsilon *
            sign_v_minus_xstar @ self.x_equilibrium
            for j in range(mode_vertices.shape[0])
        ])

    def add_continuity_constraint(self, mode_i, mode_j, common_vertices):
        """
        Add the continuity constraint (8) in the documentation above
        cᵢᵀuᵢ,ⱼ+dᵢ = cⱼᵀuᵢ,ⱼ+dⱼ
        @param mode_i The mode i's index.
        @param mode_j The mode j's index
        @param common_vertices The vertices define the intersection of mode i
        and j. common_vertices[k] is the k'th vertex.
        """
        assert (isinstance(mode_i, int))
        assert (isinstance(mode_j, int))
        assert (isinstance(common_vertices, np.ndarray))
        assert (common_vertices.shape[1] == self.x_dim)
        self.constraints.append(
            common_vertices @ (self.c[:, mode_i] - self.c[:, mode_j]) +
            (self.d[mode_i] - self.d[mode_j]) *
            np.ones(common_vertices.shape[1]) == 0)

    def solve(self, weight):
        """
        Solve the optimization in the documentation above. If the optimal cost
        is 0, then we find a Lyapunov function. Otherwise we minimizes the
        violation of the Lyapunov condition.
        @param weight The cost is (1-weight) * s1 + weight * s2.
        @return c_val, d_val, s1_val, s2_val.
        """
        assert (isinstance(weight, float))
        assert (weight > 0)
        assert (weight < 1)
        prob = cp.Problem(
            cp.Minimize((1 - weight) * self.s1 + weight * self.s2),
            self.constraints)
        prob.solve(solver="GUROBI")
        return self.c.value, self.d.value, self.s1[0].value, self.s2[0].value
