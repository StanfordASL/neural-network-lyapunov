# -*- coding: utf-8 -*-
import cvxpy as cp
import numpy as np 

class ControlLyapunovFixedActivationPath:
    """
    For a fixed activation path on a ReLU network, verify that the output of
    the network is a valid control Lyapunov function. i.e., there exists a
    control action u, such that the value function goes downhill.

    Mathematically, if we denote the ReLU output as η(x), and the dynamics as
    ẋ = Ax + Bu, then the control Lyapunov condition is
    ∃ u ∈ U, s.t ∂η/∂x * Ax + ∂η/∂x *Bu ≤ 0
    Assuming that the admissible set of control action U is a polytope, with
    vertices uᵢ*, and the ReLU network output along this activation path is 
    η(x) = gᵀx+h, while P*x ≤ q, then the control Lyapunov condition that the
    optimal cost of the following problem is no larger than 0.
    max gᵀAx + min_i gᵀBuᵢ
    s.t Px ≤ q
    """
    def __init__(self, g, P, q, A, B, u_vertices):
        """
        @param g The gradient of the ReLU output along this path.
        @param P P*x <= q is the constraint that x activates this path.
        @param q P*x <= q is the constraint that x activates this path.
        @param A The dynamics is ẋ = Ax + Bu
        @param B The dynamics is ẋ = Ax + Bu
        @param u_vertices u_vertices[:,i] is the i'th vertex of the control
        action polytope U.
        """
        xdim = g.size
        self.x = cp.Variable((xdim, 1))
        assert(g.shape[0] == xdim)
        assert(g.shape[1] == 1)
        assert(P.shape[1] == xdim)
        assert(P.shape[0] == q.shape[0])
        assert(q.shape[1] == 1)
        assert(A.shape[0] == A.shape[1])
        assert(A.shape[0] == xdim)
        assert(B.shape[0] == A.shape[0])
        assert(B.shape[1] == u_vertices.shape[0])
        cost_constant = np.min(g.T.dot(B * u_vertices))

        self.objective = g.T * (self.x) + cost_constant
        self.constraints = [P * self.x <= q]


    def construct_program(self) -> cp.Problem:
        prob = cp.Problem(cp.Maximize(self.objective), self.constraints)
        return prob

