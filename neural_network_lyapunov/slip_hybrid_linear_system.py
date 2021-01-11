import neural_network_lyapunov.spring_loaded_inverted_pendulum as\
    spring_loaded_inverted_pendulum

import numpy as np


class SlipHybridLinearSystem:
    """
    Models a SLIP model running on stepping stones as hybrid linear systems.
    """
    def __init__(self, mass, l0, k, g):
        self.slip = spring_loaded_inverted_pendulum.SLIP(mass, l0, k, g)
        self.stepping_stones = []

    def add_stepping_stone(self, left, right, height):
        """
        Add a new stepping stone to the terrain.
        @param left The left boundary of the stepping stone.
        @param right The right boundary of the stepping stone.
        @param height The height of the stepping stone.
        """
        self.stepping_stones.append(
            spring_loaded_inverted_pendulum.SteppingStone(left, right, height))

    def apex_map_linear_approximation(self, apex_state, stepping_stone,
                                      leg_angle):
        """
        Given the apex state [x;z;ẋ], and the leg angle, find the linear
        approximation of the apex-to-apex return map. Notice that here in the
        apex state, z is the vertical position above ground 0. Do not confuse
        this z with the `apex_height` in SLIP model. The `apex_height` in SLIP
        model is the height above the stepping stone to be touched on.
        @param apex_state The state of the current apex [x;z;ẋ]
        @param stepping_stone The stepping stone to be touched
        @param leg_angle The leg angle at the moment of touchdown.
        @return (A, B, c, P, q, a_t, b_t, c_t)
        The linearized apex-to-apex map is x[n+1] = A * x[n] + B * u[n] + c
        subjecto to P * [x[n];u[n]] <= q
        where x[n] is the current apex state, u[n] is the leg angle, and
        x[n+1] is the next apex state.
        The time duration from the current apex to the next is approximated as
        a_tᵀ * x[n] + b_t * u[n] + c_t
        If the SLIP cannot touch down the designated stepping stone with
        @p apex_state and @p leg_angle, then return all None

        There are three constraints for the apex-to-apex map.
        1. Initially the foot at the apex should be above the ground.
        2. At post-liftoff state, the vertical velocity is upward.
        3. At the moment of touchdown, the horizontal position of the foot
           should be within the stepping stone.
        """
        if (not self.slip.can_touch_stepping_stone(
                np.array([apex_state[0], apex_state[1], apex_state[2], 0]),
                stepping_stone, leg_angle)):
            return None, None, None, None, None, None, None, None
        (A, B, c, a_t, b_t, c_t,
         dx_pre_td_dx_apex, dx_pre_td_dleg_angle, x_pre_td,
         dx_post_lo_dx_apex, dx_post_lo_dleg_angle, x_post_lo) =\
            self.slip.apex_to_apex_gradient(
                 np.array([apex_state[0],
                           apex_state[1] - stepping_stone.height,
                           apex_state[2]]), leg_angle)
        if A is None:
            return None, None, None, None, None, None, None, None
        c = c.reshape((3, 1))
        c += -A.dot(apex_state.reshape((3, 1))) - B * leg_angle +\
            np.array([[0], [stepping_stone.height], [0]])
        c_t += -a_t.dot(apex_state) - b_t * leg_angle
        num_constraints = 2
        num_constraints += 1 if stepping_stone.left != -np.inf else 0
        num_constraints += 1 if stepping_stone.right != np.inf else 0
        P = np.zeros((num_constraints, 4))
        q = np.zeros((num_constraints, 1))
        cos_theta = np.cos(leg_angle)
        sin_theta = np.sin(leg_angle)
        # Now approximate constraint 1: z - l₀cosθ >= h
        # Namely -Δz -l₀sinθΔθ≤ z-l₀cosθ-h
        P[0, 1] = -1.
        P[0, 3] = -self.slip.l0 * sin_theta
        q[0, 0] = -self.slip.l0 * cos_theta - stepping_stone.height -\
            self.slip.l0 * sin_theta * leg_angle
        # Now approximate the constraint that at liftoff, the vertical velocity
        # should be positive.
        P[1, :3] = -dx_post_lo_dx_apex[3, :]
        P[1, 3] = -dx_post_lo_dleg_angle[3]
        q[1, 0] = x_post_lo[3] - dx_post_lo_dx_apex[3, :].dot(apex_state) -\
            dx_post_lo_dleg_angle[3] * leg_angle
        # Now approximate the constraint that the touchdown point is on the
        # stepping stone left ≤ x⁻_TD[0] + l₀sinθ ≤ right
        constraint_count = 2
        if stepping_stone.left != -np.inf:
            P[constraint_count, :3] = -dx_pre_td_dx_apex[0, :]
            P[constraint_count, 3] = -dx_pre_td_dleg_angle[0] -\
                self.slip.l0 * cos_theta
            q[constraint_count, 0] = x_pre_td[0] + self.slip.l0 * sin_theta -\
                stepping_stone.left + P[constraint_count, :3].dot(apex_state)\
                + P[constraint_count, 3] * leg_angle
            constraint_count += 1
        if stepping_stone.right != np.inf:
            P[constraint_count, :3] = dx_pre_td_dx_apex[0, :]
            P[constraint_count, 3] = dx_pre_td_dleg_angle[0] +\
                self.slip.l0 * cos_theta
            q[constraint_count, 0] = stepping_stone.right - x_pre_td[0] -\
                self.slip.l0 * sin_theta +\
                P[constraint_count, :3].dot(apex_state) +\
                P[constraint_count, 3] * leg_angle
        return (A, B, c, a_t, b_t, c_t, P, q)
