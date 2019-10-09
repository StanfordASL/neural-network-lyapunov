import numpy as np
from scipy.integrate import solve_ivp
from utils import check_shape_and_type


class SteppingStone:
    """
    A stepping stone in 2D is a flat region with a given height. Namely within
    the range x ∈ [self.left, self.right], the height of the terrain is
    self.height
    """

    def __init__(self, left, right, height):
        assert(left < right)
        self.left = left
        self.right = right
        self.height = height


class SLIP:
    """
    Defines the dynamics of the spring loaded inverted pendulum.
    """

    def __init__(self, mass, l0, k, g, dtype=np.dtype("float64")):
        """
        @param mass The mass of the system.
        @param l0 The springy leg has rest length of l0
        @param k The spring constant
        @param g The gravity acceleration.
        """
        self.mass = mass
        self.l0 = l0
        self.k = k
        self.dtype = dtype
        assert(g >= 0)
        self.g = g

    def flight_dynamics(self, state):
        """
        In the flight phase, the state is [x;y;xdot;ydot], the dynamics is
        state_dot = [xdot;ydot;0;-g]
        @param state A lenght 4 numpy array. The state of the robot in the
        flight phase.
        @return state_dot The time derivative of the state.
        """
        return np.array([state[2], state[3], 0, -self.g], dtype=self.dtype)

    def stance_dynamics(self, state):
        """
        In the stance phase, the state is [r,θ,ṙ,θ_dot,x_foot], where r is
        the leg length, and θ is the angle between the leg and the vertical
        line. x_foot is the x position of the foot.
        The dynamics is
        r̈ = rθ_dot²-gcosθ+k(l₀-r)/m
        θ_ddot = g/r*sinθ-2ṙθ_dot/r
        """
        r = state[0]
        theta = state[1]
        r_dot = state[2]
        theta_dot = state[3]
        r_ddot = r * theta_dot ** 2 - self.g * np.cos(theta) +\
            self.k * (self.l0 - r) / self.mass
        theta_ddot = self.g / r * np.sin(theta) - 2 * r_dot * theta_dot / r
        return np.array([r_dot, theta_dot, r_ddot, theta_ddot, 0],
                        dtype=self.dtype)

    def touchdown_guard(self, state, theta):
        """
        The touch down happens in the flight phase when the guard y - l0*cosθ
        crosses zero from above.
        """
        y = state[1]
        return y - self.l0 * np.cos(theta)

    def liftoff_guard(self, state):
        """
        The lift off happens in the stance phase when the garud r - l0 crosses
        zero from below.
        """
        r = state[0]
        return r - self.l0

    def apex_guard(self, state):
        """
        The apex happens when ydot crosses zero from above.
        """
        return state[3]

    def touchdown_transition(self, pre_state, theta):
        """
        At touch down, the post_impact state is
        [l0;θ;-ẋsinθ+ẏcosθ;(-ẋcosθ-ẏsinθ)/l0;x+l0sinθ]
        @param pre_state The pre-impact state [x;y;xdot;ydot]
        @param theta The leg angle.
        @return post_state The post-impact state
        """
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        xdot = pre_state[2]
        ydot = pre_state[3]
        return np.array([self.l0, theta,
                         -xdot * sin_theta + ydot * cos_theta,
                         (-xdot * cos_theta - ydot * sin_theta) / self.l0,
                         pre_state[0] + self.l0 * sin_theta],
                        dtype=self.dtype)

    def liftoff_transition(self, pre_state):
        """
        At lift off, the post lift-off state is
        [x_foot-l0*sinθ; l0*cosθ; -r*cosθ*θ_dot-ṙsinθ;-r*sinθ*θ_dot+ṙcosθ]
        @param pre_state The pre-liftoff state [r,θ,ṙ,θ_dot,x_foot]
        @return post_state The post lift-off state
        """
        x_foot = pre_state[4]
        theta = pre_state[1]
        r = pre_state[0]
        r_dot = pre_state[2]
        theta_dot = pre_state[3]
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        return np.array([x_foot - self.l0 * sin_theta,
                         self.l0 * cos_theta,
                         -r * cos_theta * theta_dot - r_dot * sin_theta,
                         -r * sin_theta * theta_dot + r_dot * cos_theta],
                        dtype=self.dtype)

    def flight_phase_energy(self, state):
        """
        Compute the total kinetic + potential energy for a flight phase state.
        """
        return 0.5 * self.mass * (state[2] ** 2 + state[3]**2)\
            + self.mass * self.g * state[1]

    def stance_phase_energy(self, state):
        """
        Compute the total kinematic + potential energy for a stance phase
        state.
        """
        return 0.5 * self.mass * ((state[0] * state[3]) ** 2 + state[2] ** 2)\
            + 0.5 * self.k * (self.l0 - state[0])**2\
            + self.mass * state[0] * np.cos(state[1]) * self.g

    def apex_map(self, pos_x, apex_height, vel_x, leg_angle):
        """
        Given the state of the SLIP at apex (ż = 0) during the flight phase,
        compute the state at the next apex.
        @param pos_x The horizontal position.
        @param apex_height The vertical height above the touchdown height,
        namely z - ground_height, where z is the apex z position, and
        ground_height is the height of the ground at the moment of touchdown.
        @param vel_x The horizontal velocity.
        @param leg_angle The angle (between the leg and z axis) at the moment
        of touchdown.
        @return (next_pos_x, next_apex_height, next_vel_x) The state at the
        next apex.
        If SLIP cannot reach next apex height (for example, if the current
        apex height is below 0, or during the stance phase, the mass touches
        the ground before the spring length increases to l0), then returns
        (None, None, None)
        """
        cos_theta = np.cos(leg_angle)
        if (apex_height - self.l0 * cos_theta < 0):
            # The foot is below the ground at the apex.
            return (None, None, None)
        if (vel_x > 0 and leg_angle < 0):
            return (None, None, None)
        if (vel_x < 0 and leg_angle > 0):
            return (None, None, None)
        t_touchdown = np.sqrt(2 * (apex_height - self.l0 * cos_theta) / self.g)
        pre_impact_state = np.array([pos_x + vel_x * t_touchdown,
                                     self.l0*cos_theta,
                                     vel_x,
                                     self.g * t_touchdown])
        post_impact_state = self.touchdown_transition(pre_impact_state,
                                                      leg_angle)

        def liftoff(t, x): return self.liftoff_guard(x)
        liftoff.terminal = True
        liftoff.direction = 1
        def hitground1(t, x): return np.pi / 2 + x[1]
        hitground1.terminal = True
        hitground1.direction = -1
        def hitground2(t, x): return x[1] - np.pi / 2
        hitground2.terminal = True
        hitground2.direction = 1
        sol_stance = solve_ivp(lambda t, x: self.stance_dynamics(x),
                               (0, np.inf),
                               post_impact_state,
                               events=[liftoff, hitground1, hitground2],
                               rtol=1e-8)
        if len(sol_stance.t_events[0]) > 0:
            pre_liftoff_state = sol_stance.y[:, -1]
        else:
            return (None, None, None)
        post_liftoff_state = self.liftoff_transition(pre_liftoff_state)
        if post_liftoff_state[3] < 0:
            # The vertical velocity is negative.
            return (None, None, None)
        t_to_apex = -post_liftoff_state[3]/self.g
        next_pos_x = post_liftoff_state[0] + post_liftoff_state[2] * t_to_apex
        next_apex_height = post_liftoff_state[1] + self.g / 2 * t_to_apex ** 2
        return (next_pos_x, next_apex_height, post_liftoff_state[2])

    def simulate(self, x0, theta_step):
        """
        Simulate the SLIP model from an intial state x0 in the flight phase.
        In the i'th flight phase, the leg angle theta is fixed to theta_step[i]
        The simulation stops end of the n'th take off, where n is
        len(theta_step)
        @param x0 The initial state in the flight phase.
        @param theta_step In the i'th flight phase, the leg angle theta is
        fixed to theta_step[i]
        @return sol A list of length 2 * len(theta_step). sol[2*i] contains
        the simulation result for the i'th flight phase. sol[2*i+1] contains
        the simulation result for the i'th ground phase.
        """
        check_shape_and_type(x0, (4,), self.dtype)
        assert(x0[1] > self.l0 * np.cos(theta_step[0]))

        x_step_start = x0
        t_step_start = 0
        sol = []
        for step in range(len(theta_step)):
            if (x_step_start[1] <= self.l0 * np.cos(theta_step[step])):
                break

            def touchdown(t, x):
                return self.touchdown_guard(x, theta_step[step])
            touchdown.terminal = True
            touchdown.direction = -1
            sol_step_flight = solve_ivp(lambda t, x: self.flight_dynamics(x),
                                        (t_step_start, t_step_start + 10),
                                        x_step_start, events=touchdown)
            sol.append(sol_step_flight)
            pre_impact_state = sol_step_flight.y[:, -1]
            post_impact_state = self.touchdown_transition(pre_impact_state,
                                                          theta_step[step])

            def liftoff(t, x): return self.liftoff_guard(x)
            liftoff.terminal = True
            liftoff.direction = 1
            def hitground1(t, x): return np.pi / 2 + x[1]
            hitground1.terminal = True
            hitground1.direction = -1
            def hitground2(t, x): return x[1] - np.pi / 2
            hitground2.terminal = True
            hitground2.direction = 1
            sol_step_stance = solve_ivp(lambda t, x: self.stance_dynamics(x),
                                        (sol_step_flight.t[-1],
                                         sol_step_flight.t[-1] + 10),
                                        post_impact_state,
                                        events=[liftoff, hitground1,
                                                hitground2],
                                        rtol=1e-8)
            sol.append(sol_step_stance)
            if (sol_step_stance.y[1, -1] + np.pi/2 < 1E-5):
                break
            if (np.pi / 2 - sol_step_stance.y[1, -1] < 1E-5):
                break
            x_step_start = self.liftoff_transition(sol_step_stance.y[:, -1])
            t_step_start = sol_step_stance.t[-1]
        return sol

    def time_to_touchdown(self, flight_state, stepping_stone, leg_angle):
        """
        For a flight state, computes the time to next touchdown on a given
        stepping stone. Returns None if the robot won't touch down on that
        stepping stone.
        @param flight_state [x;z;ẋ;ż] The state in the flight phase.
        @param stepping_stone A SteppingStone object.
        @param leg angle We assume that the leg angle (between leg and the
        vertical line is a constant).
        @return t The time to next touch down on that stepping stone. Returns
        None if the robot won't touch down on that stepping stone.
        """
        cos_theta = np.cos(leg_angle)
        sin_theta = np.sin(leg_angle)
        foot_pos_z0 = flight_state[1] - self.l0 * cos_theta
        if (stepping_stone.height > foot_pos_z0):
            return None

        t = (np.sqrt(flight_state[3] ** 2 +
                     2 * self.g * (foot_pos_z0 - stepping_stone.height)) +
             flight_state[3]) / self.g
        foot_pos_x = flight_state[0] + self.l0 * sin_theta +\
            flight_state[2] * t
        return t if foot_pos_x >= stepping_stone.left and\
            foot_pos_x <= stepping_stone.right else None

    def apex_to_touchdown_gradient(self, apex_pos_x, apex_height, apex_vel_x,
                                   leg_angle):
        """
        Computes the gradient of the pre-touchdown state x_pre_td w.r.t the
        apex state (x position, height above ground at touchdown, and x
        velocity), also the gradient of x_pre_td w.r.t the leg angle.
        The pre touchdown state can be computed from apex state in the
        closed form as
        [x + xdot * sqrt(2*(z-l0*cos_theta)/g),
         l0 * cos_theta,
         xdot,
         -g * sqrt(2*(z-l0*cos_theta)/g)]
        we can compute the gradient of the pre touchdown state w.r.t the
        apex state and the leg angle in the closed form.
        @param apex_pos_x The horizontal position of the robot at apex.
        @param apex_height The height of the robot at apex above the ground at
        touch down.
        @param apex_vel_x The horizontal velocity of the robot at apex.
        @param leg_angle The angle of the leg at touch down.
        @return (dx_pre_td_dx_apex, dx_pre_td_dleg_angle), dx_pre_td_dx_apex
        is ∂ x⁻_TD / ∂ x_apex, where x_apex = (apex_pos_x, apex_height,
        apex_vel_x). dx_pre_td_dleg_angle is ∂x⁻_TD / ∂θ.
        """
        sin_theta = np.sin(leg_angle)
        cos_theta = np.cos(leg_angle)
        dx_pre_td_dx_apex = np.zeros((4, 3))
        dx_pre_td_dleg_angle = np.zeros(4)

        dx_pre_td_dx_apex[0, 0] = 1
        dt_touchdown_dapex_height = np.sqrt(2 / self.g)\
            * 0.5 / np.sqrt(apex_height - self.l0 * cos_theta)
        dx_pre_td_dx_apex[0, 1] = apex_vel_x * dt_touchdown_dapex_height
        ground = SteppingStone(-np.inf, np.inf, 0)
        t_touchdown =\
            self.time_to_touchdown(np.array([apex_pos_x, apex_height,
                                             apex_vel_x, 0]),
                                   ground, leg_angle)
        assert(t_touchdown is not None)
        dx_pre_td_dx_apex[0, 2] = t_touchdown

        dx_pre_td_dx_apex[2, 2] = 1.
        dx_pre_td_dx_apex[3, 1] = -self.g * dt_touchdown_dapex_height

        dt_touchdown_dleg_angle = np.sqrt(2/self.g) * self.l0 * sin_theta\
            / (2 * np.sqrt(apex_height - self.l0 * cos_theta))
        dx_pre_td_dleg_angle[0] = apex_vel_x * dt_touchdown_dleg_angle
        dx_pre_td_dleg_angle[1] = -self.l0 * sin_theta
        dx_pre_td_dleg_angle[3] = -self.g * dt_touchdown_dleg_angle

        return (dx_pre_td_dx_apex, dx_pre_td_dleg_angle)

    def stance_dynamics_gradient(self, x):
        """
        Compute the gradient of the stance dynamics function w.r.t the state.
        @param x The state in the stance phase.
        @return dxdot_dx The gradient of the dynamics w.r.t the state.
        """
        sin_theta = np.sin(x[1])
        cos_theta = np.cos(x[1])
        xdot_gradient = np.zeros((5, 5))
        xdot_gradient[0, 2] = 1
        xdot_gradient[1, 3] = 1
        xdot_gradient[2, 0] = -self.k / self.mass + x[3] ** 2
        xdot_gradient[2, 1] = self.g * sin_theta
        xdot_gradient[2, 3] = 2 * x[0] * x[3]
        xdot_gradient[3, 0] = -(self.g * sin_theta - 2 * x[2] * x[3])\
            / (x[0] ** 2)
        xdot_gradient[3, 1] = self.g / x[0] * cos_theta
        xdot_gradient[3, 2] = -2 * x[3] / x[0]
        xdot_gradient[3, 3] = -2 * x[2] / x[0]
        return xdot_gradient

    def touchdown_to_liftoff_gradient(self, post_touchdown_state):
        """
        The math is explained in doc/linear_slip.tex
        Given a state right after touchdown, compute the gradient of the map
        from the post-touchdown state to the pre-liftoff state.
        @param post_touchdown_state [r,θ,ṙ,θ_dot,x_foot] right after touch
        down.
        @return dx_pre_liftoff_dx_post_touchdown The gradient of the
        pre-liftoff state w.r.t the post-touchdown state.
        """

        """
        If I use x₀ to denote the post-touchdown state, then we first need to
        evaluate ∂x(t) / ∂x₀ evaluated at the moment of lift off. We know
        d(∂x(t) / ∂x₀)/dt = ∂f/∂x * ∂x(t)/∂x₀
        which can be viewed as an ODE on the matrix ∂x(t)/∂x₀. We will
        integrate this ODE to t_liftoff
        """
        def gradient_dynamics(t, y):
            state = y[0:5]
            state_dot = self.stance_dynamics(state)
            dx_dx0 = y[5:].reshape((5, 5))
            dxdot_dx = self.stance_dynamics_gradient(state)
            dxdot_dx0 = dxdot_dx.dot(dx_dx0)
            ydot = np.concatenate((state_dot, dxdot_dx0.reshape((25,))))
            return ydot

        def liftoff(t, x): return self.liftoff_guard(x)
        liftoff.terminal = True
        liftoff.direction = 1

        y0 = np.zeros(30)
        y0[:5] = post_touchdown_state
        y0[5:] = np.eye(5).reshape((25,))
        ode_sol = solve_ivp(gradient_dynamics, (0, 100), y0, events=liftoff,
                            rtol=1e-8)
        assert(len(ode_sol.t_events) > 0)
        x_pre_lo = ode_sol.y[:5, -1].reshape((5,))
        dx_dx_post_td_lo = ode_sol.y[5:, -1].reshape((5, 5))
        xdot_pre_lo = self.stance_dynamics(x_pre_lo)
        dg_lo_dx = np.array([1, 0, 0, 0, 0])
        dt_lo_dx0 = -1.0 / (dg_lo_dx.dot(xdot_pre_lo))\
            * dg_lo_dx.reshape((1, 5)).dot(dx_dx_post_td_lo)
        return dx_dx_post_td_lo + xdot_pre_lo.reshape((5, 1)).dot(dt_lo_dx0)
