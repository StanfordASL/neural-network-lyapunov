import numpy as np
from scipy.integrate import solve_ivp
from utils import check_shape_and_type


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
        @param state A 4 x 1 column vector. The state of the robot in the
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
