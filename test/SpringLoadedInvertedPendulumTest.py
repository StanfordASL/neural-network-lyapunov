import sys
sys.path.append("..")

import utils
import SpringLoadedInvertedPendulum
from scipy.integrate import solve_ivp
import numpy as np
import unittest


class SlipTest(unittest.TestCase):
    def setUp(self):
        # Use the same setup as Underactuated Robotics.
        mass = 80
        l0 = 1
        gravity = 9.81
        dimensionless_spring_constant = 10.7
        k = dimensionless_spring_constant * mass * gravity / l0
        self.dut = SpringLoadedInvertedPendulum.SLIP(mass, l0, k, gravity)

    def test_touchdown_transition(self):
        def test_fun(pre_state):
            theta = np.arccos(pre_state[1] / self.dut.l0)
            post_state = self.dut.touchdown_transition(pre_state, theta)
            r = post_state[0]
            self.assertEqual(post_state[1], theta)
            r_dot = post_state[2]
            theta_dot = post_state[3]
            x_foot = post_state[4]
            sin_theta = np.sin(theta)
            cos_theta = np.cos(theta)
            # Check if the pre-impact position matches.
            self.assertAlmostEqual(x_foot - self.dut.l0 * sin_theta,
                                   pre_state[0], 10)
            # Check if the velocity of the post-impact state is the same as
            # the velocity of the pre-impact state
            self.assertAlmostEqual(pre_state[2] * cos_theta
                                   + pre_state[3] * sin_theta,
                                   -r * theta_dot, 10)
            self.assertAlmostEqual(r * r_dot,
                                   (pre_state[0] - x_foot) * pre_state[2]
                                   + pre_state[1] * pre_state[3], 10)
            self.assertAlmostEqual(self.dut.flight_phase_energy(pre_state),
                                   self.dut.stance_phase_energy(post_state),
                                   10)

        test_fun(np.array([0.1, 0.7, 2.1, -3.2]))
        test_fun(np.array([-0.5, 0.4, 0.5, -1.2]))

    def test_liftoff_transition(self):
        def test_fun(pre_state):
            assert(pre_state[0] == self.dut.l0)
            post_state = self.dut.liftoff_transition(pre_state)
            # Negate the x velocity of the post-impact state, and treat the
            # negated state as the pre-impact state, and compute the
            # corresponding post-impact state. This new post-impact state
            # should match with the pre lift-off state, except theta and
            # theta_dot are negated.
            negate_post_state = np.copy(post_state)
            negate_post_state[2] = -negate_post_state[2]
            negate_post_impact_state =\
                self.dut.touchdown_transition(negate_post_state,
                                              -pre_state[1])
            self.assertAlmostEqual(negate_post_impact_state[0],
                                   pre_state[0], 10)
            self.assertAlmostEqual(negate_post_impact_state[1],
                                   -pre_state[1], 10)
            self.assertAlmostEqual(negate_post_impact_state[2],
                                   pre_state[2], 10)
            self.assertAlmostEqual(negate_post_impact_state[3],
                                   -pre_state[3], 10)

            self.assertAlmostEqual(self.dut.stance_phase_energy(pre_state),
                                   self.dut.flight_phase_energy(post_state),
                                   10)

        test_fun(np.array([self.dut.l0, -0.2 * np.pi, 0.1, -0.2, 0.5]))
        test_fun(np.array([self.dut.l0, -0.3 * np.pi, 0.2, -0.4, -0.5]))

    def test_apex_map(self):
        pos_x = 0
        apex_height = 1
        vel_x = 3
        (next_pos_x, next_apex_height, next_vel_x) =\
            self.dut.apex_map(pos_x, apex_height, vel_x, np.pi / 6)
        self.assertAlmostEqual(
            self.dut.flight_phase_energy(np.array([pos_x,
                                                   apex_height,
                                                   vel_x,
                                                   0])),
            self.dut.flight_phase_energy(np.array([next_pos_x,
                                                   next_apex_height,
                                                   next_vel_x, 0])), 4)
        (next_pos_x_shift, _, _) = self.dut.apex_map(pos_x + 1,
                                                     apex_height,
                                                     vel_x, np.pi / 6)
        self.assertAlmostEqual(next_pos_x + 1, next_pos_x_shift, 5)

        def check_failure_step(bad_pos_x, bad_apex_height, bad_vel_x,
                               bad_leg_angle):
            """
            For some apex state, the robot can not reach the next apex.
            """
            (bad_next_pos_x, bad_next_apex_height, bad_next_vel_x) =\
                self.dut.apex_map(bad_pos_x, bad_apex_height, bad_vel_x,
                                  bad_leg_angle)
            self.assertIsNone(bad_next_pos_x)
            self.assertIsNone(bad_next_apex_height)
            self.assertIsNone(bad_next_vel_x)

        # Now test an apex height that is too low.
        check_failure_step(0, 0.5, 3, np.pi / 6)
        check_failure_step(1, 1, 3, -np.pi / 6)
        check_failure_step(1, 1, -3, np.pi / 6)

    def test_time_to_touchdown(self):
        def test_fun(flight_state, stepping_stone, leg_angle):
            t = self.dut.time_to_touchdown(flight_state, stepping_stone,
                                           leg_angle)
            sin_theta = np.sin(leg_angle)
            cos_theta = np.cos(leg_angle)
            if t is not None:
                foot_pos_x = flight_state[0] + self.dut.l0 * sin_theta +\
                    flight_state[2] * t
                foot_pos_z = flight_state[1] - self.dut.l0 * cos_theta +\
                    flight_state[3] * t - self.dut.g / 2 * t ** 2
                self.assertAlmostEqual(foot_pos_z, stepping_stone.height, 10)
                self.assertLessEqual(foot_pos_x, stepping_stone.right)
                self.assertGreaterEqual(foot_pos_x, stepping_stone.left)
            else:
                def touchdown(t, x):
                    return self.dut.touchdown_guard(x, leg_angle)
                touchdown.terminal = True
                touchdown.direction = -1
                sol = solve_ivp(lambda t, x: self.dut.flight_dynamics(x),
                                (0, 100), flight_state, events=touchdown)
                if sol.t_events is None:
                    return
                else:
                    self.assertEqual(len(sol.t_events), 1)
                    foot_pos_x = flight_state[0] + self.dut.l0 * sin_theta +\
                        flight_state[2] * sol.t_events[0]
                    self.assertFalse(foot_pos_x <= stepping_stone.right and
                                     foot_pos_x >= stepping_stone.left)

        test_fun(np.array([0.1, 3, 0.5, 1]), SpringLoadedInvertedPendulum.
                 SteppingStone(-np.inf, np.inf, 0), np.pi / 3)
        test_fun(np.array([0.1, 3, 0.5, 1]), SpringLoadedInvertedPendulum.
                 SteppingStone(0, 1, 0), np.pi / 3)
        test_fun(np.array([0.1, 3, 0.5, 1]), SpringLoadedInvertedPendulum.
                 SteppingStone(1, 10, 1), np.pi / 3)

    def test_apex_to_touchdown_gradient(self):
        def test_fun(apex_pos_x, apex_height, apex_vel_x, leg_angle):
            """
            The math is explained in doc/linear_slip.tex.
            We first need to compute ∂x(t) / ∂x_apex evaluated at
            t = t_touchdown. We know
            d(∂x(t) / ∂x_apex)/dt = ∂f/∂x * ∂x(t)/∂x_apex
            If we view this as an ODE on the matrix ∂x(t)/∂x_apex, then we will
            integrate this ODE to t_touchdown.
            """
            def gradient_dynamics(t, y):
                # ∂f/∂x = [0 I]
                #         [0 0]
                y_reshape = y.reshape((4, 3))
                ydot_reshape = np.zeros((4, 3))
                ydot_reshape[0:2, :] = y_reshape[2:4, :]
                return ydot_reshape.reshape((12,))

            t_touchdown = self.dut.time_to_touchdown(
                    np.array([apex_pos_x, apex_height, apex_vel_x, 0]),
                    SpringLoadedInvertedPendulum.
                    SteppingStone(-np.inf, np.inf, 0), leg_angle)
            dx_dx_apex_initial = np.zeros((4, 3))
            dx_dx_apex_initial[0:3, :] = np.eye(3)
            dx_dx_apex_initial = dx_dx_apex_initial.reshape((12,))
            ode_sol = solve_ivp(gradient_dynamics,
                                (0, t_touchdown), dx_dx_apex_initial)
            # dx_dx_apex_touchdown is ∂x(t) / ∂x_apex evaluated at
            # t = t_touchdown
            dx_dx_apex_touchdown = ode_sol.y[:, -1].reshape((4, 3))

            sin_theta = np.sin(leg_angle)
            cos_theta = np.cos(leg_angle)
            # Now we need to compute the gradient of t_touchdown (t_td)  w.r.t
            # the apex state. If we denote the touchdown guard function as
            # g_td, then
            # ∂t_TD/∂x_apex = -(∂g_TD/∂x f(x_pre_td))⁻¹∂g_TD/∂x
            #                 * dx_dx_apex_touchdown
            x_pre_td = np.array([apex_pos_x + apex_vel_x * t_touchdown,
                                 self.dut.l0 * cos_theta,
                                 apex_vel_x,
                                 -self.dut.g * t_touchdown])
            dg_TD_dx = np.array([0, 1, 0, 0])
            xdot_pre_td = self.dut.flight_dynamics(x_pre_td)
            dt_TD_dg_TD = 1.0 / (dg_TD_dx.dot(xdot_pre_td))
            dt_TD_dx_apex = -dt_TD_dg_TD * dg_TD_dx.dot(dx_dx_apex_touchdown)
            dx_pre_td_dx_apex = dx_dx_apex_touchdown +\
                xdot_pre_td.reshape((4, 1)).dot(dt_TD_dx_apex.reshape((1, 3)))

            dg_TD_dleg_angle = self.dut.l0 * sin_theta
            dx_pre_td_dleg_angle = xdot_pre_td * -dt_TD_dg_TD\
                * dg_TD_dleg_angle

            (dx_pre_td_dx_apex_expected, dx_pre_td_dleg_angle_expected) =\
                self.dut.apex_to_touchdown_gradient(apex_pos_x, apex_height,
                                                    apex_vel_x, leg_angle)

            self.assertTrue(utils.
                            compare_numpy_matrices(dx_pre_td_dx_apex,
                                                   dx_pre_td_dx_apex_expected,
                                                   1e-7, 1e-7))
            self.assertTrue(
                    utils.compare_numpy_matrices(dx_pre_td_dleg_angle,
                                                 dx_pre_td_dleg_angle_expected,
                                                 1e-7, 1e-7))

        test_fun(1, 1.5, 0.4, np.pi / 5)
        test_fun(1, 2, 0.8, np.pi / 5)
        test_fun(1, 1, 0.8, np.pi / 3)

    def test_stance_dynamics_gradient(self):
        def test_fun(x):
            grad = self.dut.stance_dynamics_gradient(x)
            grad_numerical = utils.\
                compute_numerical_gradient(self.dut.stance_dynamics, x)
            self.assertTrue(utils.compare_numpy_matrices(grad, grad_numerical,
                                                         1e-7, 1e-7))
        test_fun(np.array([1, np.pi / 5, -0.1, -0.5, 0]))
        test_fun(np.array([0.5, -np.pi / 5, -0.1, -0.5, 1]))
        test_fun(np.array([0.5, -np.pi / 4, 0.2, 0.3, 1]))

    def test_touchdown_to_liftoff_gradient(self):
        def test_fun(post_touchdown_state):
            def touchdown_to_liftoff_map(x0):
                def liftoff(t, x): return self.dut.liftoff_guard(x)
                liftoff.terminal = True
                liftoff.direction = 1
                def hitground1(t, x): return np.pi / 2 + x[1]
                hitground1.terminal = True
                hitground1.direction = -1
                def hitground2(t, x): return x[1] - np.pi / 2
                hitground2.terminal = True
                hitground2.direction = 1
                ode_sol = solve_ivp(lambda t, x: self.dut.stance_dynamics(x),
                                    (0, np.inf), x0,
                                    events=[liftoff, hitground1, hitground2],
                                    rtol=1e-12)
                if len(ode_sol.t_events[0]) > 0:
                    return ode_sol.y[:, -1]
                else:
                    raise Exception("Cannot lift off")

            grad_numerical = utils.\
                compute_numerical_gradient(touchdown_to_liftoff_map,
                                           post_touchdown_state, dx=1e-9)
            grad = self.dut.touchdown_to_liftoff_gradient(post_touchdown_state)
            self.assertTrue(utils.compare_numpy_matrices(grad, grad_numerical,
                                                         1, 1e-5))

        test_fun(np.array([self.dut.l0, np.pi / 5, -0.1, -0.5, 0]))
        test_fun(np.array([self.dut.l0, np.pi / 6, -0.2, -1.5, 0]))
        test_fun(np.array([self.dut.l0, -np.pi / 7, -0.2, -1.5, 0]))


if __name__ == "__main__":
    unittest.main()
