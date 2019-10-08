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
            The pre touchdown state can be computed from apex state in the
            closed form as
            [x + xdot * sqrt(2*(z-l0*cos_theta)/g),
             l0 * cos_theta,
             xdot,
             -g * sqrt(2*(z-l0*cos_theta)/g)]
            we can compute the gradient of the pre touchdown state w.r.t the
            apex state in the closed form.
            """
            sin_theta = np.sin(leg_angle)
            cos_theta = np.cos(leg_angle)
            dx_pre_td_dx_apex_expected = np.zeros((4, 3))
            dx_pre_td_dleg_angle_expected = np.zeros(4)

            dx_pre_td_dx_apex_expected[0, 0] = 1
            dt_touchdown_dapex_height = np.sqrt(2 / self.dut.g)\
                * 0.5 / np.sqrt(apex_height - self.dut.l0 * cos_theta)
            dx_pre_td_dx_apex_expected[0, 1] = apex_vel_x\
                * dt_touchdown_dapex_height
            ground = SpringLoadedInvertedPendulum.SteppingStone(-np.inf,
                                                                np.inf, 0)
            t_touchdown =\
                self.dut.time_to_touchdown(np.array([apex_pos_x, apex_height,
                                                     apex_vel_x, 0]),
                                           ground, leg_angle)
            assert(t_touchdown is not None)
            dx_pre_td_dx_apex_expected[0, 2] = t_touchdown

            dx_pre_td_dx_apex_expected[2, 2] = 1.
            dx_pre_td_dx_apex_expected[3, 1] = -self.dut.g\
                * dt_touchdown_dapex_height

            dt_touchdown_dleg_angle = np.sqrt(2/self.dut.g)\
                * self.dut.l0 * sin_theta\
                / (2 * np.sqrt(apex_height - self.dut.l0 * cos_theta))
            dx_pre_td_dleg_angle_expected[0] = apex_vel_x\
                * dt_touchdown_dleg_angle
            dx_pre_td_dleg_angle_expected[1] = -self.dut.l0 * sin_theta
            dx_pre_td_dleg_angle_expected[3] = -self.dut.g\
                * dt_touchdown_dleg_angle

            (dx_pre_td_dx_apex, dx_pre_td_dleg_angle) =\
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


if __name__ == "__main__":
    unittest.main()
