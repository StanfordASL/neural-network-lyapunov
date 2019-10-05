import sys
sys.path.append("..")

import unittest
import numpy as np

import SpringLoadedInvertedPendulum


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


if __name__ == "__main__":
    unittest.main()
