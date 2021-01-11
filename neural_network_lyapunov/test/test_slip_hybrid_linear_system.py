import neural_network_lyapunov.slip_hybrid_linear_system as slip_hybrid
import neural_network_lyapunov.utils as utils

import unittest
import numpy as np


class SpringHybridLinearSystemTest(unittest.TestCase):
    def setUp(self):
        # Use the same setup as Underactuated Robotics.
        mass = 80
        l0 = 1
        gravity = 9.81
        dimensionless_spring_constant = 10.7
        k = dimensionless_spring_constant * mass * gravity / l0
        self.dut = slip_hybrid.SlipHybridLinearSystem(mass, l0, k, gravity)
        self.dut.add_stepping_stone(-0.1, 0.1, 0)
        self.dut.add_stepping_stone(1, 1.5, 0.1)
        self.dut.add_stepping_stone(2, 2.5, -0.2)
        self.dut.add_stepping_stone(3, 3.5, 0.3)

    def test_apex_map_linear_approximation(self):
        def test_fun(apex_state, stepping_stone_index, leg_angle):
            (A, B, c, a_t, b_t, c_t, P, q) =\
                self.dut.apex_map_linear_approximation(
                    apex_state, self.dut.stepping_stones[stepping_stone_index],
                    leg_angle)
            terrain_height =\
                self.dut.stepping_stones[stepping_stone_index].height

            def apex_map(x0_theta):
                x0 = x0_theta[:3]
                theta = x0_theta[-1]
                if (not self.dut.slip.can_touch_stepping_stone(
                        np.array([x0[0], x0[1], x0[2], 0]),
                        self.dut.stepping_stones[stepping_stone_index],
                        leg_angle)):
                    return None
                res = np.empty(4)
                (res[0], res[1], res[2], res[3]) =\
                    self.dut.slip.apex_map(
                        x0[0], x0[1] - terrain_height, x0[2], theta)
                res[1] += terrain_height
                return res

            apex_map_res = apex_map(
                np.array(
                    [apex_state[0], apex_state[1], apex_state[2], leg_angle]))
            if (apex_map_res is None):
                self.assertIsNone(A)
                self.assertIsNone(B)
                self.assertIsNone(c)
                self.assertIsNone(a_t)
                self.assertIsNone(b_t)
                self.assertIsNone(c_t)
                self.assertIsNone(P)
                self.assertIsNone(q)
            elif (A is not None):
                # print("A is not None")
                # First check if the constant terms in the linear approximation
                # are correct.
                self.assertTrue(
                    utils.compare_numpy_matrices(apex_map_res[:3],
                                                 (A @ (apex_state.reshape(
                                                     (3, 1))) + B * leg_angle +
                                                  c).squeeze(), 1e-5, 1e-5))
                self.assertAlmostEqual(
                    apex_map_res[3],
                    a_t.dot(apex_state) + b_t * leg_angle + c_t, 5)
                # Now check if the gradient is correct.
                grad_numerical = utils.compute_numerical_gradient(
                    apex_map,
                    np.array([
                        apex_state[0], apex_state[1], apex_state[2], leg_angle
                    ]))
                self.assertTrue(
                    utils.compare_numpy_matrices(grad_numerical[:3, :3], A,
                                                 1e-5, 1e-5))
                self.assertTrue(
                    utils.compare_numpy_matrices(grad_numerical[:3, 3],
                                                 B.squeeze(), 1e-5, 1e-5))
                self.assertTrue(
                    utils.compare_numpy_matrices(grad_numerical[3, :3], a_t,
                                                 1e-5, 1e-5))
                self.assertAlmostEqual(grad_numerical[3, 3], b_t, 5)
                num_constraints = 2
                if (self.dut.stepping_stones[stepping_stone_index].left !=
                        -np.inf):
                    num_constraints += 1
                if (self.dut.stepping_stones[stepping_stone_index].right !=
                        np.inf):
                    num_constraints += 1
                self.assertEqual(P.shape, (num_constraints, 4))
                # Now check if the constraints are correct.
                # First of all, the apex_state should satisfy the constraint.
                lhs = P.dot(
                    np.array([
                        apex_state[0], apex_state[1], apex_state[2], leg_angle
                    ]))
                self.assertTrue(np.alltrue(np.less_equal(lhs, q.squeeze())))
                # Now check q - lhs.
                rhs_minus_lhs_expected = np.empty(num_constraints)
                rhs_minus_lhs_expected[0] = apex_state[1]\
                    - self.dut.slip.l0 * np.cos(leg_angle) - terrain_height
                (_, _, _, _, _, _, _, _, x_pre_td, _, _, x_post_lo) =\
                    self.dut.slip.apex_to_apex_gradient(
                        np.array([apex_state[0],
                                  apex_state[1] - terrain_height,
                                  apex_state[2]]), leg_angle)
                rhs_minus_lhs_expected[1] = x_post_lo[3]
                constraints_count = 2
                if (self.dut.stepping_stones[stepping_stone_index].left !=
                        -np.inf):
                    rhs_minus_lhs_expected[constraints_count] =\
                        x_pre_td[0] + self.dut.slip.l0 * np.sin(leg_angle) -\
                        self.dut.stepping_stones[stepping_stone_index].left
                    constraints_count += 1
                if (self.dut.stepping_stones[stepping_stone_index].right !=
                        np.inf):
                    rhs_minus_lhs_expected[constraints_count] =\
                        self.dut.stepping_stones[stepping_stone_index].right -\
                        x_pre_td[0] - self.dut.slip.l0 * np.sin(leg_angle)
                    constraints_count += 1
                self.assertTrue(
                    utils.compare_numpy_matrices(rhs_minus_lhs_expected,
                                                 q.squeeze() - lhs, 1e-5,
                                                 1e-5))

        test_fun(np.array([0, 1, 2]), 0, np.pi / 5)
        test_fun(np.array([0, 1, 3]), 1, np.pi / 5)

        # Compute the initial velocity such that the robot lands on stepping
        # stone 2
        def compute_apex_state(t_td, theta, vel_x, stepping_stone_index):
            apex_state = np.zeros(3)
            apex_state[2] = vel_x
            apex_state[0] = (
                self.dut.stepping_stones[stepping_stone_index].right +
                self.dut.stepping_stones[stepping_stone_index].left) / 2 -\
                t_td * apex_state[2] - self.dut.slip.l0 * np.sin(theta)
            apex_state[1] = self.dut.slip.g / 2 * t_td**2 +\
                self.dut.slip.l0 * np.cos(theta)
            return apex_state

        test_fun(compute_apex_state(0.5, np.pi / 7, 5, 2), 2, np.pi / 7)
        test_fun(compute_apex_state(0.4, np.pi / 6, 4, 2), 2, np.pi / 6)


if __name__ == "__main__":
    unittest.main()
