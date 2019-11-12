import unittest
import numpy as np

import robust_value_approx.spring_loaded_inverted_pendulum as\
    spring_loaded_inverted_pendulum


class SlipSimulationTest(unittest.TestCase):
    def setUp(self):
        mass = 80
        l0 = 1
        gravity = 9.81
        dimensionless_spring_constant = 10.7
        k = dimensionless_spring_constant * mass * gravity / l0
        self.dut = spring_loaded_inverted_pendulum.SLIP(mass, l0, k, gravity)

    def test_sim(self):
        x0 = np.array([0, 1, 3., 0])
        theta_step = [np.pi / 6, np.pi / 7]
        sol = self.dut.simulate(x0, theta_step)
        self.assertEqual(len(sol), 4)
        total_energy = self.dut.flight_phase_energy(x0)

        slip_x = []
        slip_y = []
        for step in range(len(theta_step)):
            for i in range(sol[2 * step].y.shape[1]):
                self.assertAlmostEqual(
                        self.dut.flight_phase_energy(sol[2 * step].y[:, i]),
                        total_energy, 2)
            slip_x.extend(sol[2 * step].y[0])
            slip_y.extend(sol[2 * step].y[1])
            # The accuracy of RK45 is not high for the stance phase.
            slip_r = np.array(sol[2 * step + 1].y[0])
            slip_theta = np.array(sol[2 * step + 1].y[1])
            slip_x_foot = np.array(sol[2 * step + 1].y[4])
            slip_x.extend(list(slip_x_foot - slip_r * np.sin(slip_theta)))
            slip_y.extend(list(slip_r * np.cos(slip_theta)))
            for i in range(sol[2 * step + 1].y.shape[1]):
                self.assertAlmostEqual(
                    self.dut.stance_phase_energy(sol[2 * step + 1].y[:, i]),
                    total_energy, 2)

        # plt.figure()
        # plt.plot(slip_x, slip_y)
        # plt.show()


if __name__ == "__main__":
    unittest.main()
