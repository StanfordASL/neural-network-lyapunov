import neural_network_lyapunov.examples.quadrotor3d.quadrotor as quadrotor
import neural_network_lyapunov.utils as utils
import neural_network_lyapunov.test.test_relu_system as test_relu_system
import neural_network_lyapunov.geometry_transform as geometry_transform
import unittest
import numpy as np
import torch


class TestQuadrotor(unittest.TestCase):
    def test_dynamics(self):
        plant = quadrotor.Quadrotor(torch.float64)
        # First make sure that at the equilibrium, xdot is zero.
        np.testing.assert_allclose(
            plant.dynamics(np.zeros((12, )), plant.hover_thrust * np.ones(
                (4, ))), np.zeros((12, )))

        def tester(x_val, u_val):
            # First test if the linear acceleration is correct.
            xdot = plant.dynamics(x_val, u_val)
            plant_input = np.array([
                u_val.sum(),
                plant.arm_length * u_val[1] - plant.arm_length * u_val[3],
                -plant.arm_length * u_val[0] + plant.arm_length * u_val[2],
                plant.z_torque_to_force_factor *
                (u_val[0] - u_val[1] + u_val[2] - u_val[3])
            ])
            rpy = x_val[3:6]
            R = geometry_transform.rpy2rotmat(rpy)
            # First check linear acceleration.
            total_force = R @ np.array([
                0, 0, plant_input[0]
            ]) - plant.mass * np.array([0, 0, plant.gravity])
            pos_ddot = total_force / plant.mass
            np.testing.assert_allclose(xdot[6:9], pos_ddot)
            # Now check angular acceleration.
            omega = x_val[9:12]
            omega_dot = xdot[9:12]
            np.testing.assert_allclose(
                plant.inertia * omega_dot +
                np.cross(omega, plant.inertia * omega), plant_input[1:])
            # Check linear velocity
            np.testing.assert_allclose(xdot[:3], x_val[6:9])
            # Now check rpy_dot. This is the tricky part. I will compute the
            # time derivative of R using angular velocity, and then compare
            # against the time derivative of R computed from rpy_dot.
            omega_hat = np.array([[0, -omega[2], omega[1]],
                                  [omega[2], 0, -omega[0]],
                                  [-omega[1], omega[0], 0]])
            Rdot = R @ omega_hat
            cos_roll = np.cos(rpy[0])
            sin_roll = np.sin(rpy[0])
            cos_pitch = np.cos(rpy[1])
            sin_pitch = np.sin(rpy[1])
            cos_yaw = np.cos(rpy[2])
            sin_yaw = np.sin(rpy[2])
            R_roll = np.array([[1., 0, 0], [0, cos_roll, -sin_roll],
                               [0, sin_roll, cos_roll]])
            R_pitch = np.array([[cos_pitch, 0, sin_pitch], [0, 1., 0],
                                [-sin_pitch, 0, cos_pitch]])
            R_yaw = np.array([[cos_yaw, -sin_yaw, 0], [sin_yaw, cos_yaw, 0],
                              [0, 0, 1.]])
            dR_droll = R_yaw @ R_pitch @ np.array([[0., 0, 0],
                                                   [0, -sin_roll, -cos_roll],
                                                   [0, cos_roll, -sin_roll]])
            dR_dpitch = R_yaw @ np.array([[-sin_pitch, 0, cos_pitch], [
                0, 0, 0
            ], [-cos_pitch, 0, -sin_pitch]]) @ R_roll
            dR_dyaw = np.array([[-sin_yaw, -cos_yaw, 0], [
                cos_yaw, -sin_yaw, 0
            ], [0, 0, 0.]]) @ R_pitch @ R_roll
            rpy_dot = xdot[3:6]
            Rdot_expected = dR_droll * rpy_dot[0] + dR_dpitch * rpy_dot[
                1] + dR_dyaw * rpy_dot[2]
            np.testing.assert_allclose(Rdot, Rdot_expected)

            # Now test with torch tensor as input.
            xdot_torch = plant.dynamics(torch.from_numpy(x_val),
                                        torch.from_numpy(u_val))
            np.testing.assert_allclose(xdot, xdot_torch.detach().numpy())

        tester(
            np.array([
                0.1, 0.2, 0.1, 0.5 * np.pi, 0.3 * np.pi, 0.4 * np.pi, 2., 1.,
                0.5, -0.2, 0.3, 0.4
            ]), np.array([1., 2., 0.5, 3.]))
        tester(
            np.array([
                0.1, -1.2, 0.1, 0.9 * np.pi, -0.3 * np.pi, -0.4 * np.pi, -2.,
                1., 1.5, 0.2, -0.5, 1.4
            ]), np.array([1., 3., 2.5, 3.]))


class TestQuadrotorWithPixhawkReLUSystem(unittest.TestCase):
    def setUp(self):
        self.dtype = torch.float64
        torch.manual_seed(0)
        dynamics_relu = utils.setup_relu((7, 15, 10, 6),
                                         params=None,
                                         negative_slope=0.1,
                                         bias=True,
                                         dtype=torch.float64)
        x_lo = torch.tensor(
            [-2, -2, -2, -0.5 * np.pi, -0.3 * np.pi, -np.pi, -3, -3, -3],
            dtype=self.dtype)
        x_up = -x_lo
        u_lo = torch.tensor([-0.5 * np.pi, -0.3 * np.pi, -np.pi, -5],
                            dtype=self.dtype)
        u_up = -u_lo
        hover_thrust = 3.
        dt = 0.03
        self.dut = quadrotor.QuadrotorWithPixhawkReLUSystem(
            self.dtype, x_lo, x_up, u_lo, u_up, dynamics_relu, hover_thrust,
            dt)

        self.assertEqual(self.dut.x_dim, 9)
        self.assertEqual(self.dut.u_dim, 4)
        np.testing.assert_allclose(self.dut.u_equilibrium,
                                   np.array([0, 0, 0, hover_thrust]))

    def test_step_forward(self):
        def compute_next_state(x_val, u_val):
            pos_current = x_val[:3]
            rpy_current = x_val[3:6]
            vel_current = x_val[6:9]
            delta_vel_rpy = self.dut.dynamics_relu(
                torch.cat((rpy_current, u_val))) - self.dut.dynamics_relu(
                    torch.cat((torch.zeros(
                        (3, ), dtype=self.dtype), self.dut.u_equilibrium)))
            vel_next = vel_current + delta_vel_rpy[:3]
            pos_next = pos_current + (vel_current + vel_next) * self.dut.dt / 2
            rpy_next = rpy_current + delta_vel_rpy[3:6]
            x_next_expected = torch.cat(
                (pos_next, rpy_next, vel_next)).detach().numpy()
            return x_next_expected

        # Test a single state/control
        x_start = torch.tensor([-0.2, 0.3, 0.4, -0.5, 1.2, 0.4, 0.3, 1.4, 0.9],
                               dtype=self.dtype)
        u_start = torch.tensor([-0.5, 1.2, 0.6, 1.3], dtype=self.dtype)
        x_next = self.dut.step_forward(x_start, u_start)
        np.testing.assert_allclose(x_next.detach().numpy(),
                                   compute_next_state(x_start, u_start))
        # Test a batch of state/control
        x_start = torch.tensor(
            [[-0.2, 0.1, 0.5, -1.2, -0.1, 0.9, 1.4, -0.3, 0.2],
             [0.5, -0.3, 1.2, -1.1, 0.8, 0.4, -0.5, 1.3, 1.2]],
            dtype=self.dtype)
        u_start = torch.tensor([[0.4, 0.3, -0.5, 1.1], [0.4, -0.3, 0.9, 1.3]],
                               dtype=self.dtype)
        x_next = self.dut.step_forward(x_start, u_start)
        for i in range(x_start.shape[0]):
            np.testing.assert_allclose(
                x_next[i].detach().numpy(),
                compute_next_state(x_start[i], u_start[i]))

    def test_add_dynamics_constraint(self):
        test_relu_system.check_add_dynamics_constraint(self.dut,
                                                       self.dut.x_equilibrium,
                                                       self.dut.u_equilibrium,
                                                       atol=1E-10)
        test_relu_system.check_add_dynamics_constraint(
            self.dut,
            torch.tensor([0.1, 0.5, -0.3, 0.2, 0.9, 1.1, 1.5, -1.1, 0.4],
                         dtype=self.dtype),
            torch.tensor([0.5, 0.8, -0.4, -1.2], dtype=self.dtype))
        test_relu_system.check_add_dynamics_constraint(
            self.dut,
            torch.tensor([-0.2, 0.7, 0.4, -1.2, 0.3, 0.4, -1.5, 1.3, 1.4],
                         dtype=self.dtype),
            torch.tensor([0.1, 0.9, 0.3, 1.5], dtype=self.dtype))


class TestQuadrotorReLUSystem(unittest.TestCase):
    def setUp(self):
        self.dtype = torch.float64
        torch.manual_seed(0)
        dynamics_relu = utils.setup_relu((10, 15, 15, 9),
                                         params=None,
                                         negative_slope=0.1,
                                         bias=True,
                                         dtype=torch.float64)
        x_lo = torch.tensor([
            -5, -5, -5, -0.5 * np.pi, -0.5 * np.pi, -np.pi, -7, -7, -7, -np.pi,
            -np.pi, -np.pi
        ],
                            dtype=self.dtype)
        x_up = -x_lo
        u_lo = torch.tensor([0., 0., 0., 0.], dtype=self.dtype)
        u_up = torch.tensor([10., 10., 10., 10.], dtype=self.dtype)
        hover_thrust = 5.
        dt = 0.01
        self.dut = quadrotor.QuadrotorReLUSystem(self.dtype, x_lo, x_up, u_lo,
                                                 u_up, dynamics_relu,
                                                 hover_thrust, dt)
        np.testing.assert_allclose(self.dut.x_equilibrium.detach().numpy(),
                                   np.zeros((12, )))
        np.testing.assert_allclose(self.dut.u_equilibrium.detach().numpy(),
                                   hover_thrust * np.ones((4, )))

    def test_step_forward(self):
        # Test equilibrium
        x_next = self.dut.step_forward(self.dut.x_equilibrium,
                                       self.dut.u_equilibrium)
        np.testing.assert_allclose(x_next.detach().numpy(),
                                   self.dut.x_equilibrium.detach().numpy())

        def compute_next_state(x_val, u_val):
            rpy_delta_posdot_angularvel_next = self.dut.dynamics_relu(
                torch.cat((x_val[3:6], x_val[9:12],
                           u_val))) - self.dut.dynamics_relu(
                               torch.cat((self.dut.x_equilibrium[3:6],
                                          self.dut.x_equilibrium[9:12],
                                          self.dut.u_equilibrium)))
            rpy_next = rpy_delta_posdot_angularvel_next[:3]
            posdot_next = x_val[6:9] + rpy_delta_posdot_angularvel_next[3:6]
            angularvel_next = rpy_delta_posdot_angularvel_next[6:9]
            pos_next = x_val[:3] + (x_val[6:9] + posdot_next) * self.dut.dt / 2
            return torch.cat(
                (pos_next, rpy_next, posdot_next, angularvel_next))

        # Test a single state/control
        x_val = torch.tensor(
            [0.5, 0.4, 1.2, 0.45, 0.3, -0.6, -2., 4., 1.5, 0.75, 0.32, 0.43],
            dtype=self.dtype)
        u_val = torch.tensor([0.24, 0.51, 0.42, 1.54], dtype=self.dtype)
        np.testing.assert_allclose(
            self.dut.step_forward(x_val, u_val).detach().numpy(),
            compute_next_state(x_val, u_val).detach().numpy())
        x_val = torch.tensor(
            [-1.5, 9.4, 10.2, 0.35, 4.3, -0.6, -3., 4., 1.5, 2.75, 1.32, 0.43],
            dtype=self.dtype)
        u_val = torch.tensor([2.4, 5.1, 4.2, 1.54], dtype=self.dtype)
        np.testing.assert_allclose(
            self.dut.step_forward(x_val, u_val).detach().numpy(),
            compute_next_state(x_val, u_val).detach().numpy())

        # Test a batch of state/control
        x_val = torch.tensor([[
            0.5, 0.4, 1.2, 0.45, 0.3, -0.6, -2., 4., 1.5, 0.75, 0.32, 0.43
        ], [0.12, 4.5, 3.2, -0.5, 0.9, 1.3, 0.25, 4.1, 0.9, 1.6, -0.7, -2.1]],
                             dtype=self.dtype)
        u_val = torch.tensor(
            [[0.24, 0.51, 0.42, 1.54], [1.32, 0.64, 4.3, 8.2]],
            dtype=self.dtype)
        x_next = self.dut.step_forward(x_val, u_val)
        for i in range(x_val.shape[0]):
            np.testing.assert_allclose(
                x_next[i].detach().numpy(),
                compute_next_state(x_val[i], u_val[i]).detach().numpy())

    def test_add_dynamics_constraint(self):
        test_relu_system.check_add_dynamics_constraint(self.dut,
                                                       self.dut.x_equilibrium,
                                                       self.dut.u_equilibrium,
                                                       atol=1E-10)
        test_relu_system.check_add_dynamics_constraint(
            self.dut,
            torch.tensor(
                [0.1, 0.5, -0.3, 0.2, 0.9, 1.1, 1.5, -1.1, 0.4, 0.5, 1.2, 2.4],
                dtype=self.dtype),
            torch.tensor([0.5, 3.8, 2.4, 1.2], dtype=self.dtype))
        test_relu_system.check_add_dynamics_constraint(
            self.dut,
            torch.tensor([
                4.1, 3.5, -4.3, 1.2, 1.1, 2.1, 1.7, -1.1, 1.4, 0.7, 2.2, -2.4
            ],
                         dtype=self.dtype),
            torch.tensor([4.5, 3.8, 6.4, 1.2], dtype=self.dtype))


if __name__ == "__main__":
    unittest.main()
