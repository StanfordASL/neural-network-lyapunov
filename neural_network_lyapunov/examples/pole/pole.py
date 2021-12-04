import torch

import numpy as np


class Pole:
    """
    Describe the system with a massless pole, one end of the pole is a sphere,
    the other end of the pole is the robot end-effector. The input to the
    system is the force applied on the end-effector.
    The state of the system is:
        [x_AB, y_AB, ẋ_A ẏ_A ż_A, ẋ_AB ẏ_AB]
        where x_AB, y_AB are delta horizontal position from the end-effector
        to the sphere.
        ẋ_A ẏ_A ż_A is the velocity of the end-effector
        ẋ_AB ẏ_AB is delta horizontal velocity from the end-effector to the
        sphere.
    The input to the system is:
        [f_x, f_y, f_z], namely force applied on the end-effector.
    """
    def __init__(self, m_sphere, m_ee, length):
        """
        Args:
          m_sphere: The mass of the sphere.
          m_ee: The mass of the end-effector.
          length: The length of the pole
        """
        self.m_sphere = m_sphere
        self.m_ee = m_ee
        self.length = length
        self.dtype = torch.float64
        self.gravity = 9.81

    def dynamics(self, x, u):
        assert (x.shape == (7, ))
        assert (u.shape == (3, ))
        x_AB = x[0]
        y_AB = x[1]
        xd_AB = x[5]
        yd_AB = x[6]
        if isinstance(x, np.ndarray):
            z_AB = np.sqrt(self.length**2 - x_AB**2 - y_AB**2)
            M = np.array([[self.m_sphere + self.m_ee, 0, 0, self.m_sphere, 0],
                          [0, self.m_sphere + self.m_ee, 0, 0, self.m_sphere],
                          [
                              0., 0., self.m_sphere + self.m_ee,
                              -self.m_sphere * x_AB / z_AB,
                              -self.m_sphere * y_AB / z_AB
                          ],
                          [
                              self.m_sphere, 0, -self.m_sphere * x_AB / z_AB,
                              self.m_sphere + self.m_sphere * (x_AB / z_AB)**2,
                              self.m_sphere * x_AB * y_AB / z_AB**2
                          ],
                          [
                              0, self.m_sphere, -self.m_sphere * y_AB / z_AB,
                              self.m_sphere * x_AB * y_AB / z_AB**2,
                              self.m_sphere + self.m_sphere * (y_AB / z_AB)**2
                          ]])
            C = np.array([
                0, 0,
                (self.m_ee + self.m_sphere) * self.gravity - self.m_sphere *
                (xd_AB**2 * (self.length**2 - y_AB**2) / z_AB**3 + yd_AB**2 *
                 (self.length**2 - x_AB**2) / z_AB**3 -
                 2 * x_AB * y_AB * xd_AB * yd_AB / z_AB**3),
                -self.m_sphere * self.gravity * x_AB / z_AB +
                self.m_sphere * x_AB *
                ((xd_AB**2 + yd_AB**2) / z_AB**2 +
                 (x_AB * xd_AB + y_AB * yd_AB)**2 / z_AB**4),
                -self.m_sphere * self.gravity * y_AB / z_AB +
                self.m_sphere * y_AB *
                ((xd_AB**2 + yd_AB**2) / z_AB**2 +
                 (x_AB * xd_AB + y_AB * yd_AB)**2 / z_AB**4)
            ])
            vdot = np.linalg.solve(M, np.array([u[0], u[1], u[2], 0, 0]) - C)
            return np.concatenate((x[5:], vdot))
        elif isinstance(x, torch.Tensor):
            z_AB = torch.sqrt(self.length**2 - x_AB**2 - y_AB**2)
            M = torch.zeros((5, 5), dtype=self.dtype)
            M[0, 0] = self.m_sphere + self.m_ee
            M[1, 1] = M[0, 0]
            M[2, 2] = M[0, 0]
            M[0, 3] = self.m_sphere
            M[3, 0] = self.m_sphere
            M[1, 4] = self.m_sphere
            M[4, 1] = self.m_sphere
            M[2, 3] = -self.m_sphere * x_AB / z_AB
            M[3, 2] = M[2, 3]
            M[2, 4] = -self.m_sphere * y_AB / z_AB
            M[4, 2] = M[2, 4]
            M[3, 3] = self.m_sphere + self.m_sphere * (x_AB / z_AB)**2
            M[3, 4] = self.m_sphere * x_AB * y_AB / z_AB**2
            M[4, 3] = M[3, 4]
            M[4, 4] = self.m_sphere + self.m_sphere * (y_AB / z_AB)**2
            C = torch.zeros((5, ), dtype=self.dtype)
            C[2] = (self.m_sphere +
                    self.m_ee) * self.gravity - self.m_sphere * (
                        xd_AB**2 *
                        (self.length**2 - y_AB**2) / z_AB**3 + yd_AB**2 *
                        (self.length**2 - x_AB**2) / z_AB**3 -
                        2 * x_AB * y_AB * xd_AB * yd_AB / z_AB**3)
            C[3] = -self.m_sphere * self.gravity * x_AB / z_AB + \
                self.m_sphere * x_AB * (
                (xd_AB**2 + yd_AB**2) / z_AB**2 +
                (x_AB * xd_AB + y_AB * yd_AB)**2 / z_AB**4)
            C[4] = -self.m_sphere * self.gravity * y_AB / z_AB + \
                self.m_sphere * y_AB * (
                (xd_AB**2 + yd_AB**2) / z_AB**2 +
                (x_AB * xd_AB + y_AB * yd_AB)**2 / z_AB**4)
            vdot = torch.linalg.solve(
                M,
                torch.cat((u, torch.zeros((2, ), dtype=self.dtype))) - C)
            return torch.cat((x[5:], vdot))

    def gradient(self, x: torch.Tensor, u: torch.Tensor):
        """
        Compute the gradient of the dynamics through autodiff.
        """
        assert (isinstance(x, torch.Tensor))
        assert (isinstance(u, torch.Tensor))
        x_requires_grad = x.requires_grad
        u_requires_grad = u.requires_grad
        x.requires_grad = True
        u.requires_grad = True
        grad = torch.autograd.functional.jacobian(self.dynamics, inputs=(x, u))
        x.requires_grad = x_requires_grad
        u.requires_grad = u_requires_grad
        return grad[0], grad[1]
