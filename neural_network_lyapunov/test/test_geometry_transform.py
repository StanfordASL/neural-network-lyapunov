import neural_network_lyapunov.geometry_transform as geometry_transform
import numpy as np
import torch
import unittest
import pybullet


class TestRpy2rotmat(unittest.TestCase):
    def test_numpy(self):
        def tester(rpy):
            R = geometry_transform.rpy2rotmat(rpy)
            np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1E-12)
            self.assertAlmostEqual(np.linalg.det(R), 1)
            R_expected = np.array(
                pybullet.getMatrixFromQuaternion(
                    pybullet.getQuaternionFromEuler(rpy))).reshape((3, 3))
            np.testing.assert_allclose(R, R_expected, atol=1E-12)

        tester(np.array([0.5, 0.4, 0.3]))
        tester(np.array([-0.5, 0.1, -0.3]))
        tester(np.array([0, 0, 0]))

    def test_torch(self):
        def tester(rpy):
            R = geometry_transform.rpy2rotmat(rpy)
            R_expected = geometry_transform.rpy2rotmat(rpy.detach().numpy())
            np.testing.assert_allclose(R.detach().numpy(),
                                       R_expected,
                                       atol=1E-12)
            # Make sure we can get the gradient.
            R.sum().backward()

        tester(
            torch.tensor([0.5, 0.3, -1.2],
                         dtype=torch.float64,
                         requires_grad=True))
        tester(
            torch.tensor([-0.9, 0.3, -1.2],
                         dtype=torch.float64,
                         requires_grad=True))
        tester(
            torch.tensor([-0.9, np.pi / 2, -1.2],
                         dtype=torch.float64,
                         requires_grad=True))


class TestRpy2rotmatGradient(unittest.TestCase):
    def check_gradient(self, rpy: torch.Tensor):
        rpy.requires_grad = True
        dR_droll, dR_dpitch, dR_dyaw = geometry_transform.rpy2rotmat_gradient(
            rpy)
        dR_droll_np, dR_dpitch_np, dR_dyaw_np = \
            geometry_transform.rpy2rotmat_gradient(rpy.detach().numpy())
        np.testing.assert_allclose(dR_droll.detach().numpy(), dR_droll_np)
        np.testing.assert_allclose(dR_dpitch.detach().numpy(), dR_dpitch_np)
        np.testing.assert_allclose(dR_dyaw.detach().numpy(), dR_dyaw_np)
        for i in range(3):
            for j in range(3):
                if rpy.grad is not None:
                    rpy.grad.zero_()
                    R = geometry_transform.rpy2rotmat(rpy)
                    R[i, j].backward()
                    self.assertEqual(dR_droll[i, j].item(), rpy.grad[0].item())
                    self.assertEqual(dR_dpitch[i, j].item(),
                                     rpy.grad[1].item())
                    self.assertEqual(dR_dyaw[i, j].item(), rpy.grad[2].item())

    def test(self):
        self.check_gradient(torch.tensor([0, 0, 0], dtype=torch.float64))
        self.check_gradient(torch.tensor([0.1, 0, 0], dtype=torch.float64))
        self.check_gradient(torch.tensor([0, 0.2, 0], dtype=torch.float64))
        self.check_gradient(torch.tensor([0, 0, 0.3], dtype=torch.float64))
        self.check_gradient(torch.tensor([-0.5, 0.4, 0.3],
                                         dtype=torch.float64))


if __name__ == "__main__":
    unittest.main()
