import unittest
import numpy as np
import torch

import neural_network_lyapunov.pybullet_data_generation as\
 pybullet_data_generation
import neural_network_lyapunov.worlds as worlds


class TestPybulletSampleGeneratorJointSpace(unittest.TestCase):

    def setUp(self):
        cb = worlds.get_load_urdf_callback(worlds.urdf_path("pendulum.urdf"))
        self.pbsg = pybullet_data_generation.PybulletSampleGenerator(cb, True)

    def test_generate_sample(self):
        x0 = torch.tensor([np.pi/2, 1.], dtype=self.pbsg.dtype)
        X, X_next, x_next = self.pbsg.generate_sample(x0, .1)
        self.assertEqual(len(X.shape), 3)
        self.assertEqual(X.shape[0], 6)
        self.assertEqual(X_next.shape[0], 3)
        self.assertEqual(x_next.shape[0], 2)

    def test_generate_rollout(self):
        x0 = torch.tensor([np.pi/2, 1.], dtype=self.pbsg.dtype)
        X, x = self.pbsg.generate_rollout(x0, .1, 10)
        self.assertEqual(len(X.shape), 4)
        self.assertEqual(X.shape, (12, 3,
                         self.pbsg.image_width, self.pbsg.image_height))
        self.assertEqual(x.shape, (11, 2))

    def test_generate_dataset(self):
        x_lo = torch.tensor([0., -1.], dtype=self.pbsg.dtype)
        x_up = torch.tensor([2*np.pi, 1.], dtype=self.pbsg.dtype)
        x_data, x_next_data, X_data, X_next_data = self.pbsg.generate_dataset(
            x_lo, x_up, .1, 10, 5)
        self.assertEqual(len(X_data.shape), 4)
        self.assertEqual(len(X_next_data.shape), 4)
        self.assertEqual(X_data.shape, (10 * 5, 6,
                         self.pbsg.image_width, self.pbsg.image_height))
        self.assertEqual(x_data.shape, (10 * 5, 2))


class TestPybulletSampleGeneratorRigidBody(unittest.TestCase):

    def setUp(self):
        cb = worlds.get_load_falling_cubes_callback()
        self.pbsg = pybullet_data_generation.PybulletSampleGenerator(cb, False)

    def test_generate_sample(self):
        x0 = torch.tensor([0, 0, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          dtype=self.pbsg.dtype)
        X, X_next, x_next = self.pbsg.generate_sample(x0, .1)
        self.assertEqual(len(X.shape), 3)
        self.assertEqual(X.shape[0], 6)
        self.assertEqual(X_next.shape[0], 3)
        self.assertEqual(x_next.shape[0], 12)

    def test_generate_rollout(self):
        x0 = torch.tensor([0, 0, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          dtype=self.pbsg.dtype)
        X, x = self.pbsg.generate_rollout(x0, .1, 10)
        self.assertEqual(len(X.shape), 4)
        self.assertEqual(X.shape, (12, 3,
                         self.pbsg.image_width, self.pbsg.image_height))
        self.assertEqual(x.shape, (11, 12))

    def test_generate_dataset(self):
        x_lo = torch.tensor([0, 0, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            dtype=self.pbsg.dtype)
        x_up = torch.tensor([0, 0, 11, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            dtype=self.pbsg.dtype)
        x_data, x_next_data, X_data, X_next_data = self.pbsg.generate_dataset(
            x_lo, x_up, .1, 10, 5)
        self.assertEqual(len(X_data.shape), 4)
        self.assertEqual(len(X_next_data.shape), 4)
        self.assertEqual(X_data.shape, (10 * 5, 6,
                         self.pbsg.image_width, self.pbsg.image_height))
        self.assertEqual(x_data.shape, (10 * 5, 12))


if __name__ == "__main__":
    unittest.main()
