import robust_value_approx.random_sample as random_sample
import robust_value_approx.value_approximation as value_approximation
import double_integrator_utils

import unittest
import torch


class ValueApproximationTest(unittest.TestCase):
    def test_value_approximation_eval(self):
        N = 5
        vf = double_integrator_utils.get_value_function(N=N)
        x0_lo = -1 * torch.ones(vf.sys.x_dim, dtype=vf.dtype)
        x0_up = 1 * torch.ones(vf.sys.x_dim, dtype=vf.dtype)
        rs_gen = random_sample.RandomSampleGenerator(vf, x0_lo, x0_up)
        (rand_data, rand_label) = rs_gen.get_random_samples(10)
        va = value_approximation.FiniteHorizonValueFunctionApproximation(
            vf, x0_lo, x0_up, 16, 1)
        val = va.eval(0, rand_data[:, :vf.sys.x_dim])
        self.assertEqual(val.shape[0], rand_data.shape[0])
        self.assertEqual(val.shape[1], 1)

    def test_value_approximation_train(self):
        N = 5
        vf = double_integrator_utils.get_value_function(N=N)
        x0_lo = -1 * torch.ones(vf.sys.x_dim, dtype=vf.dtype)
        x0_up = 1 * torch.ones(vf.sys.x_dim, dtype=vf.dtype)
        rs_gen = random_sample.RandomSampleGenerator(vf, x0_lo, x0_up)
        (rand_data, rand_label) = rs_gen.get_random_samples(10)
        va = value_approximation.FiniteHorizonValueFunctionApproximation(
            vf, x0_lo, x0_up, 16, 1)
        losses0 = va.train_step(rand_data, rand_label)
        for i in range(20):
            losses = va.train_step(rand_data, rand_label)
        for n in range(N-1):
            self.assertLess(losses[n].item(), losses0[n].item())


if __name__ == '__main__':
    unittest.main()
