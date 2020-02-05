import robust_value_approx.random_sample as random_sample
import double_integrator_utils

import unittest
import numpy as np
import torch


class RandomSampleTest(unittest.TestCase):
	def test_random_sample(self):
		N = 5
		vf = double_integrator_utils.get_value_function(N=N)
		x0_lo = -1 * torch.ones(vf.sys.x_dim, dtype=vf.dtype)
		x0_up = 1 * torch.ones(vf.sys.x_dim, dtype=vf.dtype)
		rs_gen = random_sample.RandomSampleGenerator(vf, x0_lo, x0_up)
		(rand_data, rand_label) = rs_gen.get_random_samples(10)
		for k in range(rand_data.shape[0]):
			for n in range(N-1):
				x0 = rand_data[k, vf.sys.x_dim*n:vf.sys.x_dim*(n+1)]
				vf_ = double_integrator_utils.get_value_function(N=N-n)
				V_ = vf_.get_value_function()
				v_ = V_(x0)[0]
				self.assertAlmostEqual(rand_label[k, n].item(), v_, places=5)


if __name__ == '__main__':
    unittest.main()