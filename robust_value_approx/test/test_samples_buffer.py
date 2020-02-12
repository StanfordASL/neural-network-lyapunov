import robust_value_approx.samples_buffer as samples_buffer

import unittest
import torch


class SamplesBufferTest(unittest.TestCase):
    def test_random_sample(self):
        dtype = torch.float64
        buff = samples_buffer.SamplesBuffer(10, 3, dtype)
        s = torch.rand(100, 10, dtype=dtype)
        v = torch.rand(100, 3, dtype=dtype)
        buff.add_samples(s, v)
        self.assertEqual(buff.num_samples, 100)
        rand_s, rand_v = buff.get_random_samples(20)
        self.assertEqual(rand_s.shape[0], 20)

    def test_max_size(self):
        dtype = torch.float64
        buff = samples_buffer.SamplesBuffer(10, 3, dtype)
        s = torch.rand(100, 10, dtype=dtype)
        v = torch.rand(100, 3, dtype=dtype)
        buff.add_samples(s, v)
        buff.add_samples(s, v)
        self.assertEqual(buff.num_samples, 200)
        buff = samples_buffer.SamplesBuffer(10, 3, dtype, max_size=150)
        s = torch.rand(100, 10, dtype=dtype)
        v = torch.rand(100, 3, dtype=dtype)
        buff.add_samples(s, v)
        buff.add_samples(s, v)
        self.assertEqual(buff.num_samples, 150)


if __name__ == '__main__':
    unittest.main()
