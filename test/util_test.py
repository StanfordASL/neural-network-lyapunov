import sys
sys.path.append("..")
import utils
import torch
import unittest


class test_replace_binary_continuous_product(unittest.TestCase):
    def setUp(self):
        pass

    def test(self):
        def test_fun(x_lo, x_up):
            (A_x, A_s, A_alpha, rhs) = utils.replace_binary_continuous_product(
                x_lo, x_up, torch.float64)
            # Now test if the four vertices satisfy the constraints
            points = torch.tensor([[x_up, x_up, 1],
                                   [x_lo, x_lo, 1],
                                   [x_up, 0, 0],
                                   [x_lo, 0, 0],
                                   [x_up, x_up, 0],
                                   [x_lo, x_lo, 0],
                                   [x_up, 0, 1],
                                   [x_lo, 0, 1],
                                   [x_up, x_lo - 0.1, 0],
                                   [x_lo, x_up + 0.1, 1]],
                                  dtype=torch.float64).T
            for i in range(points.shape[1]):
                lhs = A_x * points[0, i] + A_s * \
                    points[1, i] + A_alpha * points[2, i]
                satisfied = (
                    torch.abs(points[2, i] * points[0, i] - points[1, i])
                    < 1E-10)
                self.assertEqual(torch.all(lhs <= rhs + 1E-12), satisfied)

        test_fun(0, 1)
        test_fun(1, 2)
        test_fun(-1, 0)
        test_fun(-2, -1)
        test_fun(-2, 1)


if __name__ == "__main__":
    unittest.main()
