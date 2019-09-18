import ControlLyapunov
import unittest
import numpy as np
import cvxpy as cp


class TestControlLyapunovFixedActivationPath(unittest.TestCase):
    def test_constructor(self):
        # Set the problem data to arbitrary value.
        g = np.array([1, 2]).reshape((2, 1))
        P = np.array([[1, 0], [0, 2]])
        q = np.array([2, 3]).reshape((2, 1))
        A = np.array([[1, 1], [0, 1]])
        B = np.array([2, 3]).reshape((2, 1))
        d = np.array([3, 4]).reshape(2, 1)
        u_vertices = np.array([-1, 1]).reshape((1, 2))
        dut = ControlLyapunov.ControlLyapunovFixedActivationPath(
            g, P, q, A, B, d, u_vertices)
        prob = dut.construct_program()
        prob.solve()
        self.assertEqual(prob.status, cp.OPTIMAL)
        self.assertTrue(np.less_equal(P.dot(dut.x.value), q).all())
        self.assertAlmostEqual(prob.value, g.T.dot(
            dut.x.value) + np.min(g.T.dot(B.dot(u_vertices))) + g.T.dot(d), 5)


if __name__ == '__main__':
    unittest.main()
