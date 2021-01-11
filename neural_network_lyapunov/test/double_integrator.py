import torch
import numpy as np
import unittest


def double_integrator_dynamics(dtype=torch.float64):
    """
    The double integrator p̈ = u can be written as a linear system
    ẋ = Ax + Bu
    where x = [p, ṗ]. A = [0 1; 0 0], B = [0; 1]
    @return (A, B) A is a 2 x 2 matrix, B is a 2 x 1 vector.
    """
    return (torch.tensor([[0, 1], [0, 0]]).type(dtype=dtype),
            torch.tensor([[0], [1]]).type(dtype=dtype))


def double_integrator_lqr(Q, R):
    """
    Compute the LQR controller for double integrator. The LQR controller is
    u = -R⁻¹BᵀPx
    where P is the solution to the Riccati equation
    AᵀP+PA-PBR⁻¹BᵀP + Q = 0
    @param Q a positive definite symmetrix 2 x 2 matrix.
    @param R a positive scalar.
    @return (K, P) K is the control gain, K = R⁻¹BᵀP, P is the cost-to-go
    matrix.
    """
    assert (Q.dtype == R.dtype)
    assert (Q.shape[0] == 2)
    assert (Q.shape[1] == 2)
    assert (R > 0)
    assert (Q[0][1] == Q[1][0])
    """
    The LQR cost-to-go matrix P for the double integrator system satisfies the
    following equation
    R * [Q[0][0]           Q[0][1] + P[0][0]] = [P[0][1]², P[0][1]P[1][1]]
        [Q[0][1] + P[0][0] 2P[0][1]+ Q[1][1]]   [P[0][1]P[1][1], P[1][1]²]
    """
    P = torch.empty(2, 2, dtype=Q.dtype)
    P[0][1] = torch.sqrt(R * Q[0][0])
    P[1][1] = torch.sqrt(R * (2 * P[0][1] + Q[1][1]))
    P[0][0] = (P[0][1] * P[1][1] - R * Q[0][1]) / R
    if (P[0][0] < 0 or P[0][0] * P[1][1] < P[0][1] * P[0][1]):
        # P is not positive definite
        P[0][1] = -P[0][1]
        P[1][1] = torch.sqrt(R * (2 * P[0][1] + Q[1][1]))
        P[0][0] = (P[0][1] * P[1][1] - R * Q[0][1]) / R
    P[1][0] = P[0][1]
    K = (torch.tensor([[0, 1]], dtype=Q.dtype) @ P) / R
    return (K, P)


class TestDoubleIntegratorLQR(unittest.TestCase):
    def test_lqr(self):
        dtype = torch.float64
        (A, B) = double_integrator_dynamics()

        def test_lqr_util(Q, R):
            (K, P) = double_integrator_lqr(Q, R)
            # Check the Riccati equation.
            riccati_lhs = A.T @ P + P @ A - (P @ B @ B.T @ P) / R + Q
            self.assertTrue(torch.all(torch.abs(riccati_lhs) < 1E-10))
            # Check if P is positive definite.
            self.assertGreater(P[0][0], 0)
            self.assertGreater(P[1][1], 0)
            self.assertGreater(np.linalg.det(P.data.numpy()), 0)

        test_lqr_util(torch.tensor([[1, 0], [0, 10]], dtype=dtype),
                      torch.tensor(1, dtype=dtype))
        test_lqr_util(torch.tensor([[2, 0], [0, 10]], dtype=dtype),
                      torch.tensor(3, dtype=dtype))
        test_lqr_util(torch.tensor([[2, 1], [1, 10]], dtype=dtype),
                      torch.tensor(3, dtype=dtype))
        test_lqr_util(torch.tensor([[4, -1], [-1, 10]], dtype=dtype),
                      torch.tensor(3, dtype=dtype))


if __name__ == "__main__":
    unittest.main()
