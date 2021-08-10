import neural_network_lyapunov.control_affine_system as mut
import neural_network_lyapunov.gurobi_torch_mip as gurobi_torch_mip
import torch
import numpy as np
import unittest
import gurobipy


class TestLinearSystem(unittest.TestCase):
    def test(self):
        dtype = torch.float64
        A = torch.tensor([[2., 1.], [-0.5, 1.]], dtype=dtype)
        B = torch.tensor([[0.5, 0.2, 0.3], [0.1, 0.2, -1.5]], dtype=dtype)
        x_lo = torch.tensor([-2., -3], dtype=dtype)
        x_up = torch.tensor([1., 4], dtype=dtype)
        u_lo = torch.tensor([-2, -3, -5], dtype=dtype)
        u_up = torch.tensor([3, 4, 1], dtype=dtype)

        dut = mut.LinearSystem(A, B, x_lo, x_up, u_lo, u_up)
        self.assertEqual(dut.x_dim, 2)
        self.assertEqual(dut.u_dim, 3)

        mip_cnstr_f, mip_cnstr_G = dut.mixed_integer_constraints()

        prog = gurobi_torch_mip.GurobiTorchMIP(dtype)

        x = prog.addVars(dut.x_dim, lb=-gurobipy.GRB.INFINITY)
        f = prog.addVars(dut.x_dim, lb=-gurobipy.GRB.INFINITY)
        G_flat = prog.addVars(dut.x_dim * dut.u_dim, lb=-gurobipy.GRB.INFINITY)
        prog.add_mixed_integer_linear_constraints(mip_cnstr_f, x, f, "", "",
                                                  "", "", "")
        for i in range(dut.u_dim):
            prog.add_mixed_integer_linear_constraints(
                mip_cnstr_G[i], x,
                [G_flat[j * dut.u_dim + i]
                 for j in range(dut.x_dim)], "", "", "", "", "")

        x_val = np.array([1., 2.])
        for i in range(dut.x_dim):
            x[i].lb = x_val[i]
            x[i].ub = x_val[i]
        prog.gurobi_model.setParam(gurobipy.GRB.Param.OutputFlag, False)
        prog.gurobi_model.optimize()
        self.assertEqual(prog.gurobi_model.status, gurobipy.GRB.OPTIMAL)

        f_expected = A.detach().numpy() @ x_val
        np.testing.assert_allclose(
            np.array([f[i].x for i in range(dut.x_dim)]), f_expected)
        G_expected = B.detach().numpy()
        np.testing.assert_allclose(
            np.array([Gi.x for Gi in G_flat]).reshape((dut.x_dim, dut.u_dim)),
            G_expected)


if __name__ == "__main__":
    unittest.main()
