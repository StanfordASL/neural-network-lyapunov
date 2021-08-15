import neural_network_lyapunov.control_affine_system as mut
import neural_network_lyapunov.gurobi_torch_mip as gurobi_torch_mip
import neural_network_lyapunov.utils as utils
import torch
import numpy as np
import random
import unittest
import gurobipy


def get_simple_ca_system_params(dtype=torch.float64):
    A = torch.tensor([[2., 1.], [-0.5, 1.]], dtype=dtype)
    B = torch.tensor([[0.5, 0.2, 0.3], [0.1, 0.2, -1.5]], dtype=dtype)
    x_lo = torch.tensor([-2., -3], dtype=dtype)
    x_up = torch.tensor([1., 4], dtype=dtype)
    u_lo = torch.tensor([-2, -3, -5], dtype=dtype)
    u_up = torch.tensor([3, 4, 1], dtype=dtype)
    return A, B, x_lo, x_up, u_lo, u_up


class TestLinearSystem(unittest.TestCase):
    def test(self):
        dtype = torch.float64
        A, B, x_lo, x_up, u_lo, u_up = get_simple_ca_system_params(dtype)
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


class TestTrainControlAffineSystem(unittest.TestCase):
    def test(self):
        torch.manual_seed(0)
        random.seed(0)
        np.random.seed(0)

        dtype = torch.float64
        A, B, x_lo, x_up, u_lo, u_up = get_simple_ca_system_params(dtype)
        dut = mut.LinearSystem(A, B, x_lo, x_up, u_lo, u_up)

        data = []
        labels = []
        for i in range(2000):
            x = torch.rand(dut.x_dim, dtype=dtype) * (x_up - x_lo) + x_lo
            u = torch.rand(dut.u_dim, dtype=dtype) * (u_up - u_lo) + u_lo
            x_dot = dut.dynamics(x, u)
            data.append(torch.cat([x, u]).unsqueeze(0))
            labels.append(x_dot.unsqueeze(0))
        data = torch.cat(data, dim=0)
        labels = torch.cat(labels, dim=0)
        dataset = torch.utils.data.TensorDataset(data, labels)

        x_equ = torch.zeros(dut.x_dim, dtype=dtype)
        u_equ = torch.zeros(dut.u_dim, dtype=dtype)

        forward_model_f = utils.setup_relu((dut.x_dim,
                                            4 * dut.x_dim,
                                            dut.x_dim),
                                           params=None,
                                           bias=True,
                                           negative_slope=0.01,
                                           dtype=dtype)

        forward_model_G = utils.setup_relu((dut.x_dim,
                                            4 * dut.x_dim * dut.u_dim,
                                            dut.x_dim * dut.u_dim),
                                           params=None,
                                           bias=True,
                                           negative_slope=0.01,
                                           dtype=dtype)

        mut.train_control_affine_forward_model(
            forward_model_f, forward_model_G, x_equ, u_equ,
            dataset, 200, 1e-2, batch_size=100, verbose=False)

        for i in range(10):
            x_test = torch.rand(dut.x_dim, dtype=dtype) * (x_up - x_lo) + x_lo
            u_test = torch.rand(dut.u_dim, dtype=dtype) * (u_up - u_lo) + u_lo
            x_dot_exp = dut.dynamics(x_test, u_test)
            x_dot_pred = forward_model_f(x_test) +\
                forward_model_G(x_test).view(
                    (dut.x_dim, dut.u_dim)) @ u_test -\
                forward_model_f(x_equ) -\
                forward_model_G(x_equ).view((dut.x_dim, dut.u_dim)) @ u_equ

            np.testing.assert_allclose(
                x_dot_pred.detach().numpy(),
                x_dot_exp.detach().numpy(),
                rtol=.1)


if __name__ == "__main__":
    unittest.main()
