import neural_network_lyapunov.control_affine_system as mut
import neural_network_lyapunov.gurobi_torch_mip as gurobi_torch_mip
import neural_network_lyapunov.utils as utils
import neural_network_lyapunov.mip_utils as mip_utils
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

        ret = dut.mixed_integer_constraints()
        f_lo_expected, f_up_expected = mip_utils.compute_range_by_IA(
            A, torch.zeros(dut.x_dim, dtype=dtype), x_lo, x_up)
        np.testing.assert_allclose(ret.f_lo.detach().numpy(),
                                   f_lo_expected.detach().numpy())
        np.testing.assert_allclose(ret.f_up.detach().numpy(),
                                   f_up_expected.detach().numpy())
        np.testing.assert_allclose(ret.G_flat_lo.detach().numpy(),
                                   B.reshape((-1, )).detach().numpy())
        np.testing.assert_allclose(ret.G_flat_up.detach().numpy(),
                                   B.reshape((-1, )).detach().numpy())

        prog = gurobi_torch_mip.GurobiTorchMIP(dtype)

        x = prog.addVars(dut.x_dim, lb=-gurobipy.GRB.INFINITY)
        f = prog.addVars(dut.x_dim, lb=-gurobipy.GRB.INFINITY)
        G_flat = prog.addVars(dut.x_dim * dut.u_dim, lb=-gurobipy.GRB.INFINITY)
        prog.add_mixed_integer_linear_constraints(ret.mip_cnstr_f, x, f, "",
                                                  "", "", "", "")
        for i in range(dut.u_dim):
            prog.add_mixed_integer_linear_constraints(ret.mip_cnstr_G, x,
                                                      G_flat, "", "", "", "",
                                                      "")

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


    def test_is_x_stabilizable(self):
        dtype = torch.float64
        A = torch.tensor([[1, 2], [-1, 2]], dtype=dtype)
        B = torch.tensor([[1, 0, 0], [0, 1, 0]], dtype=dtype)
        x_lo = torch.tensor([-2, -3], dtype=dtype)
        x_up = -x_lo
        u_lo = torch.tensor([1, 1, 0], dtype=dtype)
        u_up = torch.tensor([2, 3, 4], dtype=dtype)
        dut = mut.LinearSystem(A, B, x_lo, x_up, u_lo, u_up)
        self.assertFalse(
            dut.can_be_equilibrium_state(torch.tensor([1, 1], dtype=dtype)))
        self.assertTrue(
            dut.can_be_equilibrium_state(torch.tensor([0, -0.75],
                                                      dtype=dtype)))


class TestSecondOrderControlAffineSystem(unittest.TestCase):
    def setUp(self):
        self.dtype = torch.float64
        self.phi_a = utils.setup_relu((2, 4, 3, 1),
                                      params=None,
                                      negative_slope=0.1,
                                      bias=True,
                                      dtype=self.dtype)
        self.phi_a[0].weight.data = torch.tensor(
            [[0.5, 1], [-1, 1], [0, 3], [1, -2]], dtype=self.dtype)
        self.phi_a[0].bias.data = torch.tensor([1, -1, 2, -1],
                                               dtype=self.dtype)
        self.phi_a[2].weight.data = torch.tensor(
            [[1., 2., -1, 3], [0, -1, -1.5, 2], [0.5, 1, -1, 2]],
            dtype=self.dtype)
        self.phi_a[2].bias.data = torch.tensor([1, 2, -1], dtype=self.dtype)
        self.phi_a[4].weight.data = torch.tensor([[2, -1, 3]],
                                                 dtype=self.dtype)
        self.phi_a[4].bias.data = torch.tensor([1], dtype=self.dtype)

        self.phi_b = utils.setup_relu((2, 4, 3),
                                      params=None,
                                      negative_slope=0.1,
                                      bias=True,
                                      dtype=self.dtype)
        self.phi_b[0].weight.data = torch.tensor(
            [[0.5, -0.1], [0.5, 1], [1, -2], [1, -1]], dtype=self.dtype)
        self.phi_b[0].bias.data = torch.tensor([1, -1, 0, 2], dtype=self.dtype)
        self.phi_b[2].weight.data = torch.tensor(
            [[1, 0, 2, -1], [0.5, -1, -2, 1], [0., 1, -1, 2]],
            dtype=self.dtype)
        self.phi_b[2].bias.data = torch.tensor([1, -1, 2], dtype=self.dtype)

    def test_dynamics(self):
        dut = mut.ReluSecondOrderControlAffineSystem(
            x_lo=torch.tensor([-2, -1], dtype=self.dtype),
            x_up=torch.tensor([1, 3], dtype=self.dtype),
            u_lo=torch.tensor([-2, -1, -3], dtype=self.dtype),
            u_up=torch.tensor([1, 2, 1], dtype=self.dtype),
            phi_a=self.phi_a,
            phi_b=self.phi_b,
            method=mip_utils.PropagateBoundsMethod.IA)
        self.assertEqual(dut.nq, 1)

        torch.manual_seed(0)
        x_samples = utils.uniform_sample_in_box(dut.x_lo, dut.x_up, 100)
        u_samples = utils.uniform_sample_in_box(dut.u_lo, dut.u_up,
                                                x_samples.shape[0])
        for i in range(x_samples.shape[0]):
            phi_a_val = self.phi_a(x_samples[i])
            phi_b_val = self.phi_b(x_samples[i])
            vdot = phi_a_val + phi_b_val.reshape(
                (dut.nq, dut.u_dim)) @ u_samples[i]
            xdot = torch.cat((x_samples[i, dut.nq:], vdot))
            np.testing.assert_allclose(
                dut.dynamics(x_samples[i], u_samples[i]).detach().numpy(),
                xdot.detach().numpy())

    def test_mixed_integer_constraints_v(self):
        for method in list(mip_utils.PropagateBoundsMethod):
            dut = mut.ReluSecondOrderControlAffineSystem(
                x_lo=torch.tensor([-2, -1], dtype=self.dtype),
                x_up=torch.tensor([1, 3], dtype=self.dtype),
                u_lo=torch.tensor([-2, -1, -3], dtype=self.dtype),
                u_up=torch.tensor([1, 2, 1], dtype=self.dtype),
                phi_a=self.phi_a,
                phi_b=self.phi_b,
                method=method)
            torch.manual_seed(0)
            x_samples = utils.uniform_sample_in_box(dut.x_lo, dut.x_up, 50)
            milp = gurobi_torch_mip.GurobiTorchMIP(self.dtype)
            x = milp.addVars(dut.x_dim, lb=-gurobipy.GRB.INFINITY)
            a = milp.addVars(dut.nq, lb=-gurobipy.GRB.INFINITY)
            b_flat = milp.addVars(dut.nq * dut.u_dim,
                                  lb=-gurobipy.GRB.INFINITY)
            mip_cnstr_a, mip_cnstr_b_flat, a_lo, a_up, b_lo, b_up = \
                dut._mixed_integer_constraints_v()
            a_slack, a_binary = milp.add_mixed_integer_linear_constraints(
                mip_cnstr_a,
                x,
                a,
                "a_slack",
                "a_binary",
                "",
                "",
                "",
                binary_var_type=gurobipy.GRB.BINARY)
            b_slack, b_binary = milp.add_mixed_integer_linear_constraints(
                mip_cnstr_b_flat,
                x,
                b_flat,
                "b_slack",
                "b_binary",
                "",
                "",
                "",
                binary_var_type=gurobipy.GRB.BINARY)
            milp.gurobi_model.setParam(gurobipy.GRB.Param.OutputFlag, False)
            for i in range(x_samples.shape[0]):
                for j in range(dut.x_dim):
                    x[j].lb = x_samples[i][j]
                    x[j].ub = x_samples[i][j]
                milp.gurobi_model.optimize()
                self.assertEqual(milp.gurobi_model.status,
                                 gurobipy.GRB.Status.OPTIMAL)
                a_val = np.array([v.x for v in a])
                b_val = np.array([v.x for v in b_flat]).reshape(
                    (dut.nq, dut.u_dim))
                a_expected = dut.a(x_samples[i])
                b_expected = dut.b(x_samples[i])
                np.testing.assert_allclose(a_val, a_expected.detach().numpy())
                np.testing.assert_allclose(b_val, b_expected.detach().numpy())

    def test_mixed_integer_constraints(self):
        for method in list(mip_utils.PropagateBoundsMethod):
            dut = mut.ReluSecondOrderControlAffineSystem(
                x_lo=torch.tensor([-2, -1], dtype=self.dtype),
                x_up=torch.tensor([1, 3], dtype=self.dtype),
                u_lo=torch.tensor([-2, -1, -3], dtype=self.dtype),
                u_up=torch.tensor([1, 2, 1], dtype=self.dtype),
                phi_a=self.phi_a,
                phi_b=self.phi_b,
                method=method)
            torch.manual_seed(0)
            x_samples = utils.uniform_sample_in_box(dut.x_lo, dut.x_up, 50)
            milp = gurobi_torch_mip.GurobiTorchMIP(self.dtype)
            x = milp.addVars(dut.x_dim, lb=-gurobipy.GRB.INFINITY)
            f = milp.addVars(dut.x_dim, lb=-gurobipy.GRB.INFINITY)
            G_flat = milp.addVars(dut.x_dim * dut.u_dim,
                                  lb=-gurobipy.GRB.INFINITY)
            ret = dut.mixed_integer_constraints()
            milp.add_mixed_integer_linear_constraints(
                ret.mip_cnstr_f,
                x,
                f,
                "f_slack",
                "f_binary",
                "",
                "",
                "",
                binary_var_type=gurobipy.GRB.BINARY)
            milp.add_mixed_integer_linear_constraints(
                ret.mip_cnstr_G,
                x,
                G_flat,
                "G_slack",
                "G_binary",
                "",
                "",
                "",
                binary_var_type=gurobipy.GRB.BINARY)
            milp.gurobi_model.setParam(gurobipy.GRB.Param.OutputFlag, False)
            for i in range(x_samples.shape[0]):
                for j in range(dut.x_dim):
                    x[j].lb = x_samples[i][j]
                    x[j].ub = x_samples[i][j]
                milp.gurobi_model.optimize()
                self.assertEqual(milp.gurobi_model.status,
                                 gurobipy.GRB.Status.OPTIMAL)
                f_val = np.array([v.x for v in f])
                G_val = np.array([v.x for v in G_flat]).reshape(
                    (dut.x_dim, dut.u_dim))
                f_expected = dut.f(x_samples[i])
                G_expected = dut.G(x_samples[i])
                np.testing.assert_allclose(f_val, f_expected.detach().numpy())
                np.testing.assert_allclose(G_val, G_expected.detach().numpy())

    def test_compute_range(self):
        for method in list(mip_utils.PropagateBoundsMethod):
            dut = mut.ReluSecondOrderControlAffineSystem(
                x_lo=torch.tensor([-2, -1], dtype=self.dtype),
                x_up=torch.tensor([1, 3], dtype=self.dtype),
                u_lo=torch.tensor([-2, -1, -3], dtype=self.dtype),
                u_up=torch.tensor([1, 2, 1], dtype=self.dtype),
                phi_a=self.phi_a,
                phi_b=self.phi_b,
                method=method)
            mip_cnstr_a = dut.relu_free_pattern_a.output_constraint(
                dut.x_lo, dut.x_up, method)
            mip_cnstr_b_flat = dut.relu_free_pattern_b.output_constraint(
                dut.x_lo, dut.x_up, method)
            ret = dut.mixed_integer_constraints()
            np.testing.assert_allclose(ret.f_lo[:dut.nq].detach().numpy(),
                                       dut.x_lo[dut.nq:].detach().numpy())
            np.testing.assert_allclose(ret.f_up[:dut.nq].detach().numpy(),
                                       dut.x_up[dut.nq:].detach().numpy())
            np.testing.assert_allclose(
                ret.f_lo[dut.nq:].detach().numpy(),
                mip_cnstr_a.nn_output_lo.detach().numpy())
            np.testing.assert_allclose(
                ret.f_up[dut.nq:].detach().numpy(),
                mip_cnstr_a.nn_output_up.detach().numpy())
            np.testing.assert_allclose(
                ret.G_flat_lo[:dut.nq * dut.u_dim].detach().numpy(), 0)
            np.testing.assert_allclose(
                ret.G_flat_up[:dut.nq * dut.u_dim].detach().numpy(), 0)
            np.testing.assert_allclose(
                ret.G_flat_lo[dut.nq * dut.u_dim:].detach().numpy(),
                mip_cnstr_b_flat.nn_output_lo.detach().numpy())
            np.testing.assert_allclose(
                ret.G_flat_up[dut.nq * dut.u_dim:].detach().numpy(),
                mip_cnstr_b_flat.nn_output_up.detach().numpy())


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

        forward_model_f = utils.setup_relu(
            (dut.x_dim, 4 * dut.x_dim, dut.x_dim),
            params=None,
            bias=True,
            negative_slope=0.01,
            dtype=dtype)

        forward_model_G = utils.setup_relu(
            (dut.x_dim, 4 * dut.x_dim * dut.u_dim, dut.x_dim * dut.u_dim),
            params=None,
            bias=True,
            negative_slope=0.01,
            dtype=dtype)

        mut.train_control_affine_forward_model(forward_model_f,
                                               forward_model_G,
                                               x_equ,
                                               u_equ,
                                               dataset,
                                               200,
                                               1e-2,
                                               batch_size=100,
                                               verbose=False)

        for i in range(10):
            x_test = torch.rand(dut.x_dim, dtype=dtype) * (x_up - x_lo) + x_lo
            u_test = torch.rand(dut.u_dim, dtype=dtype) * (u_up - u_lo) + u_lo
            x_dot_exp = dut.dynamics(x_test, u_test)
            x_dot_pred = forward_model_f(x_test) +\
                forward_model_G(x_test).view(
                    (dut.x_dim, dut.u_dim)) @ u_test -\
                forward_model_f(x_equ) -\
                forward_model_G(x_equ).view((dut.x_dim, dut.u_dim)) @ u_equ

            np.testing.assert_allclose(x_dot_pred.detach().numpy(),
                                       x_dot_exp.detach().numpy(),
                                       rtol=.1)


if __name__ == "__main__":
    unittest.main()
