import torch
import numpy as np
import gurobipy
import neural_network_lyapunov.compute_xhat as compute_xhat
import neural_network_lyapunov.gurobi_torch_mip as gurobi_torch_mip
import neural_network_lyapunov.relu_to_optimization as relu_to_optimization
import neural_network_lyapunov.utils as utils
import neural_network_lyapunov.mip_utils as mip_utils
import unittest


class TestGetXbar(unittest.TestCase):
    def setUp(self):
        self.dtype = torch.float64

    def test_get_xbar_indices(self):
        # test _get_xbar_indices(), _get_xhat_indices(), _get_xhat_value()

        self.assertEqual(compute_xhat._get_xbar_indices(3, None), [0, 1, 2])
        self.assertEqual(compute_xhat._get_xbar_indices(3, [0, 1]), [0, 1])

    def test_get_xhat_indices(self):
        self.assertEqual(compute_xhat._get_xhat_indices(3, None),
                         ([0, 1, 2], []))
        self.assertEqual(compute_xhat._get_xhat_indices(3, [0, 1]),
                         ([0, 1], [2]))
        self.assertEqual(compute_xhat._get_xhat_indices(3, [1]), ([1], [0, 2]))

    def test_get_xhat_val(self):
        np.testing.assert_allclose(
            compute_xhat._get_xhat_val(
                torch.tensor([1., 2., 3.], dtype=self.dtype),
                torch.tensor([-1, -2, -3], dtype=self.dtype),
                None).detach().numpy(), np.array([-1, -2., -3.]))
        np.testing.assert_allclose(
            compute_xhat._get_xhat_val(
                torch.tensor([1., 2., 3.], dtype=self.dtype),
                torch.tensor([-1, -2, -3], dtype=self.dtype),
                [1]).detach().numpy(), np.array([1, -2., 3.]))
        np.testing.assert_allclose(
            compute_xhat._get_xhat_val(
                torch.tensor([[1., 2., 3.], [4, 5, 6]], dtype=self.dtype),
                torch.tensor([-1, -2, -3], dtype=self.dtype),
                None).detach().numpy(), np.array([-1, -2., -3.]))
        np.testing.assert_allclose(
            compute_xhat._get_xhat_val(
                torch.tensor([[1., 2., 3.], [4, 5, 6]], dtype=self.dtype),
                torch.tensor([-1, -2, -3], dtype=self.dtype),
                [1]).detach().numpy(), np.array([[1, -2., 3.], [4, -2, 6]]))
        np.testing.assert_allclose(
            compute_xhat._get_xhat_val(
                torch.tensor([[1., 2., 3.], [4, 5, 6]], dtype=self.dtype),
                torch.tensor([-1, -2, -3],
                             dtype=self.dtype), [1, 2]).detach().numpy(),
            np.array([[1, -2., -3.], [4, -2, -3]]))


class TestComputeNetworkAtXhat(unittest.TestCase):
    def test(self):
        dtype = torch.float64

        def tester(xhat_indices):
            mip = gurobi_torch_mip.GurobiTorchMIP(dtype)
            x_var = mip.addVars(3, lb=-gurobipy.GRB.INFINITY)
            torch.manual_seed(0)
            relu = utils.setup_relu((3, 5, 4, 2),
                                    params=None,
                                    negative_slope=0.1,
                                    bias=True,
                                    dtype=dtype)
            relu_free_pattern = relu_to_optimization.ReLUFreePattern(
                relu, dtype)
            x_lb = torch.tensor([-2, -3, -1], dtype=dtype)
            x_ub = torch.tensor([1, 2, 0], dtype=dtype)
            x_equilibrium = torch.tensor([0.5, -1.5, -0.2], dtype=dtype)
            relu_z, relu_beta, Aout, Cout, xhat, output_lo,\
                output_up = compute_xhat._compute_network_at_xhat(
                    mip, x_var, x_equilibrium, relu_free_pattern, xhat_indices,
                    x_lb, x_ub, mip_utils.PropagateBoundsMethod.IA,
                    lp_relaxation=False)

            # Now fix x_var to x_val, and check the solution ϕ(x̂)
            x_val = torch.tensor([-1.2, 0.5, -0.8], dtype=dtype)
            mip.addMConstrs([torch.eye(3, dtype=dtype)], [x_var],
                            b=x_val,
                            sense=gurobipy.GRB.EQUAL)
            mip.gurobi_model.setParam(gurobipy.GRB.Param.OutputFlag, False)
            mip.gurobi_model.optimize()
            self.assertEqual(mip.gurobi_model.status,
                             gurobipy.GRB.Status.OPTIMAL)
            xhat_val = compute_xhat._get_xhat_val(x_val, x_equilibrium,
                                                  xhat_indices)
            np.testing.assert_allclose(np.array([var.x for var in xhat]),
                                       xhat_val)
            relu_z_sol = torch.tensor([var.x for var in relu_z], dtype=dtype)
            phi_xhat = Aout @ relu_z_sol + Cout
            np.testing.assert_array_less(phi_xhat.detach().numpy(),
                                         output_up.detach().numpy() + 1e-6)
            np.testing.assert_array_less(output_lo.detach().numpy() - 1E-6,
                                         phi_xhat.detach().numpy())
            phi_xhat_expected = relu(xhat_val)
            np.testing.assert_allclose(phi_xhat.detach().numpy(),
                                       phi_xhat_expected.detach().numpy())
            # Make sure x_lb and x_ub are unchanged
            np.testing.assert_allclose(x_lb.detach().numpy(),
                                       np.array([-2., -3., -1.]))
            np.testing.assert_allclose(x_ub.detach().numpy(),
                                       np.array([1., 2., 0.]))

        tester(xhat_indices=[1, 2])
        tester(xhat_indices=[0, 2])
        tester(xhat_indices=[0])

    def test_with_lp_relaxation(self):
        dtype = torch.float64
        mip = gurobi_torch_mip.GurobiTorchMIP(dtype)
        x_var = mip.addVars(3, lb=-gurobipy.GRB.INFINITY)
        torch.manual_seed(0)
        relu = utils.setup_relu((3, 5, 4, 2),
                                params=None,
                                negative_slope=0.1,
                                bias=True,
                                dtype=dtype)
        relu_free_pattern = relu_to_optimization.ReLUFreePattern(relu, dtype)
        x_lb = torch.tensor([-2, -3, -1], dtype=dtype)
        x_ub = torch.tensor([1, 2, 0], dtype=dtype)
        x_equilibrium = torch.tensor([0.5, -1.5, -0.2], dtype=dtype)
        xhat_indices = [0, 1]
        relu_z, relu_beta, Aout, Cout, xhat, output_lo,\
            output_up = compute_xhat._compute_network_at_xhat(
                mip, x_var, x_equilibrium, relu_free_pattern, xhat_indices,
                x_lb, x_ub, mip_utils.PropagateBoundsMethod.IA,
                lp_relaxation=True)
        self.assertEqual(len(mip.zeta), 0)


if __name__ == "__main__":
    unittest.main()
