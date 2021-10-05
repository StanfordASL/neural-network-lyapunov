import neural_network_lyapunov.train_lyapunov as train_lyapunov
import neural_network_lyapunov.control_lyapunov as control_lyapunov
import neural_network_lyapunov.control_affine_system as control_affine_system
import neural_network_lyapunov.r_options as r_options
import neural_network_lyapunov.utils as utils

import torch
import unittest
import gurobipy
import copy


class TestTrainControlLyapunov(unittest.TestCase):
    def setUp(self):
        self.dtype = torch.float64

        # Create a linear system.
        A = torch.tensor([[1, 2], [-2, 3]], dtype=self.dtype)
        B = torch.tensor([[1, 3, 1], [0, 1, 0]], dtype=self.dtype)
        self.linear_system = control_affine_system.LinearSystem(
            A,
            B,
            x_lo=torch.tensor([-2, -3], dtype=self.dtype),
            x_up=torch.tensor([3, 3], dtype=self.dtype),
            u_lo=torch.tensor([-1, -2, -3], dtype=self.dtype),
            u_up=torch.tensor([2, 3, 4], dtype=self.dtype))
        self.lyapunov_relu1 = utils.setup_relu((2, 4, 3, 1),
                                               params=None,
                                               negative_slope=0.1,
                                               bias=True,
                                               dtype=self.dtype)
        self.lyapunov_relu1[0].weight.data = torch.tensor(
            [[1, -1], [0, 2], [-1, 2], [-2, 1]], dtype=self.dtype)
        self.lyapunov_relu1[0].bias.data = torch.tensor([1, -1, 0, 2],
                                                        dtype=self.dtype)
        self.lyapunov_relu1[2].weight.data = torch.tensor(
            [[3, -2, 1, 0], [1, -1, 2, 3], [-2, -1, 0, 3]], dtype=self.dtype)
        self.lyapunov_relu1[2].bias.data = torch.tensor([1, -2, 3],
                                                        dtype=self.dtype)
        self.lyapunov_relu1[4].weight.data = torch.tensor([[1, 3, -2]],
                                                          dtype=self.dtype)
        self.lyapunov_relu1[4].bias.data = torch.tensor([2], dtype=self.dtype)

    def test_total_loss1(self):
        clf = control_lyapunov.ControlLyapunov(
            self.linear_system, self.lyapunov_relu1,
            control_lyapunov.SubgradientPolicy(None))
        V_lambda = 0.5
        x_equilibrium = torch.tensor([0.2, 0.5], dtype=self.dtype)
        R = torch.tensor([[1, 0.5], [-0.5, 2], [0, 1]], dtype=self.dtype)
        R_options = r_options.FixedROptions(R)
        dut = train_lyapunov.TrainLyapunovReLU(clf, V_lambda, x_equilibrium,
                                               R_options)

        positivity_state_samples = torch.tensor([[-2, 1], [2, 1]],
                                                dtype=self.dtype)
        derivative_state_samples = torch.tensor([[0, 2], [1, 1]],
                                                dtype=self.dtype)
        positivity_sample_cost_weight = 5.
        derivative_sample_cost_weight = 3.
        positivity_mip_cost_weight = 2.
        derivative_mip_cost_weight = 4.
        dut.lyapunov_derivative_mip_clamp_min = 0.
        loss, _, _, _, _, _, _, _, _, _ = dut.total_loss(
            positivity_state_samples, derivative_state_samples, None,
            positivity_sample_cost_weight, derivative_sample_cost_weight,
            positivity_mip_cost_weight, derivative_mip_cost_weight)
        loss_expected = 0
        positivity_milp, positivity_x = clf.lyapunov_positivity_as_milp(
            x_equilibrium, V_lambda, dut.lyapunov_positivity_epsilon, R=R)
        positivity_milp.gurobi_model.setParam(gurobipy.GRB.Param.OutputFlag,
                                              False)
        positivity_milp.gurobi_model.optimize()
        self.assertEqual(positivity_milp.gurobi_model.status,
                         gurobipy.GRB.Status.OPTIMAL)
        # Expect the positivity constraint is violated.
        assert (positivity_milp.gurobi_model.ObjVal > 0)
        loss_expected += positivity_mip_cost_weight *\
            positivity_milp.gurobi_model.ObjVal
        deriv_mip_return = clf.lyapunov_derivative_as_milp(
            x_equilibrium,
            V_lambda,
            dut.lyapunov_derivative_epsilon,
            dut.lyapunov_derivative_eps_type,
            R=R)
        deriv_mip_return.milp.gurobi_model.setParam(
            gurobipy.GRB.Param.OutputFlag, False)
        deriv_mip_return.milp.gurobi_model.optimize()
        self.assertEqual(deriv_mip_return.milp.gurobi_model.status,
                         gurobipy.GRB.Status.OPTIMAL)
        # Expect the derivative constraint is violated.
        assert (deriv_mip_return.milp.gurobi_model.ObjVal > 0)
        loss_expected += derivative_mip_cost_weight *\
            deriv_mip_return.milp.gurobi_model.ObjVal

        loss_expected += positivity_sample_cost_weight *\
            clf.lyapunov_positivity_loss_at_samples(
                x_equilibrium,
                positivity_state_samples,
                V_lambda,
                dut.lyapunov_positivity_epsilon,
                R=R)

        loss_expected += derivative_sample_cost_weight *\
            clf.lyapunov_derivative_loss_at_samples_and_next_states(
                V_lambda,
                dut.lyapunov_derivative_epsilon,
                derivative_state_samples,
                None,
                x_equilibrium,
                dut.lyapunov_derivative_eps_type,
                R=R)
        self.assertAlmostEqual(loss_expected.item(), loss.item())

    def test_total_loss2(self):
        # Test with the Lyapunov derivative constraint max V̇(x)+εV(x) is
        # strictly less than zero.
        # Notice that for a Lyapunov condition, it is impossible to have
        # max V̇(x)+εV(x) to be strictly less than 0, since at equilibrium x*,
        # we always have V̇(x*)+εV(x*) = 0. But for a control Lyapunov, it is
        # possible to have max V̇(x)+εV(x) < 0 since we considers minimization
        # over u when computing V̇(x)
        lyapunov_relu = copy.deepcopy(self.lyapunov_relu1)
        lyapunov_relu[0].weight.data = torch.tensor(
            [[1.3, -0.75], [-0.45, 1.5], [-1.3, 1.8], [-1.9, 1.2]],
            dtype=self.dtype)
        lyapunov_relu[0].bias.data = torch.tensor([0.6, -1.2, -0.65, 2.33],
                                                  dtype=self.dtype)
        lyapunov_relu[2].weight.data = torch.tensor(
            [[3.2, -1.7, 1., -0.2], [1.25, -0.5, 2.45, 3.2],
             [-1.65, -1., 0.05, 2.75]],
            dtype=self.dtype)
        lyapunov_relu[2].bias.data = torch.tensor([0.63, -1.9, 2.8],
                                                  dtype=self.dtype)
        lyapunov_relu[4].weight.data = torch.tensor([[0.9, 3.1, -1.8]],
                                                    dtype=self.dtype)
        lyapunov_relu[4].bias.data = torch.tensor([2], dtype=self.dtype)
        clf = control_lyapunov.ControlLyapunov(
            self.linear_system, lyapunov_relu,
            control_lyapunov.SubgradientPolicy(None))
        x_equilibrium = torch.tensor([0.2, 0.5], dtype=self.dtype)
        R = torch.tensor([[1, 0.5], [-0.5, 2], [0, 1]], dtype=self.dtype)
        R_options = r_options.FixedROptions(R)
        V_lambda = 0.5
        dut = train_lyapunov.TrainLyapunovReLU(clf, V_lambda, x_equilibrium,
                                               R_options)
        derivative_milp_return = clf.lyapunov_derivative_as_milp(
            x_equilibrium,
            V_lambda,
            dut.lyapunov_derivative_epsilon,
            dut.lyapunov_derivative_eps_type,
            R=R)
        derivative_milp_return.milp.gurobi_model.setParam(
            gurobipy.GRB.Param.OutputFlag, False)
        derivative_milp_return.milp.gurobi_model.optimize()
        assert (derivative_milp_return.milp.gurobi_model.status ==
                gurobipy.GRB.Status.OPTIMAL)
        assert (derivative_milp_return.milp.gurobi_model.ObjVal < 0)

        dut.lyapunov_derivative_mip_clamp_min = 0.

        loss, _, derivative_mip_obj, _, _, positivity_mip_loss,\
            derivative_mip_loss, _, _, _ = dut.total_loss(
                torch.empty((0, 2), dtype=self.dtype),
                torch.empty((0, 2), dtype=self.dtype),
                None, None, None, 1., 1.)
        self.assertLess(derivative_mip_obj, 0)
        self.assertEqual(derivative_mip_loss.item(), 0)
        self.assertEqual(loss.item(), positivity_mip_loss.item())

    def test_train(self):
        clf = control_lyapunov.ControlLyapunov(
            self.linear_system, self.lyapunov_relu1,
            control_lyapunov.SubgradientPolicy(None))
        V_lambda = 0.5
        x_equilibrium = torch.tensor([0.2, 0.5], dtype=self.dtype)
        R = torch.tensor([[1, 0.5], [-0.5, 2], [0, 1]], dtype=self.dtype)
        R_options = r_options.FixedROptions(R)
        dut = train_lyapunov.TrainLyapunovReLU(clf, V_lambda, x_equilibrium,
                                               R_options)

        state_samples_all = torch.empty((0, 2), dtype=self.dtype)
        dut.lyapunov_derivative_mip_clamp_min = 0.
        dut.lyapunov_positivity_convergence_tol = 1E-4
        success, loss, positivity_mip_cost, derivative_mip_cost = dut.train(
            state_samples_all)
        self.assertTrue(success)
        self.assertLessEqual(positivity_mip_cost,
                             dut.lyapunov_positivity_convergence_tol)
        self.assertLessEqual(derivative_mip_cost, 0)


if __name__ == "__main__":
    unittest.main()
