import neural_network_lyapunov.lyapunov as lyapunov
import neural_network_lyapunov.train_lyapunov as train_lyapunov
import neural_network_lyapunov.hybrid_linear_system as hybrid_linear_system
import neural_network_lyapunov.test.test_hybrid_linear_system as\
    test_hybrid_linear_system
import neural_network_lyapunov.r_options as r_options
import neural_network_lyapunov.utils as utils
import neural_network_lyapunov.relu_system as relu_system
import torch
import torch.nn as nn
import unittest
import gurobipy
import numpy as np


def setup_lyapunov_relu():
    # Construct a simple ReLU model with 2 hidden layers
    dtype = torch.float64
    linear1 = nn.Linear(2, 3)
    linear1.weight.data = torch.tensor([[-1, 0.5], [-0.3, 0.74], [-2, 1.5]],
                                       dtype=dtype)
    linear1.bias.data = torch.tensor([-0.1, 1.0, 0.5], dtype=dtype)
    linear2 = nn.Linear(3, 4)
    linear2.weight.data = torch.tensor(
        [[-1, -0.5, 1.5], [2, -1.5, 2.6], [-2, -0.3, -.4], [0.2, -0.5, 1.2]],
        dtype=dtype)
    linear2.bias.data = torch.tensor([-0.3, 0.2, 0.7, 0.4], dtype=dtype)
    linear3 = nn.Linear(4, 1)
    linear3.weight.data = torch.tensor([[-.4, .5, -.6, 0.3]], dtype=dtype)
    linear3.bias.data = torch.tensor([-0.9], dtype=dtype)
    relu1 = nn.Sequential(linear1, nn.LeakyReLU(0.1), linear2,
                          nn.LeakyReLU(0.1), linear3)
    return relu1


def setup_state_samples_all(mesh_size):
    assert (isinstance(mesh_size, tuple))
    assert (len(mesh_size) == 2)
    dtype = torch.float64
    (samples_x, samples_y) = torch.meshgrid(
        torch.linspace(-1., 1., mesh_size[0], dtype=dtype),
        torch.linspace(-1., 1., mesh_size[1], dtype=dtype))
    state_samples = [None] * (mesh_size[0] * mesh_size[1])
    for i in range(samples_x.shape[0]):
        for j in range(samples_x.shape[1]):
            state_samples[i * samples_x.shape[1] + j] = torch.tensor(
                [samples_x[i, j], samples_y[i, j]], dtype=dtype)
    return torch.stack(state_samples, dim=0)


class TestTrainLyapunovReLUMIP(unittest.TestCase):
    """
    Test solve_positivity_mip() and solve_derivative_mip() function in
    TrainLyapunovReLU
    """
    def setUp(self):
        dtype = torch.float64
        forward_relu = utils.setup_relu((2, 4, 2),
                                        params=None,
                                        negative_slope=0.1,
                                        bias=True,
                                        dtype=dtype)
        forward_relu[0].weight.data = torch.tensor(
            [[4.1, 2.0], [0.5, 2.1], [-2.3, 0.5], [0.8, 0.9]], dtype=dtype)
        forward_relu[0].bias.data = torch.tensor([0.3, -2.1, 0.5, -0.8],
                                                 dtype=dtype)
        forward_relu[2].weight.data = torch.tensor(
            [[0.2, 3.1, 0.4, 0.5], [-0.5, -2.1, 0.4, -1.3]], dtype=dtype)
        forward_relu[2].bias.data = torch.tensor([0.4, -1.2], dtype=dtype)
        x_lo = torch.tensor([-3, -2], dtype=dtype)
        x_up = torch.tensor([1.5, 2.5], dtype=dtype)
        x_equilibrium = torch.tensor([-0.5, 1.], dtype=dtype)
        system = relu_system.AutonomousReLUSystemGivenEquilibrium(
            dtype, x_lo, x_up, forward_relu, x_equilibrium)
        V_lambda = 0.1
        lyapunov_relu = setup_lyapunov_relu()
        self.lyap = lyapunov.LyapunovDiscreteTimeHybridSystem(
            system, lyapunov_relu)
        self.R_options = r_options.FixedROptions(
            torch.tensor([[1, 1], [-1, 1], [0, 1]], dtype=dtype))
        self.dut = train_lyapunov.TrainLyapunovReLU(self.lyap, V_lambda,
                                                    x_equilibrium,
                                                    self.R_options)

    def test_solve_positivity_mip(self):
        for num_solutions in (1, 10, 1000):
            self.dut.add_adversarial_state_only = True
            self.dut.lyapunov_positivity_mip_pool_solutions = num_solutions
            positivity_mip, positivity_mip_obj, positivity_mip_adversarial =\
                self.dut.solve_positivity_mip()
            self.assertLessEqual(positivity_mip_adversarial.shape[0],
                                 num_solutions)
            for i in range(positivity_mip_adversarial.shape[0]):
                positivity_mip.gurobi_model.setParam(
                    gurobipy.GRB.Param.SolutionNumber, i)
                self.assertGreater(positivity_mip.gurobi_model.PoolObjVal, 0)
            # Check the positivity_mip_obj is correct.
            self.assertEqual(positivity_mip.gurobi_model.ObjVal,
                             positivity_mip_obj)
            x_optimal = positivity_mip_adversarial[0]
            self.assertAlmostEqual(
                (self.dut.lyapunov_positivity_epsilon * torch.norm(
                    self.R_options.R() @ (x_optimal - self.dut.x_equilibrium),
                    p=1) -
                 self.lyap.lyapunov_value(x_optimal,
                                          self.dut.x_equilibrium,
                                          self.dut.V_lambda,
                                          R=self.R_options.R())).item(),
                positivity_mip_obj)
            positivity_return = self.lyap.lyapunov_positivity_as_milp(
                self.dut.x_equilibrium,
                self.dut.V_lambda,
                self.dut.lyapunov_positivity_epsilon,
                R=self.R_options.R(),
                fixed_R=True)
            mip = positivity_return[0]
            mip.gurobi_model.setParam(gurobipy.GRB.Param.OutputFlag, False)
            mip.gurobi_model.setParam(gurobipy.GRB.Param.PoolSearchMode, 2)
            mip.gurobi_model.setParam(gurobipy.GRB.Param.PoolSolutions,
                                      num_solutions)
            mip.gurobi_model.optimize()
            for i in range(num_solutions):
                mip.gurobi_model.setParam(gurobipy.GRB.Param.SolutionNumber, i)
                if i < mip.gurobi_model.SolCount and\
                        mip.gurobi_model.PoolObjVal > 0:
                    np.testing.assert_allclose(
                        np.array([v.xn for v in positivity_return[1]]),
                        positivity_mip_adversarial[i].detach().numpy())

    def test_solve_derivative_mip(self):
        for num_solutions in (1, 10, 1000):
            self.dut.lyapunov_derivative_mip_pool_solutions = num_solutions
            derivative_mip, derivative_mip_obj, derivative_mip_adversarial,\
                derivative_mip_adversarial_next = \
                self.dut.solve_derivative_mip()
            self.assertLessEqual(derivative_mip_adversarial.shape[0],
                                 num_solutions)
            np.testing.assert_allclose(
                derivative_mip_adversarial_next.detach().numpy(),
                self.lyap.system.step_forward(
                    derivative_mip_adversarial).detach().numpy())
            # Check the derivative_mip_obj is correct.
            self.assertEqual(derivative_mip.gurobi_model.ObjVal,
                             derivative_mip_obj)
            derivative_return = self.lyap.lyapunov_derivative_as_milp(
                self.dut.x_equilibrium,
                self.dut.V_lambda,
                self.dut.lyapunov_derivative_epsilon,
                self.dut.lyapunov_derivative_eps_type,
                R=self.R_options.R(),
                fixed_R=True)
            mip = derivative_return.milp
            mip.gurobi_model.setParam(gurobipy.GRB.Param.OutputFlag, False)
            mip.gurobi_model.setParam(gurobipy.GRB.Param.PoolSearchMode, 2)
            mip.gurobi_model.setParam(gurobipy.GRB.Param.PoolSolutions,
                                      num_solutions)
            mip.gurobi_model.optimize()
            for i in range(num_solutions):
                mip.gurobi_model.setParam(gurobipy.GRB.Param.SolutionNumber, i)
                if i < mip.gurobi_model.SolCount and\
                        mip.gurobi_model.PoolObjVal > 0:
                    np.testing.assert_allclose(
                        np.array([v.xn for v in derivative_return.x]),
                        derivative_mip_adversarial[i].detach().numpy())


class TestTrainLyapunovReLUAdversarial(TestTrainLyapunovReLUMIP):
    """
    Test the function TrainLyapunovReLU.train_adversarial()
    """
    def test_train_adversarial(self):
        positivity_state_samples_init = utils.get_meshgrid_samples(
            torch.from_numpy(self.lyap.system.x_lo_all),
            torch.from_numpy(self.lyap.system.x_up_all), (3, 3), torch.float64)
        derivative_state_samples_init = utils.get_meshgrid_samples(
            torch.from_numpy(self.lyap.system.x_lo_all),
            torch.from_numpy(self.lyap.system.x_up_all), (5, 5), torch.float64)
        options = train_lyapunov.TrainLyapunovReLU.AdversarialTrainingOptions()
        options.num_batches = 10
        options.num_epochs_per_mip = 5
        options.positivity_samples_pool_size = 1000
        options.derivative_samples_pool_size = 1000
        self.dut.lyapunov_positivity_mip_pool_solutions = 10
        self.dut.lyapunov_derivative_mip_pool_solutions = 20
        self.dut.add_positivity_adversarial_state = True
        self.dut.add_derivative_adversarial_state = True
        self.dut.max_iterations = 1
        self.dut.output_flag = False
        result, positivity_state_samples, derivative_state_samples,\
            positivity_state_repeatition, derivative_state_repeatition = \
            self.dut.train_adversarial(
                positivity_state_samples_init, derivative_state_samples_init,
                options)
        self.assertEqual(positivity_state_samples.shape,
                         (positivity_state_samples_init.shape[0] +
                          self.dut.lyapunov_positivity_mip_pool_solutions, 2))
        self.assertEqual(derivative_state_samples.shape,
                         (derivative_state_samples_init.shape[0] +
                          self.dut.lyapunov_derivative_mip_pool_solutions, 2))
        self.assertEqual(positivity_state_repeatition.shape,
                         (positivity_state_samples.shape[0],))
        self.assertEqual(derivative_state_repeatition.shape,
                         (derivative_state_samples.shape[0],))


class TestTrainLyapunovReLU(unittest.TestCase):
    def test_total_loss(self):
        system = test_hybrid_linear_system.setup_trecate_discrete_time_system()
        V_lambda = 0.1
        x_equilibrium = torch.tensor([0, 0], dtype=system.dtype)
        relu = setup_lyapunov_relu()
        lyapunov_hybrid_system = lyapunov.LyapunovDiscreteTimeHybridSystem(
            system, relu)
        R_options = r_options.FixedROptions(
            torch.tensor([[1, 1], [-1, 1], [0, 1]], dtype=system.dtype))
        dut = train_lyapunov.TrainLyapunovReLU(lyapunov_hybrid_system,
                                               V_lambda, x_equilibrium,
                                               R_options)
        dut.lyapunov_positivity_sample_cost_weight = 0.5
        dut.lyapunov_derivative_sample_cost_weight = 0.6
        dut.add_positivity_adversarial_state = True
        dut.add_derivative_adversarial_state = True
        dut.max_sample_pool_size = 400
        state_samples_all = setup_state_samples_all((21, 21))
        state_samples_next = torch.stack([
            system.step_forward(state_samples_all[i])
            for i in range(state_samples_all.shape[0])
        ])
        torch.manual_seed(0)
        positivity_state_samples = state_samples_all.clone()
        derivative_state_samples = state_samples_all.clone()
        derivative_state_samples_next = state_samples_next.clone()
        loss, lyapunov_positivity_mip_cost, lyapunov_derivative_mip_cost,\
            positivity_sample_loss, derivative_sample_loss,\
            positivity_mip_loss, derivative_mip_loss,\
            positivity_state_samples_new, derivative_state_samples_new,\
            derivative_state_samples_next_new = dut.total_loss(
                positivity_state_samples, derivative_state_samples,
                state_samples_next, dut.lyapunov_positivity_sample_cost_weight,
                dut.lyapunov_derivative_sample_cost_weight,
                dut.lyapunov_positivity_mip_cost_weight,
                dut.lyapunov_derivative_mip_cost_weight)

        self.assertEqual(positivity_state_samples.shape[0] + 1,
                         positivity_state_samples_new.shape[0])
        self.assertEqual(derivative_state_samples.shape[0] + 1,
                         derivative_state_samples_new.shape[0])
        self.assertEqual(derivative_state_samples_next.shape[0] + 1,
                         derivative_state_samples_next_new.shape[0])
        self.assertAlmostEqual(
            loss.item(), (positivity_sample_loss + derivative_sample_loss +
                          positivity_mip_loss + derivative_mip_loss).item())
        # Compute hinge(-V(x)) for sampled state x
        loss_expected = 0.
        loss_expected += dut.lyapunov_positivity_sample_cost_weight *\
            lyapunov_hybrid_system.lyapunov_positivity_loss_at_samples(
                x_equilibrium,
                positivity_state_samples_new[-dut.max_sample_pool_size:],
                V_lambda, dut.lyapunov_positivity_epsilon,
                R=R_options.R(), margin=dut.lyapunov_positivity_sample_margin)
        loss_expected += dut.lyapunov_derivative_sample_cost_weight *\
            lyapunov_hybrid_system.\
            lyapunov_derivative_loss_at_samples_and_next_states(
                V_lambda, dut.lyapunov_derivative_epsilon,
                derivative_state_samples_new[-dut.max_sample_pool_size:],
                derivative_state_samples_next_new[-dut.max_sample_pool_size:],
                x_equilibrium, dut.lyapunov_derivative_eps_type,
                R=R_options.R(), margin=dut.lyapunov_derivative_sample_margin)
        lyapunov_positivity_mip_return = lyapunov_hybrid_system.\
            lyapunov_positivity_as_milp(
                x_equilibrium, V_lambda, dut.lyapunov_positivity_epsilon,
                R=R_options.R(), fixed_R=R_options.fixed_R)
        lyapunov_positivity_mip = lyapunov_positivity_mip_return[0]
        lyapunov_positivity_mip.gurobi_model.setParam(
            gurobipy.GRB.Param.OutputFlag, False)
        lyapunov_positivity_mip.gurobi_model.setParam(
            gurobipy.GRB.Param.PoolSearchMode, 2)
        lyapunov_positivity_mip.gurobi_model.setParam(
            gurobipy.GRB.Param.PoolSolutions,
            dut.lyapunov_positivity_mip_pool_solutions)
        lyapunov_positivity_mip.gurobi_model.optimize()

        lyapunov_derivative_mip_return = lyapunov_hybrid_system.\
            lyapunov_derivative_as_milp(
                x_equilibrium, V_lambda, dut.lyapunov_derivative_epsilon,
                lyapunov.ConvergenceEps.ExpLower, R=R_options.R(),
                fixed_R=R_options.fixed_R, lyapunov_lower=None,
                lyapunov_upper=None)
        lyapunov_derivative_mip = lyapunov_derivative_mip_return.milp
        lyapunov_derivative_mip.gurobi_model.setParam(
            gurobipy.GRB.Param.OutputFlag, False)
        lyapunov_derivative_mip.gurobi_model.setParam(
            gurobipy.GRB.Param.PoolSearchMode, 2)
        lyapunov_derivative_mip.gurobi_model.setParam(
            gurobipy.GRB.Param.PoolSolutions,
            dut.lyapunov_derivative_mip_pool_solutions)
        lyapunov_derivative_mip.gurobi_model.optimize()

        lyapunov_positivity_mip.gurobi_model.setParam(
            gurobipy.GRB.Param.SolutionNumber, 0)
        loss_expected += dut.lyapunov_positivity_mip_cost_weight * \
            lyapunov_positivity_mip.gurobi_model.getAttr(
                gurobipy.GRB.Attr.PoolObjVal)
        lyapunov_derivative_mip.gurobi_model.setParam(
            gurobipy.GRB.Param.SolutionNumber, 0)
        loss_expected += dut.lyapunov_derivative_mip_cost_weight *\
            lyapunov_derivative_mip.gurobi_model.getAttr(
                gurobipy.GRB.Attr.PoolObjVal)

        self.assertAlmostEqual(loss.item(), loss_expected.item(), places=4)
        self.assertAlmostEqual(lyapunov_positivity_mip_cost,
                               lyapunov_positivity_mip.gurobi_model.ObjVal)
        self.assertAlmostEqual(lyapunov_derivative_mip_cost,
                               lyapunov_derivative_mip.gurobi_model.ObjVal)


class TestTrainLyapunov(unittest.TestCase):
    def setUp(self):
        self.system = \
            test_hybrid_linear_system.setup_trecate_discrete_time_system()
        self.relu = setup_lyapunov_relu()

    def test_train_value_approximator(self):
        dut = train_lyapunov.TrainValueApproximator()
        dut.max_epochs = 1000
        dut.convergence_tolerance = 0.05
        state_samples_all = setup_state_samples_all((21, 21))
        V_lambda = 0.1
        x_equilibrium = torch.tensor([0., 0.], dtype=torch.float64)
        N = 100

        def instantaneous_cost(x):
            return x @ x

        result = dut.train(self.system,
                           self.relu,
                           V_lambda,
                           x_equilibrium,
                           instantaneous_cost,
                           state_samples_all,
                           N,
                           True,
                           R=torch.eye(2, dtype=torch.float64))
        self.assertTrue(result[0])
        relu_at_equilibrium = self.relu.forward(x_equilibrium)
        # Now check the total loss.
        with torch.no_grad():
            state_cost_samples = hybrid_linear_system.\
                generate_cost_to_go_samples(
                    self.system,
                    [state_samples_all[i] for i in
                     range(state_samples_all.shape[0])], N, instantaneous_cost,
                    True)
            x0_samples = torch.stack([pair[0] for pair in state_cost_samples],
                                     dim=0)
            cost_samples = torch.stack(
                [pair[1] for pair in state_cost_samples])
            relu_output = self.relu(x0_samples).squeeze()
            error = relu_output.squeeze() - relu_at_equilibrium + \
                V_lambda * torch.norm(
                    x0_samples - x_equilibrium.reshape((1, -1)).
                    expand(x0_samples.shape[0], -1), dim=1, p=1) - cost_samples
            loss = torch.sum(error * error) / len(cost_samples)
            self.assertLessEqual(loss, dut.convergence_tolerance)


class TestClusterAdversarialStates(unittest.TestCase):
    def test1(self):
        dtype = torch.float64
        x0 = torch.tensor([0.1, 0.2], dtype=dtype)
        x1 = torch.tensor([0.5, 0.3], dtype=dtype)
        x2 = torch.tensor([0.2, -0.5], dtype=dtype)
        adversarial_states = torch.cat(
            (x0, x0, x0, x1, x1, x1, x1, x2, x2, x2, x2)).reshape((-1, 2))
        clustered_adversarial_states, repeatition = \
            train_lyapunov._cluster_adversarial_states(
                adversarial_states, 1E-10)
        np.testing.assert_allclose(
            clustered_adversarial_states.detach().numpy(),
            torch.cat((x0, x1, x2)).reshape((-1, 2)).detach().numpy())
        np.testing.assert_array_equal(repeatition, np.array([3, 4, 4]))

        adversarial_states = torch.cat((x0, x0, x0)).reshape((-1, 2))
        clustered_adversarial_states, repeatition = \
            train_lyapunov._cluster_adversarial_states(
                adversarial_states, 1E-10)
        np.testing.assert_allclose(
            clustered_adversarial_states.detach().numpy(),
            x0.reshape((-1, 2)).detach().numpy())
        np.testing.assert_array_equal(repeatition, np.array([3]))


if __name__ == "__main__":
    unittest.main()
