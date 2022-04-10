import neural_network_lyapunov.lyapunov as lyapunov
import neural_network_lyapunov.barrier as barrier
import neural_network_lyapunov.train_lyapunov_barrier as train_lyapunov_barrier
import neural_network_lyapunov.hybrid_linear_system as hybrid_linear_system
import neural_network_lyapunov.test.test_hybrid_linear_system as\
    test_hybrid_linear_system
import neural_network_lyapunov.r_options as r_options
import neural_network_lyapunov.utils as utils
import neural_network_lyapunov.relu_system as relu_system
import neural_network_lyapunov.gurobi_torch_mip as gurobi_torch_mip
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


class TestTrainerMIP(unittest.TestCase):
    """
    Test solve_positivity_mip() and solve_lyap_derivative_mip() function in
    Trainer
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
        self.dut = train_lyapunov_barrier.Trainer()
        self.dut.add_lyapunov(self.lyap, V_lambda, x_equilibrium,
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
                R=self.R_options.R())
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

    def test_solve_lyap_derivative_mip(self):
        for num_solutions in (1, 10, 1000):
            self.dut.lyapunov_derivative_mip_pool_solutions = num_solutions
            derivative_mip, derivative_mip_obj, derivative_mip_adversarial,\
                derivative_mip_adversarial_next = \
                self.dut.solve_lyap_derivative_mip()
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
                R=self.R_options.R())
            mip = derivative_return.milp
            mip.gurobi_model.setParam(gurobipy.GRB.Param.OutputFlag, False)
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


class TestTrainerAdversarial(TestTrainerMIP):
    """
    Test the function Trainer.train_adversarial()
    """
    def test_train_adversarial(self):
        positivity_state_samples_init = utils.get_meshgrid_samples(
            torch.from_numpy(self.lyap.system.x_lo_all),
            torch.from_numpy(self.lyap.system.x_up_all), (3, 3), torch.float64)
        derivative_state_samples_init = utils.get_meshgrid_samples(
            torch.from_numpy(self.lyap.system.x_lo_all),
            torch.from_numpy(self.lyap.system.x_up_all), (5, 5), torch.float64)
        options = train_lyapunov_barrier.Trainer.AdversarialTrainingOptions()
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
        self.assertLessEqual(
            positivity_state_samples.shape,
            (positivity_state_samples_init.shape[0] +
             self.dut.lyapunov_positivity_mip_pool_solutions, 2))
        self.assertLessEqual(
            derivative_state_samples.shape,
            (derivative_state_samples_init.shape[0] +
             self.dut.lyapunov_derivative_mip_pool_solutions, 2))
        self.assertEqual(positivity_state_repeatition.shape,
                         (positivity_state_samples.shape[0], ))
        self.assertEqual(derivative_state_repeatition.shape,
                         (derivative_state_samples.shape[0], ))


class TestTrainer(unittest.TestCase):
    def test_total_loss(self):
        system = test_hybrid_linear_system.setup_trecate_discrete_time_system()
        V_lambda = 0.1
        x_equilibrium = torch.tensor([0, 0], dtype=system.dtype)
        relu = setup_lyapunov_relu()
        lyapunov_hybrid_system = lyapunov.LyapunovDiscreteTimeHybridSystem(
            system, relu)
        R_options = r_options.FixedROptions(
            torch.tensor([[1, 1], [-1, 1], [0, 1]], dtype=system.dtype))
        dut = train_lyapunov_barrier.Trainer()
        dut.add_lyapunov(lyapunov_hybrid_system, V_lambda, x_equilibrium,
                         R_options)
        dut.lyapunov_positivity_sample_cost_weight = 0.5
        dut.lyapunov_derivative_sample_cost_weight = 0.6
        dut.boundary_value_gap_mip_cost_weight = 0.8
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
        total_loss_return = dut.total_loss(
            positivity_state_samples, derivative_state_samples,
            state_samples_next, dut.lyapunov_positivity_sample_cost_weight,
            dut.lyapunov_derivative_sample_cost_weight,
            dut.lyapunov_positivity_mip_cost_weight,
            dut.lyapunov_derivative_mip_cost_weight,
            dut.boundary_value_gap_mip_cost_weight)

        self.assertEqual(
            positivity_state_samples.shape[0] + 1,
            total_loss_return.lyap_loss.positivity_state_samples.shape[0])
        self.assertEqual(
            derivative_state_samples.shape[0] + 1,
            total_loss_return.lyap_loss.derivative_state_samples.shape[0])
        self.assertEqual(
            derivative_state_samples_next.shape[0] + 1,
            total_loss_return.lyap_loss.derivative_state_samples_next.shape[0])
        self.assertAlmostEqual(
            total_loss_return.loss.item(),
            (total_loss_return.lyap_loss.positivity_sample_loss +
             total_loss_return.lyap_loss.derivative_sample_loss +
             total_loss_return.lyap_loss.positivity_mip_loss +
             total_loss_return.lyap_loss.derivative_mip_loss +
             total_loss_return.lyap_loss.gap_mip_loss).item())
        # Compute hinge(-V(x)) for sampled state x
        loss_expected = 0.
        loss_expected += dut.lyapunov_positivity_sample_cost_weight *\
            lyapunov_hybrid_system.lyapunov_positivity_loss_at_samples(
                x_equilibrium,
                total_loss_return.lyap_loss.positivity_state_samples[
                    -dut.max_sample_pool_size:],
                V_lambda, dut.lyapunov_positivity_epsilon,
                R=R_options.R(), margin=dut.lyapunov_positivity_sample_margin)
        loss_expected += dut.lyapunov_derivative_sample_cost_weight *\
            lyapunov_hybrid_system.\
            lyapunov_derivative_loss_at_samples_and_next_states(
                V_lambda, dut.lyapunov_derivative_epsilon,
                total_loss_return.lyap_loss.derivative_state_samples[
                    -dut.max_sample_pool_size:],
                total_loss_return.lyap_loss.derivative_state_samples_next[
                    -dut.max_sample_pool_size:],
                x_equilibrium, dut.lyapunov_derivative_eps_type,
                R=R_options.R(), margin=dut.lyapunov_derivative_sample_margin)
        lyapunov_positivity_mip_return = lyapunov_hybrid_system.\
            lyapunov_positivity_as_milp(
                x_equilibrium, V_lambda, dut.lyapunov_positivity_epsilon,
                R=R_options.R())
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
                lyapunov_lower=None, lyapunov_upper=None)
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
        boundary_gap, _, _, _, _ = dut.solve_boundary_gap_mip()
        loss_expected += dut.boundary_value_gap_mip_cost_weight * boundary_gap

        self.assertAlmostEqual(total_loss_return.loss.item(),
                               loss_expected.item(),
                               places=4)
        self.assertAlmostEqual(total_loss_return.lyap_loss.positivity_mip_obj,
                               lyapunov_positivity_mip.gurobi_model.ObjVal)
        self.assertAlmostEqual(total_loss_return.lyap_loss.derivative_mip_obj,
                               lyapunov_derivative_mip.gurobi_model.ObjVal)

    def solve_boundary_gap_mip_tester(self, dut):
        loss, V_min_milp, V_max_milp, x_min, x_max = \
            dut.solve_boundary_gap_mip()
        # Formulate an MILP to max/min V(x) over the boundary x∈∂ℬ
        milp = gurobi_torch_mip.GurobiTorchMILP(
            dut.lyapunov_hybrid_system.system.dtype)
        x = milp.addVars(dut.lyapunov_hybrid_system.system.x_dim,
                         lb=-gurobipy.GRB.INFINITY)
        verify_region_boundary = utils.box_boundary(
            torch.from_numpy(dut.lyapunov_hybrid_system.system.x_lo_all),
            torch.from_numpy(dut.lyapunov_hybrid_system.system.x_up_all))
        milp.add_mixed_integer_linear_constraints(verify_region_boundary, x,
                                                  None, "", "", "", "", "")
        z, _, a_out, b_out, _ = \
            dut.lyapunov_hybrid_system.add_lyap_relu_output_constraint(milp, x)
        s, gamma = dut.lyapunov_hybrid_system.add_state_error_l1_constraint(
            milp, dut.x_equilibrium, x, R=dut.R_options.R())
        relu_at_equilibrium = dut.lyapunov_hybrid_system.lyapunov_relu(
            dut.x_equilibrium)
        # The objective is V(x) = ϕ(x) - ϕ(x*) + λ*|R(x−x*)|₁
        # = a_out * z + b_out -  ϕ(x*) + λ * s
        dtype = dut.lyapunov_hybrid_system.system.dtype
        milp.setObjective([
            a_out.squeeze(), dut.V_lambda * torch.ones((len(s), ), dtype=dtype)
        ], [z, s],
                          constant=b_out - relu_at_equilibrium.squeeze(),
                          sense=gurobipy.GRB.MAXIMIZE)
        milp.gurobi_model.setParam(gurobipy.GRB.Param.OutputFlag, False)
        milp.gurobi_model.optimize()
        self.assertEqual(milp.gurobi_model.status, gurobipy.GRB.Status.OPTIMAL)
        self.assertEqual(milp.gurobi_model.ObjVal, V_max_milp)
        np.testing.assert_array_equal(x_max.detach().numpy(),
                                      np.array([v.x for v in x]))
        milp.setObjective([
            a_out.squeeze(), dut.V_lambda * torch.ones((len(s), ), dtype=dtype)
        ], [z, s],
                          constant=b_out - relu_at_equilibrium.squeeze(),
                          sense=gurobipy.GRB.MINIMIZE)
        milp.gurobi_model.optimize()
        self.assertEqual(milp.gurobi_model.status, gurobipy.GRB.Status.OPTIMAL)
        self.assertEqual(milp.gurobi_model.ObjVal, V_min_milp)
        np.testing.assert_array_equal(x_min.detach().numpy(),
                                      np.array([v.x for v in x]))
        self.assertAlmostEqual(loss.item(), V_max_milp - V_min_milp)

    def test_solve_boundary_gap_mip(self):
        system = test_hybrid_linear_system.setup_trecate_discrete_time_system()
        V_lambda = 0.1
        x_equilibrium = torch.tensor([0, 0], dtype=system.dtype)
        relu = setup_lyapunov_relu()
        lyapunov_hybrid_system = lyapunov.LyapunovDiscreteTimeHybridSystem(
            system, relu)
        R_options = r_options.FixedROptions(
            torch.tensor([[1, 1], [-1, 1], [0, 1]], dtype=system.dtype))
        dut = train_lyapunov_barrier.Trainer()
        dut.add_lyapunov(lyapunov_hybrid_system, V_lambda, x_equilibrium,
                         R_options)
        self.solve_boundary_gap_mip_tester(dut)


class TestTrainerBarrier(unittest.TestCase):
    """
    Test barrier related functions in Trainer class.
    """
    def setUp(self):
        self.dtype = torch.float64
        dynamics_relu = utils.setup_relu((2, 4, 2),
                                         params=None,
                                         negative_slope=0.1,
                                         bias=True,
                                         dtype=self.dtype)
        dynamics_relu[0].weight.data = torch.tensor(
            [[1, -2], [1.2, 0.3], [0.5, 1], [0.2, 1]], dtype=self.dtype)
        dynamics_relu[0].bias.data = torch.tensor([0.5, 1, 0.2, -0.3],
                                                  dtype=self.dtype)
        dynamics_relu[2].weight.data = torch.tensor(
            [[0.1, 0.4, -0.2, 1], [0.5, 1.5, -0.2, 2]], dtype=self.dtype)
        dynamics_relu[2].bias.data = torch.tensor([0.5, 1], dtype=self.dtype)
        self.system = relu_system.AutonomousReLUSystemGivenEquilibrium(
            self.dtype,
            torch.tensor([-2, 1], dtype=self.dtype),
            torch.tensor([3, 4], dtype=self.dtype),
            dynamics_relu,
            torch.tensor([1, 3], dtype=self.dtype),
            discrete_time_flag=True)

        self.barrier_relu = utils.setup_relu((2, 3, 2, 1),
                                             params=None,
                                             negative_slope=0.1,
                                             bias=True,
                                             dtype=self.dtype)
        self.barrier_relu[0].weight.data = torch.tensor(
            [[0.2, 0.3], [1, 2], [-2, -3]], dtype=self.dtype)
        self.barrier_relu[0].bias.data = torch.tensor([0.5, -1, -0.2],
                                                      dtype=self.dtype)
        self.barrier_relu[2].weight.data = torch.tensor(
            [[0.5, 0.1, -1.5], [0.2, -0.3, 1]], dtype=self.dtype)
        self.barrier_relu[2].bias.data = torch.tensor([1, -2],
                                                      dtype=self.dtype)
        self.barrier_relu[4].weight.data = torch.tensor([[1, 3]],
                                                        dtype=self.dtype)
        self.barrier_relu[4].bias.data = torch.tensor([0.5], dtype=self.dtype)

        self.barrier_system = barrier.DiscreteTimeBarrier(
            self.system, self.barrier_relu)

    def test_barrier_sample_loss(self):
        dut = train_lyapunov_barrier.Trainer()
        dut.add_barrier(self.barrier_system,
                        x_star=(self.system.x_lo * 0.25 +
                                self.system.x_up * 0.75),
                        c=0.1,
                        barrier_epsilon=0.3)
        safe_state_samples = utils.uniform_sample_in_box(
            self.system.x_lo, (self.system.x_lo + self.system.x_up) / 2, 100)
        unsafe_state_samples = utils.uniform_sample_in_box(
            (self.system.x_lo + self.system.x_up) / 2, self.system.x_up, 200)
        derivative_state_samples = utils.uniform_sample_in_box(
            self.system.x_lo, self.system.x_up, 300)
        safe_cost_weight = 2.
        unsafe_cost_weight = 3.
        derivative_cost_weight = 4.
        safe_sample_loss, unsafe_sample_loss, derivative_sample_loss =\
            dut.barrier_sample_loss(
                safe_state_samples, unsafe_state_samples,
                derivative_state_samples, safe_cost_weight, unsafe_cost_weight,
                derivative_cost_weight)
        safe_sample_loss_expected = utils.loss_reduction(
            torch.maximum(
                -dut.barrier_system.value(safe_state_samples,
                                          dut.barrier_x_star, dut.barrier_c),
                torch.tensor(0, dtype=self.dtype)),
            dut.sample_loss_reduction) * safe_cost_weight
        self.assertAlmostEqual(safe_sample_loss.item(),
                               safe_sample_loss_expected.item())

    def test_solve_barrier_value_mip(self):
        dut = train_lyapunov_barrier.Trainer()
        dut.add_barrier(self.barrier_system,
                        x_star=(self.system.x_lo * 0.25 +
                                self.system.x_up * 0.75),
                        c=0.1,
                        barrier_epsilon=0.3)
        dut.safe_regions = [gurobi_torch_mip.MixedIntegerConstraintsReturn()]
        dut.safe_regions[0].Ain_input = -torch.eye(2, dtype=self.dtype)
        dut.safe_regions[
            0].rhs_in = self.system.x_lo * 0.3 + self.system.x_up * 0.7
        dut.unsafe_regions = [None, None]
        dut.unsafe_regions[0] = gurobi_torch_mip.MixedIntegerConstraintsReturn(
        )
        dut.unsafe_regions[0].Ain_input = torch.tensor([[1, 0]],
                                                       dtype=self.dtype)
        dut.unsafe_regions[0].rhs_in = torch.tensor(
            [self.system.x_lo[0] * 0.9 + self.system.x_up[0] * 0.1],
            dtype=self.dtype)
        dut.unsafe_regions[1] = gurobi_torch_mip.MixedIntegerConstraintsReturn(
        )
        dut.unsafe_regions[1].Ain_input = torch.tensor([[0, 1]],
                                                       dtype=self.dtype)
        dut.unsafe_regions[1].rhs_in = torch.tensor(
            [self.system.x_lo[1] * 0.8 + self.system.x_up[1] * 0.2],
            dtype=self.dtype)
        dut.barrier_value_mip_pool_solutions = 5

        # safe region
        mip, mip_obj, mip_adversarial = dut.solve_barrier_value_mip(
            safe_flag=True)
        self.assertEqual(len(mip), len(dut.safe_regions))
        self.assertEqual(len(mip_obj), len(dut.safe_regions))
        self.assertEqual(len(mip_adversarial), len(dut.safe_regions))

        # unsafe region
        mip, mip_obj, mip_adversarial = dut.solve_barrier_value_mip(
            safe_flag=False)
        self.assertEqual(len(mip), len(dut.unsafe_regions))
        self.assertEqual(len(mip_obj), len(dut.unsafe_regions))
        self.assertEqual(len(mip_adversarial), len(dut.unsafe_regions))

    def test_solve_barrier_derivative_mip(self):
        dut = train_lyapunov_barrier.Trainer()
        dut.add_barrier(self.barrier_system,
                        x_star=(self.system.x_lo * 0.25 +
                                self.system.x_up * 0.75),
                        c=0.1,
                        barrier_epsilon=0.3)
        dut.barrier_derivative_mip_pool_solutions = 10
        dut.add_adversarial_state_only = True
        milp, mip_obj, adversarial = dut.solve_barrier_derivative_mip()
        self.assertGreater(adversarial.shape[0], 1)
        np.testing.assert_array_less(
            0,
            dut.barrier_system.derivative(
                adversarial, dut.barrier_x_star, dut.barrier_c,
                dut.barrier_epsilon).detach().numpy())

    def test_barrier_loss(self):
        dut = train_lyapunov_barrier.Trainer()
        dut.add_barrier(self.barrier_system,
                        x_star=(self.system.x_lo * 0.25 +
                                self.system.x_up * 0.75),
                        c=0.1,
                        barrier_epsilon=0.3)
        dut.safe_regions = [gurobi_torch_mip.MixedIntegerConstraintsReturn()]
        dut.safe_regions[0].Ain = torch.eye(2, dtype=self.dtype)
        dut.safe_regions[0].rhs = (self.system.x_lo + self.system.x_up) / 2
        safe_state_samples = utils.uniform_sample_in_box(
            self.system.x_lo, (self.system.x_lo + self.system.x_up) / 2, 100)

        dut.unsafe_regions = [gurobi_torch_mip.MixedIntegerConstraintsReturn()]
        dut.unsafe_regions[0].Ain_input = -torch.eye(2, dtype=self.dtype)
        dut.unsafe_regions[0].rhs_in = -(self.system.x_lo * 0.25 +
                                         self.system.x_up * 0.75)
        num_unsafe_state_samples = 200
        unsafe_state_samples = utils.uniform_sample_in_box(
            self.system.x_lo * 0.25 + self.system.x_up * 0.75,
            self.system.x_up, num_unsafe_state_samples)
        num_derivative_state_samples = 300
        derivative_state_samples = utils.uniform_sample_in_box(
            self.system.x_lo, self.system.x_up, num_derivative_state_samples)

        safe_sample_cost_weight = 2.
        unsafe_sample_cost_weight = 3.
        derivative_sample_cost_weight = 4.
        safe_mip_cost_weight = 5.
        unsafe_mip_cost_weight = 6.
        derivative_mip_cost_weight = 7.

        barrier_loss = dut.compute_barrier_loss(
            safe_state_samples, unsafe_state_samples, derivative_state_samples,
            safe_sample_cost_weight, unsafe_sample_cost_weight,
            derivative_sample_cost_weight, safe_mip_cost_weight,
            unsafe_mip_cost_weight, derivative_mip_cost_weight)
        self.assertEqual(len(barrier_loss.safe_mip_obj), len(dut.safe_regions))
        self.assertEqual(len(barrier_loss.safe_mip_loss),
                         len(dut.safe_regions))
        self.assertEqual(len(barrier_loss.unsafe_mip_obj),
                         len(dut.unsafe_regions))
        self.assertEqual(len(barrier_loss.unsafe_mip_loss),
                         len(dut.unsafe_regions))
        self.assertGreater(barrier_loss.unsafe_state_samples.shape[0],
                           num_unsafe_state_samples)
        self.assertGreater(barrier_loss.derivative_state_samples.shape[0],
                           num_derivative_state_samples)


class TestTrainValueApproximator(unittest.TestCase):
    def setUp(self):
        self.system = \
            test_hybrid_linear_system.setup_trecate_discrete_time_system()
        self.relu = setup_lyapunov_relu()

    def test_train_value_approximator(self):
        dut = train_lyapunov_barrier.TrainValueApproximator()
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
            train_lyapunov_barrier._cluster_adversarial_states(
                adversarial_states, 1E-10)
        np.testing.assert_allclose(
            clustered_adversarial_states.detach().numpy(),
            torch.cat((x0, x1, x2)).reshape((-1, 2)).detach().numpy())
        np.testing.assert_array_equal(repeatition,
                                      torch.tensor([3, 4, 4], dtype=dtype))

        adversarial_states = torch.cat((x0, x0, x0)).reshape((-1, 2))
        clustered_adversarial_states, repeatition = \
            train_lyapunov_barrier._cluster_adversarial_states(
                adversarial_states, 1E-10)
        np.testing.assert_allclose(
            clustered_adversarial_states.detach().numpy(),
            x0.reshape((-1, 2)).detach().numpy())
        np.testing.assert_array_equal(repeatition,
                                      torch.tensor([3], dtype=dtype))


if __name__ == "__main__":
    unittest.main()
