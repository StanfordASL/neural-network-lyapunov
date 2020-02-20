import robust_value_approx.lyapunov as lyapunov
import robust_value_approx.train_lyapunov as train_lyapunov
import robust_value_approx.hybrid_linear_system as hybrid_linear_system
import robust_value_approx.test.test_hybrid_linear_system as\
    test_hybrid_linear_system
import torch
import torch.nn as nn
import unittest
import gurobipy


def setup_relu():
    # Construct a simple ReLU model with 2 hidden layers
    dtype = torch.float64
    linear1 = nn.Linear(2, 3)
    linear1.weight.data = torch.tensor([[-1, 0.5], [-0.3, 0.74], [-2, 1.5]],
                                       dtype=dtype)
    linear1.bias.data = torch.tensor([-0.1, 1.0, 0.5], dtype=dtype)
    linear2 = nn.Linear(3, 4)
    linear2.weight.data = torch.tensor(
            [[-1, -0.5, 1.5], [2, -1.5, 2.6], [-2, -0.3, -.4],
             [0.2, -0.5, 1.2]],
            dtype=dtype)
    linear2.bias.data = torch.tensor([-0.3, 0.2, 0.7, 0.4], dtype=dtype)
    linear3 = nn.Linear(4, 1)
    linear3.weight.data = torch.tensor([[-.4, .5, -.6, 0.3]], dtype=dtype)
    linear3.bias.data = torch.tensor([-0.9], dtype=dtype)
    relu1 = nn.Sequential(
        linear1, nn.LeakyReLU(0.1), linear2, nn.LeakyReLU(0.1), linear3,
        nn.LeakyReLU(0.1))
    return relu1


def setup_state_samples_all(mesh_size):
    assert(isinstance(mesh_size, tuple))
    assert(len(mesh_size) == 2)
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


class TestTrainLyapunovReLU(unittest.TestCase):
    def test_total_loss(self):
        system = test_hybrid_linear_system.setup_trecate_discrete_time_system()
        V_rho = 0.1
        x_equilibrium = torch.tensor([0, 0], dtype=system.dtype)
        lyapunov_hybrid_system = lyapunov.LyapunovDiscreteTimeHybridSystem(
            system)
        dut = train_lyapunov.TrainLyapunovReLU(
            lyapunov_hybrid_system, V_rho, x_equilibrium)
        dut.lyapunov_positivity_sample_cost_weight = 0.5
        dut.lyapunov_derivative_sample_cost_weight = 0.6
        relu = setup_relu()
        state_samples_all = setup_state_samples_all((21, 21))
        state_samples_next = torch.stack([
            system.step_forward(state_samples_all[i]) for i in
            range(state_samples_all.shape[0])])
        torch.manual_seed(0)
        loss, lyapunov_positivity_mip_cost, lyapunov_derivative_mip_cost,\
            positivity_sample_loss, derivative_sample_loss,\
            positivity_mip_loss, derivative_mip_loss =\
            dut.total_loss(relu, state_samples_all, state_samples_next)
        self.assertAlmostEqual(
            loss.item(),
            (positivity_sample_loss + derivative_sample_loss +
             positivity_mip_loss + derivative_mip_loss).item())
        # Compute hinge(-V(x)) for sampled state x
        loss_expected = 0.
        relu_at_equilibrium = relu.forward(x_equilibrium)
        loss_expected += dut.lyapunov_positivity_sample_cost_weight *\
            lyapunov_hybrid_system.lyapunov_positivity_loss_at_samples(
                relu, relu_at_equilibrium, x_equilibrium,
                state_samples_all, V_rho,
                dut.lyapunov_positivity_sample_margin)
        loss_expected += dut.lyapunov_derivative_sample_cost_weight *\
            lyapunov_hybrid_system.\
            lyapunov_derivative_loss_at_samples_and_next_states(
                relu, V_rho, dut.lyapunov_derivative_epsilon,
                state_samples_all, state_samples_next, x_equilibrium,
                dut.lyapunov_derivative_sample_margin)
        lyapunov_positivity_mip_return = lyapunov_hybrid_system.\
            lyapunov_positivity_as_milp(
                relu, x_equilibrium, V_rho, dut.lyapunov_positivity_epsilon)
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
                relu, x_equilibrium, V_rho, dut.lyapunov_derivative_epsilon,
                None, None)
        lyapunov_derivative_mip = lyapunov_derivative_mip_return[0]
        lyapunov_derivative_mip.gurobi_model.setParam(
            gurobipy.GRB.Param.OutputFlag, False)
        lyapunov_derivative_mip.gurobi_model.setParam(
            gurobipy.GRB.Param.PoolSearchMode, 2)
        lyapunov_derivative_mip.gurobi_model.setParam(
            gurobipy.GRB.Param.PoolSolutions,
            dut.lyapunov_derivative_mip_pool_solutions)
        lyapunov_derivative_mip.gurobi_model.optimize()

        for mip_sol_number in\
                range(dut.lyapunov_positivity_mip_pool_solutions):
            if mip_sol_number < lyapunov_positivity_mip.gurobi_model.solCount:
                lyapunov_positivity_mip.gurobi_model.setParam(
                    gurobipy.GRB.Param.SolutionNumber, mip_sol_number)
                loss_expected += dut.lyapunov_positivity_mip_cost_weight * \
                    torch.pow(torch.tensor(
                        dut.lyapunov_positivity_mip_cost_decay_rate,
                        dtype=system.dtype), mip_sol_number) * \
                    -lyapunov_positivity_mip.gurobi_model.getAttr(
                        gurobipy.GRB.Attr.PoolObjVal)
        for mip_sol_number in\
                range(dut.lyapunov_derivative_mip_pool_solutions):
            if mip_sol_number < lyapunov_derivative_mip.gurobi_model.solCount:
                lyapunov_derivative_mip.gurobi_model.setParam(
                    gurobipy.GRB.Param.SolutionNumber, mip_sol_number)
                loss_expected += dut.lyapunov_derivative_mip_cost_weight *\
                    torch.pow(torch.tensor(
                        dut.lyapunov_derivative_mip_cost_decay_rate,
                        dtype=system.dtype), mip_sol_number) *\
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
        self.relu = setup_relu()

    def test_train_value_approximator(self):
        dut = train_lyapunov.TrainValueApproximator()
        dut.max_epochs = 1000
        dut.convergence_tolerance = 0.05
        state_samples_all = setup_state_samples_all((21, 21))
        V_rho = 0.1
        x_equilibrium = torch.tensor([0., 0.], dtype=torch.float64)
        N = 100

        def instantaneous_cost(x):
            return x @ x
        result = dut.train(
            self.system, self.relu, V_rho, x_equilibrium, instantaneous_cost,
            state_samples_all, N, True)
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
            x0_samples = torch.stack([
                pair[0] for pair in state_cost_samples], dim=0)
            cost_samples = torch.stack([
                pair[1] for pair in state_cost_samples])
            relu_output = self.relu(x0_samples).squeeze()
            error = relu_output.squeeze() - relu_at_equilibrium + \
                V_rho * torch.norm(
                    x0_samples - x_equilibrium.reshape((1, -1)).
                    expand(x0_samples.shape[0], -1), dim=1, p=1) - cost_samples
            loss = torch.sum(error * error) / len(cost_samples)
            self.assertLessEqual(loss, dut.convergence_tolerance)


if __name__ == "__main__":
    unittest.main()
