"""
Test train Lyapunov function and the controller
"""
import neural_network_lyapunov.feedback_system as feedback_system
import neural_network_lyapunov.hybrid_linear_system as hybrid_linear_system
import neural_network_lyapunov.lyapunov as lyapunov
import neural_network_lyapunov.relu_system as relu_system
import neural_network_lyapunov.train_lyapunov as train_lyapunov
import neural_network_lyapunov.utils as utils
import unittest
import torch
import numpy as np


def create_hybrid_system1(dtype):
    # Create a PWL hybrid system for forward dynamics.
    system = hybrid_linear_system.HybridLinearSystem(2, 2, dtype)
    # The system has two modes
    # x[n+1] = x[n] + u[n] if [0, -10, -10, -10] <= [x[n], u[n]] <=
    # [10, 10, 10, 10]
    # x[n+1] = 0.5x[n] + 1.5u[n] if [-10, -10, -10, -10] <= [x[n], u[n]] <=
    # [0, 10, 10, 10]
    P = torch.zeros(8, 4, dtype=dtype)
    P[:4, :] = torch.eye(4, dtype=dtype)
    P[4:, :] = -torch.eye(4, dtype=dtype)
    system.add_mode(
        torch.eye(2, dtype=dtype), torch.eye(2, dtype=dtype),
        torch.tensor([0, 0], dtype=dtype), P,
        torch.tensor([10, 10, 10, 10, 0, 10, 10, 10], dtype=dtype))
    system.add_mode(
        0.5*torch.eye(2, dtype=dtype), 1.5*torch.eye(2, dtype=dtype),
        torch.tensor([0, 0], dtype=dtype), P,
        torch.tensor([0, 10, 10, 10, 10, 10, 10, 10], dtype=dtype))
    return system


class TestLyapunov(unittest.TestCase):
    def setUp(self):
        self.dtype = torch.float64
        self._create_feedback_hybrid_system()
        self._create_feedback_second_order_system()

    def _create_feedback_hybrid_system(self):
        # First system is a piecewise linear hybrid system. Neither the
        # controller network nor the Lyapunov network has bias terms.
        self.forward_system1 = create_hybrid_system1(self.dtype)
        self.controller_network1 = utils.setup_relu(
            (2, 4, 2), torch.tensor([
                0.1, 0.2, 0.3, 0.4, -0.5, 0.6, 0.7, 0.2, 0.5, 0.4, 1.5, 1.3,
                1.2, -0.8, 2.1, 0.4], dtype=self.dtype), negative_slope=0.01,
            bias=False, dtype=self.dtype)
        u_lower_limit = np.array([-10., -10.])
        u_upper_limit = np.array([10., 10.])
        self.closed_loop_system1 = feedback_system.FeedbackSystem(
            self.forward_system1, self.controller_network1,
            x_equilibrium=torch.tensor([0, 0], dtype=self.dtype),
            u_equilibrium=torch.tensor([0, 0], dtype=self.dtype),
            u_lower_limit=u_lower_limit,
            u_upper_limit=u_upper_limit)
        self.lyapunov_relu1 = utils.setup_relu(
            (2, 4, 4, 1), torch.tensor([
                0.1, 0.2, 0.3, -0.1, 0.3, 0.9, 1.2, 0.4, -0.2, -0.5, 0.5, 0.9,
                1.2, 1.4, 1.6, 2.4, 2.1, -0.6, -0.9, -.8, 2.1, -0.5, -1.2, 2.1,
                0.5, 0.4, 0.6, 1.2], dtype=self.dtype), negative_slope=0.01,
            bias=False, dtype=self.dtype)
        self.lyapunov_hybrid_system1 = \
            lyapunov.LyapunovDiscreteTimeHybridSystem(
                self.closed_loop_system1, self.lyapunov_relu1)

    def _create_feedback_second_order_system(self):
        # A second order system with nq = nv = 2, nu = 1
        forward_network_params = torch.tensor([
            0.2, 0.4, 0.1, 0.5, 0.4, -0.2, 0.4, 0.5, 0.9, -0.3, 1.2, -2.1, 0.1,
            0.45, 0.2, 0.8, 0.7, 0.3, 0.2, 0.5, 0.4, 0.8, 2.1, 0.4, 0.5, 0.2,
            -0.4, -0.5, 0.3, -2.1, 0.4, 0.2, 0.1, 0.5], dtype=self.dtype)
        forward_network = utils.setup_relu(
            (5, 4, 2), forward_network_params, negative_slope=0.01, bias=True,
            dtype=self.dtype)
        q_equilibrium = torch.tensor([0.5, 0.4], dtype=self.dtype)
        u_equilibrium = torch.tensor([0.5], dtype=self.dtype)
        dt = 0.01
        self.forward_system2 = \
            relu_system.ReLUSecondOrderSystemGivenEquilibrium(
                self.dtype, torch.tensor([-2, -2, -4, -4], dtype=self.dtype),
                torch.tensor([2, 2, 4, 4], dtype=self.dtype),
                torch.tensor([-10], dtype=self.dtype),
                torch.tensor([10], dtype=self.dtype), forward_network,
                q_equilibrium, u_equilibrium, dt)
        controller_network_params2 = torch.tensor([
            0.1, 0.4, 0.2, 0.5, -0.3, -1.2, 0.5, 0.9, 0.8, 0.7, -2.1, 0.4, 0.5,
            0.1, 1.2, 1.5, 0.4, -0.3, -2.1, -1.4, -0.9, 0.45, 0.32, 0.12, 0.78,
            -0.5], dtype=self.dtype)
        self.controller_network2 = utils.setup_relu(
            (4, 3, 2, 1), controller_network_params2, negative_slope=0.01,
            bias=True, dtype=self.dtype)
        self.closed_loop_system2 = feedback_system.FeedbackSystem(
            self.forward_system2, self.controller_network2,
            x_equilibrium=self.forward_system2.x_equilibrium,
            u_equilibrium=self.forward_system2.u_equilibrium,
            u_lower_limit=np.array([-10.]),
            u_upper_limit=np.array([10.]))
        self.lyapunov_relu2 = utils.setup_relu(
            (4, 4, 4, 1), torch.tensor([
                0.1, 0.2, 0.3, -0.1, 0.3, 0.9, 1.2, 0.4, -0.2, -0.5, 0.5, 0.9,
                1.2, 1.4, 1.6, 2.4, 2.1, -0.6, -0.9, -.8, 2.1, -0.5, -1.2, 2.1,
                0.5, 0.4, 0.6, 1.2, 0.7, 0.1, -0.2, 0.3, -1.2, 0.5, 0.1, 0.8,
                0.7, 0.2, 0.1, -0.5, 0.3, 0.6, -0.2, 1.2, 0.1],
                dtype=self.dtype), negative_slope=0.01, bias=True,
            dtype=self.dtype)
        self.lyapunov_hybrid_system2 = \
            lyapunov.LyapunovDiscreteTimeHybridSystem(
                self.closed_loop_system2, self.lyapunov_relu2)

    def compute_lyapunov_positivity_loss_at_samples(
            self, dut, state_samples, V_lambda, epsilon, margin):
        assert(isinstance(dut, lyapunov.LyapunovHybridLinearSystem))
        num_samples = state_samples.shape[0]
        losses = torch.empty((num_samples,), dtype=self.dtype)
        relu_at_equilibrium = dut.lyapunov_relu(dut.system.x_equilibrium)
        for i in range(num_samples):
            losses[i] = torch.min(torch.tensor([dut.lyapunov_value(
                state_samples[i], dut.system.x_equilibrium, V_lambda,
                relu_at_equilibrium) -
                epsilon * torch.norm(
                    state_samples[i] - dut.system.x_equilibrium, p=1) - margin,
                0.]))
        return -torch.mean(losses)

    def test_lyapunov_positivity_loss_at_samples1(self):
        state_samples = torch.tensor(
            [[0.5, 0.5], [-0.5, -0.5], [1, 1], [1, -1], [-1, 1], [-1, -1]],
            dtype=self.dtype)
        relu_at_equilibrium = self.lyapunov_relu1(
            self.closed_loop_system1.x_equilibrium)
        V_lambda = 0.1
        epsilon = 0.3
        margin = 0.2
        positivity_sample_loss = \
            self.lyapunov_hybrid_system1.lyapunov_positivity_loss_at_samples(
                relu_at_equilibrium, self.closed_loop_system1.x_equilibrium,
                state_samples, V_lambda, epsilon, margin)
        self.assertAlmostEqual(
            positivity_sample_loss.item(),
            self.compute_lyapunov_positivity_loss_at_samples(
                self.lyapunov_hybrid_system1, state_samples, V_lambda, epsilon,
                margin).item())

        optimizer = torch.optim.Adam(list(
            self.controller_network1.parameters()) +
            list(self.lyapunov_relu1.parameters()))
        optimizer = torch.optim.Adam(self.lyapunov_relu1.parameters())
        for iter_count in range(2):
            optimizer.zero_grad()
            relu_at_equilibrium = self.lyapunov_relu1(
                self.closed_loop_system1.x_equilibrium)
            loss = self.lyapunov_hybrid_system1.\
                lyapunov_positivity_loss_at_samples(
                    relu_at_equilibrium,
                    self.closed_loop_system1.x_equilibrium, state_samples,
                    V_lambda, epsilon, margin)
            loss.backward()
            optimizer.step()

    def compute_lyapunov_derivative_loss_at_samples(
            self, dut, state_samples, V_lambda, epsilon, margin):
        assert(isinstance(dut, lyapunov.LyapunovHybridLinearSystem))
        state_samples_next = torch.stack([dut.system.step_forward(
            state_samples[i]) for i in range(state_samples.shape[0])], dim=0)
        losses = torch.empty((state_samples.shape[0]), dtype=self.dtype)
        relu_at_equilibrium = dut.lyapunov_relu(dut.system.x_equilibrium)
        for i in range(state_samples.shape[0]):
            V = dut.lyapunov_value(
                state_samples[i], dut.system.x_equilibrium, V_lambda,
                relu_at_equilibrium)
            # Note, by using torch.tensor([]), we cannot do auto grad on
            # `losses[i]`. In order to do automatic differentiation, change
            # torch.max to torch.nn.HingeEmbeddingLoss
            losses[i] = torch.max(torch.tensor([dut.lyapunov_value(
                state_samples_next[i], dut.system.x_equilibrium, V_lambda,
                relu_at_equilibrium) - V + epsilon * V + margin, 0]))
        return torch.mean(losses)

    def lyapunov_derivative_loss_at_sample_tester(
            self, dut, training_params, state_samples):
        assert(isinstance(dut, lyapunov.LyapunovHybridLinearSystem))
        V_lambda = 0.1
        epsilon = 0.2
        margin = 0.3
        loss = dut.lyapunov_derivative_loss_at_samples(
            V_lambda, epsilon, state_samples, dut.system.x_equilibrium, margin)
        self.assertAlmostEqual(
            loss.item(), self.compute_lyapunov_derivative_loss_at_samples(
                dut, state_samples, V_lambda, epsilon, margin).item())

        optimizer = torch.optim.Adam(list(
            self.controller_network1.parameters()) +
            list(self.lyapunov_relu1.parameters()))
        for iter_count in range(2):
            optimizer.zero_grad()
            loss = dut.lyapunov_derivative_loss_at_samples(
                V_lambda, epsilon, state_samples, dut.system.x_equilibrium,
                margin)
            loss.backward()
            optimizer.step()

    def test_lyapunov_derivative_loss_at_samples1(self):
        state_samples = torch.tensor(
            [[0.5, 0.5], [-0.5, -0.5], [1, 1], [1, -1], [-1, 1], [-1, -1]],
            dtype=self.dtype)
        training_params = list(self.controller_network1.parameters()) +\
            list(self.lyapunov_relu1.parameters())
        self.lyapunov_derivative_loss_at_sample_tester(
            self.lyapunov_hybrid_system1, training_params, state_samples)

    def test_lyapunov_derivative_loss_at_samples2(self):
        state_samples = torch.tensor(
            [[0.5, 0.5, 0.2, 0.4], [-0.5, -0.5, 0.1, 0.6], [1, 1, -2, -2],
             [1, -1, 0.4, 0.9], [-1, 1, 0.4, 0.2], [-1, -1, -0.2, 0.4]],
            dtype=self.dtype)
        training_params = list(self.controller_network2.parameters()) +\
            list(self.lyapunov_relu2.parameters())
        self.lyapunov_derivative_loss_at_sample_tester(
            self.lyapunov_hybrid_system2, training_params, state_samples)

    def test_total_loss1(self):
        state_samples = torch.tensor(
            [[0.5, 0.5], [-0.5, -0.5], [1, 1], [1, -1], [-1, 1], [-1, -1]],
            dtype=self.dtype)
        state_samples_next = torch.stack([
            self.lyapunov_hybrid_system1.system.step_forward(
                state_samples[i]) for i in range(state_samples.shape[0])],
            dim=0)
        V_lambda = 0.1
        trainer = train_lyapunov.TrainLyapunovReLU(
            self.lyapunov_hybrid_system1, V_lambda,
            self.lyapunov_hybrid_system1.system.x_equilibrium)
        optimizer = torch.optim.Adam(list(
            self.controller_network1.parameters()) +
            list(self.lyapunov_relu1.parameters()))
        for iter_count in range(2):
            optimizer.zero_grad()
            state_samples_next = torch.stack([
                self.lyapunov_hybrid_system1.system.step_forward(
                    state_samples[i]) for i in range(state_samples.shape[0])],
                dim=0)
            loss = trainer.total_loss(
                state_samples, state_samples, state_samples_next, 1., 1., None,
                None)
            loss[0].backward()


if __name__ == "__main__":
    unittest.main()
