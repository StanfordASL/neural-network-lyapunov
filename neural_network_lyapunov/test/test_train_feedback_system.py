"""
Test train Lyapunov function and the controller
"""
import neural_network_lyapunov.feedback_system as feedback_system
import neural_network_lyapunov.hybrid_linear_system as hybrid_linear_system
import neural_network_lyapunov.lyapunov as lyapunov
import neural_network_lyapunov.relu_system as relu_system
import neural_network_lyapunov.train_lyapunov as train_lyapunov
import neural_network_lyapunov.utils as utils
import neural_network_lyapunov.test.feedback_gradient_check as\
    feedback_gradient_check
import neural_network_lyapunov.r_options as r_options
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
    system.add_mode(torch.eye(2, dtype=dtype), torch.eye(2, dtype=dtype),
                    torch.tensor([0, 0], dtype=dtype), P,
                    torch.tensor([10, 10, 10, 10, 0, 10, 10, 10], dtype=dtype))
    system.add_mode(0.5 * torch.eye(2, dtype=dtype),
                    1.5 * torch.eye(2, dtype=dtype),
                    torch.tensor([0, 0], dtype=dtype), P,
                    torch.tensor([0, 10, 10, 10, 10, 10, 10, 10], dtype=dtype))
    return system


class TestLyapunov(unittest.TestCase):
    def setUp(self):
        self.dtype = torch.float64
        self._create_feedback_hybrid_system()
        self._create_feedback_second_order_system()
        self._create_linear_feedback_second_order_system()

    def _create_feedback_hybrid_system(self):
        # First system is a piecewise linear hybrid system. Neither the
        # controller network nor the Lyapunov network has bias terms.
        self.forward_system1 = create_hybrid_system1(self.dtype)
        self.controller_network1 = utils.setup_relu(
            (2, 4, 2),
            torch.tensor([
                0.1, 0.2, 0.3, 0.4, -0.5, 0.6, 0.7, 0.2, 0.5, 0.4, 1.5, 1.3,
                1.2, -0.8, 2.1, 0.4
            ],
                         dtype=self.dtype),
            negative_slope=0.01,
            bias=False,
            dtype=self.dtype)
        u_lower_limit = np.array([-10., -10.])
        u_upper_limit = np.array([10., 10.])
        self.closed_loop_system1 = feedback_system.FeedbackSystem(
            self.forward_system1,
            self.controller_network1,
            x_equilibrium=torch.tensor([0, 0], dtype=self.dtype),
            u_equilibrium=torch.tensor([0, 0], dtype=self.dtype),
            u_lower_limit=u_lower_limit,
            u_upper_limit=u_upper_limit)
        self.lyapunov_relu1 = utils.setup_relu(
            (2, 4, 4, 1),
            torch.tensor([
                0.1, 0.2, 0.3, -0.1, 0.3, 0.9, 1.2, 0.4, -0.2, -0.5, 0.5, 0.9,
                1.2, 1.4, 1.6, 2.4, 2.1, -0.6, -0.9, -.8, 2.1, -0.5, -1.2, 2.1,
                0.5, 0.4, 0.6, 1.2
            ],
                         dtype=self.dtype),
            negative_slope=0.01,
            bias=False,
            dtype=self.dtype)
        self.lyapunov_hybrid_system1 = \
            lyapunov.LyapunovDiscreteTimeHybridSystem(
                self.closed_loop_system1, self.lyapunov_relu1)

    def _create_feedback_second_order_system(self):
        # A second order system with nq = nv = 2, nu = 1
        forward_network_params = torch.tensor([
            0.2, 0.4, 0.1, 0.5, 0.4, -0.2, 0.4, 0.5, 0.9, -0.3, 1.2, -2.1, 0.1,
            0.45, 0.2, 0.8, 0.7, 0.3, 0.2, 0.5, 0.4, 0.8, 2.1, 0.4, 0.5, 0.2,
            -0.4, -0.5, 0.3, -2.1, 0.4, 0.2, 0.1, 0.5
        ],
                                              dtype=self.dtype)
        forward_network = utils.setup_relu((5, 4, 2),
                                           forward_network_params,
                                           negative_slope=0.01,
                                           bias=True,
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
            -0.5
        ],
                                                  dtype=self.dtype)
        self.controller_network2 = utils.setup_relu((4, 3, 2, 1),
                                                    controller_network_params2,
                                                    negative_slope=0.01,
                                                    bias=True,
                                                    dtype=self.dtype)
        self.closed_loop_system2 = feedback_system.FeedbackSystem(
            self.forward_system2,
            self.controller_network2,
            x_equilibrium=self.forward_system2.x_equilibrium,
            u_equilibrium=self.forward_system2.u_equilibrium,
            u_lower_limit=np.array([-10.]),
            u_upper_limit=np.array([10.]))
        self.lyapunov_relu2 = utils.setup_relu(
            (4, 4, 4, 1),
            torch.tensor([
                0.1, 0.2, 0.3, -0.1, 0.3, 0.9, 1.2, 0.4, -0.2, -0.5, 0.5, 0.9,
                1.2, 1.4, 1.6, 2.4, 2.1, -0.6, -0.9, -.8, 2.1, -0.5, -1.2, 2.1,
                0.5, 0.4, 0.6, 1.2, 0.7, 0.1, -0.2, 0.3, -1.2, 0.5, 0.1, 0.8,
                0.7, 0.2, 0.1, -0.5, 0.3, 0.6, -0.2, 1.2, 0.1
            ],
                         dtype=self.dtype),
            negative_slope=0.01,
            bias=True,
            dtype=self.dtype)
        self.lyapunov_hybrid_system2 = \
            lyapunov.LyapunovDiscreteTimeHybridSystem(
                self.closed_loop_system2, self.lyapunov_relu2)

    def _create_linear_feedback_second_order_system(self):
        forward_network_params = torch.tensor([
            0.2, 0.4, 0.1, 0.5, 0.4, -0.2, 0.4, 0.5, 0.9, -0.3, 1.2, -2.1, 0.1,
            0.45, 0.2, 0.8, 0.7, 0.3, 0.2, 0.5, 0.4, 0.8, 2.1, 0.4, 0.5, 0.2,
            -0.4, -0.5, 0.3, -2.1, 0.4, 0.2, 0.1, 0.5
        ],
                                              dtype=self.dtype)
        forward_network = utils.setup_relu((5, 4, 2),
                                           forward_network_params,
                                           negative_slope=0.01,
                                           bias=True,
                                           dtype=self.dtype)
        q_equilibrium = torch.tensor([0.5, 0.4], dtype=self.dtype)
        u_equilibrium = torch.tensor([0.5], dtype=self.dtype)
        dt = 0.01
        self.forward_system3 = \
            relu_system.ReLUSecondOrderSystemGivenEquilibrium(
                self.dtype, torch.tensor([-2, -2, -4, -4], dtype=self.dtype),
                torch.tensor([2, 2, 4, 4], dtype=self.dtype),
                torch.tensor([-10], dtype=self.dtype),
                torch.tensor([10], dtype=self.dtype), forward_network,
                q_equilibrium, u_equilibrium, dt)
        self.controller_network3 = torch.nn.Linear(4, 1, bias=True)
        self.controller_network3.weight.data = torch.tensor(
            [[0.1, 0.4, 0.2, -0.5]], dtype=self.dtype)
        self.controller_network3.bias.data = torch.tensor([0.5],
                                                          dtype=self.dtype)
        self.closed_loop_system3 = feedback_system.FeedbackSystem(
            self.forward_system3,
            self.controller_network3,
            x_equilibrium=self.forward_system3.x_equilibrium,
            u_equilibrium=self.forward_system3.u_equilibrium,
            u_lower_limit=np.array([-10.]),
            u_upper_limit=np.array([10.]))
        self.lyapunov_relu3 = utils.setup_relu(
            (4, 4, 4, 1),
            torch.tensor([
                0.1, 0.2, 0.3, -0.1, 0.3, 0.9, 1.2, 0.4, -0.2, -0.5, 0.5, 0.9,
                1.2, 1.4, 1.6, 2.4, 2.1, -0.6, -0.9, -.8, 2.1, -0.5, -1.2, 2.1,
                0.5, 0.4, 0.6, 1.2, 0.7, 0.1, -0.2, 0.3, -1.2, 0.5, 0.1, 0.8,
                0.7, 0.2, 0.1, -0.5, 0.3, 0.6, -0.2, 1.2, 0.1
            ],
                         dtype=self.dtype),
            negative_slope=0.01,
            bias=True,
            dtype=self.dtype)
        self.lyapunov_hybrid_system3 = \
            lyapunov.LyapunovDiscreteTimeHybridSystem(
                self.closed_loop_system3, self.lyapunov_relu3)

    def compute_lyapunov_positivity_loss_at_samples(self, dut, state_samples,
                                                    V_lambda, epsilon, margin,
                                                    R):
        assert (isinstance(dut, lyapunov.LyapunovHybridLinearSystem))
        num_samples = state_samples.shape[0]
        losses = torch.empty((num_samples, ), dtype=self.dtype)
        for i in range(num_samples):
            losses[i] = torch.min(
                torch.tensor([
                    dut.lyapunov_value(state_samples[i],
                                       dut.system.x_equilibrium,
                                       V_lambda,
                                       R=R) -
                    epsilon * torch.norm(
                        R @ (state_samples[i] - dut.system.x_equilibrium), p=1)
                    - margin, 0.
                ]))
        return -torch.mean(losses)

    def test_lyapunov_positivity_loss_at_samples1(self):
        state_samples = torch.tensor(
            [[0.5, 0.5], [-0.5, -0.5], [1, 1], [1, -1], [-1, 1], [-1, -1]],
            dtype=self.dtype)
        V_lambda = 0.1
        epsilon = 0.3
        margin = 0.2
        R = torch.tensor([[1, 1], [-1, 1], [0, 1]], dtype=torch.float64)
        positivity_sample_loss = \
            self.lyapunov_hybrid_system1.lyapunov_positivity_loss_at_samples(
                self.closed_loop_system1.x_equilibrium,
                state_samples, V_lambda, epsilon, R=R, margin=margin)
        self.assertAlmostEqual(
            positivity_sample_loss.item(),
            self.compute_lyapunov_positivity_loss_at_samples(
                self.lyapunov_hybrid_system1, state_samples, V_lambda, epsilon,
                margin, R).item())

        optimizer = torch.optim.Adam(
            list(self.controller_network1.parameters()) +
            list(self.lyapunov_relu1.parameters()))
        optimizer = torch.optim.Adam(self.lyapunov_relu1.parameters())
        for iter_count in range(2):
            optimizer.zero_grad()
            loss = self.lyapunov_hybrid_system1.\
                lyapunov_positivity_loss_at_samples(
                    self.closed_loop_system1.x_equilibrium, state_samples,
                    V_lambda, epsilon, R=R, margin=margin)
            loss.backward()
            optimizer.step()

    def compute_lyapunov_derivative_loss_at_samples(self, dut, state_samples,
                                                    V_lambda, epsilon,
                                                    eps_type, margin, R):
        assert (isinstance(dut, lyapunov.LyapunovHybridLinearSystem))
        state_samples_next = torch.stack([
            dut.system.step_forward(state_samples[i])
            for i in range(state_samples.shape[0])
        ],
                                         dim=0)
        losses = torch.empty((state_samples.shape[0]), dtype=self.dtype)
        for i in range(state_samples.shape[0]):
            V = dut.lyapunov_value(state_samples[i],
                                   dut.system.x_equilibrium,
                                   V_lambda,
                                   R=R)
            # Note, by using torch.tensor([]), we cannot do auto grad on
            # `losses[i]`. In order to do automatic differentiation, change
            # torch.max to torch.nn.HingeEmbeddingLoss
            V_next = dut.lyapunov_value(state_samples_next[i],
                                        dut.system.x_equilibrium,
                                        V_lambda,
                                        R=R)
            if eps_type == lyapunov.ConvergenceEps.ExpLower:
                losses[i] = torch.max(
                    torch.tensor([V_next - V + epsilon * V + margin, 0]))
            elif eps_type == lyapunov.ConvergenceEps.ExpUpper:
                losses[i] = torch.max(
                    torch.tensor([-(V_next - V + epsilon * V) + margin, 0]))
            elif eps_type == lyapunov.ConvergenceEps.Asymp:
                losses[i] = torch.max(
                    torch.tensor([(V_next - V + epsilon * torch.norm(
                        R @ (state_samples[i] - dut.system.x_equilibrium), p=1)
                                   ) + margin, 0]))

        return torch.mean(losses)

    def lyapunov_derivative_loss_at_sample_tester(self, dut, R,
                                                  training_params,
                                                  state_samples, eps_type):
        assert (isinstance(dut, lyapunov.LyapunovHybridLinearSystem))
        V_lambda = 0.1
        epsilon = 0.2
        margin = 0.3
        loss = dut.lyapunov_derivative_loss_at_samples(
            V_lambda,
            epsilon,
            state_samples,
            dut.system.x_equilibrium,
            eps_type,
            R=R,
            margin=margin)
        self.assertAlmostEqual(
            loss.item(),
            self.compute_lyapunov_derivative_loss_at_samples(
                dut, state_samples, V_lambda, epsilon, eps_type, margin,
                R).item())

        optimizer = torch.optim.Adam(
            list(self.controller_network1.parameters()) +
            list(self.lyapunov_relu1.parameters()))
        for iter_count in range(2):
            optimizer.zero_grad()
            loss = dut.lyapunov_derivative_loss_at_samples(
                V_lambda,
                epsilon,
                state_samples,
                dut.system.x_equilibrium,
                eps_type,
                R=R,
                margin=margin)
            loss.backward()
            optimizer.step()

    def test_lyapunov_derivative_loss_at_samples1(self):
        state_samples = torch.tensor(
            [[0.5, 0.5], [-0.5, -0.5], [1, 1], [1, -1], [-1, 1], [-1, -1]],
            dtype=self.dtype)
        training_params = list(self.controller_network1.parameters()) +\
            list(self.lyapunov_relu1.parameters())
        R = torch.tensor([[1, 1], [-1, 1], [0, 1]], dtype=torch.float64)
        for eps_type in list(lyapunov.ConvergenceEps):
            self.lyapunov_derivative_loss_at_sample_tester(
                self.lyapunov_hybrid_system1, R, training_params,
                state_samples, eps_type)

    def test_lyapunov_derivative_loss_at_samples2(self):
        state_samples = torch.tensor(
            [[0.5, 0.5, 0.2, 0.4], [-0.5, -0.5, 0.1, 0.6], [1, 1, -2, -2],
             [1, -1, 0.4, 0.9], [-1, 1, 0.4, 0.2], [-1, -1, -0.2, 0.4]],
            dtype=self.dtype)
        R = torch.cat((torch.eye(4, dtype=torch.float64),
                       torch.tensor([[1, 1, 1, 1]], dtype=torch.float64)),
                      dim=0)
        training_params = list(self.controller_network2.parameters()) +\
            list(self.lyapunov_relu2.parameters())
        for eps_type in list(lyapunov.ConvergenceEps):
            self.lyapunov_derivative_loss_at_sample_tester(
                self.lyapunov_hybrid_system2, R, training_params,
                state_samples, eps_type)

    def test_lyapunov_derivative_loss_at_samples3(self):
        state_samples = torch.tensor(
            [[0.5, 0.5, 0.2, 0.4], [-0.5, -0.5, 0.1, 0.6], [1, 1, -2, -2],
             [1, -1, 0.4, 0.9], [-1, 1, 0.4, 0.2], [-1, -1, -0.2, 0.4]],
            dtype=self.dtype)
        R = torch.cat((torch.eye(4, dtype=torch.float64),
                       torch.tensor([[1, 1, 1, 1]], dtype=torch.float64)),
                      dim=0)
        training_params = list(self.controller_network3.parameters()) +\
            list(self.lyapunov_relu3.parameters())
        for eps_type in list(lyapunov.ConvergenceEps):
            self.lyapunov_derivative_loss_at_sample_tester(
                self.lyapunov_hybrid_system3, R, training_params,
                state_samples, eps_type)

    def total_loss_tester(self, lyap, R, training_params, state_samples):
        state_samples_next = torch.stack([
            lyap.system.step_forward(state_samples[i])
            for i in range(state_samples.shape[0])
        ],
                                         dim=0)
        V_lambda = 0.1
        trainer = train_lyapunov.TrainLyapunovReLU(lyap, V_lambda,
                                                   lyap.system.x_equilibrium,
                                                   r_options.FixedROptions(R))
        optimizer = torch.optim.Adam(training_params)
        for iter_count in range(2):
            optimizer.zero_grad()
            state_samples_next = torch.stack([
                lyap.system.step_forward(state_samples[i])
                for i in range(state_samples.shape[0])
            ],
                                             dim=0)
            loss_return = trainer.total_loss(state_samples, state_samples,
                                             state_samples_next, 1., 1., 1, 1,
                                             0.)
            loss_return.loss.backward()

    def test_total_loss1(self):
        state_samples = torch.tensor(
            [[0.5, 0.5], [-0.5, -0.5], [1, 1], [1, -1], [-1, 1], [-1, -1]],
            dtype=self.dtype)
        training_params = list(self.controller_network1.parameters()) +\
            list(self.lyapunov_relu1.parameters())
        R = torch.tensor([[1, 1], [-1, 1], [0, 1]], dtype=torch.float64)
        self.total_loss_tester(self.lyapunov_hybrid_system1, R,
                               training_params, state_samples)

    def test_total_loss3(self):
        state_samples = torch.tensor(
            [[0.5, 0.5, 0.2, 0.4], [-0.5, -0.5, 0.1, 0.6], [1, 1, -2, -2],
             [1, -1, 0.4, 0.9], [-1, 1, 0.4, 0.2], [-1, -1, -0.2, 0.4]],
            dtype=self.dtype)
        training_params = list(self.controller_network3.parameters()) +\
            list(self.lyapunov_relu3.parameters())
        R = torch.tensor([[1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1],
                          [1, 0, 0, 1], [0, 0, 0, 1]],
                         dtype=torch.float64)
        self.total_loss_tester(self.lyapunov_hybrid_system3, R,
                               training_params, state_samples)


class TestGradient(unittest.TestCase):
    # Tests the gradient of the loss in train_lyapunov.py
    def construct_lyap1(self):
        torch.manual_seed(0)
        dtype = torch.float64
        x_lo = torch.tensor([-2, -4], dtype=dtype)
        x_up = torch.tensor([3, 5], dtype=dtype)
        u_lo = torch.tensor([-5, -4], dtype=dtype)
        u_up = torch.tensor([3, 6], dtype=dtype)
        forward_network = utils.setup_relu((4, 4, 3, 2),
                                           params=None,
                                           negative_slope=0.1,
                                           bias=True,
                                           dtype=dtype)
        x_equilibrium = torch.tensor([0.5, 1.2], dtype=dtype)
        u_equilibrium = torch.tensor([0.4, 1.5], dtype=dtype)
        forward_system = relu_system.ReLUSystemGivenEquilibrium(
            dtype, x_lo, x_up, u_lo, u_up, forward_network, x_equilibrium,
            u_equilibrium)
        lyapunov_relu = utils.setup_relu((2, 3, 1),
                                         params=None,
                                         negative_slope=0.1,
                                         bias=True,
                                         dtype=dtype)
        controller_relu = utils.setup_relu((2, 4, 2),
                                           params=None,
                                           negative_slope=0.1,
                                           bias=True,
                                           dtype=dtype)
        closed_loop_system = feedback_system.FeedbackSystem(
            forward_system, controller_relu, x_equilibrium, u_equilibrium,
            u_lo.detach().numpy(),
            u_up.detach().numpy())
        lyap = lyapunov.LyapunovDiscreteTimeHybridSystem(
            closed_loop_system, lyapunov_relu)
        return lyap

    def test_sample_loss_grad(self):
        # Test with sample loss gradient
        lyap1 = self.construct_lyap1()
        R_val = torch.tensor([[1.5, 0.3], [-0.2, 3.1]],
                             dtype=lyap1.system.dtype)
        fixed_R_options = r_options.FixedROptions(R_val)
        V_lambda = 0.5
        x_samples = utils.uniform_sample_in_box(
            torch.from_numpy(lyap1.system.forward_system.x_lo_all),
            torch.from_numpy(lyap1.system.forward_system.x_up_all), 100)
        # Test with fixed R
        feedback_gradient_check.check_sample_loss_grad(
            lyap1,
            V_lambda,
            lyap1.system.x_equilibrium,
            fixed_R_options,
            x_samples,
            atol=1E-5,
            rtol=1E-5)
        # Test with free R.
        torch.manual_seed(0)
        search_R_options = r_options.SearchRwithSPDOptions((4, 2), 0.1)
        search_R_options.set_variable_value_directly(
            np.array([0.1, 0.5, 0.3, 0.2, -0.4, -2.1, -3.2]))
        feedback_gradient_check.check_sample_loss_grad(
            lyap1,
            V_lambda,
            lyap1.system.x_equilibrium,
            search_R_options,
            x_samples,
            atol=1E-5,
            rtol=1E-5)

    def test_positivity_mip_loss_fixed_R(self):
        lyap1 = self.construct_lyap1()
        V_lambda = 0.5
        V_epsilon = 0.3
        fixed_R_options = r_options.FixedROptions(
            torch.tensor([[0.4, 0.2], [1.3, 2.1], [-0.5, -1.4]],
                         dtype=lyap1.system.dtype))
        feedback_gradient_check.check_lyapunov_mip_loss_grad(
            lyap1,
            lyap1.system.x_equilibrium,
            V_lambda,
            V_epsilon,
            fixed_R_options,
            True,
            atol=1E-5,
            rtol=1E-5)

    def test_positivity_mip_loss_search_R(self):
        lyap1 = self.construct_lyap1()
        V_lambda = 0.5
        V_epsilon = 0.3
        torch.manual_seed(0)
        search_R_options = r_options.SearchRwithSPDOptions((4, 2), 0.2)
        search_R_options.set_variable_value_directly(
            np.array([0.1, 0.5, 0.3, 0.2, -0.4, -2.1, -3.2]))
        feedback_gradient_check.check_lyapunov_mip_loss_grad(
            lyap1,
            lyap1.system.x_equilibrium,
            V_lambda,
            V_epsilon,
            search_R_options,
            True,
            atol=1E-5,
            rtol=1E-5)

    def test_derivative_mip_loss_fixed_R(self):
        lyap1 = self.construct_lyap1()
        V_lambda = 0.5
        V_epsilon = 0.3
        fixed_R_options = r_options.FixedROptions(
            torch.tensor([[0.4, 0.2], [1.3, 2.1], [-0.5, -1.4]],
                         dtype=lyap1.system.dtype))
        feedback_gradient_check.check_lyapunov_mip_loss_grad(
            lyap1,
            lyap1.system.x_equilibrium,
            V_lambda,
            V_epsilon,
            fixed_R_options,
            False,
            atol=1E-5,
            rtol=1E-5)

    def test_derivative_mip_loss_search_R_spd(self):
        lyap1 = self.construct_lyap1()
        V_lambda = 0.5
        V_epsilon = 0.3
        torch.manual_seed(0)
        search_R_options = r_options.SearchRwithSPDOptions((4, 2), 0.2)
        search_R_options.set_variable_value_directly(
            np.array([0.1, 0.5, 0.3, 0.2, -0.4, -2.1, -3.2]))
        feedback_gradient_check.check_lyapunov_mip_loss_grad(
            lyap1,
            lyap1.system.x_equilibrium,
            V_lambda,
            V_epsilon,
            search_R_options,
            False,
            atol=1E-5,
            rtol=1E-5)

    def test_derivative_mip_loss_search_R_svd(self):
        lyap1 = self.construct_lyap1()
        V_lambda = 0.5
        V_epsilon = 0.3
        torch.manual_seed(0)
        R_val = np.array([[0.5, 0.3], [0.2, 0.1], [-0.5, 0.4], [0.3, 0.2]])
        _, Sigma_val, _ = np.linalg.svd(R_val)
        search_R_options = r_options.SearchRwithSVDOptions((4, 2),
                                                           Sigma_val * 0.9)
        search_R_options.set_variable_value(R_val)
        feedback_gradient_check.check_lyapunov_mip_loss_grad(
            lyap1,
            lyap1.system.x_equilibrium,
            V_lambda,
            V_epsilon,
            search_R_options,
            False,
            atol=1E-5,
            rtol=1E-5)


if __name__ == "__main__":
    unittest.main()
