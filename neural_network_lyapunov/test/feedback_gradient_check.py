import torch
import numpy as np
import neural_network_lyapunov.utils as utils
import neural_network_lyapunov.train_lyapunov as train_lyapunov


def check_sample_loss_grad(
    lyapunov_hybrid_system, V_lambda, x_equilibrium, x_samples, atol,
        rtol):
    dut = train_lyapunov.TrainLyapunovReLU(
        lyapunov_hybrid_system, V_lambda, x_equilibrium)
    x_next_samples = torch.cat([dut.lyapunov_hybrid_system.system.step_forward(
        x_samples[i]).reshape((1, -1)) for i in range(x_samples.shape[0])],
        dim=0)
    total_loss = dut.total_loss(
        x_samples, x_samples, x_next_samples, 1., 1., None, None)[0]
    total_loss.backward()
    controller_grad = utils.extract_relu_parameters_grad(
        lyapunov_hybrid_system.system.controller_network)
    lyapunov_grad = utils.extract_relu_parameters_grad(
        lyapunov_hybrid_system.lyapunov_relu)

    def compute_loss(
            controller_params: np.ndarray, lyapunov_params: np.ndarray):
        with torch.no_grad():
            utils.update_relu_params(
                dut.lyapunov_hybrid_system.system.controller_network,
                torch.from_numpy(controller_params))
            utils.update_relu_params(
                dut.lyapunov_hybrid_system.lyapunov_relu,
                torch.from_numpy(lyapunov_params))
            x_next_samples = torch.cat([
                lyapunov_hybrid_system.system.step_forward(
                    x_samples[i]).reshape((1, -1)) for i in
                range(x_samples.shape[0])], dim=0)
            return dut.total_loss(
                x_samples, x_samples, x_next_samples, 1., 1., None, None
                )[0].item()

    grad_numerical = utils.compute_numerical_gradient(
        compute_loss, utils.extract_relu_parameters(
            lyapunov_hybrid_system.system.controller_network).detach().numpy(),
        utils.extract_relu_parameters(
            lyapunov_hybrid_system.lyapunov_relu).detach().numpy())
    np.testing.assert_allclose(
        controller_grad, grad_numerical[0], rtol=rtol, atol=atol)
    np.testing.assert_allclose(
        lyapunov_grad, grad_numerical[1], rtol=rtol, atol=atol)


def check_lyapunov_mip_loss_grad(
    lyapunov_hybrid_system, x_equilibrium, V_lambda, V_epsilon,
        positivity_flag, atol, rtol):
    dut = train_lyapunov.TrainLyapunovReLU(
        lyapunov_hybrid_system, V_lambda, x_equilibrium)
    x_dim = x_equilibrium.shape[0]
    x_samples = torch.empty((0, x_dim), dtype=torch.float64)
    x_next_samples = torch.empty((0, x_dim), dtype=torch.float64)
    if positivity_flag:
        dut.lyapunov_positivity_epsilon = V_epsilon
        total_loss = dut.total_loss(
            x_samples, x_samples, x_next_samples, 0., 0., 1., None)[0]
    else:
        dut.lyapunov_derivative_epsilon = V_epsilon
        total_loss = dut.total_loss(
            x_samples, x_samples, x_next_samples, 0., 0., None, 1.)[0]
    total_loss.backward()

    if not positivity_flag:
        controller_grad = utils.extract_relu_parameters_grad(
            lyapunov_hybrid_system.system.controller_network)
    lyapunov_grad = utils.extract_relu_parameters_grad(
        lyapunov_hybrid_system.lyapunov_relu)

    def compute_loss(
            controller_params: np.ndarray, lyapunov_params: np.ndarray):
        with torch.no_grad():
            utils.update_relu_params(
                dut.lyapunov_hybrid_system.system.controller_network,
                torch.from_numpy(controller_params))
            utils.update_relu_params(
                dut.lyapunov_hybrid_system.lyapunov_relu,
                torch.from_numpy(lyapunov_params))
            x_next_samples = torch.empty((0, x_dim), dtype=torch.float64)
            if positivity_flag:
                dut.lyapunov_positivity_epsilon = V_epsilon
                return dut.total_loss(
                    x_samples, x_samples, x_next_samples, 0., 0., 1.,
                    None)[0].item()
            else:
                dut.lyapunov_derivative_epsilon = V_epsilon
                return dut.total_loss(
                    x_samples, x_samples, x_next_samples, 0., 0., None,
                    1.)[0].item()

    grad_numerical = utils.compute_numerical_gradient(
        compute_loss, utils.extract_relu_parameters(
            lyapunov_hybrid_system.system.controller_network).detach().numpy(),
        utils.extract_relu_parameters(
            lyapunov_hybrid_system.lyapunov_relu).detach().numpy())
    if not positivity_flag:
        np.testing.assert_allclose(
            controller_grad.detach().numpy(), grad_numerical[0], rtol=rtol,
            atol=atol)
    np.testing.assert_allclose(
        lyapunov_grad.detach().numpy(), grad_numerical[1], rtol=rtol,
        atol=atol)
