import neural_network_lyapunov.examples.rl.td3 as td3
import neural_network_lyapunov.relu_system as relu_system
import neural_network_lyapunov.lyapunov as lyapunov
import neural_network_lyapunov.feedback_system as feedback_system
import neural_network_lyapunov.train_lyapunov as train_lyapunov
import neural_network_lyapunov.utils as utils
import neural_network_lyapunov.mip_utils as mip_utils
import neural_network_lyapunov.train_utils as train_utils
import neural_network_lyapunov.r_options as r_options

import torch
import numpy as np
import gurobipy
import argparse
import os
import gym

from neural_network_lyapunov.examples.rl.td3 import \
    MLPActorCritic, MLPActor, MLPQFunction  # noqa


def rotation_matrix(theta):
    c_theta = np.cos(theta)
    s_theta = np.sin(theta)
    return torch.tensor([[c_theta, -s_theta], [s_theta, c_theta]],
                        dtype=torch.float64)


def generate_gym_env_data(env):
    """
    Generate the pairs (x[n], u[n]) -> (x[n+1])
    """
    dtype = torch.float64
    xu_tensors = []
    x_next_tensors = []

    for i in range(1000):
        x0 = env.reset()
        u0 = env.action_space.sample()
        xn, _, _, _ = env.step(u0)
        x0 = torch.tensor(x0, dtype=dtype)
        u0 = torch.tensor(u0, dtype=dtype)
        xn = torch.tensor(xn, dtype=dtype)
        xu_tensors.append(torch.cat((x0, u0)).reshape((1, -1)))
        x_next_tensors.append(xn.reshape((1, -1)))

    dataset_input = torch.cat(xu_tensors, dim=0)
    dataset_output = torch.cat(x_next_tensors, dim=0)
    return torch.utils.data.TensorDataset(dataset_input, dataset_output)


def train_forward_model(dynamics_model, model_dataset,
                        state_equilibrium, control_equilibrium):
    def compute_next_x(model, state_action):
        return model(state_action) - model(
            torch.cat((state_equilibrium, control_equilibrium))) + \
            state_equilibrium

    utils.train_approximator(model_dataset,
                             dynamics_model,
                             compute_next_x,
                             batch_size=20,
                             num_epochs=100,
                             lr=0.001)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('env', type=str)
    parser.add_argument("--search_R",
                        action="store_true",
                        help="search R when searching for controller.")
    parser.add_argument(
        "--max_iterations",
        type=int,
        default=1000,
        help="max number of iterations in searching for controller.")
    parser.add_argument("--enable_wandb", action="store_true")
    args = parser.parse_args()

    env = gym.make(args.env)
    model_dataset = generate_gym_env_data(env)

    # needs to come from the env somehow?
    x_equilibrium = torch.tensor(
        [np.cos(np.pi), np.sin(np.pi), 0], dtype=torch.float64)
    u_equilibrium = torch.tensor([0], dtype=torch.float64)
    x_lo = torch.tensor([-1., -.3, -0.5], dtype=torch.float64)
    x_up = torch.tensor([-.95, .3, 0.5], dtype=torch.float64)
    u_lo = torch.tensor([-20], dtype=torch.float64)
    u_up = torch.tensor([20], dtype=torch.float64)

    # dynamics_model = utils.setup_relu((env.observation_space.shape[0] +
    #                                    env.action_space.shape[0], 5, 5,
    #                                    env.observation_space.shape[0]),
    #                                   params=None,
    #                                   negative_slope=0.01,
    #                                   bias=True,
    #                                   dtype=torch.float64)
    # train_forward_model(dynamics_model, model_dataset,
    #                     x_equilibrium, u_equilibrium)
    # torch.save(dynamics_model, 'gym_forward_model.pt')
    dynamics_model = torch.load('gym_forward_model.pt')

    V_lambda = 0.8

    actor_critic = torch.load('td3_Pendulum-v0_actor_critic.pt')
    controller_relu = actor_critic.actor.controller_relu.double()

    lyapunov_relu = utils.setup_relu((env.observation_space.shape[0],
                                      8, 8, 6, 1),
                                     params=None,
                                     negative_slope=0.1,
                                     bias=True,
                                     dtype=torch.float64)

    forward_system = relu_system.ReLUSystemGivenEquilibrium(
        torch.float64, x_lo, x_up, u_lo, u_up, dynamics_model,
        x_equilibrium, u_equilibrium)

    # closed_loop_system = feedback_system.ActorFeedbackSystem(
    closed_loop_system = feedback_system.FeedbackSystem(
        forward_system, controller_relu, forward_system.x_equilibrium,
        forward_system.u_equilibrium,
        u_lo.detach().numpy(),
        u_up.detach().numpy())

    lyapunov_hybrid_system = lyapunov.LyapunovDiscreteTimeHybridSystem(
        closed_loop_system, lyapunov_relu)

    R = 0.01 * torch.eye(3, dtype=torch.float64)
    if args.search_R:
        R_options = r_options.SearchRwithSPDOptions(R.shape, epsilon=0.01)
        R_options.set_variable_value(R.detach().numpy())
    else:
        R_options = r_options.FixedROptions(R)

    if args.enable_wandb:
        train_utils.wandb_config_update(args, lyapunov_relu, controller_relu,
                                        x_lo, x_up, u_lo, u_up)

    dut = train_lyapunov.TrainLyapunovReLU(lyapunov_hybrid_system, V_lambda,
                                           closed_loop_system.x_equilibrium,
                                           R_options)

    dut.lyapunov_positivity_mip_pool_solutions = 1
    dut.lyapunov_derivative_mip_pool_solutions = 1
    dut.lyapunov_derivative_convergence_tol = 1E-5
    dut.max_iterations = args.max_iterations
    dut.lyapunov_positivity_epsilon = 0.5
    dut.lyapunov_derivative_epsilon = 0.001
    dut.lyapunov_derivative_eps_type = lyapunov.ConvergenceEps.ExpLower

    dut.enable_wandb = args.enable_wandb
    dut.train(torch.empty((0, 3), dtype=torch.float64))

    pass
