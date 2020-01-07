import double_integrator
import robust_value_approx.hybrid_linear_system as hybrid_linear_system
import robust_value_approx.value_to_optimization as value_to_optimization
import robust_value_approx.utils as utils

import sys
import torch
import torch.nn as nn


def get_value_function():
    dt = 1.
    dtype = torch.float64
    (A_c, B_c) = double_integrator.double_integrator_dynamics(dtype)
    x_dim = A_c.shape[1]
    u_dim = B_c.shape[1]
    A = torch.eye(x_dim, dtype=dtype) + dt * A_c
    B = dt * B_c
    sys = hybrid_linear_system.HybridLinearSystem(x_dim, u_dim, dtype)
    c = torch.zeros(x_dim, dtype=dtype)
    P = torch.cat((-torch.eye(x_dim+u_dim),
                   torch.eye(x_dim+u_dim)), 0).type(dtype)
    x_lo = -10. * torch.ones(x_dim, dtype=dtype)
    x_up = 10. * torch.ones(x_dim, dtype=dtype)
    u_lo = -1. * torch.ones(u_dim, dtype=dtype)
    u_up = 1. * torch.ones(u_dim, dtype=dtype)
    q = torch.cat((-x_lo, -u_lo, x_up, u_up), 0).type(dtype)
    sys.add_mode(A, B, c, P, q)
    R = torch.eye(sys.u_dim)
    Q = torch.eye(sys.x_dim)
    N = 5
    vf = value_to_optimization.ValueFunction(sys, N, x_lo, x_up, u_lo, u_up)
    vf.set_cost(Q=Q, R=R)
    vf.set_terminal_cost(Qt=Q, Rt=R)
    return vf


def generate_data():
    print("generating data...")
    vf = get_value_function()
    x0_lo = -1 * torch.ones(vf.sys.x_dim, dtype=vf.dtype)
    x0_up = 1 * torch.ones(vf.sys.x_dim, dtype=vf.dtype)
    num_breaks = [50] * vf.sys.x_dim
    x_samples, v_samples = vf.get_value_sample_grid(x0_lo, x0_up, num_breaks)
    torch.save(x_samples, 'data/double_integrator_x_samples.pt')
    torch.save(v_samples, 'data/double_integrator_v_samples.pt')


def generate_model():
    print("training a model...")
    x_samples = torch.load('data/double_integrator_x_samples.pt')
    v_samples = torch.load('data/double_integrator_v_samples.pt')
    nn_width = 64
    model = nn.Sequential(nn.Linear(x_samples.shape[1], nn_width),
                          nn.ReLU(), nn.Linear(nn_width, nn_width),
                          nn.ReLU(), nn.Linear(nn_width, nn_width),
                          nn.ReLU(), nn.Linear(nn_width, 1))
    model.double()
    utils.train_model(model, x_samples, v_samples,
                      num_epoch=1000, batch_size=100, learning_rate=1e-3)
    torch.save(model, 'data/double_integrator_model.pt')


def generate_q_data():
    print("generating q data...")
    vf = get_value_function()
    x0_lo = -1 * torch.ones(vf.sys.x_dim, dtype=vf.dtype)
    x0_up = 1 * torch.ones(vf.sys.x_dim, dtype=vf.dtype)
    u0_lo = vf.u_lo.clone()
    u0_up = vf.u_up.clone()
    x_num_breaks = [10] * vf.sys.x_dim
    u_num_breaks = [10] * vf.sys.u_dim
    x_samples, u_samples, v_samples = vf.get_q_sample_grid(x0_lo, x0_up,
                                                           x_num_breaks,
                                                           u0_lo, u0_up,
                                                           u_num_breaks)
    print("Generated " + str(x_samples.shape[0]) + " datapoints")
    torch.save(x_samples, 'data/double_integrator_q_x_samples.pt')
    torch.save(u_samples, 'data/double_integrator_q_u_samples.pt')
    torch.save(v_samples, 'data/double_integrator_q_v_samples.pt')


def generate_q_model():
    print("training a q model...")
    x_samples = torch.load('data/double_integrator_q_x_samples.pt')
    u_samples = torch.load('data/double_integrator_q_u_samples.pt')
    v_samples = torch.load('data/double_integrator_q_v_samples.pt')
    xu_samples = torch.cat((x_samples, u_samples), 1)
    nn_width = 64
    model = nn.Sequential(nn.Linear(xu_samples.shape[1], nn_width),
                          nn.ReLU(), nn.Linear(nn_width, nn_width),
                          nn.ReLU(), nn.Linear(nn_width, nn_width),
                          nn.ReLU(), nn.Linear(nn_width, 1))
    model.double()
    utils.train_model(model, xu_samples, v_samples,
                      num_epoch=300, batch_size=100, learning_rate=1e-3)
    torch.save(model, 'data/double_integrator_q_model.pt')


if __name__ == "__main__":
    command = sys.argv[1]
    if command == "data":
        generate_data()
    elif command == "model":
        generate_model()
    elif command == "qdata":
        generate_q_data()
    elif command == "qmodel":
        generate_q_model()
