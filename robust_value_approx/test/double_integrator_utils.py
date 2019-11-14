import double_integrator
import robust_value_approx.hybrid_linear_system as hybrid_linear_system
import robust_value_approx.value_to_optimization as value_to_optimization
import robust_value_approx.utils as utils

import sys
import torch
import torch.nn as nn


def generate_data():
    print("generating data...")

    dtype = torch.float64
    (A, B) = double_integrator.double_integrator_dynamics(dtype)
    x_dim = A.shape[1]
    u_dim = B.shape[1]
    sys = hybrid_linear_system.HybridLinearSystem(x_dim, u_dim, dtype)

    c = torch.zeros(x_dim, dtype=dtype)
    x_lo = -1. * torch.ones(x_dim, dtype=dtype)
    x_up = 1. * torch.ones(x_dim, dtype=dtype)
    u_lo = -1. * torch.ones(u_dim, dtype=dtype)
    u_up = 1. * torch.ones(u_dim, dtype=dtype)
    P = torch.cat((-torch.eye(x_dim+u_dim),
                   torch.eye(x_dim+u_dim)), 0).type(dtype)
    q = torch.cat((-x_lo, -u_lo, x_up, u_up), 0).type(dtype)
    sys.add_mode(A, B, c, P, q)

    # value function
    N = 5
    vf = value_to_optimization.ValueFunction(sys, N, x_lo, x_up, u_lo, u_up)
    R = torch.eye(sys.u_dim)
    vf.set_cost(R=R)
    vf.set_terminal_cost(Rt=R)
    xN = torch.ones(x_dim, dtype=dtype)
    vf.set_constraints(xN=xN)

    x0_lo = x_lo
    x0_up = x_up
    num_breaks = [20] * x_dim

    x_samples, v_samples = vf.get_sample_grid(x0_lo, x0_up, num_breaks)

    torch.save(x_samples, 'double_integrator_x_samples.pt')
    torch.save(v_samples, 'double_integrator_v_samples.pt')


def generate_model():
    print("training a model...")

    x_samples = torch.load('double_integrator_x_samples.pt')
    v_samples = torch.load('double_integrator_v_samples.pt')

    nn_width = 36
    model = nn.Sequential(nn.Linear(x_samples.shape[1], nn_width),
                          nn.ReLU(), nn.Linear(nn_width, nn_width),
                          nn.ReLU(), nn.Linear(nn_width, 1), nn.ReLU())
    model.double()

    utils.train_model(model, x_samples, v_samples,
                      num_epoch=1000, batch_size=100, learning_rate=1e-2)

    torch.save(model, 'double_integrator_model.pt')


if __name__ == "__main__":
    command = sys.argv[1]
    if command == "data":
        generate_data()
    elif command == "model":
        generate_model()
