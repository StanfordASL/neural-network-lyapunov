import double_integrator
import robust_value_approx.control_lyapunov as control_lyapunov
import torch.nn as nn
import torch
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # noqa

"""
This file exercise whole control lyapunov workflow on a toy double integrator
system:
    1. Find the LQR cost-to-go for the system.
    2. Use ReLU network to do regression on some sampled states, and the
       corresponding cost-to-go.
    3. Verify that the ReLU network satisfies the control Lyapunov condition.
"""


def generate_cost_to_go_mesh(num_samples, x_lo, x_up):
    """
    We will generate a mesh of states within a range, and evaluate the
    cost-to-go for each of the sampled state.
    @param num_samples A length 2 array. num_samples[i] is the number of
    samples along x[i]
    @param x_lo The lower bound of the sample region.
    @param x_up The upper bound of the sample region.
    @return (x_samples, cost_samples) x_sample is a 2 x N numpy matrix, where
    N is the number of samples, cost_samples is a 1 x N numpy vector.
    cost_samples[:, i] is the cost-to-go for x_samples[:, i]
    """
    assert(len(num_samples) == 2)
    assert(x_lo.numel() == 2)
    assert(x_up.numel() == 2)
    assert(torch.all(x_lo <= x_up))
    (pos_mesh, vel_mesh) = np.meshgrid(np.linspace(
        x_lo[0], x_up[0], num_samples[0]),
        np.linspace(x_lo[1], x_up[1], num_samples[1]))
    x_samples = np.vstack((np.reshape(pos_mesh, -1), np.reshape(vel_mesh, -1)))
    Q = torch.tensor([[1, 0], [0, 10]], dtype=torch.float64)
    R = torch.tensor(2, dtype=torch.float64)
    (K, P) = double_integrator.double_integrator_lqr(Q, R)

    cost_samples = (np.sum((P.numpy().dot(x_samples))
                           * x_samples, axis=0)).reshape(1, -1)

    return (x_samples, cost_samples)


def draw_cost_to_go(ax, x_lo, x_up):
    (x_samples, cost_samples) = generate_cost_to_go_mesh([10, 20], x_lo, x_up)
    ax.plot_surface(x_samples[0].reshape(20, 10), x_samples[1].reshape(
        20, 10), cost_samples.reshape(20, 10), rstride=1, cstride=1)


def relu_training(num_samples, x_lo, x_up):
    model = nn.Sequential(nn.Linear(2, 12), nn.ReLU(),
                          nn.Linear(12, 12), nn.ReLU(), nn.Linear(12, 1))
    model.double()
    loss_fn = nn.MSELoss(reduction="sum")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    # model.cuda()
    device = next(model.parameters()).device
    (x_samples, cost_samples) = \
        generate_cost_to_go_mesh(num_samples, x_lo, x_up)
    x_tensor = torch.from_numpy(x_samples.T).to(device)
    cost_tensor = torch.from_numpy(cost_samples.T).to(device)
    num_epoch = 1000
    for epoch in range(num_epoch):
        cost_pred = model(x_tensor)
        loss = loss_fn(cost_pred, cost_tensor)/cost_samples.size
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return model


def verify_control_lyapunov(model, x_lo, x_up):
    dtype = torch.float64
    verifier = control_lyapunov.ControlLyapunovFreeActivationPattern(
        model, dtype)
    (A_dyn, B_dyn) = double_integrator.double_integrator_dynamics(dtype)
    d_dyn = torch.tensor([[0], [0]], dtype=dtype)
    u_vertices = torch.tensor([[-10, 10]], dtype=dtype)

    (c1, c2, Ain1, Ain2, Ain3, Ain4, Ain5, rhs) =\
        verifier.generate_program_verify_continuous_affine_system(
        A_dyn, B_dyn, d_dyn, u_vertices, x_lo, x_up)

    x = cp.Variable(2)
    s = cp.Variable(c1.numel())
    t = cp.Variable()
    alpha = cp.Variable(Ain4.shape[1], boolean=True)
    beta = cp.Variable(Ain5.shape[1], boolean=True)
    constraint1 = Ain1.to_dense().detach().numpy() * x\
        + Ain2.to_dense().detach().numpy() * s\
        + Ain3.to_dense().detach().numpy().squeeze() * t\
        + Ain4.to_dense().detach().numpy() * alpha\
        + Ain5.to_dense().detach().numpy() * beta\
        <= rhs.squeeze()
    (Ain6, Ain7, Ain8, rhs_in, Aeq6, Aeq7, Aeq8, rhs_eq, _, _, _, _, _, _) =\
        verifier.relu_free_pattern.output_constraint(model, x_lo, x_up)
    z = cp.Variable(Ain7.shape[1])
    constraints = [
        constraint1, Ain6.detach().numpy() * x + Ain7.detach().numpy() * z
        + Ain8.detach().numpy() * beta <= rhs_in.squeeze().detach().numpy(),
        Aeq6.detach().numpy() * x + Aeq7.detach().numpy() * z +
        Aeq8.detach().numpy() * beta == rhs_eq.squeeze().detach().numpy()]
    objective = cp.Maximize(
        c1.detach().numpy().T * s + t + c2.detach().numpy().T * alpha)
    cp_prob = cp.Problem(objective, constraints)
    cp_prob.solve(solver=cp.GUROBI, verbose=True)
    if (cp_prob.value <= 0):
        print("Satisfies control Lyapunov.")
        return True


if __name__ == "__main__":
    x_lo = torch.tensor([-1, -1], dtype=torch.float64)
    x_up = torch.tensor([1, 1], dtype=torch.float64)
    fig = plt.figure()
    ax = fig.gca(projection="3d")
    draw_cost_to_go(ax, x_lo, x_up)
    plt.show()

    model = relu_training([5, 5], x_lo, x_up)

    verify_control_lyapunov(model, x_lo, x_up)
