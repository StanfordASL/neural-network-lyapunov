import neural_network_lyapunov.examples.pole.pole as pole
import neural_network_lyapunov.utils as utils
import neural_network_lyapunov.integrator as integrator

import torch
import argparse


def generate_dataset(plant, dt):
    x_lo = torch.tensor([-plant.length, -plant.length, -2, -2, -1, -2, -2],
                        dtype=plant.dtype)
    x_up = -x_lo
    x_samples = utils.uniform_sample_in_box(x_lo, x_up, 500000)
    # Keep only state samples that |x_AB² + y_AB²] < 0.7 * length (azimuth
    # angle larger than 45 degrees).
    x_samples = x_samples[torch.norm(x_samples[:, :2], p=2, dim=1) < 0.7 *
                          plant.length, :]
    u_lo = (plant.m_sphere + plant.m_ee) * plant.gravity * torch.tensor(
        [-1, -1, 0.5], dtype=plant.dtype)
    u_up = (plant.m_sphere + plant.m_ee) * plant.gravity * torch.tensor(
        [1, 1, 1.5], dtype=plant.dtype)
    u_samples = utils.uniform_sample_in_box(u_lo, u_up, x_samples.shape[0])
    x_next_samples = torch.empty_like(x_samples, dtype=plant.dtype)
    with torch.no_grad():
        for i in range(x_samples.shape[0]):

            def controller(x):
                return u_samples[i]

            x_next_samples[i] = integrator.rk4_constant_control(
                plant.dynamics,
                controller,
                x_samples[i],
                dt,
                constant_control_steps=1)[0]

    # x_next *might* contain Nan if x_AB² + y_AB² > l in the next state.
    x_next_is_nan = torch.any(torch.isnan(x_next_samples), dim=1)
    xu_samples = torch.cat((x_samples, u_samples), dim=1)

    return torch.utils.data.TensorDataset(xu_samples[~x_next_is_nan, :],
                                          x_next_samples[~x_next_is_nan, :])


def learn_relu_dynamics(dynamics_model, dynamics_dataset, u_z_equilibrium,
                        num_epochs, lr):
    xu_inputs, x_next_outputs = dynamics_dataset[:]

    def compute_v_next(model, xu):
        return dynamics_model(torch.cat(
            (xu[:, :2], xu[:, 5:]), dim=1)) - dynamics_model(
                torch.cat((torch.zeros(6, dtype=torch.float64),
                           torch.tensor([u_z_equilibrium],
                                        dtype=torch.float64)))) + xu[:, 2:7]

    dataset = torch.utils.data.TensorDataset(xu_inputs, x_next_outputs[:, 2:])
    utils.train_approximator(dataset,
                             dynamics_model,
                             compute_v_next,
                             batch_size=50,
                             num_epochs=num_epochs,
                             lr=lr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="pole dynamics training")
    parser.add_argument("--generate_dynamics_data", action="store_true")
    parser.add_argument("--load_dynamics_data",
                        type=str,
                        default=None,
                        help="path to load dynamics data")
    parser.add_argument("--train_dynamics_model",
                        type=str,
                        default=None,
                        help="path to save trained dynamics_model")
    args = parser.parse_args()

    plant = pole.Pole(m_sphere=0.1649, m_ee=0.2, length=0.82)
    dt = 0.01
    if args.generate_dynamics_data:
        data = generate_dataset(plant, dt)

    if args.load_dynamics_data is not None:
        data = torch.load(args.load_dynamics_data)

    if args.train_dynamics_model is not None:
        dynamics_relu = utils.setup_relu((7, 12, 12, 5),
                                         params=None,
                                         negative_slope=0.1,
                                         bias=True,
                                         dtype=torch.float64)
        u_z_equilibrium = (plant.m_sphere + plant.m_ee) * plant.gravity
        learn_relu_dynamics(dynamics_relu,
                            data,
                            u_z_equilibrium,
                            num_epochs=200,
                            lr=0.003)
        linear_layer_width, negative_slope, bias = \
            utils.extract_relu_structure(dynamics_relu)
        torch.save(
            {
                "linear_layer_width": linear_layer_width,
                "state_dict": dynamics_relu.state_dict(),
                "negative_slope": negative_slope,
                "bias": bias,
                "u_z_equilibrium": u_z_equilibrium,
                "dt": dt
            }, args.train_dynamics_model)

    pass
