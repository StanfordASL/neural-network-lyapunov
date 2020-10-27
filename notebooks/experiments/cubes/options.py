import torch
import numpy as np

import neural_network_lyapunov.worlds as worlds
import neural_network_lyapunov.encoders as encoders


x_dim = 12
z_dim = 6
dtype = torch.float64
default = dict(
    dtype=dtype,

    # data
    world_cb=worlds.get_load_falling_cubes_callback(),
    joint_space=False,
    camera_eye_position=[0, -.5, .15],
    camera_target_position=[0, 0, .1],
    camera_up_vector=[0, 0, 1],
    grayscale=False,
    image_width=72,
    image_height=72,

    dataset_x_lo=torch.tensor([-.1, 0, .1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              dtype=dtype),
    dataset_x_up=torch.tensor([.1, 0, .15, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              dtype=dtype),
    dataset_noise=.1,

    dataset_dt=.1,
    dataset_N=5,
    dataset_num_rollouts=25,
    dataset_num_val_rollouts=10,

    batch_size=50,

    long_horizon_N=20,
    long_horizon_num_val_rollouts=10,

    # state space
    x_dim=x_dim,
    x_equilibrium=torch.tensor([0, 0, .15, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                               dtype=dtype),
    x_lo_stable=None,
    x_up_stable=None,

    V_lambda=0.,
    V_eps=0.,

    dyn_nn_width=None,
    lyap_nn_width=None,

    # image space
    z_dim=z_dim,
    z_equilibrium=torch.zeros(z_dim, dtype=dtype),
    z_lo_stable=-10.*torch.ones(z_dim, dtype=dtype),
    z_up_stable=10.*torch.ones(z_dim, dtype=dtype),

    encoder_class=encoders.LinearEncoder1,
    decoder_class=encoders.LinearDecoder1,
    use_bce=True,
    use_variational=True,
    kl_loss_weight=1e-3,
    decoded_equilibrium_loss_weight=0.,

    latent_dyn_nn_width=(z_dim,) + (20, 20,) + (z_dim,),
    latent_lyap_nn_width=(z_dim,) + (20, 20,) + (1,),
)

variants = dict(
    unstable=dict(
        device="cpu",
        lyap_loss_optimal=False,
        lyap_loss_warmstart=True,
        lyap_loss_freq=0,
        lyap_pos_loss_at_samples_weight=0.,
        lyap_der_loss_at_samples_weight=0.,
        lyap_pos_loss_weight=0.,
        lyap_der_loss_weight=0.,
    ),
    stable_samples=dict(
        device="cpu",
        lyap_loss_optimal=False,
        lyap_loss_warmstart=True,
        lyap_loss_freq=0,
        lyap_pos_loss_at_samples_weight=1.,
        lyap_der_loss_at_samples_weight=1.,
        lyap_pos_loss_weight=0.,
        lyap_der_loss_weight=0.,
    ),
    stable=dict(
        device="cpu",
        lyap_loss_optimal=False,
        lyap_loss_warmstart=True,
        lyap_loss_freq=5,
        lyap_pos_loss_at_samples_weight=1.,
        lyap_der_loss_at_samples_weight=1.,
        lyap_pos_loss_weight=1.,
        lyap_der_loss_weight=1.,
    ),
)
