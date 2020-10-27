import torch
import numpy as np

import neural_network_lyapunov.worlds as worlds
import neural_network_lyapunov.encoders as encoders


x_dim = 2
z_dim = 3
dtype = torch.float64
default = dict(
    dtype=dtype,

    # data
    world_cb=worlds.get_load_urdf_callback(worlds.urdf_path("pendulum.urdf")),
    joint_space=True,
    camera_eye_position=[0, -3, 0],
    camera_target_position=[0, 0, 0],
    camera_up_vector=[0, 0, 1],
    grayscale=True,
    image_width=48,
    image_height=48,

    dataset_x_lo=torch.tensor([0., -5], dtype=dtype),
    dataset_x_up=torch.tensor([2.*np.pi, 5], dtype=dtype),
    dataset_noise=torch.tensor([0.01, 0.01]),
    dataset_dt=.1,
    dataset_N=5,
    dataset_num_rollouts=100,
    dataset_num_val_rollouts=10,

    batch_size=100,

    long_horizon_N=250,
    long_horizon_num_val_rollouts=25,

    # state space
    x_dim=x_dim,
    x_equilibrium=torch.tensor([np.pi, 0], dtype=dtype),
    x_lo_stable=torch.tensor([np.pi/4, -2.5], dtype=dtype),
    x_up_stable=torch.tensor([np.pi + 3*np.pi/4, 2.5], dtype=dtype),

    V_lambda=0.,
    V_eps=0.001,

    dyn_nn_width=(x_dim,) + (10, 10,) + (x_dim,),
    lyap_nn_width=(x_dim,) + (10, 10,) + (1,),

    # image space
    z_dim=z_dim,
    z_equilibrium=torch.zeros(z_dim, dtype=dtype),
    z_lo_stable=-10.*torch.ones(z_dim, dtype=dtype),
    z_up_stable=10.*torch.ones(z_dim, dtype=dtype),

    encoder_class=encoders.LinearEncoder1,
    decoder_class=encoders.LinearDecoder1,
    use_bce=True,
    use_variational=True,
    kl_loss_weight=1e-5,
    decoded_equilibrium_loss_weight=1e-4,

    latent_dyn_nn_width=(z_dim,) + (10, 10,) + (z_dim,),
    latent_lyap_nn_width=(z_dim,) + (10, 10,) + (1,),
)

variants = dict(
    unstable=dict(
        device="cuda:0",
        lyap_loss_optimal=False,
        lyap_loss_warmstart=True,
        lyap_loss_freq=0,
        lyap_pos_loss_at_samples_weight=0.,
        lyap_der_loss_at_samples_weight=0.,
        lyap_pos_loss_weight=0.,
        lyap_der_loss_weight=0.,
    ),
    stable_samples=dict(
        device="cuda:0",
        lyap_loss_optimal=False,
        lyap_loss_warmstart=True,
        lyap_loss_freq=0,
        lyap_pos_loss_at_samples_weight=.1,
        lyap_der_loss_at_samples_weight=.1,
        lyap_pos_loss_weight=0.,
        lyap_der_loss_weight=0.,
    ),
    stable=dict(
        device="cuda:1",
        lyap_loss_optimal=True,
        lyap_loss_warmstart=False,
        lyap_loss_freq=10,
        lyap_pos_loss_at_samples_weight=.1,
        lyap_der_loss_at_samples_weight=.1,
        lyap_pos_loss_weight=.1,
        lyap_der_loss_weight=.1,
    ),
)
