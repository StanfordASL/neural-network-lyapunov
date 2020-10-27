import torch
import numpy as np

import neural_network_lyapunov.worlds as worlds
import neural_network_lyapunov.encoders as encoders


x_dim = 12
z_dim = 10
dtype = torch.float64
default = dict(
    dtype=dtype,

    # data
    world_cb=worlds.get_load_cluttered_table_callback(),
    joint_space=False,
    camera_eye_position=[0, -.5, .88],
    camera_target_position=[0, 0, .725],
    camera_up_vector=[0, 0, 1],
    grayscale=False,
    image_width=92,
    image_height=92,

    dataset_x_lo=torch.tensor([-.08, -.04, .25 + .625 - .07,
                               -np.pi/4, -np.pi/4, -np.pi/4,
                               0, 0, 0, -5, -5, -5], dtype=dtype),
    dataset_x_up=torch.tensor([.08, .04, .25 + .625,
                               np.pi/4, np.pi/4, np.pi/4,
                               0, 0, 0, 5, 5, 5], dtype=dtype),
    dataset_noise=torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                               dtype=dtype),
    dataset_dt=.1,
    dataset_N=5,
    dataset_num_rollouts=250,
    dataset_num_val_rollouts=10,

    batch_size=60,

    long_horizon_N=25,
    long_horizon_num_val_rollouts=25,

    # state space
    x_dim=x_dim,
    x_equilibrium=torch.tensor([0, 0, .25 + .625, 0, 0,
                                0, 0, 0, 0, 0, 0, 0], dtype=dtype),
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

    encoder_class=encoders.CNNEncoder2,
    decoder_class=encoders.CNNDecoder2,
    use_bce=True,
    use_variational=True,
    kl_loss_weight=1e-5,

    latent_dyn_nn_width=(z_dim,) + (20, 20,) + (z_dim,),
    latent_lyap_nn_width=(z_dim,) + (20, 20,) + (1,),
)

variants = dict(
    unstable=dict(
        device="cuda:0",
        lyap_loss_optimal=False,
        lyap_loss_freq=0,
        lyap_pos_loss_at_samples_weight=0.,
        lyap_der_loss_at_samples_weight=0.,
        lyap_pos_loss_weight=0.,
        lyap_der_loss_weight=0.,
    ),
    stable=dict(
        device="cuda:1",
        lyap_loss_optimal=False,
        lyap_loss_freq=10,
        lyap_pos_loss_at_samples_weight=.1,
        lyap_der_loss_at_samples_weight=.1,
        lyap_pos_loss_weight=.1,
        lyap_der_loss_weight=.1,
    ),
    stable_samples=dict(
        device="cuda:0",
        lyap_loss_optimal=False,
        lyap_loss_freq=0,
        lyap_pos_loss_at_samples_weight=.1,
        lyap_der_loss_at_samples_weight=.1,
        lyap_pos_loss_weight=0.,
        lyap_der_loss_weight=0.,
    ),
)
