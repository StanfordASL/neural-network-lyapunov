import unittest
import torch
import numpy as np
import os
import shutil

import neural_network_lyapunov.dynamics_learning as dynamics_learning
import neural_network_lyapunov.worlds as worlds
import neural_network_lyapunov.encoders as encoders
import neural_network_lyapunov.relu_system as relu_system
import neural_network_lyapunov.lyapunov as lyapunov
import neural_network_lyapunov.pybullet_data_generation as\
 pybullet_data_generation
import neural_network_lyapunov.utils as utils


z_dim = 2
dtype = torch.float64
opt_default = dict(
    dtype=dtype,

    # data
    world_cb=worlds.get_load_urdf_callback(worlds.urdf_path("pendulum.urdf")),
    joint_space=True,
    camera_eye_position=[0, -3, 0],
    camera_target_position=[0, 0, 0],
    camera_up_vector=[0, 0, 1],
    grayscale=True,
    image_width=64,
    image_height=64,

    x_dim=2,
    x_equilibrium=torch.tensor([np.pi, 0], dtype=dtype),

    dataset_x_lo=torch.tensor([0., -5.], dtype=dtype),
    dataset_x_up=torch.tensor([2.*np.pi, 5.], dtype=dtype),
    dataset_noise=torch.tensor([.1, .1]),
    dataset_dt=.1,
    dataset_N=1,
    dataset_num_rollouts=25,
    dataset_num_val_rollouts=10,

    batch_size=50,

    long_horizon_N=10,
    long_horizon_num_val_rollouts=10,

    V_lambda=0.,
    V_eps=0.1,

    # dynamics nn
    dyn_nn_width=(z_dim, z_dim*5, z_dim*3, z_dim),

    # lyapunov nn
    lyap_nn_width=(z_dim, z_dim*5, z_dim*3, 1),

    # encoder (image-space learning)
    encoder_class=encoders.CNNEncoder2,
    decoder_class=encoders.CNNDecoder2,
    use_bce=True,
    use_variational=False,
    kl_loss_weight=1.,
    decoded_equilibrium_loss_weight=1e-3,
    z_dim=z_dim,
    z_lo=-1.*torch.ones(z_dim, dtype=dtype),
    z_up=torch.ones(z_dim, dtype=dtype),
    z_equilibrium=torch.zeros(z_dim, dtype=dtype),
)
opt_variants = dict(
    unstable=dict(
        lyap_loss_optimal=True,
        lyap_loss_warmstart=False,
        lyap_loss_freq=0,
        lyap_pos_loss_at_samples_weight=0.,
        lyap_der_loss_at_samples_weight=0.,
        lyap_pos_loss_weight=0.,
        lyap_der_loss_weight=0.,
    ),
    stable=dict(
        lyap_loss_optimal=True,
        lyap_loss_warmstart=False,
        lyap_loss_freq=2,
        lyap_pos_loss_at_samples_weight=1.,
        lyap_der_loss_at_samples_weight=1.,
        lyap_pos_loss_weight=1.,
        lyap_der_loss_weight=1.,
    ),
)


class TestDynamicsLearning(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(123)
        opt = dynamics_learning.DynamicsLearningOptions(opt_default)
        opt.set_options(opt_variants["stable"])
        self.opt = opt
        self.x_data = torch.rand(
            (2*opt.batch_size, opt.x_dim), dtype=opt.dtype)
        self.x_next_data = torch.rand(
            (2*opt.batch_size, opt.x_dim), dtype=opt.dtype)
        if opt.grayscale:
            num_channels = 1
        else:
            num_channels = 3
        self.X_data = torch.rand(
            (2*opt.batch_size, 2*num_channels,
             opt.image_width, opt.image_height), dtype=opt.dtype)
        self.X_next_data = torch.rand(
            (2*opt.batch_size, num_channels,
             opt.image_width, opt.image_height), dtype=opt.dtype)
        self.x_train_dataloader = pybullet_data_generation.get_dataloader(
            self.x_data, self.x_next_data, opt.batch_size)
        self.x_validation_dataloader = pybullet_data_generation.get_dataloader(
            self.x_data, self.x_next_data, opt.batch_size)
        self.X_train_dataloader = pybullet_data_generation.get_dataloader(
            self.X_data, self.X_next_data, opt.batch_size)
        self.X_validation_dataloader = pybullet_data_generation.get_dataloader(
            self.X_data, self.X_next_data, opt.batch_size)
        self.X_equilibrium = torch.rand(
            (2*num_channels, opt.image_width, opt.image_height),
            dtype=opt.dtype)
        self.dyn_nn_model = utils.setup_relu(
            opt.dyn_nn_width, negative_slope=0., dtype=opt.dtype)
        self.lyap_nn_model = utils.setup_relu(
            opt.lyap_nn_width, negative_slope=0., dtype=opt.dtype)
        self.relu_sys = relu_system.AutonomousReLUSystemGivenEquilibrium(
            opt.dtype, opt.dataset_x_lo, opt.dataset_x_up,
            self.dyn_nn_model, opt.x_equilibrium)
        self.lyap = lyapunov.LyapunovDiscreteTimeHybridSystem(
            self.relu_sys, self.lyap_nn_model)
        self.ss_dyn_learner = dynamics_learning.StateSpaceDynamicsLearning(
            self.x_train_dataloader, self.x_validation_dataloader,
            self.lyap, opt)
        self.encoder = opt.encoder_class(
            opt.z_dim, opt.image_width, opt.image_height, opt.grayscale)
        self.decoder = opt.decoder_class(
            opt.z_dim, opt.image_width, opt.image_height, opt.grayscale)
        self.latent_dyn_learner = dynamics_learning.\
            LatentSpaceDynamicsLearning(
                self.X_train_dataloader, self.X_validation_dataloader,
                self.lyap, opt,
                self.encoder, self.decoder,
                decoded_equilibrium=self.X_equilibrium)

    def test_lyapunov_loss(self):
        for dyn_learner, data in [(self.ss_dyn_learner, self.x_data),
                                  (self.latent_dyn_learner, self.X_data)]:
            self.opt.set_option("lyap_loss_optimal", True)
            lyap_pos_loss, lyap_der_loss = dyn_learner.lyapunov_loss()
            self.opt.set_option("lyap_loss_optimal", False)
            lyap_pos_loss_sub, lyap_der_loss_sub = dyn_learner.lyapunov_loss()
            lyap_pos_loss_samp, lyap_der_loss_samp = dyn_learner.\
                lyapunov_loss_at_samples(data)
            self.assertLessEqual(lyap_pos_loss_samp.item(),
                                 lyap_pos_loss.item())
            self.assertLessEqual(lyap_pos_loss_sub.item(),
                                 lyap_pos_loss.item())
            self.assertGreaterEqual(lyap_pos_loss.item(), 0.)
            self.assertGreaterEqual(lyap_pos_loss_sub.item(), 0.)
            self.assertGreaterEqual(lyap_pos_loss_samp.item(), 0.)
            self.assertGreaterEqual(lyap_der_loss.item(), 0.)
            self.assertGreaterEqual(lyap_der_loss_sub.item(), 0.)
            self.assertGreaterEqual(lyap_der_loss_samp.item(), 0.)

    def test_adversarial_samples(self):
        for dyn_learner in [self.ss_dyn_learner, self.latent_dyn_learner]:
            for optimality in [True, False]:
                self.opt.set_option("lyap_loss_optimal", optimality)
                z_adv_pos, z_adv_der = dyn_learner.adversarial_samples()
                self.assertGreaterEqual(z_adv_pos.shape[0], 1)
                self.assertGreaterEqual(z_adv_der.shape[0], 1)
                for k in range(z_adv_pos.shape[0]):
                    V = self.lyap.lyapunov_value(
                        z_adv_pos[k, :], self.lyap.system.x_equilibrium,
                        self.opt.V_lambda)
                    self.assertLessEqual(
                        V.item() - self.opt.V_eps * torch.norm(
                            z_adv_pos[k, :] - self.lyap.system.x_equilibrium,
                            p=1), 0.)
                for k in range(z_adv_der.shape[0]):
                    dV = self.lyap.lyapunov_derivative(
                        z_adv_der[k, :], self.lyap.system.x_equilibrium,
                        self.opt.V_lambda, self.opt.V_eps)
                    [self.assertGreaterEqual(dv.item(), 0.) for dv in dV]

    def test_dynamics_loss(self):
        loss = self.ss_dyn_learner.dynamics_loss(self.x_data, self.x_next_data)
        self.assertGreaterEqual(loss, 0.)
        loss = self.ss_dyn_learner.dynamics_loss(
            self.x_data, self.lyap.system.step_forward(self.x_data))
        self.assertEqual(loss, 0.)
        for variational in [True, False]:
            self.opt.set_option("use_variational", variational)
            loss = self.latent_dyn_learner.dynamics_loss(
                self.X_data, self.X_next_data)
            self.assertGreaterEqual(loss, 0.)

    def test_kl_div(self):
        z_mu = torch.zeros((5, self.opt.z_dim), dtype=self.opt.dtype)
        z_log_var = torch.log(
            torch.ones((5, self.opt.z_dim), dtype=self.opt.dtype))
        kl = self.latent_dyn_learner.kl_loss(z_mu, z_log_var)
        self.assertEqual(kl.item(), 0.)
        z_mu += (torch.rand(z_mu.shape, dtype=self.opt.dtype) - .5) * 2
        kl1 = self.latent_dyn_learner.kl_loss(z_mu, z_log_var)
        self.assertGreater(kl1, kl)
        z_log_var -= (torch.rand(
            z_log_var.shape, dtype=self.opt.dtype) - .5) * 2
        kl2 = self.latent_dyn_learner.kl_loss(z_mu, z_log_var)
        self.assertGreater(kl2, kl1)

    def test_save_load(self):
        self.ss_dyn_learner.save(".")
        dl = dynamics_learning.StateSpaceDynamicsLearning.load(
            ".", self.x_train_dataloader, self.x_validation_dataloader,
            self.opt)
        x1 = self.ss_dyn_learner.lyap.lyapunov_relu(self.x_data)
        x2 = dl.lyap.lyapunov_relu(self.x_data)
        self.assertTrue(torch.all(x1 == x2))
        os.remove("./dyn_learner.pkl")
        self.opt.set_option("use_variational", False)
        self.latent_dyn_learner.save(".")
        dl = dynamics_learning.LatentSpaceDynamicsLearning.load(
            ".", self.x_train_dataloader, self.x_validation_dataloader,
            self.opt)
        x1, _, _, _, _ = self.latent_dyn_learner.vae_forward(self.X_data)
        x2, _, _, _, _ = dl.vae_forward(self.X_data)
        self.assertTrue(torch.all(x1 == x2))
        os.remove("./dyn_learner.pkl")

    def test_rollout(self):
        x0 = self.x_data[0, :]
        roll, V_roll = self.ss_dyn_learner.rollout(x0, 10)
        self.assertEqual(roll.shape, (10+1, self.opt.x_dim))
        self.assertEqual(V_roll.shape, (10+1,))
        self.assertTrue(torch.all(x0 == roll[0, :]))
        for decode in [True, False]:
            X0 = self.X_data[0, :]
            roll, V_roll, z_roll = self.latent_dyn_learner.rollout(
                X0, 10, decode_intermediate=decode)
            num_channels = int(X0.shape[0]/2)
            self.assertEqual(
                roll.shape, (10+2, num_channels,
                             self.opt.image_width, self.opt.image_height))
            self.assertEqual(V_roll.shape, (10+1,))
            self.assertEqual(z_roll.shape, (10+1, self.opt.z_dim))
            self.assertTrue(torch.all(X0[:num_channels, :] == roll[0, :]))
            self.assertTrue(torch.all(X0[num_channels:, :] == roll[1, :]))

    def test_rollout_loss(self):
        x0 = self.x_data[0, :]
        roll, V_roll = self.ss_dyn_learner.rollout(x0, 10)
        loss = self.ss_dyn_learner.rollout_loss(roll)
        self.assertEqual(loss.shape, (11,))
        self.assertEqual(loss[0].item(), 0.)
        self.assertTrue(torch.all(loss >= 0))
        for decode in [True, False]:
            self.opt.set_option("use_bce", False)
            X0 = self.X_data[0, :]
            roll, _, _ = self.latent_dyn_learner.rollout(X0, 10, decode)
            loss = self.latent_dyn_learner.rollout_loss(roll, decode)
            self.assertEqual(loss.shape, (12,))
            self.assertEqual(loss[0].item(), 0.)
            self.assertEqual(loss[1].item(), 0.)
            self.assertTrue(torch.all(loss >= 0))
            self.opt.set_option("use_bce", True)
            X0 = self.X_data[0, :]
            roll, _, _ = self.latent_dyn_learner.rollout(X0, 10, decode)
            loss = self.latent_dyn_learner.rollout_loss(roll, decode)
            self.assertEqual(loss.shape, (12,))
            self.assertTrue(torch.all(loss > 0))
            self.assertTrue(torch.all(loss[0] < loss[2:]))
            self.assertTrue(torch.all(loss[1] < loss[2:]))

    def test_rollout_validation_loss(self):
        rollouts = [torch.rand((10, self.opt.x_dim), dtype=self.opt.dtype)
                    for i in range(4)]
        loss = self.ss_dyn_learner.rollout_validation(rollouts)
        self.assertEqual(loss.shape, (10, ))
        rollouts = [torch.rand(
            (10, self.X_next_data.shape[1],
             self.opt.image_width, self.opt.image_height),
            dtype=self.opt.dtype)
                    for i in range(4)]
        loss = self.latent_dyn_learner.rollout_validation(rollouts)
        self.assertEqual(loss.shape, (10, ))
        if torch.cuda.is_available():
            loss_cuda = self.latent_dyn_learner.rollout_validation(
                rollouts, device='cuda')
            self.assertEqual(loss_cuda.shape, (10, ))
            self.assertTrue(torch.all(torch.abs(loss - loss_cuda) < 1e-6))

    def test_train(self):
        if torch.cuda.is_available():
            devices = ['cpu', 'cuda']
        else:
            devices = ['cpu']
        x_test = torch.ones(self.opt.x_dim, dtype=self.opt.dtype)
        z_test = torch.ones(self.opt.z_dim, dtype=self.opt.dtype)
        for device in devices:
            for var_name in opt_variants.keys():
                self.opt.set_options(opt_variants[var_name])
                x_pre = self.ss_dyn_learner.lyap.system.step_forward(x_test)
                v_pre = self.ss_dyn_learner.lyap.lyapunov_relu.forward(x_test)
                self.ss_dyn_learner.reset_optimizer()
                self.ss_dyn_learner.train(2, validate=True, device=device)
                x_post = self.ss_dyn_learner.lyap.system.step_forward(x_test)
                v_post = self.ss_dyn_learner.lyap.lyapunov_relu.forward(x_test)
                self.assertFalse(torch.all(x_pre == x_post))
                if var_name == "unstable":
                    self.assertTrue(torch.all(v_pre == v_post))
                elif var_name == "stable":
                    self.assertFalse(torch.all(v_pre == v_post))
                self.ss_dyn_learner.reset_optimizer(lyapunov_only=True)
                self.ss_dyn_learner.train(2, validate=True, device=device)
                # testing lyapunov training only
                x_post2 = self.ss_dyn_learner.lyap.system.step_forward(x_test)
                v_post2 = self.ss_dyn_learner.lyap.lyapunov_relu.forward(
                    x_test)
                self.assertTrue(torch.all(x_post == x_post2))
                if var_name == "unstable":
                    self.assertTrue(torch.all(v_post == v_post2))
                elif var_name == "stable":
                    self.assertFalse(torch.all(v_post == v_post2))
                z_pre = self.latent_dyn_learner.lyap.system.step_forward(
                    z_test)
                v_pre = self.latent_dyn_learner.lyap.lyapunov_relu.forward(
                    z_test)
                self.latent_dyn_learner.reset_optimizer()
                self.latent_dyn_learner.train(2, validate=True, device=device)
                z_post = self.latent_dyn_learner.lyap.system.step_forward(
                    z_test)
                v_post = self.latent_dyn_learner.lyap.lyapunov_relu.forward(
                    z_test)
                self.assertFalse(torch.all(z_pre == z_post))
                if var_name == "unstable":
                    self.assertTrue(torch.all(v_pre == v_post))
                elif var_name == "stable":
                    self.assertFalse(torch.all(v_pre == v_post))
                self.latent_dyn_learner.reset_optimizer(lyapunov_only=True)
                self.latent_dyn_learner.train(2, validate=True, device=device)
                z_post2 = self.latent_dyn_learner.lyap.system.step_forward(
                    z_test)
                v_post2 = self.latent_dyn_learner.lyap.lyapunov_relu.forward(
                    z_test)
                self.assertTrue(torch.all(z_post == z_post2))
                if var_name == "unstable":
                    self.assertTrue(torch.all(v_post == v_post2))
                elif var_name == "stable":
                    self.assertFalse(torch.all(v_post == v_post2))
        shutil.rmtree("runs")

    def test_early_term(self):
        for dyn_learner, data in [(self.ss_dyn_learner, self.x_data),
                                  (self.latent_dyn_learner, self.X_data)]:
            self.opt.set_option("lyap_loss_optimal", True)
            lyap_pos_loss1, lyap_der_loss1 = dyn_learner.lyapunov_loss()
            self.opt.set_option("lyap_loss_optimal", False)
            lyap_pos_loss2, lyap_der_loss2 = dyn_learner.lyapunov_loss(
                pos_threshold=lyap_pos_loss1, der_threshold=lyap_der_loss1)
            lyap_pos_loss3, lyap_der_loss3 = dyn_learner.lyapunov_loss()
            self.assertEqual(lyap_pos_loss1, lyap_pos_loss2)
            self.assertEqual(lyap_der_loss1, lyap_der_loss2)
            self.assertLess(lyap_pos_loss3, lyap_pos_loss1)
            self.assertLess(lyap_der_loss3, lyap_der_loss1)


if __name__ == "__main__":
    unittest.main()
