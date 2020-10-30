import os
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import gurobipy
import pickle

import neural_network_lyapunov.lyapunov as lyapunov
import neural_network_lyapunov.relu_system as relu_system
import neural_network_lyapunov.utils as utils
from neural_network_lyapunov.gurobi_torch_mip import IncorrectActiveConstraint


class DynamicsLearningOptions():
    def __init__(self, options_dict):
        """
        Container for training options. Gets passed to DynamicsLearning
        constructor
        """
        self.options = options_dict

    def set_option(self, name, value):
        self.options[name] = value

    def set_options(self, options_dict):
        for key in options_dict.keys():
            self.options[key] = options_dict[key]

    def __getattr__(self, attr):
        if attr not in self.options.keys():
            raise AttributeError("Option " + attr + " missing")
        return self.options[attr]


class DynamicsLearning:
    def __init__(self, train_dataloader, validation_dataloader,
                 lyap, learning_opt):
        """
        Helper class to train dynamics using lyapunov regularization
        @param train_dataloader torch Dataloader for training
        @param validation_dataloader torch Dataloader for validation
        @param lyap instance of LyapunovHybridLinearSystem class
        @param learning_opt instance of DynamicsLearningOptions class
        """
        assert(isinstance(lyap, lyapunov.LyapunovHybridLinearSystem))
        assert(
            isinstance(
                lyap.system, relu_system.AutonomousReLUSystemGivenEquilibrium)
            or isinstance(
                lyap.system,
                relu_system.AutonomousResidualReLUSystemGivenEquilibrium))
        assert(isinstance(learning_opt, DynamicsLearningOptions))
        self.train_dataloader = train_dataloader
        self.validation_dataloader = validation_dataloader
        self.lyap = lyap
        self.opt = learning_opt
        self.optimizer = None
        self.writer = None
        self.log_suffix = ""
        self.n_iter = 0.
        self.lyap_pos_x_adv = lyap.system.x_equilibrium
        self.lyap_der_x_adv = lyap.system.x_equilibrium
        self.kl_loss_weight = utils.SigmoidAnneal(
            self.opt.dtype, self.opt.kl_weight_lo, self.opt.kl_weight_up,
            self.opt.kl_weight_center_step, self.opt.kl_weight_steps_lo_to_up)

    def reset_optimizer(self, lyapunov_only=False):
        """
        resets the optimizers
        @param lyapunov_only boolean set to true to ONLY train the lyapunov.
        This is mostly useful for benchmarks by training a lyapunov "after-
        the-fact" and finding adversarial examples
        """
        if lyapunov_only:
            params_list = self.get_lyapunov_parameters()
        else:
            params_list = self.get_trainable_parameters()
        params = [{'params': p} for p in params_list]
        self.optimizer = torch.optim.Adam(params)
        self.writer = SummaryWriter()
        self.log_suffix = ""
        self.n_iter = 0

    def get_lyapunov_parameters(self):
        """
        return a list of lists of parameters that correpond to the lyapunov
        only
        """
        params = [self.lyap.lyapunov_relu.parameters()]
        return params

    def lyapunov_to_device(self, device):
        """
        move all the parameters related to training the lyapunov to device
        @param device string for the device to move everything to
        e.g. 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc.
        """
        self.lyap.lyapunov_relu.to(device)
        self.lyap.system.dynamics_relu.to(device)
        self.lyap.system.x_equilibrium = self.lyap.system.x_equilibrium.to(
            device)

    def adversarial_samples(self):
        """
        computes states the lead to violations of the lyapunov conditions
        @return z_adv_pos tensor [num_samples, x/z_dim] where the lyapunov
        violates the positivity constraint
        @return z_adv_der tensor [num_samples, x/z_dim] where the lyapunov
        violate the derivative constraint
        """
        lyap_pos_mip, x_var = self.lyap.lyapunov_positivity_as_milp(
            self.lyap.system.x_equilibrium, self.opt.V_lambda, self.opt.V_eps)
        lyap_pos_mip.gurobi_model.setParam(gurobipy.GRB.Param.OutputFlag,
                                           False)
        if self.opt.lyap_loss_optimal:
            lyap_pos_mip.gurobi_model.optimize()
        else:
            lyap_pos_mip.gurobi_model.optimize(
                utils.get_gurobi_terminate_if_callback())
        num_sol = lyap_pos_mip.gurobi_model.solCount
        z_adv_pos = []
        for i in range(num_sol):
            lyap_pos_mip.gurobi_model.setParam(
                gurobipy.GRB.Param.SolutionNumber, i)
            obj = lyap_pos_mip.gurobi_model.PoolObjVal
            if obj > 0:
                z_adv_pos.append(torch.tensor(
                    [r.xn for r in x_var], dtype=self.opt.dtype).unsqueeze(0))
        if num_sol > 0:
            z_adv_pos = torch.cat(z_adv_pos, dim=0)
        lyap_der_mip_ = self.lyap.lyapunov_derivative_as_milp(
            self.lyap.system.x_equilibrium, self.opt.V_lambda, self.opt.V_eps)
        lyap_der_mip = lyap_der_mip_[0]
        x_var = lyap_der_mip_[1]
        lyap_der_mip.gurobi_model.setParam(gurobipy.GRB.Param.OutputFlag,
                                           False)
        if self.opt.lyap_loss_optimal:
            lyap_der_mip.gurobi_model.optimize()
        else:
            lyap_der_mip.gurobi_model.optimize(
                utils.get_gurobi_terminate_if_callback())
        num_sol = lyap_der_mip.gurobi_model.solCount
        z_adv_der = []
        for i in range(num_sol):
            lyap_der_mip.gurobi_model.setParam(
                gurobipy.GRB.Param.SolutionNumber, i)
            obj = lyap_der_mip.gurobi_model.PoolObjVal
            if obj > 0:
                z_adv_der.append(torch.tensor(
                    [r.xn for r in x_var], dtype=self.opt.dtype).unsqueeze(0))
        if num_sol > 0:
            z_adv_der = torch.cat(z_adv_der, dim=0)
        return z_adv_pos, z_adv_der

    def lyapunov_loss(self, lyap_pos_threshold=0., lyap_der_threshold=0.):
        """
        compute the Lyapunov losses
        @param lyap_pos_threshold float the thresold used when computing an
        adversarial example for violation of the positivity of the lyapunov
        function. The MILP is terminated when its objective reaches
        lyap_pos_threshold.
        @param lyap_der_threshold float the thresold used when computing an
        adversarial example for violation of the derivative constraint of the
        lyapunov function. The MILP is terminated when its objective reaches
        lyap_der_threshold.
        @return lyap_pos_loss tensor of the lyapunov loss for the positivity
        constraint
        @return lyap_der_loss tensor of the lypunov loss for the derivative
        constraint
        """
        if self.opt.lyap_loss_warmstart:
            lyap_pos_mip, x_var = self.lyap.lyapunov_positivity_as_milp(
                self.lyap.system.x_equilibrium, self.opt.V_lambda,
                self.opt.V_eps, x_warmstart=self.lyap_pos_x_adv)
        else:
            lyap_pos_mip, x_var = self.lyap.lyapunov_positivity_as_milp(
                self.lyap.system.x_equilibrium, self.opt.V_lambda,
                self.opt.V_eps)
        lyap_pos_mip.gurobi_model.setParam(gurobipy.GRB.Param.OutputFlag,
                                           False)
        if self.opt.lyap_loss_optimal:
            lyap_pos_mip.gurobi_model.optimize()
            lyap_pos_loss = lyap_pos_mip.\
                compute_objective_from_mip_data_and_solution()
        else:
            lyap_pos_mip.gurobi_model.optimize(
                utils.get_gurobi_terminate_if_callback(
                    threshold=lyap_pos_threshold))
            try:
                lyap_pos_loss = lyap_pos_mip.\
                    compute_objective_from_mip_data_and_solution()
            except IncorrectActiveConstraint:
                print("WARNING: Cannot find the right " +
                      "constraints to get gradient.")
                lyap_pos_loss = torch.tensor(0, dtype=self.opt.dtype)
        if self.opt.lyap_loss_warmstart:
            self.lyap_pos_x_adv = torch.tensor(
                [var.X for var in x_var], dtype=self.opt.dtype)
        if self.opt.lyap_loss_warmstart:
            lyap_der_mip_return = self.lyap.lyapunov_derivative_as_milp(
                self.lyap.system.x_equilibrium,
                self.opt.V_lambda,
                self.opt.V_eps,
                x_warmstart=self.lyap_der_x_adv)
        else:
            lyap_der_mip_return = self.lyap.lyapunov_derivative_as_milp(
                self.lyap.system.x_equilibrium,
                self.opt.V_lambda,
                self.opt.V_eps)
        lyap_der_mip = lyap_der_mip_return[0]
        x_var = lyap_der_mip_return[1]
        lyap_der_mip.gurobi_model.setParam(gurobipy.GRB.Param.OutputFlag,
                                           False)
        if self.opt.lyap_loss_optimal:
            lyap_der_mip.gurobi_model.optimize()
            lyap_der_loss = lyap_der_mip.\
                compute_objective_from_mip_data_and_solution()
        else:
            lyap_der_mip.gurobi_model.optimize(
                utils.get_gurobi_terminate_if_callback(
                    threshold=lyap_der_threshold))
            try:
                lyap_der_loss = lyap_der_mip.\
                    compute_objective_from_mip_data_and_solution()
            except IncorrectActiveConstraint:
                print("WARNING: Cannot find the right " +
                      "constraints to get gradient.")
                lyap_der_loss = torch.tensor(0, dtype=self.opt.dtype)
        if self.opt.lyap_loss_warmstart:
            self.lyap_der_x_adv = torch.tensor(
                [var.X for var in x_var], dtype=self.opt.dtype)
        return (self.opt.lyap_pos_loss_weight * lyap_pos_loss,
                self.opt.lyap_der_loss_weight * lyap_der_loss)

    def lyapunov_loss_at_samples(self, x_all, x_lo, x_up):
        """
        computes lyapunov loss at the provided samples
        @param x_all tensor (num_samples, x/z dim)
        @param x_lo tensor upper bound outside which the loss is 0
        @param x_up tensor lower bound outside which the loss is 0
        @return sample_pos_loss tensor lyapunov positivity loss at x
        @return sample_der_loss tensor lyapunov derivative loss at x
        """
        x = x_all[torch.all(x_all <= x_up.to(x_all.device), dim=1), :]
        x = x[torch.all(x >= x_lo.to(x_all.device), dim=1), :]
        if x.shape[0] == 0:
            sample_der_loss = torch.zeros(1, dtype=self.opt.dtype).to(
                x_all.device)
            sample_pos_loss = torch.zeros(1, dtype=self.opt.dtype).to(
                x_all.device)
            return sample_pos_loss, sample_der_loss
        x_next = self.lyap.system.step_forward(x)
        relu_at_equilibrium = self.lyap.lyapunov_relu.forward(
            self.lyap.system.x_equilibrium)
        sample_pos_loss = \
            self.lyap.lyapunov_positivity_loss_at_samples(
                relu_at_equilibrium, self.lyap.system.x_equilibrium, x,
                self.opt.V_lambda, self.opt.V_eps)
        sample_der_loss = \
            self.lyap.lyapunov_derivative_loss_at_samples_and_next_states(
                self.opt.V_lambda, self.opt.V_eps, x, x_next,
                self.lyap.system.x_equilibrium)
        return (self.opt.lyap_pos_loss_at_samples_weight * sample_pos_loss,
                self.opt.lyap_der_loss_at_samples_weight * sample_der_loss)

    def rollout_validation(self, rollouts, device='cpu'):
        """
        computes the mean loss along a list of rollouts
        @param rollouts list of tensors, each one of them a rollout
        @param device where to run the computation (e.g. 'cuda')
        """
        assert(isinstance(rollouts, list))
        assert(len(rollouts) >= 1)
        self.all_to_device(device)
        validation_loss = torch.zeros(
            rollouts[0].shape[0], dtype=self.opt.dtype).to(device)
        with torch.no_grad():
            for rollout_expected in rollouts:
                rollout_expected = rollout_expected.to(device)
                validation_loss += self.rollout_loss(rollout_expected)
        self.all_to_device('cpu')
        validation_loss = validation_loss.to('cpu')
        return validation_loss

    def log(self, name, value):
        if self.writer is not None:
            if self.log_suffix != "":
                self.writer.add_scalar(
                    name + '/' + self.log_suffix, value, self.n_iter)
            else:
                self.writer.add_scalar(name, value, self.n_iter)

    def train(self, num_epoch, validate=False, device='cpu',
              save_rate=0, save_path=None):
        """
        trains all the trainable parameters in the model
        @param num_epoch int number of epochs
        @param validate boolean set to True to run the validation loss at
        each epoch
        @param device string device where to do the training ('cpu' or
        'cuda:0' etc). Note that the lyapunov training will always happen
        on the CPU regardless of that setting, because gurobi cannot run
        on GPU
        @param save_rate int save the networks every save_rate epoch, will
        also save at the end of training if save_path is set.
        @param save_path string path where to save the models
        """
        self.all_to_device(device)
        if self.optimizer is None:
            self.reset_optimizer()
        try:
            for epoch_i in range(num_epoch):
                self.log_suffix = "train"
                for x, x_next in self.train_dataloader:
                    x = x.to(device)
                    x_next = x_next.to(device)
                    self.optimizer.zero_grad()
                    dyn_loss = self.dynamics_loss(x, x_next)
                    lyap_pos_loss_at_samples, lyap_der_loss_at_samples = self.\
                        lyapunov_loss_at_samples(x)
                    loss = dyn_loss + lyap_pos_loss_at_samples +\
                        lyap_der_loss_at_samples
                    loss.backward()
                    self.optimizer.step()
                    self.log('Dynamics', dyn_loss)
                    self.log('LyapunovPosSamples', lyap_pos_loss_at_samples)
                    self.log('LyapunovDerSamples', lyap_der_loss_at_samples)
                    if ((self.opt.lyap_loss_freq > 0) and
                       ((self.n_iter % self.opt.lyap_loss_freq) == 0)):
                        with torch.no_grad():
                            (lyap_pos_loss_at_samples,
                             lyap_der_loss_at_samples) =\
                                self.lyapunov_loss_at_samples(x)
                        lyap_pos_threshold = lyap_pos_loss_at_samples.item()
                        lyap_der_threshold = lyap_der_loss_at_samples.item()
                        self.optimizer.zero_grad()
                        self.lyapunov_to_device('cpu')
                        lyap_pos_loss, lyap_der_loss = self.lyapunov_loss(
                            lyap_pos_threshold=lyap_pos_threshold,
                            lyap_der_threshold=lyap_der_threshold)
                        loss = (lyap_pos_loss + lyap_der_loss)
                        loss.backward()
                        self.lyapunov_to_device(device)
                        loss = loss.to(device)
                        self.optimizer.step()
                        self.log('LyapunovPos', lyap_pos_loss)
                        self.log('LyapunovDer', lyap_der_loss)
                    self.n_iter += 1
                if validate:
                    self.log_suffix = "validate"
                    with torch.no_grad():
                        dyn_loss = 0.
                        lyap_pos_loss_at_samples = 0.
                        lyap_der_loss_at_samples = 0.
                        n_samples = 0.
                        for x, x_next in self.validation_dataloader:
                            x = x.to(device)
                            x_next = x_next.to(device)
                            dyn_loss_ = self.dynamics_loss(x, x_next)
                            (lyap_pos_loss_at_samples_,
                             lyap_der_loss_at_samples_) =\
                                self.lyapunov_loss_at_samples(x)
                            dyn_loss += dyn_loss_ * x.shape[0]
                            lyap_pos_loss_at_samples +=\
                                lyap_pos_loss_at_samples_ * x.shape[0]
                            lyap_der_loss_at_samples +=\
                                lyap_der_loss_at_samples_ * x.shape[0]
                            n_samples += x.shape[0]
                        dyn_loss /= n_samples
                        lyap_pos_loss_at_samples /= n_samples
                        lyap_der_loss_at_samples /= n_samples
                        self.log('Dynamics', dyn_loss)
                        self.log('LyapunovPosSamples',
                                 lyap_pos_loss_at_samples)
                        self.log('LyapunovDerSamples',
                                 lyap_der_loss_at_samples)
                if save_rate > 0 and epoch_i % save_rate == 0:
                    assert(save_path is not None)
                    self.save(save_path)
        except KeyboardInterrupt:
            pass
        self.all_to_device('cpu')
        if save_path is not None:
            self.save(save_path)


class StateSpaceDynamicsLearning(DynamicsLearning):
    def __init__(self, train_dataloader, validation_dataloader,
                 lyap, learning_opt):
        super(StateSpaceDynamicsLearning, self).__init__(
            train_dataloader, validation_dataloader, lyap, learning_opt)
        self.mse_loss = nn.MSELoss()

    def get_trainable_parameters(self):
        """
        return a list of lists of parameters that are trainable
        """
        params = [self.lyap.lyapunov_relu.parameters(),
                  self.lyap.system.dynamics_relu.parameters()]
        return params

    def all_to_device(self, device):
        """
        moves all the relevant parameters to device (e.g. 'cpu', 'cuda')
        """
        self.lyapunov_to_device(device)

    def save(self, save_path):
        """
        helper function to save relevant parameters
        @param save_path string for where to save (prefix)
        """
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        with open(os.path.join(save_path, 'dyn_learner.pkl'), 'wb') as output:
            pickle.dump(self.lyap, output, pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, load_path,
             train_dataloader, validation_dataloader, learning_opt):
        """
        load from what was saved using the save method
        @param load_path string path to load dyn_learner from
        @param train_dataloader torch Dataloader for training
        @param validation_dataloader torch Dataloader for validation
        @param learning_opt instance of DynamicsLearningOptions
        """
        with open(os.path.join(load_path, 'dyn_learner.pkl'), 'rb') as input:
            lyap = pickle.load(input)
        return cls(
            train_dataloader, validation_dataloader, lyap, learning_opt)

    def lyapunov_loss_at_samples(self, x):
        """
        computes the lyapunov loss at the samples.
        """
        return super(StateSpaceDynamicsLearning, self).\
            lyapunov_loss_at_samples(
                x, self.opt.x_lo_stable, self.opt.x_up_stable)

    def dynamics_loss(self, x, x_next):
        """
        computes the dynamics loss in state space (this is just L2 on the
        predicted states)
        """
        x_next_pred = self.lyap.system.step_forward(x)
        loss = self.mse_loss(x_next_pred, x_next)
        return loss

    def rollout(self, x_init, N):
        """
        generates a rollout with the learned dynamics for N step
        @param x_init tensor of dim x_dim
        @param N int number of steps to take
        @return x_traj [N+1, x_dim] rollout
        @return V_traj tensor N+1 of the lyapunov value along the rollout
        """
        assert(len(x_init.shape) == 1)
        assert(x_init.shape[0] == self.opt.x_dim)
        x_traj = torch.zeros(
            N+1, self.opt.x_dim, dtype=self.opt.dtype).to(x_init.device)
        V_traj = []
        x_traj[0, :] = x_init
        V_traj.append(self.lyap.lyapunov_value(
            x_init, self.lyap.system.x_equilibrium, self.opt.V_lambda).item())
        for n in range(N):
            with torch.no_grad():
                x_next_pred = self.lyap.system.step_forward(x_traj[n, :])
                x_traj[n+1, :] = x_next_pred
                V_traj.append(self.lyap.lyapunov_value(
                    x_next_pred, self.lyap.system.x_equilibrium,
                    self.opt.V_lambda).item())
        V_traj = torch.tensor(V_traj, dtype=self.opt.dtype)
        return x_traj, V_traj

    def rollout_loss(self, rollout_expected):
        """
        computes the reconstruction loss for a rollout
        @param rollout_expected tensor [N+1, x_dim]
        @return loss tensor of dim [N+1] of L2 reconstruction loss along the
        rollout
        """
        x0 = rollout_expected[0, :]
        rollout_pred, _ = self.rollout(x0, rollout_expected.shape[0] - 1)
        loss = (rollout_expected - rollout_pred).pow(2).mean(dim=[1])
        return loss


class LatentSpaceDynamicsLearning(DynamicsLearning):
    def __init__(self, train_dataloader, validation_dataloader,
                 lyap, learning_opt, encoder, decoder,
                 decoded_equilibrium=None):
        super(LatentSpaceDynamicsLearning, self).__init__(
            train_dataloader, validation_dataloader, lyap, learning_opt)
        self.encoder = encoder.type(self.opt.dtype)
        self.decoder = decoder.type(self.opt.dtype)
        if decoded_equilibrium is not None:
            assert(len(decoded_equilibrium.shape) == 3)
            assert(decoded_equilibrium.shape[1:] ==
                   (self.opt.image_width, self.opt.image_height))
            assert(decoded_equilibrium.shape[0] == 2 or
                   decoded_equilibrium.shape[0] == 6)
        self.decoded_equilibrium = decoded_equilibrium
        self.bce_loss = nn.BCELoss(reduction='mean')
        self.bce_loss_none = nn.BCELoss(reduction='none')

    def get_trainable_parameters(self):
        """
        return a list of lists of parameters that are trainable
        """
        params = [self.lyap.lyapunov_relu.parameters(),
                  self.lyap.system.dynamics_relu.parameters(),
                  self.encoder.parameters(),
                  self.decoder.parameters()]
        return params

    def all_to_device(self, device):
        """
        moves all the relevant parameters to device (e.g. 'cpu', 'cuda')
        """
        self.lyapunov_to_device(device)
        self.encoder.to(device)
        self.decoder.to(device)
        if self.decoded_equilibrium is not None:
            self.decoded_equilibrium = self.decoded_equilibrium.to(device)

    def save(self, save_path):
        """
        helper function to save relevant parameters
        @param save_path string for where to save (prefix)
        """
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        with open(os.path.join(save_path, 'dyn_learner.pkl'), 'wb') as output:
            pickle.dump(self.lyap, output, pickle.HIGHEST_PROTOCOL)
            pickle.dump(self.encoder, output, pickle.HIGHEST_PROTOCOL)
            pickle.dump(self.decoder, output, pickle.HIGHEST_PROTOCOL)
            pickle.dump(
                self.decoded_equilibrium, output, pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, load_path,
             train_dataloader, validation_dataloader, learning_opt):
        """
        load from what was saved using the save method
        @param load_path string path to load dyn_learner from
        @param train_dataloader torch Dataloader for training
        @param validation_dataloader torch Dataloader for validation
        @param learning_opt instance of DynamicsLearningOptions
        """
        with open(os.path.join(load_path, 'dyn_learner.pkl'), 'rb') as input:
            lyap = pickle.load(input)
            encoder = pickle.load(input)
            decoder = pickle.load(input)
            decoded_equilibrium = pickle.load(input)
        return cls(
            train_dataloader, validation_dataloader, lyap, learning_opt,
            encoder, decoder, decoded_equilibrium=decoded_equilibrium)

    def reparam(self, z_mu, z_log_var):
        """
        gets a sample in latent space
        @param z_mu tensor mean of the samples
        @param z_log_var tensor log of the variance of the samples
        @return a sample z in latent space, as a tensor
        """
        z_std = torch.exp(0.5 * z_log_var)
        eps = torch.randn(z_mu.shape, dtype=z_mu.dtype).to(z_mu.device)
        z = eps * z_std + z_mu
        return z

    def vae_forward(self, x):
        """
        full forward pass of the AE/VAE
        @param x tensor input
        @return x_decoded tensor, x encoded and then decoded
        @return x_next_pred_decoded, x encoded, pass through dynamics and
        then decoded
        @return z_mu tensor, mean of the latent samples (if VAE), otherwise
        encoded x
        @return z_log_var tensor, log of the variance of the latent samples
        (if VAE) otherwise just None
        """
        z_mu, z_log_var = self.encoder(x)
        if self.opt.use_variational:
            z = self.reparam(z_mu, z_log_var)
        else:
            z = z_mu
            z_log_var = None
        x_decoded = self.decoder(z)
        z_next = self.lyap.system.step_forward(z)
        x_next_pred_decoded = self.decoder(z_next)
        return x_decoded, x_next_pred_decoded, z_mu, z_log_var, z_next

    def lyapunov_loss_at_samples(self, x):
        """
        computes the lyapunov loss at the samples. This first encodes
        the samples and then calls the function of the same name in the
        parent class DynamicsLearning
        """
        z_mu, z_log_var = self.encoder(x)
        if self.opt.use_variational:
            z = self.reparam(z_mu, z_log_var)
        else:
            z = z_mu
        return super(LatentSpaceDynamicsLearning, self).\
            lyapunov_loss_at_samples(
                z, self.opt.z_lo_stable, self.opt.z_up_stable)

    def kl_loss(self, z_mu, z_log_var):
        """
        compute the KL divergence from the samples produced
        @param z_mu tensor mean of the samples
        @param z_log_var tensor log of the variance of the samples
        @return weighted KL divergence
        """
        loss = torch.mean(-.5 * torch.sum(-torch.pow(z_mu, 2) -
                          torch.exp(z_log_var) + z_log_var + 1., dim=1))
        weighted_loss = self.kl_loss_weight(self.n_iter) * loss
        self.log("KL/original", loss)
        self.log("KL/weighted", weighted_loss)
        return weighted_loss

    def reconstruction_loss(self, x, x_decoded):
        """
        computes the reconstruction loss using either L2 loss
        or binary cross entropy
        @param x tensor label
        @param x_decoded tensor predicted label after encoding-decoding
        """
        if self.opt.use_bce:
            loss = self.bce_loss(x_decoded, x)
        else:
            loss = (x - x_decoded).pow(2).mean(dim=[1, 2, 3])[0]
        return loss

    def dynamics_loss(self, x, x_next):
        """
        computes the dynamics loss
        @param x tensor initial state
        @param x_next tensor next state
        """
        if x_next.shape[1] < x.shape[1]:
            x_next_ = torch.cat((x[:, x_next.shape[1]:, :, :], x_next), dim=1)
        else:
            x_next_ = x_next
        x_decoded, x_next_pred_decoded, z_mu, z_log_var, z_next =\
            self.vae_forward(x)
        enc_loss = self.reconstruction_loss(x, x_decoded)
        self.log("Dynamics/encoding", enc_loss)
        dyn_loss = self.reconstruction_loss(x_next_, x_next_pred_decoded)
        self.log("Dynamics/latent_dyn", dyn_loss)
        loss = enc_loss + dyn_loss
        if self.opt.use_variational:
            loss += self.kl_loss(z_mu, z_log_var)
        if self.decoded_equilibrium is not None:
            decoded_equilibrium_pred = self.decoder.forward(
                self.lyap.system.x_equilibrium.unsqueeze(0))
            dec_equ_loss = self.opt.decoded_equilibrium_loss_weight *\
                self.reconstruction_loss(
                    self.decoded_equilibrium.unsqueeze(0),
                    decoded_equilibrium_pred)
            self.log("DecodedEquilibirum", dec_equ_loss)
            loss += dec_equ_loss
        return loss

    def rollout(self, x_init, N, decode_intermediate=False):
        """
        generates a rollout with the learned dynamics for N step
        @param x_init tensor of dim [2*num_channels, width, height]
        @param N int number of steps to take
        @param decode_intermediate boolean if True, will pass the latent
        state through the decoder, then ecoder before taking another step
        @return x_traj [N+1, num_channels, width height] rollout
        @return V_traj tensor N+1 of the lyapunov value along the rollout
        @return z_traj tensor [N+1, z_dim] rollout in latent space
        """
        assert(len(x_init.shape) == 3)
        num_channels = int(x_init.shape[0]/2)
        x_traj = torch.zeros(
            N+2, num_channels, x_init.shape[1], x_init.shape[2],
            dtype=self.opt.dtype).to(x_init.device)
        x_traj[0, :] = x_init[:num_channels, :, :]
        x_traj[1, :] = x_init[num_channels:, :, :]
        z_traj = []
        V_traj = []
        z_traj.append(self.encoder(x_init.unsqueeze(0))[0])
        V_traj.append(self.lyap.lyapunov_value(
            z_traj[-1].squeeze(), self.lyap.system.x_equilibrium,
            self.opt.V_lambda).item())
        for n in range(N):
            with torch.no_grad():
                if not decode_intermediate:
                    z = self.lyap.system.step_forward(z_traj[-1])
                    x_traj[n+2, :] = self.decoder(z)[0, num_channels:, :, :]
                    z_traj.append(z)
                    V_traj.append(
                        self.lyap.lyapunov_value(
                            z.squeeze(), self.lyap.system.x_equilibrium,
                            self.opt.V_lambda).item())
                else:
                    _, x_next_pred_decoded, _, _, z_next =\
                        self.vae_forward(
                            torch.cat(
                                (x_traj[n, :], x_traj[n+1, :]),
                                dim=0).unsqueeze(0))
                    x_traj[n+2, :] =\
                        x_next_pred_decoded[0, num_channels:, :, :]
                    z_traj.append(z_next)
                    V_traj.append(
                        self.lyap.lyapunov_value(
                            z_next.squeeze(), self.lyap.system.x_equilibrium,
                            self.opt.V_lambda).item())
        V_traj = torch.tensor(V_traj, dtype=self.opt.dtype)
        z_traj = torch.cat(z_traj).detach()
        return x_traj, V_traj, z_traj

    def rollout_loss(self, rollout_expected, decode_intermediate=False):
        """
        computes the reconstruction loss for a rollout
        @param rollout_expected tensor [N+1, num_channels, width, height]
        @return loss tensor of dim [N+2] of reconstruction loss along the
        rollout (either L2 or binary cross entropy depending on the options)
        """
        x0 = torch.cat([rollout_expected[0, :], rollout_expected[1, :]], dim=0)
        rollout_pred, _, _ = self.rollout(
            x0, rollout_expected.shape[0] - 2,
            decode_intermediate=decode_intermediate)
        if self.opt.use_bce:
            loss = self.bce_loss_none(rollout_expected, rollout_pred).mean(
                dim=[1, 2, 3])
        else:
            loss = (rollout_expected - rollout_pred).pow(2).mean(dim=[1, 2, 3])
        return loss
