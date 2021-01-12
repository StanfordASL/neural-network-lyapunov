import neural_network_lyapunov.adversarial_sample as adversarial_sample

import torch
import copy
import numpy as np
from torch.utils.tensorboard import SummaryWriter


class AdversarialWithBaselineTrainingOptions:
    def __init__(self):
        # total number of iteration to be done
        self.num_iter_desired = 1000
        # number of training steps between each generation of new adv samples
        self.num_steps_between_sampling = 100
        # number of random samples to initialize the buffer with
        self.init_buffer_size = 0
        # number of training step to do initially
        # (should have a nonzero init buffer size)
        self.init_num_train_steps = 0
        # number of additional random samples generated each sampling
        self.num_rand_extra = 0
        # number of adversarial optimization run at each sampling
        self.num_x_adv_opt = 1
        # maximum number of iterations to find adversarial samples
        self.x_adv_max_iter = 5
        # convergence tolerance for adversarial samples
        self.x_adv_conv_tol = 1e-5
        # learning rate used to find adversarial examples
        self.x_adv_lr = .001
        # batch size during training
        self.batch_size = 10
        # size of the buffer when training
        self.max_buffer_size = 100000
        # whether to project some of the baseline samples to the state limits
        self.baseline_use_limits = False


class AdversarialWithBaseline:
    def __init__(self,
                 vf,
                 x0_lo,
                 x0_up,
                 nn_width=16,
                 nn_depth=1,
                 x_samples_validation=None,
                 v_samples_validation=None,
                 x_samples_init=None,
                 v_samples_init=None):
        """
        Trains a neural network to approximate a value function using
        adversarial training. Also trains a baseline for comparison, that
        trains over samples that are generated during training. The two
        networks therefore solve the same number of cost-to-go optimization
        problems.
        @param vf an instance of ValueFunction class
        @param x0_lo Tensor lower bound of model input
        @param x0_up Tensor upper bound of model input
        Data:
        @param x_samples_validation, state input to validate over
        @param v_samples_validation, corresponding value functions
        @param x_samples_init, initialization of the sample buffer (state)
        @param v_samples_init, initialization of the sample buffer (value)
        Model Params:
        @param nn_width Integer number of units per layers
        @param nn_depth Integer number of HIDDEN neural network layers
        """
        self.vf = vf
        self.V = vf.get_value_function()
        self.dtype = vf.dtype
        self.x_dim = vf.sys.x_dim
        self.x0_lo = x0_lo
        self.x0_up = x0_up
        self.x_samples_validation = x_samples_validation
        self.v_samples_validation = v_samples_validation
        if x_samples_init is not None or v_samples_init is not None:
            assert (x_samples_init is not None)
            assert (v_samples_init is not None)
            assert (isinstance(x_samples_init, torch.Tensor))
            assert (isinstance(v_samples_init, torch.Tensor))
            assert (x_samples_init.dtype == self.dtype)
            assert (v_samples_init.dtype == self.dtype)
            assert (x_samples_init.shape[1] == self.x_dim)
            assert (v_samples_init.shape[1] == 1)
            self.adv_data_buffer = x_samples_init.clone()
            self.adv_label_buffer = v_samples_init.clone()
            self.rand_data_buffer = x_samples_init.clone()
            self.rand_label_buffer = v_samples_init.clone()
        else:
            self.adv_data_buffer = torch.Tensor(0, self.x_dim).type(self.dtype)
            self.adv_label_buffer = torch.Tensor(0, 1).type(self.dtype)
            self.rand_data_buffer = torch.Tensor(0,
                                                 self.x_dim).type(self.dtype)
            self.rand_label_buffer = torch.Tensor(0, 1).type(self.dtype)
        nn_layers = [torch.nn.Linear(self.x_dim, nn_width), torch.nn.ReLU()]
        for i in range(nn_depth):
            nn_layers += [torch.nn.Linear(nn_width, nn_width), torch.nn.ReLU()]
        nn_layers += [torch.nn.Linear(nn_width, 1)]
        self.baseline_model = torch.nn.Sequential(*nn_layers).double()
        self.baseline_optimizer = torch.optim.Adam(
            self.baseline_model.parameters())
        self.baseline_mse = torch.nn.MSELoss(reduction="mean")
        self.robust_model = copy.deepcopy(self.baseline_model)
        self.robust_optimizer = torch.optim.Adam(
            self.robust_model.parameters())
        self.robust_mse = torch.nn.MSELoss(reduction="mean")
        self.as_generator = adversarial_sample.AdversarialSampleGenerator(
            vf, x0_lo, x0_up)
        # logging
        self.writer = SummaryWriter()
        self.num_total_iter_done = 0
        self.x_adv_opt_log = []
        self.robust_loss_log = []
        self.baseline_loss_log = []
        self.robust_val_loss_log = []
        self.baseline_val_loss_log = []

    def get_random_x0(self):
        """
        @returns a Tensor between the initial state bounds of the training
        """
        return torch.rand(self.x_dim, dtype=self.dtype) *\
            (self.x0_up - self.x0_lo) + self.x0_lo

    def get_limit_x0(self):
        x0_rand = torch.rand(self.x_dim, dtype=self.dtype) *\
            (self.x0_up - self.x0_lo) + self.x0_lo
        dim = np.random.choice(self.x_dim, 1)
        if np.random.rand(1) >= .5:
            x0_rand[dim] = self.x0_lo[dim]
        else:
            x0_rand[dim] = self.x0_up[dim]
        return x0_rand

    def get_random_samples(self, n):
        """
        @param n Integer number of samples
        @return rand_data Tensor with random initial states
        @return rand_label Tensor with corresponding labels
        """
        rand_data = torch.zeros(n, self.x_dim, dtype=self.dtype)
        rand_label = torch.zeros(n, 1, dtype=self.dtype)
        k = 0
        while k < n:
            rand_data[k, :] = self.get_random_x0()
            rand_v = self.V(rand_data[k, :])[0]
            if rand_v is not None:
                rand_label[k, 0] = rand_v
                k += 1
        return (rand_data, rand_label)

    def get_limit_samples(self, n):
        rand_data = torch.zeros(n, self.x_dim, dtype=self.dtype)
        rand_label = torch.zeros(n, 1, dtype=self.dtype)
        k = 0
        while k < n:
            rand_data[k, :] = self.get_limit_x0()
            rand_v = self.V(rand_data[k, :])[0]
            if rand_v is not None:
                rand_label[k, 0] = rand_v
                k += 1
        return (rand_data, rand_label)

    def get_adversarial_samples(self, opt):
        """
        @param TODO
        """
        assert (isinstance(opt, AdversarialWithBaselineTrainingOptions))
        adv_data_ = torch.Tensor(0, self.x_dim).type(self.dtype)
        adv_label_ = torch.Tensor(0, 1).type(self.dtype)
        self.x_adv_opt_log = []
        k = 0
        while k < opt.num_x_adv_opt:
            x_adv0 = self.get_random_x0()
            adv = self.as_generator.get_squared_bound_sample(
                self.robust_model,
                max_iter=opt.x_adv_max_iter,
                conv_tol=opt.x_adv_conv_tol,
                learning_rate=opt.x_adv_lr,
                x_adv0=x_adv0)
            (_, adv_data_k, adv_label_k, _) = adv
            if adv_data_k.shape[0] == 0:
                print("Warning: Initial sample was infeasible")
                continue
            else:
                k += 1
            if adv_data_k.shape[0] <= 1 and adv_data_k.shape[0] > 0:
                print("Warning: no descent on adversarial sample")
            adv_data_ = torch.cat((adv_data_, adv_data_k), axis=0)
            adv_label_ = torch.cat((adv_label_, adv_label_k), axis=0)
            # just for plotting
            with torch.no_grad():
                self.x_adv_opt_log.append(
                    (adv_data_k.clone(), adv_label_k.clone()))
        return (adv_data_, adv_label_)

    def train(self, opt, logging=True):
        """
        @param opt instance of AdversarialWithBaselineTrainingOptions
        @param logging Boolean whether or not to log with tensorboard
        """
        assert (isinstance(opt, AdversarialWithBaselineTrainingOptions))
        if self.adv_data_buffer.shape[0] < opt.init_buffer_size:
            (rand_data_, rand_label_
             ) = self.get_random_samples(opt.init_buffer_size -
                                         self.adv_data_buffer.shape[0])
            self.adv_data_buffer = torch.cat(
                (self.adv_data_buffer, rand_data_), axis=0)
            self.adv_label_buffer = torch.cat(
                (self.adv_label_buffer, rand_label_), axis=0)
            self.rand_data_buffer = torch.cat(
                (self.rand_data_buffer, rand_data_), axis=0)
            self.rand_label_buffer = torch.cat(
                (self.rand_label_buffer, rand_label_), axis=0)
        if self.num_total_iter_done == 0:
            if opt.init_num_train_steps > 0:
                assert (self.adv_data_buffer.shape[0] > 0)
                assert (self.adv_data_buffer.shape[0] ==
                        self.rand_data_buffer.shape[0])
                assert (self.adv_label_buffer.shape[0] ==
                        self.rand_label_buffer.shape[0])
                for init_iter_num in range(opt.init_num_train_steps):
                    data_i = np.random.choice(
                        self.adv_data_buffer.shape[0],
                        min(self.adv_data_buffer.shape[0], opt.batch_size),
                        replace=False)
                    adv_data = self.adv_data_buffer[data_i, :]
                    adv_label = self.adv_label_buffer[data_i, :]
                    y_pred_adv = self.robust_model(adv_data)
                    robust_loss = self.robust_mse(y_pred_adv, adv_label)
                    self.robust_optimizer.zero_grad()
                    robust_loss.backward()
                    self.robust_optimizer.step()
                    # pick batch size data from buffer randomly
                    rand_data = self.rand_data_buffer[data_i, :]
                    rand_label = self.rand_label_buffer[data_i, :]
                    y_pred_rand = self.baseline_model(rand_data)
                    baseline_loss = self.baseline_mse(y_pred_rand, rand_label)
                    self.baseline_optimizer.zero_grad()
                    baseline_loss.backward()
                    self.baseline_optimizer.step()
                    self.writer.add_scalars(
                        'InitialTrain', {
                            'baseline': baseline_loss.item(),
                            'robust': robust_loss.item()
                        }, init_iter_num)
                    with torch.no_grad():
                        baseline_validation_loss = self.baseline_mse(
                            self.baseline_model(self.x_samples_validation),
                            self.v_samples_validation)
                        robust_validation_loss = self.robust_mse(
                            self.robust_model(self.x_samples_validation),
                            self.v_samples_validation)
                    bvl = baseline_validation_loss.item()
                    rvl = robust_validation_loss.item()
                    self.writer.add_scalars('InitialValidation', {
                        'baseline': bvl,
                        'robust': rvl
                    }, init_iter_num)
                    self.writer.flush()
        iter_num = 0
        while iter_num < opt.num_iter_desired:
            assert (self.adv_data_buffer.shape[0] ==
                    self.rand_data_buffer.shape[0])
            assert (self.adv_label_buffer.shape[0] ==
                    self.rand_label_buffer.shape[0])
            (adv_data_, adv_label_) = self.get_adversarial_samples(opt)
            num_new_adv_samples = adv_data_.shape[0]
            if opt.baseline_use_limits and num_new_adv_samples > 1:
                (rand_data_, rand_label_) = self.get_random_samples(1)
                (rand_data_lim,
                 rand_label_lim) = self.get_limit_samples(num_new_adv_samples -
                                                          1)
                # (rand_data_, rand_label_) = self.get_random_samples(
                #     num_new_adv_samples-1)
                # (rand_data_lim, rand_label_lim) = self.get_limit_samples(1)
                rand_data_ = torch.cat((rand_data_, rand_data_lim), axis=0)
                rand_label_ = torch.cat((rand_label_, rand_label_lim), axis=0)
            else:
                (rand_data_,
                 rand_label_) = self.get_random_samples(num_new_adv_samples)
            if (num_new_adv_samples + self.adv_data_buffer.shape[0] >
                    opt.max_buffer_size):
                self.adv_data_buffer = self.adv_data_buffer[
                    num_new_adv_samples:, :]
                self.adv_label_buffer = self.adv_label_buffer[
                    num_new_adv_samples:, :]
                self.rand_data_buffer = self.rand_data_buffer[
                    num_new_adv_samples:, :]
                self.rand_label_buffer = self.rand_label_buffer[
                    num_new_adv_samples:, :]
            self.adv_data_buffer = torch.cat((self.adv_data_buffer, adv_data_),
                                             axis=0)
            self.adv_label_buffer = torch.cat(
                (self.adv_label_buffer, adv_label_), axis=0)
            self.rand_data_buffer = torch.cat(
                (self.rand_data_buffer, rand_data_), axis=0)
            self.rand_label_buffer = torch.cat(
                (self.rand_label_buffer, rand_label_), axis=0)
            # adding additional random samples to both
            (rand_data_,
             rand_label_) = self.get_random_samples(opt.num_rand_extra)
            if (opt.num_rand_extra + self.adv_data_buffer.shape[0] >
                    opt.max_buffer_size):
                self.adv_data_buffer = self.adv_data_buffer[
                    opt.num_rand_extra:, :]
                self.adv_label_buffer = self.adv_label_buffer[
                    opt.num_rand_extra:, :]
                self.rand_data_buffer = self.rand_data_buffer[
                    opt.num_rand_extra:, :]
                self.rand_label_buffer = self.rand_label_buffer[
                    opt.num_rand_extra:, :]
            self.adv_data_buffer = torch.cat(
                (self.adv_data_buffer, rand_data_), axis=0)
            self.adv_label_buffer = torch.cat(
                (self.adv_label_buffer, rand_label_), axis=0)
            self.rand_data_buffer = torch.cat(
                (self.rand_data_buffer, rand_data_), axis=0)
            self.rand_label_buffer = torch.cat(
                (self.rand_label_buffer, rand_label_), axis=0)
            for learning_iter_num in range(opt.num_steps_between_sampling):
                assert (self.adv_data_buffer.shape[0] ==
                        self.rand_data_buffer.shape[0])
                assert (self.adv_label_buffer.shape[0] ==
                        self.rand_label_buffer.shape[0])
                data_i = np.random.choice(self.adv_data_buffer.shape[0],
                                          min(self.adv_data_buffer.shape[0],
                                              opt.batch_size),
                                          replace=False)
                adv_data = self.adv_data_buffer[data_i, :]
                adv_label = self.adv_label_buffer[data_i, :]
                y_pred_adv = self.robust_model(adv_data)
                robust_loss = self.robust_mse(y_pred_adv, adv_label)
                self.robust_optimizer.zero_grad()
                robust_loss.backward()
                self.robust_optimizer.step()
                # pick batch size data from buffer randomly
                rand_data = self.rand_data_buffer[data_i, :]
                rand_label = self.rand_label_buffer[data_i, :]
                y_pred_rand = self.baseline_model(rand_data)
                baseline_loss = self.baseline_mse(y_pred_rand, rand_label)
                self.baseline_optimizer.zero_grad()
                baseline_loss.backward()
                self.baseline_optimizer.step()
                if logging:
                    self.writer.add_scalars(
                        'Train', {
                            'baseline': baseline_loss.item(),
                            'robust': robust_loss.item()
                        }, self.num_total_iter_done)
                    with torch.no_grad():
                        baseline_validation_loss = self.baseline_mse(
                            self.baseline_model(self.x_samples_validation),
                            self.v_samples_validation)
                        robust_validation_loss = self.robust_mse(
                            self.robust_model(self.x_samples_validation),
                            self.v_samples_validation)
                    bvl = baseline_validation_loss.item()
                    rvl = robust_validation_loss.item()
                    self.writer.add_scalars('Validation', {
                        'baseline': bvl,
                        'robust': rvl
                    }, self.num_total_iter_done)
                    self.writer.flush()
                    self.robust_loss_log.append(robust_loss.item())
                    self.baseline_loss_log.append(baseline_loss.item())
                    self.robust_val_loss_log.append(rvl)
                    self.baseline_val_loss_log.append(bvl)
                # keeping track of iterations
                self.num_total_iter_done += 1
                iter_num += 1
                if iter_num >= opt.num_iter_desired:
                    break

    def get_state(self):
        """
        Returns various states of the learning process for logging
        """
        with torch.no_grad():
            state = dict()
            state["adv_data_buffer"] = self.adv_data_buffer.clone()
            state["adv_label_buffer"] = self.adv_label_buffer.clone()
            state["rand_data_buffer"] = self.rand_data_buffer.clone()
            state["rand_label_buffer"] = self.rand_label_buffer.clone()
            state["robust_loss_log"] = copy.deepcopy(self.robust_loss_log)
            state["robust_val_loss_log"] = copy.deepcopy(
                self.robust_val_loss_log)
            state["robust_buffer_loss"] = torch.pow(
                self.adv_label_buffer.squeeze() -
                self.robust_model(self.adv_data_buffer).squeeze(), 2).clone()
            state["robust_val_loss"] = torch.pow(
                self.v_samples_validation.squeeze() -
                self.robust_model(self.x_samples_validation).squeeze(),
                2).clone()
            state["baseline_loss_log"] = copy.deepcopy(self.baseline_loss_log)
            state["baseline_val_loss_log"] = copy.deepcopy(
                self.baseline_val_loss_log)
            state["baseline_buffer_loss"] = torch.pow(
                self.rand_label_buffer.squeeze() -
                self.baseline_model(self.rand_data_buffer).squeeze(),
                2).clone()
            state["baseline_val_loss"] = torch.pow(
                self.v_samples_validation.squeeze() -
                self.baseline_model(self.x_samples_validation).squeeze(),
                2).clone()
            x_adv_opt_loss_log = []
            for (x, label) in self.x_adv_opt_log:
                x_adv_opt_loss_log.append(
                    (x.clone(),
                     torch.pow(
                         label.squeeze() - self.robust_model(x).squeeze(),
                         2).clone()))
            state["x_adv_opt_loss_log"] = x_adv_opt_loss_log
        return state

    def train_no_benchmark(self, opt, logging=True):
        """
        @param opt instance of AdversarialWithBaselineTrainingOptions
        @param logging Boolean whether or not to log with tensorboard
        """
        assert (isinstance(opt, AdversarialWithBaselineTrainingOptions))
        if self.adv_data_buffer.shape[0] < opt.init_buffer_size:
            (rand_data_, rand_label_
             ) = self.get_random_samples(opt.init_buffer_size -
                                         self.adv_data_buffer.shape[0])
            self.adv_data_buffer = torch.cat(
                (self.adv_data_buffer, rand_data_), axis=0)
            self.adv_label_buffer = torch.cat(
                (self.adv_label_buffer, rand_label_), axis=0)
        if self.num_total_iter_done == 0:
            if opt.init_num_train_steps > 0:
                for init_iter_num in range(opt.init_num_train_steps):
                    data_i = np.random.choice(
                        self.adv_data_buffer.shape[0],
                        min(self.adv_data_buffer.shape[0], opt.batch_size),
                        replace=False)
                    adv_data = self.adv_data_buffer[data_i, :]
                    adv_label = self.adv_label_buffer[data_i, :]
                    y_pred_adv = self.robust_model(adv_data)
                    robust_loss = self.robust_mse(y_pred_adv, adv_label)
                    self.robust_optimizer.zero_grad()
                    robust_loss.backward()
                    self.robust_optimizer.step()
                    self.writer.add_scalars('InitialTrain',
                                            {'robust': robust_loss.item()},
                                            init_iter_num)
                    with torch.no_grad():
                        robust_validation_loss = self.robust_mse(
                            self.robust_model(self.x_samples_validation),
                            self.v_samples_validation)
                    rvl = robust_validation_loss.item()
                    self.writer.add_scalars('InitialValidation',
                                            {'robust': rvl}, init_iter_num)
                    self.writer.flush()
        iter_num = 0
        while iter_num < opt.num_iter_desired:
            (adv_data_, adv_label_) = self.get_adversarial_samples(opt)
            num_new_adv_samples = adv_data_.shape[0]
            if (num_new_adv_samples + self.adv_data_buffer.shape[0] >
                    opt.max_buffer_size):
                self.adv_data_buffer = self.adv_data_buffer[
                    num_new_adv_samples:, :]
                self.adv_label_buffer = self.adv_label_buffer[
                    num_new_adv_samples:, :]
            self.adv_data_buffer = torch.cat((self.adv_data_buffer, adv_data_),
                                             axis=0)
            self.adv_label_buffer = torch.cat(
                (self.adv_label_buffer, adv_label_), axis=0)
            # adding additional random samples to both
            (rand_data_,
             rand_label_) = self.get_random_samples(opt.num_rand_extra)
            if (opt.num_rand_extra + self.adv_data_buffer.shape[0] >
                    opt.max_buffer_size):
                self.adv_data_buffer = self.adv_data_buffer[
                    opt.num_rand_extra:, :]
                self.adv_label_buffer = self.adv_label_buffer[
                    opt.num_rand_extra:, :]
            self.adv_data_buffer = torch.cat(
                (self.adv_data_buffer, rand_data_), axis=0)
            self.adv_label_buffer = torch.cat(
                (self.adv_label_buffer, rand_label_), axis=0)
            for learning_iter_num in range(opt.num_steps_between_sampling):
                data_i = np.random.choice(self.adv_data_buffer.shape[0],
                                          min(self.adv_data_buffer.shape[0],
                                              opt.batch_size),
                                          replace=False)
                adv_data = self.adv_data_buffer[data_i, :]
                adv_label = self.adv_label_buffer[data_i, :]
                y_pred_adv = self.robust_model(adv_data)
                robust_loss = self.robust_mse(y_pred_adv, adv_label)
                self.robust_optimizer.zero_grad()
                robust_loss.backward()
                self.robust_optimizer.step()
                if logging:
                    self.writer.add_scalars('Train',
                                            {'robust': robust_loss.item()},
                                            self.num_total_iter_done)
                    with torch.no_grad():
                        robust_validation_loss = self.robust_mse(
                            self.robust_model(self.x_samples_validation),
                            self.v_samples_validation)
                    rvl = robust_validation_loss.item()
                    self.writer.add_scalars('Validation', {'robust': rvl},
                                            self.num_total_iter_done)
                    self.writer.flush()
                    self.robust_loss_log.append(robust_loss.item())
                    self.robust_val_loss_log.append(rvl)
                # keeping track of iterations
                self.num_total_iter_done += 1
                iter_num += 1
                if iter_num >= opt.num_iter_desired:
                    break
