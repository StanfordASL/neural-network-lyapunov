import robust_value_approx.adversarial_sample as adversarial_sample

import torch
import copy
import numpy as np
from torch.utils.tensorboard import SummaryWriter


class AdversarialWithBaselineTrainingOptions:
    def __init__(self):
        # number of pure l2 fit iteration to be done over the dataset
        self.num_l2_iter_desired = 1000
        # number of adversarial iteration to be done
        self.num_robust_iter_desired = 1500
        # loss is (1 - robust_weight)*MSE(data)+robust_weight*MSE(adversaries)
        self.robust_weight = .5
        # number of iterations to find adversarial examples
        self.x_adv_num_iter = 3
        # learning rate used to find adversarial examples
        self.x_adv_lr = .25
        # number of randomly generated samples per iteration for the baseline
        self.num_x_rand = self.x_adv_num_iter
        # size of the buffer to keep a few old adversarial examples
        self.max_buffer_size = 5 * self.x_adv_num_iter


class AdversarialWithBaseline:
    def __init__(self, vf, x0_lo, x0_up,
                 x_samples_train, v_samples_train,
                 x_samples_validation, v_samples_validation,
                 nn_width=16, nn_depth=1,
                 batch_size=30, learning_rate=1e-3):
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
        @param x_samples_train, state input to train over
        @param v_samples_train, corresponding value functions
        @param x_samples_validation, state input to validate over
        @param v_samples_validation, corresponding value functions
        Model Params:
        @param nn_width Integer number of units per layers
        @param nn_depth Integer number of HIDDEN neural network layers
        Training Params:
        @param batch_size Integer batch size used duing training
        @param learning_rate Float learning rate (Adam is used)
        """
        self.vf = vf
        self.V = vf.get_value_function()
        self.x_dim = vf.sys.x_dim
        self.x0_lo = x0_lo
        self.x0_up = x0_up
        self.dtype = vf.dtype
        self.train_data_set = torch.utils.data.TensorDataset(x_samples_train,
                                                             v_samples_train)
        self.train_data_loader = torch.utils.data.DataLoader(
            self.train_data_set, batch_size=batch_size, shuffle=True)
        self.x_samples_validation = x_samples_validation
        self.v_samples_validation = v_samples_validation
        nn_layers = [torch.nn.Linear(self.x_dim, nn_width), torch.nn.ReLU()]
        for i in range(nn_depth):
            nn_layers += [torch.nn.Linear(nn_width, nn_width), torch.nn.ReLU()]
        nn_layers += [torch.nn.Linear(nn_width, 1)]
        self.baseline_model = torch.nn.Sequential(*nn_layers).double()
        self.baseline_optimizer = torch.optim.Adam(
            self.baseline_model.parameters(), lr=learning_rate)
        self.baseline_mse = torch.nn.MSELoss(reduction="mean")
        self.robust_model = copy.deepcopy(self.baseline_model)
        self.robust_optimizer = torch.optim.Adam(
            self.robust_model.parameters(), lr=learning_rate)
        self.robust_mse = torch.nn.MSELoss(reduction="mean")
        self.as_generator = adversarial_sample.AdversarialSampleGenerator(
            vf, x0_lo, x0_up)
        self.writer = SummaryWriter()
        self.num_total_iter_done = 0
        self.rand_data_buffer = torch.Tensor(0, self.x_dim).type(self.dtype)
        self.rand_label_buffer = torch.Tensor(0, 1).type(self.dtype)
        self.adv_data_buffer = torch.Tensor(0, self.x_dim).type(self.dtype)
        self.adv_label_buffer = torch.Tensor(0, 1).type(self.dtype)

    def get_random_x0(self):
        """
        @returns a Tensor between the initial state bounds of the training
        """
        return torch.rand(self.x_dim, dtype=self.dtype) *\
            (self.x0_up - self.x0_lo) + self.x0_lo

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
        return(rand_data, rand_label)

    def train(self, opt, logging=True):
        """
        @param opt instance of AdversarialWithBaselineTrainingOptions
        @param logging Boolean whether or not to log with tensorboard
        """
        assert(isinstance(opt, AdversarialWithBaselineTrainingOptions))
        num_l2_iter_done = 0
        num_robust_iter_done = 0
        training_done = False
        while True:
            if training_done:
                break
            for batch_data, batch_label in self.train_data_loader:
                if (num_l2_iter_done >= opt.num_l2_iter_desired and
                        num_robust_iter_done >= opt.num_robust_iter_desired):
                    training_done = True
                    break
                if num_l2_iter_done < opt.num_l2_iter_desired:
                    # baseline
                    y_pred_baseline = self.baseline_model(batch_data)
                    baseline_loss = self.baseline_mse(
                        y_pred_baseline, batch_label)
                    # robust
                    y_pred_robust = self.robust_model(batch_data)
                    robust_loss = self.robust_mse(y_pred_robust, batch_label)
                    num_l2_iter_done += 1
                else:
                    # baseline
                    (rand_data_, rand_label_) = self.get_random_samples(
                        opt.num_x_rand)
                    self.rand_data_buffer = torch.cat(
                        (self.rand_data_buffer, rand_data_), axis=0)
                    self.rand_label_buffer = torch.cat(
                        (self.rand_label_buffer, rand_label_), axis=0)
                    rand_data_i = np.random.choice(
                        self.rand_data_buffer.shape[0], opt.num_x_rand,
                        replace=False)
                    rand_data = self.rand_data_buffer[rand_data_i, :]
                    rand_label = self.rand_label_buffer[rand_data_i, :]
                    y_pred_rand = self.baseline_model(rand_data)
                    y_pred_baseline = self.baseline_model(batch_data)
                    baseline_loss = (1. - opt.robust_weight) *\
                        self.baseline_mse(y_pred_baseline, batch_label) +\
                        opt.robust_weight *\
                        self.baseline_mse(y_pred_rand, rand_label)
                    # robust
                    (epsilon_squ,
                     adv_data_,
                     adv_label_) = self.as_generator.get_squared_bound_sample(
                        self.robust_model, num_iter=opt.x_adv_num_iter,
                        learning_rate=opt.x_adv_lr,
                        x_adv0=self.get_random_x0())
                    self.adv_data_buffer = torch.cat(
                        (self.adv_data_buffer, adv_data_), axis=0)
                    self.adv_label_buffer = torch.cat(
                        (self.adv_label_buffer, adv_label_), axis=0)
                    adv_data_i = np.random.choice(
                        self.adv_data_buffer.shape[0], opt.x_adv_num_iter,
                        replace=False)
                    adv_data = self.adv_data_buffer[adv_data_i, :]
                    adv_label = self.adv_label_buffer[adv_data_i, :]
                    y_pred_adv = self.robust_model(adv_data)
                    y_pred_robust = self.robust_model(batch_data)
                    robust_loss = (1. - opt.robust_weight) *\
                        self.robust_mse(y_pred_robust, batch_label) +\
                        opt.robust_weight *\
                        self.robust_mse(y_pred_adv, adv_label)
                    num_robust_iter_done += 1
                self.baseline_optimizer.zero_grad()
                baseline_loss.backward()
                self.baseline_optimizer.step()
                self.robust_optimizer.zero_grad()
                robust_loss.backward()
                self.robust_optimizer.step()
                # keeping track of iterations
                self.num_total_iter_done += 1
                if self.rand_data_buffer.shape[0] >= opt.max_buffer_size:
                    self.rand_data_buffer = self.rand_data_buffer[
                        -opt.max_buffer_size:, :]
                    self.rand_label_buffer = self.rand_label_buffer[
                        -opt.max_buffer_size:, :]
                if self.adv_data_buffer.shape[0] >= opt.max_buffer_size:
                    self.adv_data_buffer = self.adv_data_buffer[
                        -opt.max_buffer_size:, :]
                    self.adv_label_buffer = self.adv_label_buffer[
                        -opt.max_buffer_size:, :]
                if logging:
                    self.writer.add_scalars('Train',
                                            {'baseline': baseline_loss.item(),
                                             'robust': robust_loss.item()},
                                            self.num_total_iter_done)
                    if num_robust_iter_done > 0:
                        with torch.no_grad():
                            baseline_validation_loss = self.baseline_mse(
                                self.baseline_model(self.x_samples_validation),
                                self.v_samples_validation)
                            robust_validation_loss = self.robust_mse(
                                self.robust_model(self.x_samples_validation),
                                self.v_samples_validation)
                        bvl = baseline_validation_loss.item()
                        rvl = robust_validation_loss.item()
                        self.writer.add_scalars('Validation',
                                                {'baseline': bvl,
                                                 'robust': rvl},
                                                self.num_total_iter_done)
