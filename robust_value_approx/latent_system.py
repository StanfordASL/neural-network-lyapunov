import gurobipy
import torch
import torch.nn as nn
import torch.distributions as distributions

import robust_value_approx.relu_system as relu_system


class Encoder(nn.Module):

    def __init__(self, z_dim, use_conv, image_width, image_height, grayscale):
        super(Encoder, self).__init__()
        self.z_dim = z_dim
        self.use_conv = use_conv
        self.image_width = image_width
        self.image_height = image_height
        self.grayscale = grayscale
        if self.grayscale:
            self.num_channels_in = 2
        else:
            self.num_channels_in = 6
        if use_conv:
            conv = [
                nn.Conv2d(self.num_channels_in, 32, 5, stride=1, padding=0),
                nn.Conv2d(32, 32, 5, stride=2, padding=0),
                nn.Conv2d(32, 32, 5, stride=2, padding=0),
                nn.Conv2d(32, 10, 5, stride=2, padding=0),
            ]
            self.conv = nn.ModuleList(conv)
            with torch.no_grad():
                x_tmp = torch.rand((1, self.num_channels_in,
                                    self.image_width, self.image_height))
                for c_layer in self.conv:
                    x_tmp = c_layer(x_tmp)
                conv_out_size = x_tmp.shape[1] * x_tmp.shape[2] * \
                    x_tmp.shape[3]
            linear = [
                nn.Linear(conv_out_size, 500),
                nn.Linear(500, self.z_dim * 2),
            ]
            self.linear = nn.ModuleList(linear)
        else:
            self.conv = []
            linear = [
                nn.Linear(self.num_channels_in * self.image_width *
                          self.image_height, 500),
                nn.Linear(500, 500),
                nn.Linear(500, self.z_dim * 2),
            ]
            self.linear = nn.ModuleList(linear)
        self.relu = nn.ReLU()

    def forward(self, x):
        for c_layer in self.conv:
            x = self.relu(c_layer(x))
        x = torch.flatten(x, start_dim=1)
        for l_layer in self.linear[:-1]:
            x = self.relu(l_layer(x))
        x = self.linear[-1](x)
        return x[:, :self.z_dim], x[:, self.z_dim:]


class Decoder(nn.Module):

    def __init__(self, z_dim, use_conv, image_width, image_height, grayscale):
        super(Decoder, self).__init__()
        self.z_dim = z_dim
        self.use_conv = use_conv
        self.image_width = image_width
        self.image_height = image_height
        self.grayscale = grayscale
        if grayscale:
            self.num_channels_out = 2
        else:
            self.num_channels_out = 6
        if use_conv:
            width_in = self.image_width - 4
            height_in = self.image_height - 4
            for k in range(2):
                width_in = int(width_in/2)
                height_in = int(height_in/2)
                width_in = int(width_in - 4)
                height_in = int(height_in - 4)
            self.height_in = height_in
            self.width_in = width_in
            linear = [
                nn.Linear(self.z_dim, 200),
                nn.Linear(200, 1000),
                nn.Linear(1000, self.width_in * self.height_in),
            ]
            self.linear = nn.ModuleList(linear)
            conv = [
                nn.ConvTranspose2d(1, 32, 5, stride=1, padding=0),
                nn.Upsample(scale_factor=2),
                nn.ConvTranspose2d(32, 32, 5, stride=1, padding=0),
                nn.Upsample(scale_factor=2),
                nn.ConvTranspose2d(32, self.num_channels_out, 5,
                                   stride=1, padding=0),
            ]
            self.conv = nn.ModuleList(conv)
        else:
            linear = [
                nn.Linear(self.z_dim, 500),
                nn.Linear(500, 500),
                nn.Linear(500, self.num_channels_out * self.image_width *
                          self.image_height),
            ]
            self.linear = nn.ModuleList(linear)
            self.conv = []
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        for l_layer in self.linear[:-1]:
            x = self.relu(l_layer(x))
        x = self.linear[-1](x)
        if self.use_conv:
            x = x.view(x.shape[0], 1, self.width_in, self.height_in)
            for c_layer in self.conv[:-1]:
                x = self.relu(c_layer(x))
            x = self.conv[-1](x)
        else:
            x = x.view(x.shape[0], self.num_channels_out,
                       self.image_width, self.image_height)
        x = self.sigmoid(x)
        return x


class LatentAutonomousReLUSystem():

    def __init__(self, relu_system, lyapunov,
                 use_conv=False, use_bce=True, use_variational=True,
                 image_width=48, image_height=48, grayscale=True):
        self.relu_system = relu_system
        self.lyapunov = lyapunov
        self.dtype = self.relu_system.dtype
        self.z_dim = self.relu_system.x_dim
        self.V_lambda = 0.
        self.V_eps = .01
        self.z_equilibrium = torch.zeros(self.z_dim, dtype=self.dtype)
        self.image_width = image_width
        self.image_height = image_height
        self.grayscale = grayscale
        self.use_conv = use_conv
        self.use_bce = use_bce
        self.use_variational = use_variational
        self.bce_loss = nn.BCELoss(reduction='mean')
        self.encoder = Encoder(self.z_dim, use_conv=use_conv,
                               image_width=self.image_width,
                               image_height=self.image_height,
                               grayscale=self.grayscale).type(self.dtype)
        self.decoder = Decoder(self.z_dim, use_conv=use_conv,
                               image_width=self.image_width,
                               image_height=self.image_height,
                               grayscale=self.grayscale).type(self.dtype)
        if self.grayscale:
            self.num_channels_in = 2
        else:
            self.num_channels_in = 6
        self.sigmoid = torch.nn.Sigmoid()

    def check_input(self, x):
        assert(len(x.shape) == 4)
        assert(x.shape[1:] == (self.num_channels_in, self.image_width,
                               self.image_height))
        batch_size = x.shape[0]
        return batch_size

    def vae_loss(self, x, x_next, x_decoded, x_next_pred_decoded,
                 z_mu, z_log_var):
        loss = 0.
        x_next_ = torch.cat((x[:, int(self.num_channels_in/2):, :, :],
                             x_next), dim=1)
        if self.use_bce:
            loss += self.bce_loss(x_decoded, x)
            loss += self.bce_loss(x_next_pred_decoded, x_next_)
        else:
            loss += (x - x_decoded).pow(2).mean(dim=[1, 2, 3])[0]
            loss += (x_next_ - x_next_pred_decoded).pow(2).mean(
                dim=[1, 2, 3])[0]
        if self.use_variational:
            loss += torch.mean(-.5 * torch.sum(-torch.pow(z_mu, 2) -
                               torch.exp(z_log_var) + z_log_var + 1., dim=1))
        return loss

    def lyapunov_loss(self):
        lyap_pos_mip = self.lyapunov.lyapunov_positivity_as_milp(
            self.z_equilibrium, self.V_lambda, self.V_eps)[0]
        lyap_pos_mip.gurobi_model.setParam(gurobipy.GRB.Param.OutputFlag,
                                           False)
        lyap_pos_mip.gurobi_model.optimize()
        lyap_der_mip = self.lyapunov.lyapunov_derivative_as_milp(
            self.z_equilibrium, self.V_lambda, self.V_eps)[0]
        lyap_der_mip.gurobi_model.setParam(gurobipy.GRB.Param.OutputFlag,
                                           False)
        lyap_der_mip.gurobi_model.optimize()
        loss = -lyap_pos_mip.compute_objective_from_mip_data_and_solution() +\
            lyap_der_mip.compute_objective_from_mip_data_and_solution()
        return loss

    def vae_forward(self, x):
        batch_size = self.check_input(x)

        z_mu, z_log_var = self.encoder(x)
        if self.use_variational:
            z_std = torch.exp(0.5 * z_log_var)
            eps = torch.autograd.Variable(torch.Tensor(z_mu.shape).normal_())
            z = eps * z_std + z_mu
        else:
            z = z_mu

        z_next = self.relu_system.dynamics_relu(z)

        x_decoded = self.decoder(z)
        x_next_pred_decoded = self.decoder(z_next)

        return x_decoded, x_next_pred_decoded, z_mu, z_log_var

    def rollout(self, x_init, N, clamp=False):
        assert(len(x_init.shape) == 3)
        x_traj = torch.zeros(N+2, int(x_init.shape[0]/2), x_init.shape[1],
                             x_init.shape[2], dtype=self.dtype)
        x_traj[0, :] = x_init[:int(x_init.shape[0]/2), :, :]
        x_traj[1, :] = x_init[int(x_init.shape[0]/2):, :, :]
        for n in range(N):
            with torch.no_grad():
                x_decoded, x_next_pred_decoded, z_mu, z_log_var = \
                    self.vae_forward(torch.cat((x_traj[n, :], x_traj[n+1, :]),
                                               dim=0).unsqueeze(0))
                if clamp:
                    x_next_pred_decoded = torch.clamp(
                        x_next_pred_decoded, 0, 1)
                x_traj[n+2, :] = x_next_pred_decoded[0,
                                                     int(x_init.shape[0]/2):,
                                                     :, :]
        return x_traj
