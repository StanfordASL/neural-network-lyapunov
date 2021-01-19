import torch.nn as nn
import torch
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
# from torch.optim.lr_scheduler import StepLR
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import argparse


class LocalityNet(nn.Module):
    def __init__(self, input_dim, output_dim, nf, neighbors=5):
        super(LocalityNet, self).__init__()
        self.c = nn.Conv1d(1, 1, neighbors, padding=int(
            (neighbors - 1) / 2))  # ,padding_mode='circular')
        self.fc1 = nn.Linear(input_dim, nf)
        self.fc2 = nn.Linear(nf, nf)
        self.fc3 = nn.Linear(nf, output_dim)

    def forward(self, d, u):
        d1 = self.c(d)
        x = torch.cat((d1, u), dim=2)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def train_depth_model(u,
                      d,
                      dn,
                      nf,
                      config,
                      delta_dynamics=False,
                      local=False,
                      asymmetric_loss=None,
                      resume=False,
                      resume_file=None):
    """
    @param nf: hidden layer neuron
    @param delta_dynamics: bool. True: train different between current and next
    depth measurements
    @param local: bool. True: train with locality model(CNN); False: train
    with FNN
    @param asymmetric_loss: 'dist','underapprox'.
    'dist': weighted loss.smaller dn has larger loss.
    'underapprox': less cost on errors when prediction is closer than actual
    measurement. Asymmetric cost such that the model favors conservative
    estimation of obstacle distances.
    @param resume: bool. True: resume training a pre-trained model
    """
    input_dim = u.shape[1] + d.shape[1]
    model_name = "depth_3_layer_" + str(nf) + "_neuron"

    if delta_dynamics:
        model_name += "_delta_dynamics"
    if asymmetric_loss == "dist":
        model_name += "_dist_weighted"
    elif asymmetric_loss == "underapprox":
        model_name += "_under_approx"

    if local:
        # Use locality model
        model = LocalityNet(input_dim, d.shape[1], nf)
        device = next(model.parameters()).device
        u_tensor = torch.from_numpy(u).to(device)
        d_tensor = torch.from_numpy(d).to(device)
        dn_tensor = torch.from_numpy(dn).to(device)
        u_tensor.resize_((u.shape[0], 1, u.shape[1]))
        d_tensor.resize_((d.shape[0], 1, d.shape[1]))
        dn_tensor.resize_((dn.shape[0], 1, dn.shape[1]))
        model_name += "_local_model"
    else:
        model = nn.Sequential(  # nn.BatchNorm1d(input_dim),
            nn.Linear(input_dim, nf),
            nn.LeakyReLU(0.1),  # nn.ReLU(),
            nn.Linear(nf, nf),
            nn.LeakyReLU(0.1),  # nn.ReLU(),
            # nn.Linear(nf * 2, nf), nn.LeakyReLU(0.1), #nn.ReLU(),
            # nn.Linear(nf * 2, nf), nn.ReLU(),
            nn.Linear(nf, d.shape[1]))
        device = next(model.parameters()).device
        u_tensor = torch.from_numpy(u).to(device)
        d_tensor = torch.from_numpy(d).to(device)
        dn_tensor = torch.from_numpy(dn).to(device)

    # Resume training
    if resume:
        assert (isinstance(resume_file, str))
        model.load_state_dict(torch.load('depth_model/' + resume_file))
        model.eval()
    model.double()

    if config["loss_type"] == "mse":
        if asymmetric_loss is not None:
            loss_fn = nn.MSELoss(reduction="none")
        else:
            loss_fn = nn.MSELoss()  # reduction="sum")
    elif config["loss_type"] == "l1":
        if asymmetric_loss is not None:
            loss_fn = nn.L1Loss(reduction="none")
        else:
            loss_fn = nn.L1Loss()
        model_name += "_l1"
    elif config["loss_type"] == "smoothL1":
        if asymmetric_loss is not None:
            loss_fn = nn.SmoothL1Loss(reduction="none")
        else:
            loss_fn = nn.SmoothL1Loss()
        model_name += "_smoothL1"

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    num_epoch = config["num_epoch"]
    # numEnvs = 100 - 2
    batch_size = config['batch_size']
    num_samples = d.shape[0]

    for epoch in range(num_epoch):
        if epoch % 100 == 0:
            print(epoch)
        index = np.random.randint(0, num_samples, batch_size)

        if local:
            d_pred = model(d_tensor[index, :], u_tensor[index, :])
        else:
            d_pred = model(
                torch.cat((d_tensor[index, :], u_tensor[index, :]), 1))
        if delta_dynamics:
            target = dn_tensor[index, :] - \
                d_tensor[index, :]
        else:
            target = dn_tensor[index, :]

        if asymmetric_loss == "dist":
            weights = calculate_weight(dn[index, :], std=1.6)
            loss_vector = loss_fn(d_pred, target)
            loss = torch.sum(weights * loss_vector) / torch.numel(loss_vector)
            # print(torch.sum(loss_fn(d_pred, dn_tensor-d_tensor))
            #       /torch.numel(dn_tensor))
        elif asymmetric_loss == "underapprox":
            weights = calculate_weight(d_pred - target,
                                       "underapprox",
                                       slope=0.2)
            loss_vector = loss_fn(d_pred, target)
            loss = torch.sum(weights * loss_vector) / torch.numel(loss_vector)
        else:
            loss = loss_fn(d_pred, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        writer.add_scalar("Loss/train_" + model_name, loss, epoch)

    writer.flush()
    return model, model_name


def car_dynamics_training(u, x, xn, nf):
    # TODO(lu): combine with train_unicycle_demo.train_forward_model
    # Input to NN:  [theta, velocity, turning_rate]
    # Output of NN: [delta_position_x, delta_position_y]
    input_dim = u.shape[1] + 1
    model = nn.Sequential(
        nn.Linear(input_dim, nf),
        nn.ReLU(),
        nn.Linear(nf, nf),
        nn.ReLU(),
        # nn.Linear(nf, nf), nn.ReLU(),
        nn.Linear(nf, 2))
    model.double()
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    device = next(model.parameters()).device
    u_tensor = torch.from_numpy(u).to(device)
    x_tensor = torch.from_numpy(x).to(device)
    xn_tensor = torch.from_numpy(xn).to(device)
    num_epoch = 1000
    numEnvs = 100 - 2
    T_horizon = 1000
    for epoch in range(num_epoch):
        if epoch % 100 == 0:
            print(epoch)
        for i in range(numEnvs):
            delta_p_pred = model(
                torch.cat((x_tensor[i * T_horizon:(i + 1) * T_horizon, 2].view(
                    T_horizon,
                    1), u_tensor[i * T_horizon:(i + 1) * T_horizon, :]), 1))
            # x[n+ 1] =φdyn(x[n],u[n])−φdyn(x∗,u∗) + x∗
            loss = loss_fn(
                delta_p_pred - model(torch.from_numpy(np.zeros(
                    (T_horizon, 3)))),
                xn_tensor[i * T_horizon:(i + 1) * T_horizon, :-1] -
                x_tensor[i * T_horizon:(i + 1) * T_horizon, :-1])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        writer.add_scalar("Loss/train_car_delta_dynamics", loss, epoch)
    writer.flush()
    return model


def visualize_model(depth_model,
                    u,
                    d,
                    next_d,
                    local=False,
                    delta_dynamics=False,
                    model_name=None):
    u_tensor = torch.from_numpy(u.T)
    d_tensor = torch.from_numpy(d.T)
    if local:
        u_tensor.resize_(1, 1, u.shape[0])
        d_tensor.resize_(1, 1, d.shape[0])
        pred = depth_model(d_tensor, u_tensor)[0, 0]
    else:
        pred = depth_model(torch.cat((d_tensor, u_tensor)))
    if delta_dynamics:
        d_pred = pred.detach().numpy() + d
    else:
        d_pred = pred.detach().numpy()
    plt.subplot(121)
    plt.plot(d_pred, label='next')
    plt.plot(d, 'r--', label='current')
    plt.xlabel('Ray Index')
    plt.ylabel('Depth Measurement (m)')
    plt.title('Depth Sensor Model Prediction')
    plt.legend(loc='best')
    plt.subplot(122)
    plt.plot(next_d, label='next')
    plt.plot(d, 'r--', label='current')
    plt.xlabel('Ray Index')
    plt.ylabel('Depth Measurement (m)')
    plt.title('Actual Depth Sensor')
    plt.legend(loc='best')
    if model_name is not None:
        assert (isinstance(model_name, str))
        plt.savefig(model_name + '.png')
    plt.show()
    # plt.subplot(133)
    plt.plot(next_d, label='actual')
    plt.plot(d_pred, label='prediction')
    plt.xlabel('Ray Index')
    plt.ylabel('Depth Measurement (m)')
    plt.title('Actual vs Prediction at next time step')
    plt.legend(loc='best')
    if model_name is not None:
        plt.savefig(model_name + '_compare.png')
    plt.show()


def standardize(X, axis=0):
    mean = np.mean(X, axis=axis)
    std = np.std(X, axis=axis)
    X = X - mean
    X = X / std + 1e-10
    return X


def whiten(X):
    import math
    means = np.mean(X, axis=0)
    norms = (np.linalg.norm(X - means, axis=0) / math.sqrt(X.shape[0]))
    return (X - means) / norms


def down_sample(X, factor=1):
    low = 25  # 0
    high = 76  # 100
    return X[:, np.arange(low, high, factor).astype(int)]


def calculate_weight(d, option="normal", std=1, slope=0.1):
    # TODO: different weights
    if option == "normal":
        weights = torch.from_numpy(10 * scipy.stats.norm.pdf(d, scale=std))
    elif option == "underapprox":
        leaky_relu = nn.LeakyReLU(slope)
        weights = torch.abs(leaky_relu(d))
    return weights


def load_config():
    config = {}
    config['num_epoch'] = 25000
    config['batch_size'] = 256  # 1024
    config["loss_type"] = "mse"
    config['scheduler'] = {'enabled': False, 'step_size': 700, 'gamma': 0.1}
    return config


if __name__ == "__main__":
    u = np.load('data/u_100_env_0.01dt_uniform_sample.npy')
    d = np.load('data/depth_100_env_0.01dt_uniform_sample.npy')
    x = np.load('data/state_100_env_0.01dt_uniform_sample.npy')
    dn = np.load('data/next_depth_100_env_0.01dt_uniform_sample.npy')
    xn = np.load('data/next_state_100_env_0.01dt_uniform_sample.npy')
    # Truncate depth measurements above 5m
    d[np.where(d > 5)] = 5
    dn[np.where(dn > 5)] = 5

    d = down_sample(d, 2)
    dn = down_sample(dn, 2)
    # u = standardize(u)
    # d = standardize(d)
    config = load_config()
    nf = 100
    nf_car = 8
    delta_dynamics = True
    local = True

    parser = argparse.ArgumentParser()
    parser.add_argument("--save_model", action="store_true")
    parser.add_argument("--visualize_model",
                        type=str,
                        help="path to the model file for visualization")
    args = parser.parse_args()

    if args.visualize_model is not None:
        input_dim = u.shape[1] + d.shape[1]
        if local:
            model = LocalityNet(input_dim, d.shape[1], nf)
        else:
            model = nn.Sequential(
                nn.Linear(input_dim, nf),
                nn.LeakyReLU(0.1),
                # nn.Linear(nf, nf), nn.LeakyReLU(0.1),
                nn.Linear(nf, d.shape[1]))
        model.load_state_dict(torch.load('depth_model/' +
                                         args.visualize_model))
        model.eval()
        model.double()
        index = 800
        visualize_model(model,
                        u[index, :],
                        d[index, :],
                        dn[index, :],
                        local=local,
                        delta_dynamics=delta_dynamics)
        index = 22
        visualize_model(model,
                        u[index, :],
                        d[index, :],
                        dn[index, :],
                        local=local,
                        delta_dynamics=delta_dynamics)
    else:
        writer = SummaryWriter(filename_suffix="sensor_dynamics")
        depth_model, model_name = train_depth_model(
            u,
            d,
            dn,
            nf,
            config,
            local=local,
            asymmetric_loss="underapprox",
            delta_dynamics=delta_dynamics)
        # car_model = car_dynamics_training(u, x, xn, nf_car)
        # torch.save(car_model.state_dict(), 'car_model/'+args.model_name)
        writer.close()
        index = 8
        visualize_model(depth_model,
                        u[index, :],
                        d[index, :],
                        dn[index, :],
                        local=local,
                        delta_dynamics=delta_dynamics)
        index = 22
        visualize_model(depth_model,
                        u[index, :],
                        d[index, :],
                        dn[index, :],
                        local=local,
                        delta_dynamics=delta_dynamics,
                        model_name=model_name)

        if args.save_model:
            torch.save(depth_model.state_dict(), 'depth_model/' + model_name)
