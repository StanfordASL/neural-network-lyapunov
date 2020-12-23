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


def train_depth_model(u, d, dn, nf, config, delta_dynamics=False, local=False,
                      weighted=False, resume=False, resume_file=None):
    """
    @param nf: hidden layer neuron
    @param delta_dynamics: bool. True: train different between current and next
    depth measurements
    @param local: bool. True: train with locality model(CNN); False: train
    with FNN
    @param weighted: bool. True: weighted loss.
    @param resume: bool. True: resume training a pre-trained model
    """
    input_dim = u.shape[1] + d.shape[1]
    model_name = "depth_2_layer_" + str(nf) + "_neuron"

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
            nn.Linear(input_dim, nf), nn.LeakyReLU(0.1),  # nn.ReLU(),
            # nn.Linear(nf, nf), nn.LeakyReLU(0.1),  # nn.ReLU(),
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
        loss_fn = nn.MSELoss()  # reduction="sum")
    elif config["loss_type"] == "l1":
        loss_fn = nn.L1Loss()
    elif config["loss_type"] == "smoothl1":
        loss_fn = nn.SmoothL1Loss()

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
            d_pred = model(d_tensor[index, :],
                           u_tensor[index, :])
        else:
            d_pred = model(torch.cat((d_tensor[index, :],
                                      u_tensor[index, :]), 1))
        if delta_dynamics:
            target = dn_tensor[index, :] - \
                d_tensor[index, :]
            model_name += "_delta_dynamics"
        else:
            target = dn_tensor[index, :]

        if weighted:
            weights = calculate_weight(
                d[index, :])
            diff = d_pred - target
            loss = loss_fn(weights * diff, torch.zeros_like(diff))
            model_name += '_weighted_loss'
        else:
            loss = loss_fn(d_pred, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        writer.add_scalar(
            "Loss/train_" + model_name,
            loss,
            epoch)

    writer.flush()
    return model, model_name


def car_dynamics_training(u, x, xn, nf):
    # TODO(lu): combine with train_dubins_car_demo.train_forward_model
    # Input to NN:  [theta, velocity, turning_rate]
    # Output of NN: [delta_position_x, delta_position_y]
    input_dim = u.shape[1] + 1
    model = nn.Sequential(nn.Linear(input_dim, nf), nn.ReLU(),
                          nn.Linear(nf, nf), nn.ReLU(),
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
            delta_p_pred = model(torch.cat((x_tensor[i * T_horizon:(i + 1) *
                                                     T_horizon,
                                                     2].view(T_horizon, 1),
                                            u_tensor[i * T_horizon:(i + 1) *
                                                     T_horizon, :]), 1))
            # x[n+ 1] =φdyn(x[n],u[n])−φdyn(x∗,u∗) + x∗
            loss = loss_fn(
                delta_p_pred - model(torch.from_numpy(
                    np.zeros((T_horizon, 3)))),
                xn_tensor[i * T_horizon:(i + 1) * T_horizon, :-1] -
                x_tensor[i * T_horizon:(i + 1) * T_horizon, :-1])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        writer.add_scalar("Loss/train_car_delta_dynamics", loss, epoch)
    writer.flush()
    return model


def visualize_model(depth_model, u, d, next_d):
    u_tensor = torch.from_numpy(u.T)
    d_tensor = torch.from_numpy(d.T)
    d_pred = depth_model(torch.cat((u_tensor, d_tensor)))
    plt.subplot(121)
    plt.plot(d_pred.detach().numpy())
    plt.title('Depth Sensor Model Prediction')
    plt.subplot(122)
    plt.plot(next_d)
    plt.title('Actual Depth Sensor')
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
    norms = (np.linalg.norm(X - means, axis=0)
             / math.sqrt(X.shape[0]))
    return (X - means) / norms


def load_config():
    config = {}
    config['num_epoch'] = 25000
    config['batch_size'] = 256
    config["loss_type"] = "mse"
    config['scheduler'] = {'enabled': False, 'step_size': 700, 'gamma': 0.1}
    return config


def down_sample(X, factor=1):
    low = 24  # 0
    high = 74  # 100
    return X[:, np.arange(low, high, factor).astype(int)]


def calculate_weight(d, option="normal"):
    # TODO: different weights
    if option == "normal":
        weights = torch.from_numpy(10 * scipy.stats.norm.pdf(d))
    return weights


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
    nf = 50
    nf_car = 8

    parser = argparse.ArgumentParser()
    parser.add_argument("--save_model", action="store_true")
    args = parser.parse_args()

    writer = SummaryWriter(filename_suffix="sensor_dynamics")
    depth_model, model_name = train_depth_model(
        u, d, dn, nf, config)
    if args.save_model:
        torch.save(depth_model.state_dict(), 'depth_model/' + model_name)
    # car_model = car_dynamics_training(u, x, xn, nf_car)
    # torch.save(car_model.state_dict(), 'car_model/'+args.model_name)
    writer.close()
    # index = 888
    # visualize_model(depth_model, u[:, index],
    #                 d[:, index], d[:, index + 1])
