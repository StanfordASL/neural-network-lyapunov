import neural_network_lyapunov.worlds as worlds
import neural_network_lyapunov.utils as utils
import neural_network_lyapunov.control_affine_system as mut

import pytinydiffsim as pd
import argparse
import torch
import yaml


def load_multibody(cfg):
    world = pd.TinyWorld()

    urdf_data = pd.TinyUrdfParser().load_urdf(
        worlds.urdf_path(cfg["world"]["urdf"]))

    mb = pd.TinyMultiBody(cfg["world"]["floating"])
    urdf2mb = pd.UrdfToMultiBody2()
    urdf2mb.convert2(urdf_data, world, mb)

    actuation_mask = [
        j.joint_name in cfg["world"]["actuated_joints"] for
        j in urdf_data.joints]

    return mb, actuation_mask


def generate_dataset(cfg, mb, actuation_mask):
    dtype = torch.float64

    q_dim = mb.q.size()
    qd_dim = mb.qd.size()
    tau_dim = mb.tau.size()
    gravity = pd.Vector3(0., 0., -9.81)

    q = pd.VectorX(q_dim)
    qd = pd.VectorX(qd_dim)
    tau = pd.VectorX(tau_dim)
    u_dim = sum(actuation_mask)

    xu_lo = torch.cat([torch.tensor(cfg["data"]["q_lo"], dtype=dtype),
                       torch.tensor(cfg["data"]["qd_lo"], dtype=dtype),
                       torch.tensor(cfg["data"]["u_lo"], dtype=dtype)])
    xu_up = torch.cat([torch.tensor(cfg["data"]["q_up"], dtype=dtype),
                       torch.tensor(cfg["data"]["qd_up"], dtype=dtype),
                       torch.tensor(cfg["data"]["u_up"], dtype=dtype)])

    samples = utils.uniform_sample_in_box(
        xu_lo, xu_up, cfg["data"]["num_samples"])

    data = []
    labels = []
    for k in range(samples.shape[0]):
        q_sample = samples[k, :q_dim]
        qd_sample = samples[k, q_dim:q_dim+qd_dim]
        u_sample = samples[k, q_dim+qd_dim:q_dim+qd_dim+u_dim]
        for i in range(q_dim):
            q[i] = q_sample[i]
        for i in range(qd_dim):
            qd[i] = qd_sample[i]
        j = 0
        for i in range(tau_dim):
            if actuation_mask[i]:
                tau[i] = u_sample[j]
                j += 1
            else:
                tau[i] = 0
        assert(j == u_dim)

        mb.q = q
        mb.qd = qd
        mb.tau = tau
        pd.forward_dynamics(mb, gravity)
        qdd_sample = torch.tensor(
            [mb.qdd[i] for i in range(mb.qdd.size())], dtype=dtype)
        label = torch.cat([qd_sample, qdd_sample])
        data.append(samples[k, :].unsqueeze(0))
        labels.append(label.unsqueeze(0))

    data = torch.cat(data, dim=0)
    labels = torch.cat(labels, dim=0)
    dataset = torch.utils.data.TensorDataset(data, labels)

    return dataset


def get_models(cfg):
    dtype = torch.float64
    x_dim = len(cfg['data']['q_lo']) + len(cfg['data']['qd_lo'])
    u_dim = len(cfg['data']['u_lo'])
    hid_f = tuple([x_dim] + cfg['train']['f_hid'] + [x_dim])
    hid_G = tuple([x_dim] + cfg['train']['G_hid'] + [x_dim * u_dim])

    forward_model_f = utils.setup_relu(hid_f,
                                       params=None,
                                       bias=True,
                                       negative_slope=0.01,
                                       dtype=dtype)
    forward_model_G = utils.setup_relu(hid_G,
                                       params=None,
                                       bias=True,
                                       negative_slope=0.01,
                                       dtype=dtype)

    return forward_model_f, forward_model_G


def train_models(cfg, dataset, forward_model_f, forward_model_G,
                 verbose=False):
    dtype = torch.float64
    x_equ = torch.tensor(cfg['train']['x_equ'], dtype=dtype)
    u_equ = torch.tensor(cfg['train']['u_equ'], dtype=dtype)
    mut.train_control_affine_forward_model(
        forward_model_f, forward_model_G, x_equ, u_equ,
        dataset, cfg['train']['epoch'], cfg['train']['lr'],
        batch_size=cfg['train']['batch_size'],
        verbose=verbose)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="trains control affine dynamics models using " +
        "Tiny Differentiable Simulator (~Bullet)")
    parser.add_argument("cfg_path", type=str)
    parser.add_argument("--generate_dataset", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    with open(args.cfg_path, 'r') as cfg_file:
        cfg = yaml.safe_load(cfg_file)

    if args.generate_dataset:
        mb, actuation_mask = load_multibody(cfg)
        dataset = generate_dataset(cfg, mb, actuation_mask)
        torch.save(dataset, cfg['world']['name'] + "_dataset.pt")
    else:
        dataset = torch.load(cfg['world']['name'] + "_dataset.pt")

    forward_model_f, forward_model_G = get_models(cfg)
    test_loss = train_models(
        cfg, dataset, forward_model_f, forward_model_G, verbose=args.verbose)

    models = [forward_model_f, forward_model_G]
    torch.save(models, cfg['world']['name'] + "_models.pt")
