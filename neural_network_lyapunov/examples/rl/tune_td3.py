import gym
import torch
import os
from ray import tune
from ray.tune.suggest.hyperopt import HyperOptSearch

import neural_network_lyapunov.examples.rl.td3 as td3
from neural_network_lyapunov.examples.quadrotor2d.quadrotor2d_env import \
    Quadrotor2DEnv


def train_td3(config, checkpoint_dir=None):

    def env_fn():
        if config['env'] == 'quadrotor2d':
            return Quadrotor2DEnv()
        else:
            return gym.make(config['env'])

    ac, pi_optimizer, q_optimizer, mean_ret = td3.td3(
        env_fn,
        actor_critic=td3.MLPActorCritic,
        ac_kwargs=dict(hidden_sizes=[config['ac_hid']]*config['ac_l']),
        gamma=config['gamma'],
        seed=config['seed'],
        epochs=config['epochs'],
        max_ep_len=config['max_ep_len'],
        pi_lr=config['pi_lr'],
        q_lr=config['q_lr'],
        batch_size=config['batch_size'],
        polyak=config['polyak'],
        act_noise=config['act_noise'],
        target_noise=config['target_noise'],
        policy_delay=config['policy_delay'],
        num_test_episodes=config['num_test_episodes'],
        steps_per_epoch=config['steps_per_epoch'],
        start_steps=config['start_steps'],
        update_after=config['update_after'],
        update_every=config['update_every'],
        save_model=False,
        continuous_test=False,
        log_tensorboard=False,
        log_console=False) 

    with tune.checkpoint_dir(step=config['epochs']) as checkpoint_dir:
        path = os.path.join(checkpoint_dir, "checkpoint")
        torch.save((ac.state_dict(), pi_optimizer.state_dict(),
                    q_optimizer.state_dict()), path)

    tune.report(mean_ret=mean_ret)


if __name__ == "__main__":
    config = {
        'env': "Pendulum-v0",
        'ac_hid': tune.grid_search([5, 10, 15]),
        'ac_l': tune.grid_search([1, 2]),
        'gamma': 0.99,
        'seed': 0,
        'epochs': 25,
        'max_ep_len': 1000,
        'pi_lr': 1e-3,
        'q_lr': 1e-3,
        'batch_size': 100,
        'polyak': .995,
        'act_noise': tune.uniform(0.05, .75),
        'target_noise': tune.uniform(0.05, .75),
        'policy_delay': 2,
        'num_test_episodes': 30,
        'max_ep_len': 1000,
        'steps_per_epoch': 4000,
        'start_steps': 10000,
        'update_after': 1000,
        'update_every': 50,
    }

    result = tune.run(
        train_td3,
        config=config,
        num_samples=10,
        metric='mean_ret',
        mode='max')

    best_trial = result.get_best_trial("mean_ret", "max", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final mean reward: {}".format(
        best_trial.last_result["mean_ret"]))
