"""
adapted from: https://github.com/openai/spinningup
"""
import torch
import gym
import numpy as np
import matplotlib.pyplot as plt

from neural_network_lyapunov.examples.rl.td3 import \
    MLPActorCritic, MLPActor, MLPQFunction  # noqa
from neural_network_lyapunov.examples.quadrotor2d.quadrotor2d_env import \
    Quadrotor2DEnv


def load_policy_and_env(env, fpath):
    if env == 'quadrotor2d':
        env_inst = Quadrotor2DEnv()
    else:
        env_inst = gym.make(env)
    ac = torch.load(fpath)

    def get_action(x):
        with torch.no_grad():
            x = torch.as_tensor(x, dtype=torch.float32)
            action = ac.act(x)
        return action

    return env_inst, get_action


def run_policy(env, get_action, max_ep_len=None, num_episodes=100,
               render=False):
    episodes = [[]]
    actions = [[]]
    o, r, d, ep_ret, ep_len, n = env.reset(), 0, False, 0, 0, 0
    while n < num_episodes:
        a = get_action(o)
        o, r, d, _ = env.step(a)
        if render:
            env.render()
        ep_ret += r
        ep_len += 1
        episodes[-1].append(o)
        actions[-1].append(a)
        if d or (ep_len == max_ep_len):
            print('Episode %d \t EpRet %.3f \t EpLen %d' % (n, ep_ret, ep_len))
            episodes.append([])
            actions.append([])
            o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
            n += 1
    plt.figure()
    for k in range(len(episodes) - 1):
        rollout = np.array(episodes[k])
        # plt.plot(rollout[:, 0], rollout[:, 1])
        plt.plot(rollout[:, 0], 'r')
        plt.plot(rollout[:, 1], 'b')
    # plt.xlim([env.x_lo[0], env.x_up[0]])
    # plt.ylim([env.x_lo[1], env.x_up[1]])
    plt.show()


if __name__ == '__main__':
    seed = 123
    torch.manual_seed(seed)
    np.random.seed(seed)

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('env', type=str)
    parser.add_argument('fpath', type=str)
    parser.add_argument('--len', '-l', type=int, default=100)
    parser.add_argument('--episodes', '-n', type=int, default=10)
    parser.add_argument('--render', '-r', action='store_true')
    args = parser.parse_args()
    env, get_action = load_policy_and_env(args.env, args.fpath)
    run_policy(env, get_action, args.len, args.episodes, render=args.render)
