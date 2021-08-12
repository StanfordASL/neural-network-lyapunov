"""
adapted from: https://github.com/openai/spinningup
"""
import itertools
import numpy as np
import torch
import torch.nn as nn
import gym
from torch.optim import Adam
from copy import deepcopy
from torch.utils.tensorboard import SummaryWriter

from neural_network_lyapunov.examples.quadrotor2d.quadrotor2d_env import \
    Quadrotor2DEnv
import neural_network_lyapunov.utils as utils


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


class MLPActor(nn.Module):
    def __init__(self, act_low, act_high, obs_equ, act_equ,
                 hidden_sizes, dtype):
        super().__init__()
        self.act_low = act_low
        self.act_high = act_high
        self.obs_equ = obs_equ
        self.act_equ = act_equ
        controller_sizes = tuple([obs_equ.shape[0]] + list(
            hidden_sizes) + [act_equ.shape[0]])
        self.controller_relu = utils.setup_relu(controller_sizes,
                                                params=None,
                                                negative_slope=0.01,
                                                bias=True,
                                                dtype=dtype)

    def forward(self, obs):
        output_raw = self.controller_relu(obs)
        output_equ = self.controller_relu(self.obs_equ)
        output_clip = (torch.clip(output_raw - output_equ, -1, 1.) + 1.) * .5
        output = output_clip * (self.act_high - self.act_low) + self.act_equ
        return output


class MLPQFunction(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes, dtype):
        super().__init__()
        q_sizes = tuple([obs_dim + act_dim] + list(hidden_sizes) + [1])
        self.q = utils.setup_relu(q_sizes,
                                  params=None,
                                  negative_slope=0.01,
                                  bias=True,
                                  dtype=dtype)

    def forward(self, obs, act):
        q = self.q(torch.cat([obs, act], dim=-1))
        return torch.squeeze(q, -1)


class MLPActorCritic(nn.Module):
    def __init__(self, observation_space, action_space, obs_equ, act_equ,
                 hidden_sizes):
        super().__init__()

        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]
        act_low = torch.tensor(action_space.low)
        act_high = torch.tensor(action_space.high)

        # build policy and value functions
        self.actor = MLPActor(
            act_low, act_high, obs_equ, act_equ, hidden_sizes, act_low.dtype)
        self.q1 = MLPQFunction(obs_dim, act_dim, hidden_sizes, act_low.dtype)
        self.q2 = MLPQFunction(obs_dim, act_dim, hidden_sizes, act_low.dtype)

    def act(self, obs):
        with torch.no_grad():
            return self.actor(obs).numpy()


class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for TD3 agents.
    """
    def __init__(self, obs_dim, act_dim, size):
        self.obs_buf = np.zeros(
            combined_shape(size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros(
            combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(
            combined_shape(size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs=self.obs_buf[idxs],
                     obs2=self.obs2_buf[idxs],
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs],
                     done=self.done_buf[idxs])
        return {k: torch.as_tensor(v, dtype=torch.float32) for
                k, v in batch.items()}


def td3(env_fn, actor_critic=MLPActorCritic, ac_kwargs=dict(), ac_state=None, 
        seed=0, steps_per_epoch=4000, epochs=100,
        replay_size=int(1e6), gamma=0.99,
        polyak=0.995, pi_lr=1e-3, q_lr=1e-3, batch_size=100, start_steps=10000,
        update_after=1000, update_every=50, act_noise=0.1, target_noise=0.2,
        policy_delay=2, num_test_episodes=10, max_ep_len=1000,
        exp_name='td3', continuous_test=True, save_model=True,
        log_tensorboard=True, log_console=True):
    """
    Twin Delayed Deep Deterministic Policy Gradient (TD3)


    Args:
        env_fn : A function which creates a copy of the environment.
            The environment must satisfy the OpenAI Gym API.

        actor_critic: The constructor method for a PyTorch Module with an
            ``act`` method, a ``pi`` module, a ``q1`` module, and a ``q2``
            module. The ``act`` method and ``pi`` module should accept batches
            of observations as inputs, and ``q1`` and ``q2`` should accept a
            batch of observations and a batch of actions as inputs.

        ac_kwargs (dict): Any kwargs appropriate for the ActorCritic object
            you provided to TD3.

        ac_state (dict): State to start the actor critic model with (can be
        generated using torch.save(model.state_dict(), PATH) and 
        torch.load(PATH))

        seed (int): Seed for random number generators.

        steps_per_epoch (int): Number of steps of interaction
          (state-action pairs) for the agent and the environment in each epoch.

        epochs (int): Number of epochs to run and train agent.

        replay_size (int): Maximum length of replay buffer.

        gamma (float): Discount factor. (Always between 0 and 1.)

        polyak (float): Interpolation factor in polyak averaging for target
            networks. (Always between 0 and 1, usually close to 1.)

        pi_lr (float): Learning rate for policy.

        q_lr (float): Learning rate for Q-networks.

        batch_size (int): Minibatch size for SGD.

        start_steps (int): Number of steps for uniform-random action selection,
            before running real policy. Helps exploration.

        update_after (int): Number of env interactions (at the very beginning)
            to collect before starting to do gradient descent updates.
            Ensures replay buffer is full enough for useful updates.

        update_every (int): Number of env interactions that should elapse
            between gradient descent updates. Note: Regardless of how long
            you wait between updates, the ratio of env steps to gradient steps
            is locked to 1.

        act_noise (float): Stddev for Gaussian exploration noise added to
            policy at training time. (At test time, no noise is added.)

        target_noise (float): Stddev for smoothing noise added to target
            policy.

        policy_delay (int): Policy will only be updated once every
            policy_delay times for each update of the Q-networks.

        num_test_episodes (int): Number of episodes to test the deterministic
            policy at the end of each epoch.

        max_ep_len (int): Maximum length of trajectory / episode / rollout.

    """
    if log_tensorboard:
        writer = SummaryWriter()

    torch.manual_seed(seed)
    np.random.seed(seed)

    env, test_env = env_fn(), env_fn()
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape[0]
    act_low = torch.tensor(env.action_space.low)
    act_high = torch.tensor(env.action_space.high)

    # Create actor-critic module and target networks
    if hasattr(env, 'obs_equ'):
        obs_equ = env.obs_equ
    else:
        obs_equ = torch.zeros(env.observation_space.shape)
    if hasattr(env, 'act_equ'):
        act_equ = env.act_equ
    else:
        act_equ = torch.zeros(env.action_space.shape)
    ac = actor_critic(
        env.observation_space, env.action_space, obs_equ, act_equ, **ac_kwargs)
    if ac_state is not None:
        ac.load_state_dict(ac_state)
    ac_targ = deepcopy(ac)

    # Freeze target networks with respect to optimizers
    for p in ac_targ.parameters():
        p.requires_grad = False

    # List of parameters for both Q-networks (save this for convenience)
    q_params = itertools.chain(ac.q1.parameters(), ac.q2.parameters())

    # Experience buffer
    replay_buffer = ReplayBuffer(
        obs_dim=obs_dim, act_dim=act_dim, size=replay_size)

    # Set up function for computing TD3 Q-losses
    def compute_loss_q(data):
        o, a, r, o2, d = (data['obs'], data['act'], data['rew'],
                          data['obs2'], data['done'])

        q1 = ac.q1(o, a)
        q2 = ac.q2(o, a)

        # Bellman backup for Q functions
        with torch.no_grad():
            pi_targ = ac_targ.actor(o2)

            # Target policy smoothing
            epsilon = torch.randn_like(pi_targ) * target_noise * \
                (act_high - act_low)
            a2 = pi_targ + epsilon
            a2 = torch.max(torch.min(a2, act_high), act_low)

            # Target Q-values
            q1_pi_targ = ac_targ.q1(o2, a2)
            q2_pi_targ = ac_targ.q2(o2, a2)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            backup = r + gamma * (1 - d) * q_pi_targ

        # MSE loss against Bellman backup
        loss_q1 = ((q1 - backup)**2).mean()
        loss_q2 = ((q2 - backup)**2).mean()
        loss_q = loss_q1 + loss_q2

        # Useful info for logging
        loss_info = dict(Q1Vals=q1.detach().numpy(),
                         Q2Vals=q2.detach().numpy())

        return loss_q, loss_info

    # Set up function for computing TD3 pi loss
    def compute_loss_pi(data):
        o = data['obs']
        q1_pi = ac.q1(o, ac.actor(o))
        return -q1_pi.mean()

    # Set up optimizers for policy and q-function
    pi_optimizer = Adam(ac.actor.parameters(), lr=pi_lr)
    q_optimizer = Adam(q_params, lr=q_lr)

    def update(data, timer, step):
        # First run one gradient descent step for Q1 and Q2
        q_optimizer.zero_grad()
        loss_q, loss_info = compute_loss_q(data)
        loss_q.backward()
        q_optimizer.step()

        if log_tensorboard:
            writer.add_scalar('Qloss', loss_q.item(), step)

        # Possibly update pi and target networks
        if timer % policy_delay == 0:

            # Freeze Q-networks so you don't waste computational effort
            # computing gradients for them during the policy learning step.
            for p in q_params:
                p.requires_grad = False

            # Next run one gradient descent step for pi.
            pi_optimizer.zero_grad()
            loss_pi = compute_loss_pi(data)
            loss_pi.backward()
            pi_optimizer.step()

            if log_tensorboard:
                writer.add_scalar('Piloss', loss_pi.item(), step)

            # Unfreeze Q-networks so you can optimize it at next DDPG step.
            for p in q_params:
                p.requires_grad = True

            # Finally, update target networks by polyak averaging.
            with torch.no_grad():
                for p, p_targ in zip(ac.parameters(), ac_targ.parameters()):
                    p_targ.data.mul_(polyak)
                    p_targ.data.add_((1 - polyak) * p.data)

    def get_action(o, noise_scale):
        a = ac.act(torch.as_tensor(o, dtype=torch.float32))
        a += noise_scale * np.random.randn(act_dim) * \
            (env.action_space.high - env.action_space.low)
        return np.maximum(np.minimum(
            a, env.action_space.high), env.action_space.low)

    def test_agent():
        mean_ret = 0.
        for j in range(num_test_episodes):
            o, d, ep_ret, ep_len = test_env.reset(), False, 0, 0
            while not(d or (ep_len == max_ep_len)):
                # Take deterministic actions at test time (noise_scale=0)
                o, r, d, _ = test_env.step(get_action(o, 0))
                ep_ret += r
                ep_len += 1
            mean_ret += ep_ret
        mean_ret /= float(num_test_episodes)
        return mean_ret

    # Prepare for interaction with environment
    total_steps = steps_per_epoch * epochs
    o, ep_ret, ep_len = env.reset(), 0, 0

    # Main loop: collect experience in env and update/log each epoch
    for t in range(total_steps):
        # Until start_steps have elapsed, randomly sample actions
        # from a uniform distribution for better exploration. Afterwards,
        # use the learned policy (with some noise, via act_noise).
        if t > start_steps:
            a = get_action(o, act_noise)
        else:
            a = env.action_space.sample()

        # Step the env
        o2, r, d, _ = env.step(a)
        ep_ret += r
        ep_len += 1

        # Ignore the "done" signal if it comes from hitting the time
        # horizon (that is, when it's an artificial terminal signal
        # that isn't based on the agent's state)
        d = False if ep_len == max_ep_len else d

        # Store experience to replay buffer
        replay_buffer.store(o, a, r, o2, d)

        # Super critical, easy to overlook step: make sure to update
        # most recent observation!
        o = o2

        # End of trajectory handling
        if d or (ep_len == max_ep_len):
            o, ep_ret, ep_len = env.reset(), 0, 0

        # Update handling
        if t >= update_after and t % update_every == 0:
            for j in range(update_every):
                batch = replay_buffer.sample_batch(batch_size)
                update(data=batch, timer=j, step=t)

        # End of epoch handling
        if (t+1) % steps_per_epoch == 0:
            epoch = (t+1) // steps_per_epoch

            # Test the performance of the deterministic version of the agent.
            if continuous_test:
                mean_ret = test_agent()
                if log_tensorboard:
                    writer.add_scalar('TD3AvgRet', mean_ret, epoch)
                if log_console:
                    print("Epoch %s: %s" % (str(epoch), str(mean_ret)))

            if save_model:
                torch.save(ac, exp_name + '_actor_critic.pt')

    mean_ret = test_agent()

    return ac, pi_optimizer, q_optimizer, mean_ret


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('env', type=str)
    parser.add_argument('--hid', type=int, default=10)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--exp_name', type=str, default='td3')
    parser.add_argument('--load_ac', type=str, default='')
    args = parser.parse_args()

    def env_fn():
        if args.env == 'quadrotor2d':
            return Quadrotor2DEnv()
        else:
            return gym.make(args.env)

    if args.load_ac == '':
        ac_state = None
    else:
        ac_state = torch.load(args.load_ac).state_dict()

    td3(env_fn, actor_critic=MLPActorCritic,
        ac_kwargs=dict(hidden_sizes=[args.hid]*args.l),
        ac_state=ac_state,
        gamma=args.gamma, seed=args.seed, epochs=args.epochs,
        max_ep_len=1000,
        pi_lr=1e-3, q_lr=1e-3,
        polyak=.995,
        act_noise=0.1,
        target_noise=0.2,
        steps_per_epoch=4000,
        start_steps=10000,
        update_after=1000,
        update_every=50,
        exp_name=args.exp_name + '_' + args.env)
