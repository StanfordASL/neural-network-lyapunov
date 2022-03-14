import torch
import numpy as np
import gym

import neural_network_lyapunov.examples.quadrotor2d.quadrotor_2d as \
    quadrotor_2d


class Quadrotor2DEnv(gym.Env):
    def __init__(self):
        super(Quadrotor2DEnv, self).__init__()
        self.dtype = torch.float32
        self.dt = .01
        self.x_lo = torch.tensor([-0.3, -0.3, -np.pi * 0.3, -1.5, -1.5, -0.9],
                                 dtype=self.dtype)
        self.x_up = -self.x_lo
        self.u_lo = torch.tensor([0, 0], dtype=self.dtype)
        self.u_up = torch.tensor([8, 8], dtype=self.dtype)
        self.action_space = gym.spaces.Box(low=self.u_lo.detach().numpy(),
                                           high=self.u_up.detach().numpy())
        self.observation_space = gym.spaces.Box(
            low=self.x_lo.detach().numpy(), high=self.x_up.detach().numpy())
        self.system = quadrotor_2d.Quadrotor2D(self.dtype)
        self.obs_equ = torch.zeros((6, ), dtype=self.dtype)
        self.act_equ = self.system.u_equilibrium
        self.x_current = torch.zeros((6, ), dtype=self.dtype)
        self.t_current = 0.
        self.lqr_Q = torch.diag(
            torch.tensor([10, 10, 10, 1, 1, self.system.length / 2. / np.pi],
                         dtype=self.dtype))
        self.lqr_R = torch.tensor([[0.1, 0.05], [0.05, 0.1]], dtype=self.dtype)

    def step(self, action_np):
        action = torch.tensor(action_np, dtype=self.dtype)
        x_next = torch.tensor(self.system.next_pose(self.x_current, action,
                                                    self.dt),
                              dtype=self.dtype)
        observation = x_next.detach().numpy()
        act_delta = (action - self.act_equ)
        obs_delta = (x_next - self.obs_equ)
        reward = -(act_delta).dot(self.lqr_R @ act_delta).item() - \
            obs_delta.dot(self.lqr_Q @ obs_delta).item()
        done = False
        self.x_current = x_next.clone()
        self.t_current += self.dt
        info = dict()
        return observation, reward, done, info

    def reset(self):
        self.x_current = torch.rand(6, dtype=self.dtype) *\
            (self.x_up - self.x_lo) + self.x_lo
        self.t_current = 0.
        observation = self.x_current.detach().numpy()
        return observation
