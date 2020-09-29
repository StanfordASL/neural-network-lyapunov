import os
import torch
import numpy as np
import pybullet as pb
import matplotlib.pyplot as plt


def show_sample(X_sample, X_next_sample=None, clamp=False):
    if clamp:
        X_sample = torch.clamp(X_sample, 0, 1)
        if X_next_sample is not None:
            X_next_sample = torch.clamp(X_next_sample)
    if X_sample.shape[0] == 6:
        num_channels = 3
        cmap = None
    elif X_sample.shape[0] == 3:
        num_channels = 3
        cmap = None
    elif X_sample.shape[0] == 2:
        num_channels = 1
        cmap = 'gray'
    elif X_sample.shape[0] == 1:
        num_channels = 1
        cmap = 'gray'
    else:
        raise(NotImplementedError)
    fig = plt.figure(figsize=(10, 10))
    if X_next_sample is not None:
        fig.add_subplot(1, 3, 1)
        plt.imshow(X_sample[:num_channels, :, :].to(
            'cpu').detach().numpy().transpose(1, 2, 0),
            cmap=cmap, vmin=0, vmax=1)
        fig.add_subplot(1, 3, 2)
        plt.imshow(X_sample[num_channels:, :, :].to(
            'cpu').detach().numpy().transpose(1, 2, 0),
            cmap=cmap, vmin=0, vmax=1)
        fig.add_subplot(1, 3, 3)
        plt.imshow(X_next_sample[:num_channels, :, :].to(
            'cpu').detach().numpy().transpose(1, 2, 0),
            cmap=cmap, vmin=0, vmax=1)
    else:
        fig.add_subplot(1, 2, 1)
        plt.imshow(X_sample[:num_channels, :, :].to(
            'cpu').detach().numpy().transpose(1, 2, 0),
            cmap=cmap, vmin=0, vmax=1)
        if X_sample.shape[0] == 2 or X_sample.shape[0] == 6:
            fig.add_subplot(1, 2, 2)
            plt.imshow(X_sample[num_channels:, :, :].to(
                'cpu').detach().numpy().transpose(1, 2, 0),
                cmap=cmap, vmin=0, vmax=1)
    plt.show()


def load_urdf_callback(urdf):
    def cb(pb):
        robot_id = pb.loadURDF(urdf, flags=pb.URDF_USE_SELF_COLLISION)
        return robot_id
    return cb


class PybulletSampleGenerator:
    def __init__(self, load_world_cb, joint_space,
                 image_width=80, image_height=80, grayscale=False,
                 camera_eye_position=[0, -3, 0],
                 camera_target_position=[0, 0, 0],
                 camera_up_vector=[0, 0, 1],
                 dtype=torch.float64):
        self.dtype = dtype

        # self.physics_client = pb.connect(pb.GUI)
        self.physics_client = pb.connect(pb.DIRECT)
        pb.setGravity(0, 0, -9.8)
        pb.setTimeStep(1./240.)
        pb.setPhysicsEngineParameter(enableFileCaching=0)

        self.grayscale = grayscale
        self.grayscale_weight = [.2989, .5870, .1140]
        if self.grayscale:
            self.num_channels = 1
        else:
            self.num_channels = 3
        self.image_width = image_width
        self.image_height = image_height
        self.view_matrix = pb.computeViewMatrix(
            cameraEyePosition=camera_eye_position,
            cameraTargetPosition=camera_target_position,
            cameraUpVector=camera_up_vector)
        self.projection_matrix = pb.computeProjectionMatrixFOV(
            fov=45.0,
            aspect=1.0,
            nearVal=0.1,
            farVal=3.1)

        self.robot_id = load_world_cb(pb)
        self.joint_space = joint_space

        if self.joint_space:
            self.num_joints = pb.getNumJoints(self.robot_id)
            self.x_dim = 2 * self.num_joints
            for i in range(self.num_joints):
                pb.setJointMotorControl2(self.robot_id, i, pb.VELOCITY_CONTROL,
                                         force=0)
        else:
            self.x_dim = 12

    def __del__(self):
        pb.disconnect(self.physics_client)

    def generate_sample(self, x0, dt):
        assert(isinstance(x0, torch.Tensor))
        assert(len(x0) == self.x_dim)

        num_step = int(dt*240.*.5)

        X = np.zeros((6, self.image_width, self.image_height), dtype=np.uint8)
        X_next = np.zeros((3, self.image_width, self.image_height),
                          dtype=np.uint8)

        if self.joint_space:
            q0 = x0[:self.num_joints]
            v0 = x0[self.num_joints:self.x_dim]
            for i in range(len(q0)):
                pb.resetJointState(self.robot_id, i, q0[i], v0[i])
        else:
            pos0 = x0[:3]
            orn0 = pb.getQuaternionFromEuler(x0[3:6])
            vel0 = x0[6:9]
            w0 = x0[9:12]
            pb.resetBasePositionAndOrientation(self.robot_id, pos0, orn0)
            pb.resetBaseVelocity(self.robot_id, vel0, w0)

        width0, height0, rgb0, depth0, seg0 = pb.getCameraImage(
            width=self.image_width,
            height=self.image_height,
            viewMatrix=self.view_matrix,
            projectionMatrix=self.projection_matrix)

        for k in range(num_step):
            pb.stepSimulation()

        width1, height1, rgb1, depth1, seg1 = pb.getCameraImage(
            width=self.image_width,
            height=self.image_height,
            viewMatrix=self.view_matrix,
            projectionMatrix=self.projection_matrix)

        for k in range(num_step):
            pb.stepSimulation()

        if self.joint_space:
            state = pb.getJointStates(self.robot_id, list(range(len(q0))))
            q1 = state[0][:len(q0)]
            v1 = state[0][len(q0):len(q0)+len(v0)]
            q1 = torch.tensor(q1, dtype=self.dtype)
            v1 = torch.tensor(v1, dtype=self.dtype)
            x1 = torch.cat((q1, v1))
        else:
            pos1, orn1_quat = pb.getBasePositionAndOrientation(self.robot_id)
            orn1 = pb.getEulerFromQuaternion(orn1_quat)
            vel1, w1 = pb.getBaseVelocity(self.robot_id)
            pos1 = torch.tensor(pos1, dtype=self.dtype)
            orn1 = torch.tensor(orn1, dtype=self.dtype)
            vel1 = torch.tensor(vel1, dtype=self.dtype)
            w1 = torch.tensor(w1, dtype=self.dtype)
            x1 = torch.cat((pos1, orn1, vel1, w1))

        width2, height2, rgb2, depth2, seg2 = pb.getCameraImage(
            width=self.image_width,
            height=self.image_height,
            viewMatrix=self.view_matrix,
            projectionMatrix=self.projection_matrix)

        X[:3, :, :] = rgb0[:, :, :3].transpose(2, 0, 1)
        X[3:, :, :] = rgb1[:, :, :3].transpose(2, 0, 1)
        X_next[:3, :, :] = rgb2[:, :, :3].transpose(2, 0, 1)

        X = torch.tensor(X, dtype=torch.float64)
        X_next = torch.tensor(X_next, dtype=torch.float64)

        if self.grayscale:
            X_gray = torch.zeros(2, X.shape[1], X.shape[2],
                                 dtype=torch.float64)
            X_gray[0, :] = self.grayscale_weight[0] * X[0, :, :] +\
                self.grayscale_weight[1] * X[1, :, :] +\
                self.grayscale_weight[2] * X[2, :, :]
            X_gray[1, :] = self.grayscale_weight[0] * X[3, :, :] +\
                self.grayscale_weight[1] * X[4, :, :] +\
                self.grayscale_weight[2] * X[5, :, :]
            X_next_gray = torch.zeros(1, X_next.shape[1], X_next.shape[2],
                                      dtype=torch.float64)
            X_next_gray[0, :, :] = \
                self.grayscale_weight[0] * X_next[0, :, :] +\
                self.grayscale_weight[1] * X_next[1, :, :] +\
                self.grayscale_weight[2] * X_next[2, :, :]
            X = X_gray
            X_next = X_next_gray

        X /= 255.
        X_next /= 255.
        X = torch.clamp(X, 0., 1.)
        X_next = torch.clamp(X_next, 0., 1.)

        X = X.type(self.dtype)
        X_next = X_next.type(self.dtype)

        return X, X_next, x1

    def generate_rollout(self, x0, dt, N):
        X_data = torch.empty((N+2, self.num_channels,
                             self.image_width, self.image_height),
                             dtype=self.dtype)
        x_data = torch.empty((N+1, self.x_dim), dtype=self.dtype)
        X, _, _ = self.generate_sample(x0, dt)
        X_data[0, :] = X[:self.num_channels, :]
        X_data[1, :] = X[self.num_channels:, :]
        x_data[0, :] = x0
        for n in range(N):
            _, X_next, x_next = self.generate_sample(x_data[n], dt)
            X_data[n+2] = X_next
            x_data[n+1] = x_next
        return X_data, x_data

    def generate_dataset(self, x_lo, x_up, dt, N, num_samples):
        assert(N >= 1)
        X_data = torch.empty((num_samples * N, 2*self.num_channels,
                             self.image_width, self.image_height),
                             dtype=self.dtype)
        X_next_data = torch.empty((num_samples * N, self.num_channels,
                                  self.image_width, self.image_height),
                                  dtype=self.dtype)
        x_data = torch.empty((num_samples * N, self.x_dim), dtype=self.dtype)
        x_next_data = torch.empty((num_samples * N, self.x_dim),
                                  dtype=self.dtype)
        for i in range(num_samples):
            x0 = torch.rand(self.x_dim) * (x_up - x_lo) + x_lo
            X_data_rollout, x_data_rollout = self.generate_rollout(x0, dt, N)
            for n in range(N):
                X_data[i * N + n, :self.num_channels, :] = X_data_rollout[n, :]
                X_data[i * N + n, self.num_channels:, :] = X_data_rollout[
                    n+1, :]
                X_next_data[i * N + n, :] = X_data_rollout[n+2, :]
                x_data[i * N + n, :] = x_data_rollout[n, :]
                x_next_data[i * N + n, :] = x_data_rollout[n+1, :]
        return x_data, x_next_data, X_data, X_next_data
