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


class PybulletSampleGenerator:

    def __init__(self, urdf,
                 image_width=80, image_height=80,
                 dtype=torch.float64):
        self.dtype = dtype

        # physicsClient = pb.connect(pb.GUI)
        self.physics_client = pb.connect(pb.DIRECT)
        pb.setGravity(0, 0, 9.8)
        pb.setTimeStep(1./240.)

        self.image_width = image_width
        self.image_height = image_height
        self.view_matrix = pb.computeViewMatrix(
            cameraEyePosition=[0, -3, 0],
            cameraTargetPosition=[0, 0, 0],
            cameraUpVector=[0, 0, 1])
        self.projection_matrix = pb.computeProjectionMatrixFOV(
            fov=45.0,
            aspect=1.0,
            nearVal=0.1,
            farVal=3.1)

        pb.setAdditionalSearchPath(os.path.dirname(os.path.realpath(__file__)))
        self.robot_id = pb.loadURDF(urdf, flags=pb.URDF_USE_SELF_COLLISION)
        self.num_joints = pb.getNumJoints(self.robot_id)
        self.x_dim = 2 * self.num_joints
        for i in range(self.num_joints):
            pb.setJointMotorControl2(self.robot_id, i, pb.VELOCITY_CONTROL,
                                     force=0)

    def __del__(self):
        pb.disconnect(self.physics_client)

    def generate_sample(self, x0, dt):
        assert(isinstance(x0, torch.Tensor))
        assert(len(x0) == self.x_dim)
        q0 = x0[:self.num_joints]
        v0 = x0[self.num_joints:self.x_dim]

        num_step = int(dt*240.*.5)

        X = np.zeros((6, self.image_width, self.image_height), dtype=np.uint8)
        X_next = np.zeros((3, self.image_width, self.image_height),
                          dtype=np.uint8)

        for i in range(len(q0)):
            pb.resetJointState(self.robot_id, i, q0[i], v0[i])

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

        state = pb.getJointStates(self.robot_id, list(range(len(q0))))
        q1 = state[0][:len(q0)]
        v1 = state[0][len(q0):len(q0)+len(v0)]

        width2, height2, rgb2, depth2, seg2 = pb.getCameraImage(
            width=self.image_width,
            height=self.image_height,
            viewMatrix=self.view_matrix,
            projectionMatrix=self.projection_matrix)

        X[:3, :, :] = rgb0[:, :, :3].transpose(2, 0, 1)
        X[3:, :, :] = rgb1[:, :, :3].transpose(2, 0, 1)
        X_next[:3, :, :] = rgb2[:, :, :3].transpose(2, 0, 1)

        X = torch.tensor(X, dtype=torch.uint8)
        X_next = torch.tensor(X_next, dtype=torch.uint8)
        q1 = torch.tensor(q1, dtype=self.dtype)
        v1 = torch.tensor(v1, dtype=self.dtype)
        x1 = torch.cat((q1, v1))

        return X, X_next, x1

    def generate_dataset(self, x_lo, x_up, dt, num_samples, grayscale=False):
        X_data = torch.empty((num_samples, 6,
                             self.image_width, self.image_height))
        X_next_data = torch.empty((num_samples, 3,
                                  self.image_width, self.image_height))
        x_data = torch.empty((num_samples, self.x_dim))
        x_next_data = torch.empty((num_samples, self.x_dim))
        for i in range(num_samples):
            x0 = torch.rand(self.x_dim) * (x_up - x_lo) + x_lo
            x_data[i, :] = x0
            X_data[i, :, :, :], X_next_data[i, :, :, :], x_next_data[i, :] =\
                self.generate_sample(x0, dt)
        if grayscale:
            grayscale_weight = [.2989, .5870, .1140]
            X_data_gray = torch.zeros(X_data.shape[0], 2, X_data.shape[2],
                                      X_data.shape[3])
            X_data_gray[:, 0, :, :] = grayscale_weight[0] * X_data[:, 0, :, :]\
                + grayscale_weight[1] * X_data[:, 1, :, :] +\
                grayscale_weight[2] * X_data[:, 2, :, :]
            X_data_gray[:, 1, :, :] = grayscale_weight[0] *\
                X_data[:, 3, :, :] + grayscale_weight[1] *\
                X_data[:, 4, :, :] + grayscale_weight[2] * X_data[:, 5, :, :]
            X_next_data_gray = torch.zeros(X_next_data.shape[0],
                                           1, X_next_data.shape[2],
                                           X_next_data.shape[3])
            X_next_data_gray[:, 0, :, :] = grayscale_weight[0] *\
                X_next_data[:, 0, :, :] + grayscale_weight[1] *\
                X_next_data[:, 1, :, :] + grayscale_weight[2] *\
                X_next_data[:, 2, :, :]
            X_data = X_data_gray
            X_next_data = X_next_data_gray
        X_data /= 255.
        X_next_data /= 255.
        x_data = x_data.type(self.dtype)
        x_next_data = x_next_data.type(self.dtype)
        X_data = X_data.type(self.dtype)
        X_next_data = X_next_data.type(self.dtype)
        return x_data, x_next_data, X_data, X_next_data
