import torch
import numpy as np


def rpy2rotmat(rpy):
    if isinstance(rpy, np.ndarray):
        cos_roll = np.cos(rpy[0])
        sin_roll = np.sin(rpy[0])
        cos_pitch = np.cos(rpy[1])
        sin_pitch = np.sin(rpy[1])
        cos_yaw = np.cos(rpy[2])
        sin_yaw = np.sin(rpy[2])
        R_roll = np.array([[1., 0, 0], [0, cos_roll, -sin_roll],
                           [0, sin_roll, cos_roll]])
        R_pitch = np.array([[cos_pitch, 0, sin_pitch], [0, 1., 0],
                            [-sin_pitch, 0, cos_pitch]])
        R_yaw = np.array([[cos_yaw, -sin_yaw, 0], [sin_yaw, cos_yaw, 0],
                          [0, 0, 1.]])
        return R_yaw @ R_pitch @ R_roll
    elif isinstance(rpy, torch.Tensor):
        cos_roll = torch.cos(rpy[0])
        sin_roll = torch.sin(rpy[0])
        cos_pitch = torch.cos(rpy[1])
        sin_pitch = torch.sin(rpy[1])
        cos_yaw = torch.cos(rpy[2])
        sin_yaw = torch.sin(rpy[2])
        R_roll = torch.zeros((3, 3), dtype=rpy.dtype)
        R_roll[0, 0] = 1
        R_roll[1, 1] = cos_roll
        R_roll[1, 2] = -sin_roll
        R_roll[2, 1] = sin_roll
        R_roll[2, 2] = cos_roll
        R_pitch = torch.zeros((3, 3), dtype=rpy.dtype)
        R_pitch[1, 1] = 1
        R_pitch[0, 0] = cos_pitch
        R_pitch[0, 2] = sin_pitch
        R_pitch[2, 0] = -sin_pitch
        R_pitch[2, 2] = cos_pitch
        R_yaw = torch.zeros((3, 3), dtype=rpy.dtype)
        R_yaw[0, 0] = cos_yaw
        R_yaw[0, 1] = -sin_yaw
        R_yaw[1, 0] = sin_yaw
        R_yaw[1, 1] = cos_yaw
        R_yaw[2, 2] = 1
        return R_yaw @ R_pitch @ R_roll
