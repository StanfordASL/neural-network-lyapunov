import torch
import numpy

class DubinsCar:
    """
    A simple Dubin's car model, that the state is [pos_x, pos_y, theta], and
    the control is [vel, thetadot], where vel is the velocity along the heading
    direction of the car.
    """

    def __init__(self, dtype):
        self.dtype= dtype

    def dynamics(self, x, u):
        """
        Compute xdot of the Dubins car.
        """
        theta = x[2]
        vel = u[0]
        thetadot = u[1]
        if isinstance(x, np.ndarray):
            return np.array([vel*np.cos(theta), vel * np.sin(theta), thetadot])
        elif isinstance(x, torch.Tensor):
            return torch.tensor(
                [vel * torch.cos(theta), vel * torch.sin(theta), thetadot],
                dtype=self.dtype)

    def next_pose(self, x, u, dt):
        """
        Computes the next pose of the car after dt.
        """
        result = scipy.integrate.solve_ivp(
            lambda t, x_val: self.dynamics(x_val, u), [0, dt], x)
        return result.y[:, -1]
