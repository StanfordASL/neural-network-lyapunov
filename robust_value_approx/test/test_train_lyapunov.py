import robust_value_approx.train_lyapunov as train_lyapunov
import robust_value_approx.hybrid_linear_system as hybrid_linear_system
import torch
import torch.nn as nn
import unittest


def setup_discrete_time_system():
    """
    The piecewise affine system is from "Analysis of discrete-time
    piecewise affine and hybrid systems" by Giancarlo Ferrari-Trecate
    et.al.
    """
    dtype = torch.float64
    system = hybrid_linear_system.AutonomousHybridLinearSystem(
        2, dtype)
    system.add_mode(
        torch.tensor([[-0.999, 0], [-0.139, 0.341]], dtype=dtype),
        torch.zeros((2,), dtype=dtype),
        torch.tensor([[1, 0], [-1, 0], [0, 1], [0, -1]], dtype=dtype),
        torch.tensor([1, 0, 0, 1], dtype=dtype))
    system.add_mode(
        torch.tensor([[0.436, 0.323], [0.388, -0.049]], dtype=dtype),
        torch.zeros((2,), dtype=dtype),
        torch.tensor([[1, 0], [-1, 0], [0, 1], [0, -1]], dtype=dtype),
        torch.tensor([1, 0, 1, 0], dtype=dtype))
    system.add_mode(
        torch.tensor([[-0.457, 0.215], [0.491, 0.49]], dtype=dtype),
        torch.zeros((2,), dtype=dtype),
        torch.tensor([[1, 0], [-1, 0], [0, 1], [0, -1]], dtype=dtype),
        torch.tensor([0, 1, 0, 1], dtype=dtype))
    system.add_mode(
        torch.tensor([[-0.022, 0.344], [0.458, 0.271]], dtype=dtype),
        torch.zeros((2,), dtype=dtype),
        torch.tensor([[1, 0], [-1, 0], [0, 1], [0, -1]], dtype=dtype),
        torch.tensor([0, 1, 1, 0], dtype=dtype))
    return system


def setup_relu():
    # Construct a simple ReLU model with 2 hidden layers
    dtype = torch.float64
    linear1 = nn.Linear(2, 3)
    linear1.weight.data = torch.tensor([[-1, 0.5], [-0.3, 0.74], [-2, 1.5]],
                                       dtype=dtype)
    linear1.bias.data = torch.tensor([-0.1, 1.0, 0.5], dtype=dtype)
    linear2 = nn.Linear(3, 4)
    linear2.weight.data = torch.tensor(
            [[-1, -0.5, 1.5], [2, -1.5, 2.6], [-2, -0.3, -.4],
             [0.2, -0.5, 1.2]],
            dtype=dtype)
    linear2.bias.data = torch.tensor([-0.3, 0.2, 0.7, 0.4], dtype=dtype)
    linear3 = nn.Linear(4, 1)
    linear3.weight.data = torch.tensor([[-.4, .5, -.6, 0.3]], dtype=dtype)
    linear3.bias.data = torch.tensor([-0.9], dtype=dtype)
    relu1 = nn.Sequential(
        linear1, nn.ReLU(), linear2, nn.ReLU(), linear3, nn.ReLU())
    return relu1


def setup_state_samples_all(mesh_size):
    assert(isinstance(mesh_size, tuple))
    assert(len(mesh_size) == 2)
    dtype = torch.float64
    (samples_x, samples_y) = torch.meshgrid(
        torch.linspace(-1., 1., mesh_size[0], dtype=dtype),
        torch.linspace(-1., 1., mesh_size[1], dtype=dtype))
    state_samples = [None] * (mesh_size[0] * mesh_size[1])
    for i in range(samples_x.shape[0]):
        for j in range(samples_x.shape[1]):
            state_samples[i * samples_x.shape[1] + j] = torch.tensor(
                [samples_x[i, j], samples_y[i, j]], dtype=dtype)
    return state_samples


class TestTrainLyapunov(unittest.TestCase):
    def setUp(self):
        self.system = setup_discrete_time_system()
        self.relu = setup_relu()

    def test_train_value_approximator(self):
        state_samples_all = setup_state_samples_all((21, 21))
        options = train_lyapunov.TrainValueApproximatorOptions()
        options.max_epochs = 1000
        options.convergence_tolerance = 0.05
        result = train_lyapunov.train_value_approximator(
            self.system, self.relu, lambda x: x @ x, state_samples_all,
            options)
        self.assertTrue(result)


if __name__ == "__main__":
    unittest.main()
