import neural_network_lyapunov.examples.car.unicycle as unicycle
import neural_network_lyapunov.examples.car.rrt_star as rrt_star
import numpy as np
import torch

if __name__ == "__main__":
    x_goal = np.array([0., 0., 0.])
    plant = unicycle.Unicycle(torch.float64)
    u_lo = np.array([-3, -np.pi * 0.25])
    u_up = np.array([6, np.pi * 0.25])
    dut = rrt_star.RrtStar(plant, 3, u_lo, u_up, x_goal)
    dut.grow_tree(20)
    pass
