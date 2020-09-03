import torch
import robust_value_approx.relu_to_optimization as relu_to_optimization


class AutonomousReLUSystem:
    """
    This system models an autonomous using a feedforward
    neural network with ReLU activations
    x[n+1] = relu(x[n])
    or
    x_dot = relu(x)
    """

    def __init__(self, dtype, x_lo, x_up, dynamics_relu):
        """
        @param dtype The torch datatype
        @param x_lo, x_up torch tensor that lower and upper bound the state
        @param dynamics_relu torch model that represents the dynamics
        """
        assert(len(x_lo) == len(x_up))
        assert(torch.all(x_up >= x_lo))
        assert(dynamics_relu[0].in_features == dynamics_relu[-1].out_features)
        self.dtype = dtype
        self.x_lo = x_lo
        self.x_up = x_up
        self.x_dim = len(self.x_lo)
        self.dynamics_relu = dynamics_relu
        self.dynamics_relu_free_pattern = relu_to_optimization.ReLUFreePattern(
            dynamics_relu, dtype)

    @property
    def x_lo_all(self):
        return self.x_lo.detach().numpy()

    @property
    def x_up_all(self):
        return self.x_up.detach().numpy()

    def mixed_integer_constraints(self):
        """
        @return Aout_s, Cout,
                Ain_x, Ain_s, Ain_gamma, rhs_in,
                Aeq_x, Aeq_s, Aeq_gamma, rhs_eq
                such that
                x[n+1] = Aout_s @ s + Cout or x_dot = Aout_s @ s + Cout
                s.t.
                Ain_x @ x + Ain_s @ s + Ain_gamma @ gamma <= rhs_in
                Aeq_x @ x + Aeq_s @ s + Aeq_gamma @ gamma == rhs_eq
        """
        (Ain_x, Ain_s, Ain_gamma, rhs_in, Aeq_x, Aeq_s, Aeq_gamma, rhs_eq,
         Aout_s, Cout, z_pre_relu_lo, z_pre_relu_up, z_post_relu_lo,
         z_post_relu_up) = self.dynamics_relu_free_pattern.output_constraint(
            self.x_lo, self.x_up)

        return (Aout_s, Cout,
                Ain_x, Ain_s, Ain_gamma, rhs_in,
                Aeq_x, Aeq_s, Aeq_gamma, rhs_eq)

    def possible_dx(self, x):
        assert(isinstance(x, torch.Tensor))
        assert(len(x) == self.x_dim)
        return [self.dynamics_relu(x)]
