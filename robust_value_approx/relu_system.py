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
        (result, z_pre_relu_lo, z_pre_relu_up, z_post_relu_lo,
         z_post_relu_up) = self.dynamics_relu_free_pattern.output_constraint(
            self.x_lo, self.x_up)
        return result

    def possible_dx(self, x):
        assert(isinstance(x, torch.Tensor))
        assert(len(x) == self.x_dim)
        return [self.dynamics_relu(x)]


class ReLUSystem:
    """
    This class represents either a discrete-time controlled system, whose
    dynamics is
    x[n+1] = f(x[n], u[n])
    or a continuous time system with dynamics
    ẋ = f(x, u)
    where f is a feed-forward neural network with (leaky) ReLU activation
    functions. x[n+1], x[n], and u[n] (or ẋ, x, u) satisfy piecewise linear
    relationship, hence when they are bounded, they satisfy some mixed-integer
    linear constraint.
    """
    def __init__(self, dtype, x_lo, x_up, u_lo, u_up, dynamics_relu):
        """
        @param x_lo The lower bound of x[n] and x[n+1]. This is only used in
        forming the mixed-integer linear constraints.
        @param x_up The upper bound of x[n] and x[n+1]. This is only used in
        forming the mixed-integer linear constraints.
        @param u_lo The lower bound of u[n]. This is only used in
        forming the mixed-integer linear constraints.
        @param u_up The upper bound of u[n]. This is only used in
        forming the mixed-integer linear constraints.
        @param dynamics_relu A feedforward neural network with (leaky) ReLU
        activation units. The input to the network is the concatenation
        [x[n], u[n]], and the output of the network is x[n+1].
        """
        assert(len(x_lo.shape) == 1)
        assert(x_lo.shape == x_up.shape)
        assert(torch.all(x_lo <= x_up))
        self.x_dim = x_lo.numel()
        self.x_lo = x_lo
        self.x_up = x_up
        assert(len(u_lo.shape) == 1)
        assert(u_lo.shape == u_up.shape)
        assert(torch.all(u_lo <= u_up))
        self.u_dim = u_lo.numel()
        self.u_lo = u_lo
        self.u_up = u_up
        assert(dynamics_relu[0].in_features == self.x_dim + self.u_dim)
        assert(dynamics_relu[-1].out_features == self.x_dim)
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
                Ain_x, Ain_u, Ain_s, Ain_gamma, rhs_in,
                Aeq_x, Aeq_u, Aeq_s, Aeq_gamma, rhs_eq
                such that
                x[n+1] = Aout_s @ s + Cout or x_dot = Aout_s @ s + Cout
                s.t.
                Ain_x @ x + Ain_u @ u + Ain_s @ s + Ain_gamma @ gamma <= rhs_in
                Aeq_x @ x + Aeq_u @ u + Aeq_s @ s + Aeq_gamma @ gamma == rhs_eq
        """
        xu_lo = torch.cat((self.x_lo, self.u_lo))
        xu_up = torch.cat((self.x_up, self.u_up))
        (result, z_pre_relu_lo, z_pre_relu_up, z_post_relu_lo,
         z_post_relu_up) = self.dynamics_relu_free_pattern.output_constraint(
            xu_lo, xu_up)

        return result
