import torch
import neural_network_lyapunov.relu_to_optimization as relu_to_optimization
import neural_network_lyapunov.gurobi_torch_mip as gurobi_torch_mip


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

    def mixed_integer_constraints(self) ->\
            gurobi_torch_mip.MixedIntegerConstraintsReturn:
        """
        @return mixed-integer linear constraints MixedIntegerConstraintsReturn
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

    def step_forward(self, x_start):
        assert(isinstance(x_start, torch.Tensor))
        return self.dynamics_relu(x_start)


class AutonomousReLUSystemGivenEquilibrium:
    """
    This system models an autonomous system with known equilibirum x* using
    a feedforward neural network with ReLU activations
    x[n+1] = ϕ(x[n]) − ϕ(x*) + x*
    where ϕ is a feedforward (leaky) ReLU network.
    """

    def __init__(self, dtype, x_lo, x_up, dynamics_relu, x_equilibrium):
        """
        @param dtype The torch datatype
        @param x_lo, x_up torch tensor that lower and upper bound the state
        @param dynamics_relu torch model that represents the dynamics
        @param x_equilibrium The equilibrium state.
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
        assert(x_equilibrium.shape == (self.x_dim,))
        self.x_equilibrium = x_equilibrium

    @property
    def x_lo_all(self):
        return self.x_lo.detach().numpy()

    @property
    def x_up_all(self):
        return self.x_up.detach().numpy()

    def mixed_integer_constraints(self):
        """
        @return mixed-integer linear constraints MixedIntegerConstraintsReturn
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
        result.Cout += -self.dynamics_relu(self.x_equilibrium) +\
            self.x_equilibrium

        return result

    def possible_dx(self, x):
        assert(isinstance(x, torch.Tensor))
        assert(len(x) == self.x_dim)
        return [self.step_forward(x)]

    def step_forward(self, x_start):
        assert(isinstance(x_start, torch.Tensor))
        return self.dynamics_relu(x_start) - \
            self.dynamics_relu(self.x_equilibrium) + self.x_equilibrium


class AutonomousResidualReLUSystemGivenEquilibrium:
    """
    This system models an autonomous system with known equilibirum x* using
    a feedforward neural network with ReLU activations. The neural network
    learn the residual dynamics
    x[n+1] = ϕ(x[n]) − ϕ(x*) + x[n]
    where ϕ is a feedforward (leaky) ReLU network.
    """

    def __init__(self, dtype, x_lo, x_up, dynamics_relu, x_equilibrium):
        """
        @param dtype The torch datatype
        @param x_lo, x_up torch tensor that lower and upper bound the state
        @param dynamics_relu torch model that represents the dynamics
        @param x_equilibrium The equilibrium state.
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
        assert(x_equilibrium.shape == (self.x_dim,))
        self.x_equilibrium = x_equilibrium

    @property
    def x_lo_all(self):
        return self.x_lo.detach().numpy()

    @property
    def x_up_all(self):
        return self.x_up.detach().numpy()

    def mixed_integer_constraints(self):
        """
        @return mixed-integer linear constraints MixedIntegerConstraintsReturn
                Ain_x, Ain_s, Ain_gamma, rhs_in,
                Aeq_x, Aeq_s, Aeq_gamma, rhs_eq
                such that
                x[n+1] = Aout_input @ x + Aout_s @ s + Cout
                 or x_dot = Aout_input @ x + Aout_s @ s + Cout
                s.t.
                Ain_x @ x + Ain_s @ s + Ain_gamma @ gamma <= rhs_in
                Aeq_x @ x + Aeq_s @ s + Aeq_gamma @ gamma == rhs_eq
        """
        (result, z_pre_relu_lo, z_pre_relu_up, z_post_relu_lo,
         z_post_relu_up) = self.dynamics_relu_free_pattern.output_constraint(
            self.x_lo, self.x_up)
        result.Cout += -self.dynamics_relu(self.x_equilibrium)
        if result.Aout_input is None:
            result.Aout_input = torch.eye(self.x_dim, dtype=self.dtype)
        else:
            result.Aout_input += torch.eye(self.x_dim, dtype=self.dtype)

        return result

    def possible_dx(self, x):
        assert(isinstance(x, torch.Tensor))
        assert(len(x) == self.x_dim)
        return [self.step_forward(x)]

    def step_forward(self, x_start):
        assert(isinstance(x_start, torch.Tensor))
        return self.dynamics_relu(x_start) - \
            self.dynamics_relu(self.x_equilibrium) + x_start


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
        self.dtype = dtype
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
        @return mixed-integer linear constraints MixedIntegerConstraintsReturn
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

    def step_forward(self, x_start, u_start):
        assert(isinstance(x_start, torch.Tensor))
        assert(isinstance(u_start, torch.Tensor))
        return self.dynamics_relu(torch.cat((x_start, u_start)))

    def possible_dx(self, x, u):
        assert(isinstance(x, torch.Tensor))
        assert(len(x) == self.x_dim)
        assert(isinstance(u, torch.Tensor))
        assert(u.shape == (self.u_dim,))
        return [self.dynamics_relu(torch.cat((x, u)))]


class ReLUSystemGivenEquilibrium:
    """
    Represent a forward dynamical system with given equilibrium x*, u*. The
    dynamics is
    x[n+1] = ϕ(x[n], u[n]) − ϕ(x*, u*) + x*
    where ϕ is a feedforward (leaky) ReLU network.
    x[n+1], x[n] and u[n] satisfy a piecewise affine relationship. When x[n],
    u[n] are bounded, we can write this relationship using mixed-integer linear
    constraints.
    """

    def __init__(
        self, dtype, x_lo, x_up, u_lo, u_up, dynamics_relu, x_equilibrium,
            u_equilibrium):
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
        @param x_equilibrium The equilibrium state.
        @param u_equilibrium The control at the equilibrium.
        """
        assert(len(x_lo.shape) == 1)
        assert(x_lo.shape == x_up.shape)
        assert(torch.all(x_lo <= x_up))
        self.dtype = dtype
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
        assert(x_equilibrium.shape == (self.x_dim,))
        assert(torch.all(x_lo <= x_equilibrium))
        assert(torch.all(x_up >= x_equilibrium))
        self.x_equilibrium = x_equilibrium
        assert(u_equilibrium.shape == (self.u_dim,))
        assert(torch.all(u_lo <= u_equilibrium))
        assert(torch.all(u_up >= u_equilibrium))
        self.u_equilibrium = u_equilibrium

    @property
    def x_lo_all(self):
        return self.x_lo.detach().numpy()

    @property
    def x_up_all(self):
        return self.x_up.detach().numpy()

    def mixed_integer_constraints(self):
        """
        The relationship between x[n], u[n] and x[n+1] can be captured by mixed
        -integer linear constraints.
        """
        xu_lo = torch.cat((self.x_lo, self.u_lo))
        xu_up = torch.cat((self.x_up, self.u_up))
        (result, z_pre_relu_lo, z_pre_relu_up, z_post_relu_lo,
         z_post_relu_up) = self.dynamics_relu_free_pattern.output_constraint(
            xu_lo, xu_up)
        result.Aout_slack = result.Aout_slack.reshape((self.x_dim, -1))
        result.Cout = result.Cout.reshape((-1))
        result.Cout += -self.dynamics_relu(
            torch.cat((self.x_equilibrium, self.u_equilibrium))) +\
            self.x_equilibrium

        return result

    def step_forward(self, x_start, u_start):
        assert(isinstance(x_start, torch.Tensor))
        assert(isinstance(u_start, torch.Tensor))
        x_next = self.dynamics_relu(torch.cat((x_start, u_start))) - \
            self.dynamics_relu(torch.cat(
                (self.x_equilibrium, self.u_equilibrium))) + self.x_equilibrium
        return x_next

    def possible_dx(self, x, u):
        assert(isinstance(x, torch.Tensor))
        assert(len(x) == self.x_dim)
        assert(isinstance(u, torch.Tensor))
        assert(u.shape == (self.u_dim,))
        return [self.step_forward(x, u)]


class ReLUSecondOrderSystemGivenEquilibrium:
    """
    For a second order system
    q̇ = v
    v̇ = f(q, v, u)
    We use a fully connected network with (leaky) ReLU activation unit ϕ to
    approximate its second order dynamics (in discrete time), as
    v[n+1] = ϕ(q[n], v[n], u[n]) − ϕ(q*, v*, u*)
    For the update on q, we use mid-point interpolation
    q[n+1] = q[n] + (v[n] + v[n+1]) / 2 * dt
    @note at the equilibrium, v should be 0.
    """
    def __init__(
        self, dtype, x_lo: torch.Tensor, x_up: torch.Tensor,
        u_lo: torch.Tensor, u_up: torch.Tensor, dynamics_relu,
        q_equilibrium: torch.Tensor, u_equilibrium: torch.Tensor,
            dt: float):
        """
        @param x_lo The lower bound of state x = [q; v].
        @param x_up The upper bound of state x = [q; v].
        @param u_lo The lower bound of the input u.
        @param u_up The upper bound of the input u.
        @param dynamics_relu A fully connected network with (leaky) ReLU
        activation units. The input to the network is (q, v, u), the output of
        the network is of same dimension as v.
        @param q_equilibrium The equilibrium position.
        @param u_equilibrium The control at the equilibrium.
        @param dt The integration time step.
        """
        self.dtype = dtype
        self.x_dim = x_lo.numel()
        assert(isinstance(x_lo, torch.Tensor))
        self.x_lo = x_lo
        assert(isinstance(x_up, torch.Tensor))
        assert(x_up.shape == (self.x_dim,))
        self.x_up = x_up
        self.u_dim = u_lo.numel()
        assert(isinstance(u_lo, torch.Tensor))
        self.u_lo = u_lo
        assert(isinstance(u_up, torch.Tensor))
        assert(u_up.shape == (self.u_dim,))
        self.u_up = u_up
        assert(dynamics_relu[0].in_features == self.x_dim + self.u_dim)
        self.nv = dynamics_relu[-1].out_features
        self.nq = self.x_dim - self.nv
        self.dynamics_relu = dynamics_relu
        self.dynamics_relu_free_pattern = relu_to_optimization.ReLUFreePattern(
            dynamics_relu, dtype)
        assert(isinstance(q_equilibrium, torch.Tensor))
        assert(q_equilibrium.shape == (self.nq,))
        self.q_equilibrium = q_equilibrium
        self.x_equilibrium = torch.cat((
            self.q_equilibrium, torch.zeros((self.nv,), dtype=self.dtype)))
        assert(torch.all(self.x_equilibrium >= self.x_lo))
        assert(torch.all(self.x_equilibrium <= self.x_up))
        assert(isinstance(u_equilibrium, torch.Tensor))
        assert(u_equilibrium.shape == (self.u_dim,))
        self.u_equilibrium = u_equilibrium
        assert(torch.all(self.u_equilibrium >= self.u_lo))
        assert(torch.all(self.u_equilibrium <= self.u_up))
        assert(isinstance(dt, float))
        assert(dt > 0)
        self.dt = dt

    @property
    def x_lo_all(self):
        return self.x_lo.detach().numpy()

    @property
    def x_up_all(self):
        return self.x_up.detach().numpy()

    def mixed_integer_constraints(self) ->\
            gurobi_torch_mip.MixedIntegerConstraintsReturn:
        """
        The relationship between x[n], u[n] and x[n+1] can be captured by mixed
        -integer linear constraints.
        Please refer to gurobi_torch_mip.MixedIntegerConstraintsReturn for the
        meaning of each term in the output.
        """
        # `result` contains the mixed integer linear constraint to write
        # ϕ(q[n], v[n], u[n]) as a function
        # ϕ(q[n], v[n], u[n]) = result.Aout_slack * s + result.Cout
        # whee s is the slack variable.
        # For the constraint v[n+1] = ϕ(q[n], v[n], u[n]) − ϕ(q*, v*, u*)
        # This is equivalent to
        # v[n+1] = result.Aout_slack * s  + result.Cout - ϕ(q*, v*, u*)
        (result, z_pre_relu_lo, z_pre_relu_up, z_post_relu_lo,
         z_post_relu_up) = self.dynamics_relu_free_pattern.output_constraint(
            torch.cat((self.x_lo, self.u_lo)),
            torch.cat((self.x_up, self.u_up)))
        assert(result.Aout_input is None)
        assert(result.Aout_binary is None)
        if (len(result.Aout_slack.shape) == 1):
            result.Aout_slack = result.Aout_slack.reshape((1, -1))
        if (len(result.Cout.shape) == 0):
            result.Cout = result.Cout.reshape((-1))
        result.Cout -= self.dynamics_relu(torch.cat((
            self.x_equilibrium, self.u_equilibrium)))
        # We also need to add the output constraint
        # q[n+1] = q[n] + (v[n] + v[n+1]) * dt / 2
        #        = [I dt/2*I 0] * [q[n]; v[n]; u[n]] +
        #          + (result.Aout_slack*dt/2) * s + result.Cout*dt/2
        result.Aout_input = torch.cat((torch.cat((
            torch.eye(self.nq, dtype=self.dtype),
            self.dt/2 * torch.eye(self.nv, dtype=self.dtype),
            torch.zeros((self.nq, self.u_dim), dtype=self.dtype)), dim=1),
            torch.zeros((self.nv, self.x_dim + self.u_dim), dtype=self.dtype)),
            dim=0)
        result.Aout_slack = torch.cat((
            result.Aout_slack * self.dt / 2, result.Aout_slack), dim=0)
        result.Cout = torch.cat((result.Cout * self.dt / 2, result.Cout),
                                dim=0)
        return result

    def step_forward(self, x_start, u_start):
        # Compute x[n+1] according to
        # v[n+1] = ϕ(q[n], v[n], u[n]) − ϕ(q*, v*, u*)
        # q[n+1] = q[n] + (v[n] + v[n+1]) * dt / 2
        assert(isinstance(x_start, torch.Tensor))
        assert(isinstance(u_start, torch.Tensor))
        v_next = self.dynamics_relu(torch.cat((x_start, u_start))) - \
            self.dynamics_relu(torch.cat(
                (self.x_equilibrium, self.u_equilibrium)))
        q_next = x_start[:self.nq] + (x_start[self.nq:] + v_next) * self.dt / 2
        x_next = torch.cat((q_next, v_next))
        return x_next

    def possible_dx(self, x, u):
        assert(isinstance(x, torch.Tensor))
        assert(len(x) == self.x_dim)
        assert(isinstance(u, torch.Tensor))
        assert(u.shape == (self.u_dim,))
        return [self.step_forward(x, u)]
