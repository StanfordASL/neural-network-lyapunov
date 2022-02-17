import torch
import neural_network_lyapunov.relu_to_optimization as relu_to_optimization
import neural_network_lyapunov.hybrid_linear_system as hybrid_linear_system
import neural_network_lyapunov.gurobi_torch_mip as gurobi_torch_mip
import neural_network_lyapunov.mip_utils as mip_utils
import gurobipy


class ReLUDynamicsConstraintReturn(
        hybrid_linear_system.DynamicsConstraintReturn):
    def __init__(self,
                 slack,
                 binary,
                 x_next_lb_IA=None,
                 x_next_ub_IA=None,
                 x_next_bound_prog=None,
                 x_next_bound_var=None,
                 x_var=None):
        super(ReLUDynamicsConstraintReturn,
              self).__init__(slack, binary, x_next_lb_IA, x_next_ub_IA,
                             x_next_bound_prog, x_next_bound_var, x_var)
        self.nn_input = None
        self.nn_input_lo = None
        self.nn_input_up = None
        self.nn_output_lo = None
        self.nn_output_up = None
        self.relu_input_lo = None
        self.relu_input_up = None
        self.relu_output_lo = None
        self.relu_output_up = None

    def from_mip_cnstr_return(self, mip_cnstr_return: relu_to_optimization.
                              ReLUMixedIntegerConstraintsReturn,
                              nn_input: list):
        assert (isinstance(
            mip_cnstr_return,
            relu_to_optimization.ReLUMixedIntegerConstraintsReturn))
        self.nn_input = nn_input
        self.nn_input_lo = mip_cnstr_return.nn_input_lo
        self.nn_input_up = mip_cnstr_return.nn_input_up
        self.nn_output_lo = mip_cnstr_return.nn_output_lo
        self.nn_output_up = mip_cnstr_return.nn_output_up
        self.relu_input_lo = mip_cnstr_return.relu_input_lo
        self.relu_input_up = mip_cnstr_return.relu_input_up
        self.relu_output_lo = mip_cnstr_return.relu_output_lo
        self.relu_output_up = mip_cnstr_return.relu_output_up


def _add_dynamics_constraint_autonmous(mip_cnstr_return,
                                       x_next_lb_IA,
                                       x_next_ub_IA,
                                       mip: gurobi_torch_mip.GurobiTorchMIP,
                                       x_var,
                                       x_next_var,
                                       slack_var_name,
                                       binary_var_name,
                                       network_bound_propagate_method,
                                       binary_var_type=gurobipy.GRB.BINARY):
    """
    Args:
        mip_cnstr_return: Returned from mixed_integer_constraints() function.
        It encodes the constraint between x and x_next.
    """
    dtype = mip.dtype
    x_dim = len(x_var)
    slack, binary = mip.add_mixed_integer_linear_constraints(
        mip_cnstr_return, x_var, x_next_var, slack_var_name, binary_var_name,
        "dynamics_ineq", "dynamics_eq", "dynamics_output", binary_var_type)
    ret = ReLUDynamicsConstraintReturn(slack, binary)
    ret.from_mip_cnstr_return(mip_cnstr_return, x_var)
    if network_bound_propagate_method in (
            mip_utils.PropagateBoundsMethod.IA,
            mip_utils.PropagateBoundsMethod.IA_MIP):
        ret.x_next_lb_IA = x_next_lb_IA
        ret.x_next_ub_IA = x_next_ub_IA
    if network_bound_propagate_method in (
            mip_utils.PropagateBoundsMethod.LP,
            mip_utils.PropagateBoundsMethod.MIP,
            mip_utils.PropagateBoundsMethod.IA_MIP):
        ret.x_next_bound_prog = gurobi_torch_mip.GurobiTorchMILP(dtype)
        ret.x_next_bound_prog.gurobi_model.setParam(
            gurobipy.GRB.Param.OutputFlag, False)
        ret.x_next_bound_var = ret.x_next_bound_prog.addVars(
            x_dim, lb=-gurobipy.GRB.INFINITY)
        ret.x_var = ret.x_next_bound_prog.addVars(x_dim,
                                                  lb=-gurobipy.GRB.INFINITY)
        x_next_bound_prog_binary_type = gurobipy.GRB.CONTINUOUS if \
            network_bound_propagate_method == \
            mip_utils.PropagateBoundsMethod.LP else gurobipy.GRB.BINARY
        ret.x_next_bound_prog.add_mixed_integer_linear_constraints(
            mip_cnstr_return, ret.x_var, ret.x_next_bound_var, "", "", "", "",
            "", x_next_bound_prog_binary_type)
    return ret


class AutonomousReLUSystem:
    """
    This system models an autonomous using a feedforward
    neural network with ReLU activations
    x[n+1] = relu(x[n])
    or
    x_dot = relu(x)
    """
    def __init__(
            self,
            dtype,
            x_lo,
            x_up,
            dynamics_relu,
            network_bound_propagate_method=mip_utils.PropagateBoundsMethod.IA):
        """
        @param dtype The torch datatype
        @param x_lo, x_up torch tensor that lower and upper bound the state
        @param dynamics_relu torch model that represents the dynamics
        """
        assert (len(x_lo) == len(x_up))
        assert (torch.all(x_up >= x_lo))
        assert (dynamics_relu[0].in_features == dynamics_relu[-1].out_features)
        self.dtype = dtype
        self.x_lo = x_lo
        self.x_up = x_up
        self.x_dim = len(self.x_lo)
        self.dynamics_relu = dynamics_relu
        self.dynamics_relu_free_pattern = relu_to_optimization.ReLUFreePattern(
            dynamics_relu, dtype)
        self.network_bound_propagate_method = network_bound_propagate_method

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
        result = self.dynamics_relu_free_pattern.output_constraint(
            self.x_lo, self.x_up, self.network_bound_propagate_method)
        return result

    def possible_dx(self, x):
        assert (isinstance(x, torch.Tensor))
        assert (len(x) == self.x_dim)
        return [self.dynamics_relu(x)]

    def step_forward(self, x_start):
        assert (isinstance(x_start, torch.Tensor))
        return self.dynamics_relu(x_start)

    def add_dynamics_constraint(self,
                                mip: gurobi_torch_mip.GurobiTorchMIP,
                                x_var,
                                x_next_var,
                                slack_var_name,
                                binary_var_name,
                                binary_var_type=gurobipy.GRB.BINARY):
        mip_cnstr_return = self.mixed_integer_constraints()
        if self.network_bound_propagate_method in (
                mip_utils.PropagateBoundsMethod.IA,
                mip_utils.PropagateBoundsMethod.IA_MIP):
            x_next_lb_IA = mip_cnstr_return.nn_output_lo
            x_next_ub_IA = mip_cnstr_return.nn_output_up
        else:
            x_next_lb_IA = None
            x_next_ub_IA = None
        return _add_dynamics_constraint_autonmous(
            mip_cnstr_return, x_next_lb_IA, x_next_ub_IA, mip, x_var,
            x_next_var, slack_var_name, binary_var_name,
            self.network_bound_propagate_method, binary_var_type)


class AutonomousReLUSystemGivenEquilibrium:
    """
    This system models an autonomous system with known equilibirum x* using
    a feedforward neural network with ReLU activations
    For discrete-time system
    x[n+1] = ϕ(x[n]) − ϕ(x*) + x*
    For continuous-time system
    ẋ = ϕ(x) − ϕ(x*)
    where ϕ is a feedforward (leaky) ReLU network.
    """
    def __init__(
            self,
            dtype,
            x_lo,
            x_up,
            dynamics_relu,
            x_equilibrium,
            discrete_time_flag=True,
            network_bound_propagate_method=mip_utils.PropagateBoundsMethod.IA):
        """
        @param dtype The torch datatype
        @param x_lo, x_up torch tensor that lower and upper bound the state
        @param dynamics_relu torch model that represents the dynamics
        @param x_equilibrium The equilibrium state.
        """
        assert (len(x_lo) == len(x_up))
        assert (torch.all(x_up >= x_lo))
        assert (dynamics_relu[0].in_features == dynamics_relu[-1].out_features)
        self.dtype = dtype
        self.x_lo = x_lo
        self.x_up = x_up
        self.x_dim = len(self.x_lo)
        self.dynamics_relu = dynamics_relu
        self.dynamics_relu_free_pattern = relu_to_optimization.ReLUFreePattern(
            dynamics_relu, dtype)
        assert (x_equilibrium.shape == (self.x_dim, ))
        self.x_equilibrium = x_equilibrium
        self.discrete_time_flag = discrete_time_flag
        self.network_bound_propagate_method = network_bound_propagate_method

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
        result = self.dynamics_relu_free_pattern.output_constraint(
            self.x_lo, self.x_up, self.network_bound_propagate_method)
        if self.discrete_time_flag:
            result.Cout += -self.dynamics_relu(self.x_equilibrium) +\
                self.x_equilibrium
        else:
            result.Cout += -self.dynamics_relu(self.x_equilibrium)
        return result

    def possible_dx(self, x):
        assert (isinstance(x, torch.Tensor))
        assert (len(x) == self.x_dim)
        return [self.step_forward(x)]

    def step_forward(self, x_start):
        assert (isinstance(x_start, torch.Tensor))
        if self.discrete_time_flag:
            return self.dynamics_relu(x_start) - \
                self.dynamics_relu(self.x_equilibrium) + self.x_equilibrium
        else:
            return self.dynamics_relu(x_start) - \
                self.dynamics_relu(self.x_equilibrium)

    def add_dynamics_constraint(self,
                                mip: gurobi_torch_mip.GurobiTorchMIP,
                                x_var,
                                x_next_var,
                                slack_var_name,
                                binary_var_name,
                                binary_var_type=gurobipy.GRB.BINARY):
        mip_cnstr_return = self.mixed_integer_constraints()
        if self.network_bound_propagate_method in (
                mip_utils.PropagateBoundsMethod.IA,
                mip_utils.PropagateBoundsMethod.IA_MIP):
            relu_at_equilibrium = self.dynamics_relu(self.x_equilibrium)
            x_next_lb_IA = mip_cnstr_return.nn_output_lo - relu_at_equilibrium\
                + (self.x_equilibrium if self.discrete_time_flag else 0.)
            x_next_ub_IA = mip_cnstr_return.nn_output_up - relu_at_equilibrium\
                + (self.x_equilibrium if self.discrete_time_flag else 0.)
        else:
            x_next_lb_IA = None
            x_next_ub_IA = None
        return _add_dynamics_constraint_autonmous(
            mip_cnstr_return, x_next_lb_IA, x_next_ub_IA, mip, x_var,
            x_next_var, slack_var_name, binary_var_name,
            self.network_bound_propagate_method, binary_var_type)


class AutonomousResidualReLUSystemGivenEquilibrium:
    """
    This system models an autonomous system with known equilibirum x* using
    a feedforward neural network with ReLU activations. The neural network
    learn the residual dynamics
    Discrete-time system
    x[n+1] = ϕ(x[n]) − ϕ(x*) + x[n]
    Continuous-time system
    ẋ = ϕ(x) − ϕ(x*)
    where ϕ is a feedforward (leaky) ReLU network.
    """
    def __init__(
            self,
            dtype,
            x_lo,
            x_up,
            dynamics_relu,
            x_equilibrium,
            discrete_time_flag=True,
            network_bound_propagate_method=mip_utils.PropagateBoundsMethod.IA):
        """
        @param dtype The torch datatype
        @param x_lo, x_up torch tensor that lower and upper bound the state
        @param dynamics_relu torch model that represents the dynamics
        @param x_equilibrium The equilibrium state.
        """
        assert (len(x_lo) == len(x_up))
        assert (torch.all(x_up >= x_lo))
        assert (dynamics_relu[0].in_features == dynamics_relu[-1].out_features)
        self.dtype = dtype
        self.x_lo = x_lo
        self.x_up = x_up
        self.x_dim = len(self.x_lo)
        self.dynamics_relu = dynamics_relu
        self.dynamics_relu_free_pattern = relu_to_optimization.ReLUFreePattern(
            dynamics_relu, dtype)
        assert (x_equilibrium.shape == (self.x_dim, ))
        self.x_equilibrium = x_equilibrium
        self.discrete_time_flag = discrete_time_flag
        self.network_bound_propagate_method = network_bound_propagate_method

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
        result = self.dynamics_relu_free_pattern.output_constraint(
            self.x_lo, self.x_up, self.network_bound_propagate_method)
        result.Cout += -self.dynamics_relu(self.x_equilibrium)
        if self.discrete_time_flag:
            if result.Aout_input is None:
                result.Aout_input = torch.eye(self.x_dim, dtype=self.dtype)
            else:
                result.Aout_input += torch.eye(self.x_dim, dtype=self.dtype)

        return result

    def possible_dx(self, x):
        assert (isinstance(x, torch.Tensor))
        assert (len(x) == self.x_dim)
        return [self.step_forward(x)]

    def step_forward(self, x_start):
        assert (isinstance(x_start, torch.Tensor))
        if self.discrete_time_flag:
            return self.dynamics_relu(x_start) - \
                self.dynamics_relu(self.x_equilibrium) + x_start
        else:
            return self.dynamics_relu(x_start) - \
                self.dynamics_relu(self.x_equilibrium)

    def add_dynamics_constraint(self,
                                mip: gurobi_torch_mip.GurobiTorchMIP,
                                x_var,
                                x_next_var,
                                slack_var_name,
                                binary_var_name,
                                binary_var_type=gurobipy.GRB.BINARY):
        mip_cnstr_return = self.mixed_integer_constraints()
        if self.network_bound_propagate_method in (
                mip_utils.PropagateBoundsMethod.IA,
                mip_utils.PropagateBoundsMethod.IA_MIP):
            relu_at_equilibrium = self.dynamics_relu(self.x_equilibrium)
            x_next_lb_IA = mip_cnstr_return.nn_output_lo - relu_at_equilibrium\
                + (self.x_lo if self.discrete_time_flag else 0.)
            x_next_ub_IA = mip_cnstr_return.nn_output_up - relu_at_equilibrium\
                + (self.x_up if self.discrete_time_flag else 0.)
        else:
            x_next_lb_IA = None
            x_next_ub_IA = None
        return _add_dynamics_constraint_autonmous(
            mip_cnstr_return, x_next_lb_IA, x_next_ub_IA, mip, x_var,
            x_next_var, slack_var_name, binary_var_name,
            self.network_bound_propagate_method, binary_var_type)


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
        assert (len(x_lo.shape) == 1)
        assert (x_lo.shape == x_up.shape)
        assert (torch.all(x_lo <= x_up))
        self.dtype = dtype
        self.x_dim = x_lo.numel()
        self.x_lo = x_lo
        self.x_up = x_up
        assert (len(u_lo.shape) == 1)
        assert (u_lo.shape == u_up.shape)
        assert (torch.all(u_lo <= u_up))
        self.u_dim = u_lo.numel()
        self.u_lo = u_lo
        self.u_up = u_up
        assert (dynamics_relu[0].in_features == self.x_dim + self.u_dim)
        assert (dynamics_relu[-1].out_features == self.x_dim)
        self.dynamics_relu = dynamics_relu
        self.dynamics_relu_free_pattern = relu_to_optimization.ReLUFreePattern(
            dynamics_relu, dtype)

    @property
    def x_lo_all(self):
        return self.x_lo.detach().numpy()

    @property
    def x_up_all(self):
        return self.x_up.detach().numpy()

    def mixed_integer_constraints(self, u_lo=None, u_up=None):
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
        if u_lo is None:
            u_lo = self.u_lo
        if u_up is None:
            u_up = self.u_up
        xu_lo = torch.cat((self.x_lo, u_lo))
        xu_up = torch.cat((self.x_up, u_up))
        result = self.dynamics_relu_free_pattern.output_constraint(
            xu_lo, xu_up, mip_utils.PropagateBoundsMethod.IA)

        return result

    def step_forward(self, x_start, u_start):
        assert (isinstance(x_start, torch.Tensor))
        assert (isinstance(u_start, torch.Tensor))
        return self.dynamics_relu(torch.cat((x_start, u_start), dim=-1))

    def possible_dx(self, x, u):
        assert (isinstance(x, torch.Tensor))
        assert (isinstance(u, torch.Tensor))
        return [self.dynamics_relu(torch.cat((x, u), dim=-1))]

    def add_dynamics_constraint(self,
                                mip,
                                x_var,
                                x_next_var,
                                u_var,
                                slack_var_name,
                                binary_var_name,
                                additional_u_lo: torch.Tensor = None,
                                additional_u_up: torch.Tensor = None,
                                binary_var_type=gurobipy.GRB.BINARY):
        return _add_dynamics_mip_constraints(mip, self, x_var, x_next_var,
                                             u_var, slack_var_name,
                                             binary_var_name, additional_u_lo,
                                             additional_u_up, binary_var_type)


class ReLUSystemGivenEquilibrium:
    """
    Represent a forward dynamical system with given equilibrium x*, u*. The
    dynamics is
    discrete time system:
    x[n+1] = ϕ(x[n], u[n]) − ϕ(x*, u*) + x*
    continuous time system
    ẋ = ϕ(x, u) − ϕ(x*, u*)
    where ϕ is a feedforward (leaky) ReLU network.
    x[n+1], x[n] and u[n] satisfy a piecewise affine relationship. When x[n],
    u[n] are bounded, we can write this relationship using mixed-integer linear
    constraints.
    """
    def __init__(self, dtype, x_lo, x_up, u_lo, u_up, dynamics_relu,
                 x_equilibrium, u_equilibrium, discrete_time_flag: bool):
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
        assert (len(x_lo.shape) == 1)
        assert (x_lo.shape == x_up.shape)
        assert (torch.all(x_lo <= x_up))
        self.dtype = dtype
        self.x_dim = x_lo.numel()
        self.x_lo = x_lo
        self.x_up = x_up
        assert (len(u_lo.shape) == 1)
        assert (u_lo.shape == u_up.shape)
        assert (torch.all(u_lo <= u_up))
        self.u_dim = u_lo.numel()
        self.u_lo = u_lo
        self.u_up = u_up
        assert (dynamics_relu[0].in_features == self.x_dim + self.u_dim)
        assert (dynamics_relu[-1].out_features == self.x_dim)
        self.dynamics_relu = dynamics_relu
        self.dynamics_relu_free_pattern = relu_to_optimization.ReLUFreePattern(
            dynamics_relu, dtype)
        assert (x_equilibrium.shape == (self.x_dim, ))
        assert (torch.all(x_lo <= x_equilibrium))
        assert (torch.all(x_up >= x_equilibrium))
        self.x_equilibrium = x_equilibrium
        assert (u_equilibrium.shape == (self.u_dim, ))
        assert (torch.all(u_lo <= u_equilibrium))
        assert (torch.all(u_up >= u_equilibrium))
        self.u_equilibrium = u_equilibrium
        self.discrete_time_flag = discrete_time_flag

    @property
    def x_lo_all(self):
        return self.x_lo.detach().numpy()

    @property
    def x_up_all(self):
        return self.x_up.detach().numpy()

    def mixed_integer_constraints(self, u_lo=None, u_up=None):
        """
        The relationship between x[n], u[n] and x[n+1] can be captured by mixed
        -integer linear constraints.
        """
        if u_lo is None:
            u_lo = self.u_lo
        if u_up is None:
            u_up = self.u_up
        xu_lo = torch.cat((self.x_lo, u_lo))
        xu_up = torch.cat((self.x_up, u_up))
        result = self.dynamics_relu_free_pattern.output_constraint(
            xu_lo, xu_up, mip_utils.PropagateBoundsMethod.IA)
        result.Aout_slack = result.Aout_slack.reshape((self.x_dim, -1))
        result.Cout = result.Cout.reshape((-1))
        result.Cout += -self.dynamics_relu(
            torch.cat((self.x_equilibrium, self.u_equilibrium)))
        if self.discrete_time_flag:
            result.Cout += self.x_equilibrium

        return result

    def step_forward(self, x_start, u_start):
        assert (isinstance(x_start, torch.Tensor))
        assert (isinstance(u_start, torch.Tensor))
        x_next = self.dynamics_relu(torch.cat((x_start, u_start), dim=-1)) - \
            self.dynamics_relu(torch.cat(
                (self.x_equilibrium, self.u_equilibrium)))
        if self.discrete_time_flag:
            x_next += self.x_equilibrium
        return x_next

    def possible_dx(self, x, u):
        assert (isinstance(x, torch.Tensor))
        assert (isinstance(u, torch.Tensor))
        return [self.step_forward(x, u)]

    def add_dynamics_constraint(self,
                                mip,
                                x_var,
                                x_next_var,
                                u_var,
                                slack_var_name,
                                binary_var_name,
                                additional_u_lo: torch.Tensor = None,
                                additional_u_up: torch.Tensor = None,
                                binary_var_type=gurobipy.GRB.BINARY):
        return _add_dynamics_mip_constraints(mip, self, x_var, x_next_var,
                                             u_var, slack_var_name,
                                             binary_var_name, additional_u_lo,
                                             additional_u_up, binary_var_type)


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
    def __init__(self, dtype, x_lo: torch.Tensor, x_up: torch.Tensor,
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
        assert (isinstance(x_lo, torch.Tensor))
        self.x_lo = x_lo
        assert (isinstance(x_up, torch.Tensor))
        assert (x_up.shape == (self.x_dim, ))
        self.x_up = x_up
        self.u_dim = u_lo.numel()
        assert (isinstance(u_lo, torch.Tensor))
        self.u_lo = u_lo
        assert (isinstance(u_up, torch.Tensor))
        assert (u_up.shape == (self.u_dim, ))
        self.u_up = u_up
        assert (dynamics_relu[0].in_features == self.x_dim + self.u_dim)
        self.nv = dynamics_relu[-1].out_features
        self.nq = self.x_dim - self.nv
        self.dynamics_relu = dynamics_relu
        self.dynamics_relu_free_pattern = relu_to_optimization.ReLUFreePattern(
            dynamics_relu, dtype)
        assert (isinstance(q_equilibrium, torch.Tensor))
        assert (q_equilibrium.shape == (self.nq, ))
        self.q_equilibrium = q_equilibrium
        self.x_equilibrium = torch.cat(
            (self.q_equilibrium, torch.zeros((self.nv, ), dtype=self.dtype)))
        assert (torch.all(self.x_equilibrium >= self.x_lo))
        assert (torch.all(self.x_equilibrium <= self.x_up))
        assert (isinstance(u_equilibrium, torch.Tensor))
        assert (u_equilibrium.shape == (self.u_dim, ))
        self.u_equilibrium = u_equilibrium
        assert (torch.all(self.u_equilibrium >= self.u_lo))
        assert (torch.all(self.u_equilibrium <= self.u_up))
        assert (isinstance(dt, float))
        assert (dt > 0)
        self.dt = dt

    @property
    def x_lo_all(self):
        return self.x_lo.detach().numpy()

    @property
    def x_up_all(self):
        return self.x_up.detach().numpy()

    def mixed_integer_constraints(self, u_lo=None, u_up=None) ->\
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
        if u_lo is None:
            u_lo = self.u_lo
        if u_up is None:
            u_up = self.u_up
        result = self.dynamics_relu_free_pattern.output_constraint(
            torch.cat((self.x_lo, u_lo)), torch.cat((self.x_up, u_up)),
            mip_utils.PropagateBoundsMethod.IA)
        assert (result.Aout_input is None)
        assert (result.Aout_binary is None)
        if (len(result.Aout_slack.shape) == 1):
            result.Aout_slack = result.Aout_slack.reshape((1, -1))
        if (len(result.Cout.shape) == 0):
            result.Cout = result.Cout.reshape((-1))
        result.Cout -= self.dynamics_relu(
            torch.cat((self.x_equilibrium, self.u_equilibrium)))
        # We also need to add the output constraint
        # q[n+1] = q[n] + (v[n] + v[n+1]) * dt / 2
        #        = [I dt/2*I 0] * [q[n]; v[n]; u[n]] +
        #          + (result.Aout_slack*dt/2) * s + result.Cout*dt/2
        result.Aout_input = torch.cat(
            (torch.cat((torch.eye(self.nq, dtype=self.dtype),
                        self.dt / 2 * torch.eye(self.nv, dtype=self.dtype),
                        torch.zeros((self.nq, self.u_dim), dtype=self.dtype)),
                       dim=1),
             torch.zeros(
                 (self.nv, self.x_dim + self.u_dim), dtype=self.dtype)),
            dim=0)
        result.Aout_slack = torch.cat(
            (result.Aout_slack * self.dt / 2, result.Aout_slack), dim=0)
        result.Cout = torch.cat((result.Cout * self.dt / 2, result.Cout),
                                dim=0)
        return result

    def step_forward(self, x_start, u_start):
        # Compute x[n+1] according to
        # v[n+1] = ϕ(q[n], v[n], u[n]) − ϕ(q*, v*, u*)
        # q[n+1] = q[n] + (v[n] + v[n+1]) * dt / 2
        assert (isinstance(x_start, torch.Tensor))
        assert (isinstance(u_start, torch.Tensor))
        v_next = self.dynamics_relu(torch.cat((x_start, u_start), dim=-1)) - \
            self.dynamics_relu(torch.cat(
                (self.x_equilibrium, self.u_equilibrium)))
        if len(x_start.shape) == 1:
            q_next = x_start[:self.nq] + (x_start[self.nq:] + v_next) *\
                self.dt / 2
        else:
            # batch of x_start and u_start
            q_next = x_start[:, :self.nq] + (x_start[:, self.nq:] + v_next) *\
                self.dt / 2
        x_next = torch.cat((q_next, v_next), dim=-1)
        return x_next

    def possible_dx(self, x, u):
        assert (isinstance(x, torch.Tensor))
        assert (isinstance(u, torch.Tensor))
        return [self.step_forward(x, u)]

    def add_dynamics_constraint(self,
                                mip,
                                x_var,
                                x_next_var,
                                u_var,
                                slack_var_name,
                                binary_var_name,
                                additional_u_lo: torch.Tensor = None,
                                additional_u_up: torch.Tensor = None,
                                binary_var_type=gurobipy.GRB.BINARY):
        return _add_dynamics_mip_constraints(mip, self, x_var, x_next_var,
                                             u_var, slack_var_name,
                                             binary_var_name, additional_u_lo,
                                             additional_u_up, binary_var_type)


class ReLUSecondOrderResidueSystemGivenEquilibrium:
    """
    A second order system whose dynamics is represented by
    q[n+1] = q[n] + (v[n] + v[n+1]) * dt / 2
    v[n+1] - v[n] = ϕ(x̅[n], u[n]) − ϕ(x̅*, u*)
    Note that for the update on the velocity, we use the network to only
    represent the "residue" part v[n+1] - v[n], not v[n+1] directly.
    x̅ is a partial state of x. For many system, the "residue" part only
    depends on part of the state. For example, for a system that is shift
    invariant (such as a car), its acceleration doesn't depend on the location
    of the car.
    """
    def __init__(self, dtype, x_lo: torch.Tensor, x_up: torch.Tensor,
                 u_lo: torch.Tensor, u_up: torch.Tensor,
                 dynamics_relu: torch.nn.Sequential,
                 q_equilibrium: torch.Tensor, u_equilibrium: torch.Tensor,
                 dt: float, network_input_x_indices: list):
        """
        @param dynamics_relu A fully connected network that takes the input as
        a partial state and the control, and outputs the change of velocity
        v[n+1] - v[n] = ϕ(x̅[n], u[n]) − ϕ(x̅*, u*)
        @param q_equilibrium The equilibrium position.
        @param u_equilibrium The control at the equilibrium.
        @param dt The integration time step.
        @param network_input_x_indices The partial state
        x̅=x[network_input_x_indices]
        """
        self.dtype = dtype
        self.x_dim = x_lo.numel()
        assert (isinstance(x_lo, torch.Tensor))
        self.x_lo = x_lo
        assert (isinstance(x_up, torch.Tensor))
        assert (x_up.shape == (self.x_dim, ))
        self.x_up = x_up
        self.u_dim = u_lo.numel()
        assert (isinstance(u_lo, torch.Tensor))
        self.u_lo = u_lo
        assert (isinstance(u_up, torch.Tensor))
        assert (u_up.shape == (self.u_dim, ))
        self.u_up = u_up
        assert (isinstance(network_input_x_indices, list))
        assert (dynamics_relu[0].in_features == len(network_input_x_indices) +
                self.u_dim)
        self.nv = dynamics_relu[-1].out_features
        self.nq = self.x_dim - self.nv
        self.dynamics_relu = dynamics_relu
        self.dynamics_relu_free_pattern = relu_to_optimization.ReLUFreePattern(
            dynamics_relu, dtype)
        assert (isinstance(q_equilibrium, torch.Tensor))
        assert (q_equilibrium.shape == (self.nq, ))
        self.q_equilibrium = q_equilibrium
        self.x_equilibrium = torch.cat(
            (self.q_equilibrium, torch.zeros((self.nv, ), dtype=self.dtype)))
        assert (torch.all(self.x_equilibrium >= self.x_lo))
        assert (torch.all(self.x_equilibrium <= self.x_up))
        assert (isinstance(u_equilibrium, torch.Tensor))
        assert (u_equilibrium.shape == (self.u_dim, ))
        self.u_equilibrium = u_equilibrium
        assert (torch.all(self.u_equilibrium >= self.u_lo))
        assert (torch.all(self.u_equilibrium <= self.u_up))
        assert (isinstance(dt, float))
        assert (dt > 0)
        self.dt = dt
        self._network_input_x_indices = network_input_x_indices
        self.network_bound_propagate_method = \
            mip_utils.PropagateBoundsMethod.IA

    @property
    def x_lo_all(self):
        return self.x_lo.detach().numpy()

    @property
    def x_up_all(self):
        return self.x_up.detach().numpy()

    def step_forward(self, x_start, u_start):
        """
        Compute x[n+1] according to
        q[n+1] = q[n] + (v[n] + v[n+1]) * dt / 2
        v[n+1] - v[n] = ϕ(x̅[n], u[n]) − ϕ(x̅*, u*)
        """
        assert (isinstance(x_start, torch.Tensor))
        assert (isinstance(u_start, torch.Tensor))
        if len(x_start.shape) == 1:
            q_start = x_start[:self.nq]
            v_start = x_start[self.nq:]
            v_next = v_start + self.dynamics_relu(torch.cat((
                x_start[self._network_input_x_indices], u_start))) -\
                self.dynamics_relu(torch.cat((self.x_equilibrium[
                    self._network_input_x_indices], self.u_equilibrium)))
            q_next = q_start + (v_start + v_next) * self.dt / 2
            return torch.cat((q_next, v_next))
        elif len(x_start.shape) == 2:
            # batch of data.
            q_start = x_start[:, :self.nq]
            v_start = x_start[:, self.nq:]
            v_next = v_start + self.dynamics_relu(torch.cat((x_start[
                :, self._network_input_x_indices], u_start), dim=-1)) -\
                self.dynamics_relu(torch.cat((self.x_equilibrium[
                    self._network_input_x_indices], self.u_equilibrium)))
            q_next = q_start + (v_start + v_next) * self.dt / 2
            return torch.cat((q_next, v_next), dim=-1)

    def possible_dx(self, x, u):
        assert (isinstance(x, torch.Tensor))
        assert (isinstance(u, torch.Tensor))
        return [self.step_forward(x, u)]

    def add_dynamics_constraint(self,
                                mip: gurobi_torch_mip.GurobiTorchMIP,
                                x_var,
                                x_next_var,
                                u_var,
                                slack_var_name,
                                binary_var_name,
                                additional_u_lo: torch.Tensor = None,
                                additional_u_up: torch.Tensor = None,
                                binary_var_type=gurobipy.GRB.BINARY):
        u_lo = self.u_lo if additional_u_lo is None else torch.max(
            self.u_lo, additional_u_lo)
        u_up = self.u_up if additional_u_up is None else torch.min(
            self.u_up, additional_u_up)
        mip_cnstr_result = self.dynamics_relu_free_pattern.\
            output_constraint(torch.cat((self.x_lo[
                self._network_input_x_indices], u_lo)), torch.cat((
                    self.x_up[self._network_input_x_indices], u_up)),
                self.network_bound_propagate_method)
        # First add mip_cnstr_result, but don't impose the constraint on the
        # output of the network (we will impose the constraint separately)
        input_vars = [x_var[i] for i in self._network_input_x_indices] + u_var
        forward_slack, forward_binary = \
            mip.add_mixed_integer_linear_constraints(
                mip_cnstr_result, input_vars, None, slack_var_name,
                binary_var_name, "residue_forward_dynamics_ineq",
                "residue_forward_dynamics_eq", None, binary_var_type)
        # We want to impose the constraint
        # v[n+1] = v[n] + ϕ(x̅[n], u[n]) − ϕ(x̅*, u*)
        #        = v[n] + Aout_slack * s + Cout - ϕ(x̅*, u*)
        assert (mip_cnstr_result.Aout_input is None)
        assert (mip_cnstr_result.Aout_binary is None)

        if len(mip_cnstr_result.Aout_slack.shape) == 1:
            mip_cnstr_result.Aout_slack = mip_cnstr_result.Aout_slack.reshape(
                (1, -1))
        if len(mip_cnstr_result.Cout.shape) == 0:
            mip_cnstr_result.Cout = mip_cnstr_result.Cout.reshape((-1))
        v_next = x_next_var[self.nq:]
        v_curr = x_var[self.nq:]
        mip.addMConstrs(
            [
                torch.eye(self.nv, dtype=self.dtype),
                -torch.eye(self.nv, dtype=self.dtype),
                -mip_cnstr_result.Aout_slack
            ], [v_next, v_curr, forward_slack],
            b=mip_cnstr_result.Cout - self.dynamics_relu(
                torch.cat((self.x_equilibrium[self._network_input_x_indices],
                           self.u_equilibrium))),
            sense=gurobipy.GRB.EQUAL,
            name="residue_forward_dynamics_output")
        # Now add the constraint
        # q[n+1] - q[n] = (v[n+1] + v[n]) * dt / 2
        q_next = x_next_var[:self.nq]
        q_curr = x_var[:self.nq]
        mip.addMConstrs([
            torch.eye(self.nq,
                      dtype=self.dtype), -torch.eye(self.nq, dtype=self.dtype),
            -self.dt / 2 * torch.eye(self.nv, dtype=self.dtype),
            -self.dt / 2 * torch.eye(self.nv, dtype=self.dtype)
        ], [q_next, q_curr, v_next, v_curr],
                        b=torch.zeros((self.nv), dtype=self.dtype),
                        sense=gurobipy.GRB.EQUAL,
                        name="update_q_next")
        ret = ReLUDynamicsConstraintReturn(forward_slack, forward_binary)
        ret.from_mip_cnstr_return(mip_cnstr_result, input_vars)
        return ret


def _add_dynamics_mip_constraints(mip,
                                  relu_system,
                                  x_var,
                                  x_next_var,
                                  u_var,
                                  slack_var_name,
                                  binary_var_name,
                                  additional_u_lo: torch.Tensor = None,
                                  additional_u_up: torch.Tensor = None,
                                  binary_var_type=gurobipy.GRB.BINARY):
    u_lo = relu_system.u_lo if additional_u_lo is None else torch.max(
        relu_system.u_lo, additional_u_lo)
    u_up = relu_system.u_up if additional_u_up is None else torch.min(
        relu_system.u_up, additional_u_up)
    mip_cnstr = relu_system.mixed_integer_constraints(u_lo, u_up)
    input_vars = x_var + u_var
    slack, binary = mip.add_mixed_integer_linear_constraints(
        mip_cnstr, input_vars, x_next_var, slack_var_name, binary_var_name,
        "relu_forward_dynamics_ineq", "relu_forward_dynamics_eq",
        "relu_forward_dynamics_output", binary_var_type)
    ret = ReLUDynamicsConstraintReturn(slack, binary)
    ret.from_mip_cnstr_return(mip_cnstr, input_vars)
    return ret
