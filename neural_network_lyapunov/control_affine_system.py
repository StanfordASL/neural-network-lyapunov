import torch
import gurobipy
import neural_network_lyapunov.gurobi_torch_mip as gurobi_torch_mip
import neural_network_lyapunov.utils as utils
import neural_network_lyapunov.mip_utils as mip_utils
import neural_network_lyapunov.relu_to_optimization as relu_to_optimization


class ControlAffineSystemConstraintReturn:
    """
    The return type of
    ControlPiecewiseAffineSystem::mixed_integer_constraints() function.
    """
    def __init__(self):
        # A MixedIntegerConstraintsReturn object with f as the output.
        self.mip_cnstr_f = None
        # A MixedIntegerConstraintsReturn object with the flat vector
        # G.reshape((-1,)) as the output.
        self.mip_cnstr_G = None
        # The lower bound of f.
        self.f_lo = None
        # The upper bound of f.
        self.f_up = None
        # The lower bound of the flat vector G.reshape((-1,))
        self.G_flat_lo = None
        # The upper bound of the flat vector G.reshape((-1,))
        self.G_flat_up = None


class ControlPiecewiseAffineSystem:
    """
    Represent a continuous-time control-affine system
    ẋ=f(x)+G(x)u
    u_lo <= u <= u_up
    Notice that the dynamics ẋ is an affine function of u.
    We will assume that given u, ẋ is a (piecewise) affine function of x,
    hence we can write the dynamics constraint using (mixed-integer) linear
    constraints.
    """
    def __init__(self, x_lo: torch.Tensor, x_up: torch.Tensor,
                 u_lo: torch.Tensor, u_up: torch.Tensor):
        """
        Args:
          x_lo, x_up: We will constrain the state to be within the box
          x_lo <= x <= x_up.
          u_lo, u_up: The input limits.
        """
        assert (len(x_lo.shape) == 1)
        assert (x_lo.shape == x_up.shape)
        assert (torch.all(x_lo <= x_up))
        self.x_lo = x_lo
        self.x_up = x_up

        assert (len(u_lo.shape) == 1)
        assert (u_lo.shape == u_up.shape)
        assert (torch.all(u_lo <= u_up))
        self.u_lo = u_lo
        self.u_up = u_up

    @property
    def dtype(self):
        return self.x_lo.dtype

    @property
    def x_dim(self):
        return self.x_lo.numel()

    @property
    def u_dim(self):
        return self.u_lo.numel()

    @property
    def x_lo_all(self):
        return self.x_lo.detach().numpy()

    @property
    def x_up_all(self):
        return self.x_up.detach().numpy()

    def mixed_integer_constraints(self) -> ControlAffineSystemConstraintReturn:
        """
        Returns the mixed-integer linear constraints on f(x) and G(x), together
        with the bounds on f(x) and G(x)
        """
        raise NotImplementedError

    def dynamics(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """
        Computes ẋ=f(x)+G(x) * clamp(u, u_lo, u_up)
        """
        u_clamped = torch.max(torch.min(u, self.u_up), self.u_lo)
        return self.f(x) + self.G(x) @ u_clamped

    def f(self, x):
        """
        The dynamics is ẋ=f(x)+G(x)u
        """
        raise NotImplementedError

    def G(self, x):
        """
        The dynamics is ẋ=f(x)+G(x)u
        """
        raise NotImplementedError

    def can_be_equilibrium_state(self, x: torch.Tensor) -> bool:
        """
        Checks if the system can remain equilibrium at x. Namely
        ∃ u ∈ [u_lo, u_up] such that f(x) + G(x)u = 0
        """
        f = self.f(x)
        G = self.G(x)
        model = gurobipy.Model()
        u = model.addMVar(self.u_dim,
                          lb=self.u_lo.detach().numpy(),
                          ub=self.u_up.detach().numpy())
        model.addMConstrs(G.detach().numpy(),
                          u,
                          sense=gurobipy.GRB.EQUAL,
                          b=-f.detach().numpy())
        model.setParam(gurobipy.GRB.Param.OutputFlag, False)
        model.optimize()
        return model.status == gurobipy.GRB.OPTIMAL


class LinearSystem(ControlPiecewiseAffineSystem):
    """
    A linear system ẋ = A*x+B*u
    We create this system to test synthesizing control Lyapunov functions.
    """
    def __init__(self, A: torch.Tensor, B: torch.Tensor, x_lo: torch.Tensor,
                 x_up: torch.Tensor, u_lo: torch.Tensor, u_up: torch.Tensor):
        super(LinearSystem, self).__init__(x_lo, x_up, u_lo, u_up)
        assert (A.shape == (self.x_dim, self.x_dim))
        assert (B.shape == (self.x_dim, self.u_dim))
        self.A = A
        self.B = B

    def mixed_integer_constraints(self):
        # f = A*x
        # G = B
        ret = ControlAffineSystemConstraintReturn()
        ret.mip_cnstr_f = gurobi_torch_mip.MixedIntegerConstraintsReturn()
        ret.mip_cnstr_f.Aout_input = self.A
        ret.mip_cnstr_G = gurobi_torch_mip.MixedIntegerConstraintsReturn()
        ret.mip_cnstr_G.Cout = self.B.reshape((-1, ))
        # Independent of method (IA/MIP/LP), they all compute the same range
        # for f=A*x.
        ret.f_lo, ret.f_up = mip_utils.compute_range_by_IA(
            self.A, torch.zeros((self.x_dim, ), dtype=self.dtype), self.x_lo,
            self.x_up)
        ret.G_flat_lo = self.B.reshape((-1, ))
        ret.G_flat_up = self.B.reshape((-1, ))
        return ret

    def f(self, x):
        return self.A @ x

    def G(self, x):
        return self.B


class SecondOrderControlAffineSystem(ControlPiecewiseAffineSystem):
    """
    A second-order system
    q̇ = v
    v̇ = a(x) + b(x)u
    where the state is x = [q, v].
    """
    def __init__(self, x_lo, x_up, u_lo, u_up):
        super(SecondOrderControlAffineSystem,
              self).__init__(x_lo, x_up, u_lo, u_up)
        assert (self.x_dim % 2 == 0)
        self.nq = int(self.x_dim / 2)

    def a(self, x):
        raise NotImplementedError

    def b(self, x):
        raise NotImplementedError

    def f(self, x):
        v = x[self.nq:]
        return torch.cat((v, self.a(x)))

    def G(self, x):
        return torch.vstack((torch.zeros((self.nq, self.u_dim),
                                         dtype=self.dtype), self.b(x)))

    def _mixed_integer_constraints_v(self):
        """
        Return the mixed-integer constraints on a(x) and b(x).reshape((-1,)).
        """
        raise NotImplementedError

    def mixed_integer_constraints(self):
        mip_cnstr_a, mip_cnstr_b_flat, a_lo, a_up, b_lo, b_up = \
            self._mixed_integer_constraints_v()
        # We want mip_cnstr_f to be the same as mip_cnstr_a, except for the
        # output constraint.
        ret = ControlAffineSystemConstraintReturn()
        ret.mip_cnstr_f = gurobi_torch_mip.MixedIntegerConstraintsReturn()
        for field in mip_cnstr_a.__dict__.keys():
            if field not in ("Aout_input", "Aout_slack", "Aout_binary",
                             "Cout"):
                ret.mip_cnstr_f.__dict__[field] = mip_cnstr_a.__dict__[field]

        def get_tensor(tensor, row, col):
            # Return @p tensor if @p tensor is not None
            # else return torch.zeros((row, col))
            return tensor if tensor is not None else torch.zeros(
                (row, col), dtype=self.dtype)

        ret.mip_cnstr_f.Aout_input = torch.vstack(
            (torch.hstack((torch.zeros((self.nq, self.nq), dtype=self.dtype),
                           torch.eye(self.nq, dtype=self.dtype))),
             get_tensor(mip_cnstr_a.Aout_input, self.nq, self.x_dim)))
        if mip_cnstr_a.Aout_slack is not None:
            ret.mip_cnstr_f.Aout_slack = torch.vstack(
                (torch.zeros((self.nq, mip_cnstr_a.Aout_slack.shape[1]),
                             dtype=self.dtype), mip_cnstr_a.Aout_slack))
        if mip_cnstr_a.Aout_binary is not None:
            ret.mip_cnstr_f.Aout_binary = torch.vstack(
                (torch.zeros((self.nq, mip_cnstr_a.Aout_binary.shape[1]),
                             dtype=self.dtype), mip_cnstr_a.Aout_binary))
        if mip_cnstr_a.Cout is not None:
            ret.mip_cnstr_f.Cout = torch.cat(
                (torch.zeros(self.nq, dtype=self.dtype), mip_cnstr_a.Cout))

        ret.mip_cnstr_G = gurobi_torch_mip.MixedIntegerConstraintsReturn()
        for field in mip_cnstr_b_flat.__dict__.keys():
            if field not in ("Aout_input", "Aout_slack", "Aout_binary",
                             "Cout"):
                ret.mip_cnstr_G.__dict__[field] = mip_cnstr_b_flat.__dict__[
                    field]
        if mip_cnstr_b_flat.Aout_input is not None:
            ret.mip_cnstr_G.Aout_input = torch.vstack(
                (torch.zeros((self.nq * self.u_dim, self.nq),
                             dtype=self.dtype), mip_cnstr_b_flat.Aout_input))
        if mip_cnstr_b_flat.Aout_slack is not None:
            ret.mip_cnstr_G.Aout_slack = torch.vstack((torch.zeros(
                (self.nq * self.u_dim, mip_cnstr_b_flat.Aout_slack.shape[1]),
                dtype=self.dtype), mip_cnstr_b_flat.Aout_slack))
        if mip_cnstr_b_flat.Aout_binary is not None:
            ret.mip_cnstr_G.Aout_binary = torch.vstack((torch.zeros(
                (self.nq * self.u_dim, mip_cnstr_b_flat.Aout_binary.shape[1]),
                dtype=self.dtype), mip_cnstr_b_flat.Aout_binary))
        if mip_cnstr_b_flat.Cout is not None:
            ret.mip_cnstr_G.Cout = torch.cat(
                (torch.zeros(self.nq * self.u_dim,
                             dtype=self.dtype), mip_cnstr_b_flat.Cout))
        ret.f_lo = torch.cat((self.x_lo[self.nq:], a_lo))
        ret.f_up = torch.cat((self.x_up[self.nq:], a_up))
        ret.G_flat_lo = torch.cat((torch.zeros(self.nq * self.u_dim,
                                               dtype=self.dtype), b_lo))
        ret.G_flat_up = torch.cat((torch.zeros(self.nq * self.u_dim,
                                               dtype=self.dtype), b_up))
        return ret


class ReluSecondOrderControlAffineSystem(SecondOrderControlAffineSystem):
    """
    A second order system, that
    v̇=ϕ_a(x) + ϕ_b(x) * u
    where ϕ_a, ϕ_b are both neural networks with (leaky) ReLU activation unit.
    """
    def __init__(self, x_lo, x_up, u_lo, u_up, phi_a, phi_b,
                 method: mip_utils.PropagateBoundsMethod):
        """
        Args:
          phi_a: A neural network that maps x to ϕ_a(x).
          phi_b: A neural network that maps x to a flat vector
          [ϕ_b(x).row(0), ϕ_b(x).row(1), ..., ϕ_b(x).row(nq-1)]
          method: The method to propagate the bounds in the ReLU networks phi_a
          and phi_b.
        """
        super(ReluSecondOrderControlAffineSystem,
              self).__init__(x_lo, x_up, u_lo, u_up)
        assert (phi_a[0].in_features == self.x_dim)
        assert (phi_a[-1].out_features == self.nq)
        assert (phi_b[0].in_features == self.x_dim)
        assert (phi_b[-1].out_features == self.u_dim * self.nq)
        self.phi_a = phi_a
        self.phi_b = phi_b
        self.method = method
        self.relu_free_pattern_a = relu_to_optimization.ReLUFreePattern(
            self.phi_a, self.dtype)
        self.relu_free_pattern_b = relu_to_optimization.ReLUFreePattern(
            self.phi_b, self.dtype)

    def a(self, x):
        return self.phi_a(x)

    def b(self, x):
        return self.phi_b(x).reshape((self.nq, self.u_dim))

    def _mixed_integer_constraints_v(self):
        mip_cnstr_a = self.relu_free_pattern_a.output_constraint(
            self.x_lo, self.x_up, self.method)
        mip_cnstr_b_flat = self.relu_free_pattern_b.output_constraint(
            self.x_lo, self.x_up, self.method)
        a_lo = mip_cnstr_a.nn_output_lo
        a_up = mip_cnstr_a.nn_output_up
        b_lo = mip_cnstr_b_flat.nn_output_lo
        b_up = mip_cnstr_b_flat.nn_output_up
        return mip_cnstr_a, mip_cnstr_b_flat, a_lo, a_up, b_lo, b_up


class SecondOrderControlAffineWEquilibriumSystem(SecondOrderControlAffineSystem
                                                 ):
    """
    A second order system, that
    v̇=ϕ_a(x)-ϕ_a(x*)-ϕ_b(x*)*u* + ϕ_b(x) * u
    where ϕ_a, ϕ_b are both neural networks with (leaky) ReLU activation unit.
    x*, u* are the state/control at the equilibrium.
    """
    def __init__(self, x_lo, x_up, u_lo, u_up, phi_a, phi_b,
                 x_equilibrium: torch.Tensor, u_equilibrium: torch.Tensor,
                 method: mip_utils.PropagateBoundsMethod):
        """
        Args:
          phi_a: A neural network that maps x to ϕ_a(x).
          phi_b: A neural network that maps x to a flat vector
          [ϕ_b(x).row(0), ϕ_b(x).row(1), ..., ϕ_b(x).row(nq-1)]
          method: The method to propagate the bounds in the ReLU networks phi_a
          and phi_b.
        """
        super(SecondOrderControlAffineWEquilibriumSystem,
              self).__init__(x_lo, x_up, u_lo, u_up)
        assert (phi_a[0].in_features == self.x_dim)
        assert (phi_a[-1].out_features == self.nq)
        assert (phi_b[0].in_features == self.x_dim)
        assert (phi_b[-1].out_features == self.u_dim * self.nq)
        assert (isinstance(x_equilibrium, torch.Tensor))
        assert (x_equilibrium.shape == (self.x_dim, ))
        assert (isinstance(u_equilibrium, torch.Tensor))
        assert (u_equilibrium.shape == (self.u_dim, ))
        assert (torch.all(x_equilibrium <= x_up))
        assert (torch.all(x_equilibrium >= x_lo))
        assert (torch.all(u_equilibrium >= u_lo))
        assert (torch.all(u_equilibrium <= u_up))
        assert (torch.all(x_equilibrium[self.nq:] == 0))
        self.phi_a = phi_a
        self.phi_b = phi_b
        self.x_equilibrium = x_equilibrium
        self.u_equilibrium = u_equilibrium
        self.method = method
        self.relu_free_pattern_a = relu_to_optimization.ReLUFreePattern(
            self.phi_a, self.dtype)
        self.relu_free_pattern_b = relu_to_optimization.ReLUFreePattern(
            self.phi_b, self.dtype)

    def a(self, x):
        return self.phi_a(x) - self.phi_a(
            self.x_equilibrium) - self.phi_b(self.x_equilibrium).reshape(
                (self.nq, self.u_dim)) @ self.u_equilibrium

    def b(self, x):
        return self.phi_b(x).reshape((self.nq, self.u_dim))

    def _mixed_integer_constraints_v(self):
        mip_cnstr_a = self.relu_free_pattern_a.output_constraint(
            self.x_lo, self.x_up, self.method)
        mip_cnstr_b_flat = self.relu_free_pattern_b.output_constraint(
            self.x_lo, self.x_up, self.method)
        a_delta = -self.phi_a(self.x_equilibrium) - self.phi_b(
            self.x_equilibrium).reshape(
                (self.nq, self.u_dim)) @ self.u_equilibrium
        mip_cnstr_a.Cout += a_delta
        a_lo = mip_cnstr_a.nn_output_lo + a_delta
        a_up = mip_cnstr_a.nn_output_up + a_delta
        b_lo = mip_cnstr_b_flat.nn_output_lo
        b_up = mip_cnstr_b_flat.nn_output_up
        return mip_cnstr_a, mip_cnstr_b_flat, a_lo, a_up, b_lo, b_up


def train_control_affine_forward_model(forward_model_f,
                                       forward_model_G,
                                       x_equ,
                                       u_equ,
                                       model_dataset,
                                       num_epochs,
                                       lr,
                                       batch_size=50,
                                       verbose=True):
    """
    Helper function to train neural networks that approximate continuous time
    dynamics as
    ẋ = ϕ_f(x) + ϕ_G(x) u - ϕ_f(x*) - ϕ_G(x*) u*
    @param forward_model_f Feedforward network ϕ_f
    @param forward_model_G Feedforward network ϕ_G
    @param x_equ Tensor shape (x_dim,) with the system equilibrium state
    @param u_equ Tensor shape (u_dim,) with the system equilibrium control
    @param model_dataset TensorDataset with data, label = ([x|u], ẋ)
    @param num_epochs int number of training epochs
    @param lr float learning rate
    @param batch_size int
    """
    x_dim = x_equ.shape[0]
    u_dim = u_equ.shape[0]

    def compute_x_dot(forward_model_f, network_input, forward_model_G):
        x, u = torch.split(network_input, [x_dim, u_dim], dim=1)
        x_dot = forward_model_f(x) +\
            (forward_model_G(x).view((x.shape[0], x_dim, u_dim)) @
                u.unsqueeze(2)).squeeze(2) -\
            forward_model_f(x_equ) -\
            forward_model_G(x_equ).view((x_dim, u_dim)) @ u_equ
        return x_dot

    utils.train_approximator(
        model_dataset,
        forward_model_f,
        compute_x_dot,
        batch_size=batch_size,
        num_epochs=num_epochs,
        lr=lr,
        additional_variable=list(forward_model_G.parameters()),
        output_fun_args=dict(forward_model_G=forward_model_G),
        verbose=verbose)


def add_system_constraint(system: ControlPiecewiseAffineSystem,
                          mip: gurobi_torch_mip.GurobiTorchMIP,
                          x: list,
                          f: list,
                          Gt: list,
                          *,
                          binary_var_type=gurobipy.GRB.BINARY):
    """
    Add the (mixed-integer linear) constraints of f(x) and G(x) to mip.

    Args:
      x: The state variable.
      f: The variable for f(x).
      Gt: Gt is a 2D list. len(G) = u_dim. len(G[i]) = x_dim. Gt is the
      transpose of system.G(x).
    """
    assert (len(f) == system.x_dim)
    assert (len(Gt) == system.u_dim)
    # Add constraint that x_lo <= x <= x_up
    mip.addMConstrs([torch.eye(system.x_dim, dtype=system.dtype)], [x],
                    gurobipy.GRB.LESS_EQUAL,
                    system.x_up,
                    name="x_up")
    mip.addMConstrs([torch.eye(system.x_dim, dtype=system.dtype)], [x],
                    gurobipy.GRB.GREATER_EQUAL,
                    system.x_lo,
                    name="x_lo")
    # Set the bounds of x
    for i in range(system.x_dim):
        if x[i].lb < system.x_lo[i].item():
            x[i].lb = system.x_lo[i].item()
        if x[i].ub > system.x_up[i].item():
            x[i].ub = system.x_up[i].item()
    mip_cnstr_ret = system.mixed_integer_constraints()
    slack_f, binary_f = mip.add_mixed_integer_linear_constraints(
        mip_cnstr_ret.mip_cnstr_f, x, f, "slack_f", "binary_f", "f_ineq",
        "f_eq", "f_output", binary_var_type)
    G_flat = [None] * system.x_dim * system.u_dim
    for i in range(system.x_dim):
        for j in range(system.u_dim):
            G_flat[i * system.u_dim + j] = Gt[j][i]
    slack_G, binary_G = mip.add_mixed_integer_linear_constraints(
        mip_cnstr_ret.mip_cnstr_G, x, G_flat, "slack_G", "binary_G", "G_ineq",
        "G_eq", "G_out", binary_var_type)
    return mip_cnstr_ret, slack_f, slack_G, binary_f, binary_G
