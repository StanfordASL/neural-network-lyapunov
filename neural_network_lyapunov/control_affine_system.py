import torch
import neural_network_lyapunov.gurobi_torch_mip as gurobi_torch_mip
import neural_network_lyapunov.utils as utils
import neural_network_lyapunov.mip_utils as mip_utils


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

    def mixed_integer_constraints(self):
        """
        Returns the mixed-integer linear constraints on f(x) and G(x).
        Return:
          (mip_cnstr_f, mip_cnstr_G):
          mip_cnstr_f: A MixedIntegerConstraintsReturn object with f as the
          output.
          mip_cnstr_G: A MixedIntegerConstraintsReturn object with the flat
          vector G.reshape((-1,)) as the output.
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

    def compute_f_range_ia(self) -> (torch.Tensor, torch.Tensor):
        """
        Compute the range of f through interval arithemetics (IA).

        Return:
          f_lo, f_up: The lower and upper bound of f.
        """
        raise NotImplementedError

    def compute_G_range_ia(self) -> (torch.Tensor, torch.Tensor):
        """
        Compute the range of G through interval arithemetics (IA).

        Return:
          G_lo, G_up: G_lo/G_up is the lower/upper bound of G.reshape((-1,))
        """
        raise NotImplementedError


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
        mip_cnstr_f = gurobi_torch_mip.MixedIntegerConstraintsReturn()
        mip_cnstr_f.Aout_input = self.A
        mip_cnstr_G = gurobi_torch_mip.MixedIntegerConstraintsReturn()
        mip_cnstr_G.Cout = self.B.reshape((-1, ))
        return (mip_cnstr_f, mip_cnstr_G)

    def f(self, x):
        return self.A @ x

    def G(self, x):
        return self.B

    def compute_f_range_ia(self):
        return mip_utils.compute_range_by_IA(
            self.A, torch.zeros((self.x_dim, ), dtype=self.dtype), self.x_lo,
            self.x_up)

    def compute_G_range_ia(self):
        G_lo = self.B.reshape((-1, ))
        G_up = self.B.reshape((-1, ))
        return G_lo, G_up


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
