import torch
import neural_network_lyapunov.gurobi_torch_mip as gurobi_torch_mip


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
          mip_cnstr_G: A list (of size u_dim) of MixedIntegerConstraintsReturn
          object such mip_cnstr_G[i] has G.col(i) as the output.
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
        mip_cnstr_G = [None] * self.u_dim
        for i in range(self.u_dim):
            mip_cnstr_G[i] = gurobi_torch_mip.MixedIntegerConstraintsReturn()
            mip_cnstr_G[i].Cout = self.B[:, i]
        return (mip_cnstr_f, mip_cnstr_G)

    def f(self, x):
        return self.A @ x

    def G(self, x):
        return self.B
