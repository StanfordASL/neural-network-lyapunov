import neural_network_lyapunov.control_affine_system as control_affine_system
import neural_network_lyapunov.relu_to_optimization as relu_to_optimization
import neural_network_lyapunov.mip_utils as mip_utils
import neural_network_lyapunov.gurobi_torch_mip as gurobi_torch_mip
import torch


class ControlAffineUnicycle(control_affine_system.ControlPiecewiseAffineSystem
                            ):
    """
    The state is [x, y, cosθ, sinθ]
    The control is [vel, θdot]
    The dynamics is
    xdot = cosθ * u[0]
    ydot = sinθ * u[0]
    dcosθ/dt = -sinθ * u[1]
    dsinθ/dt = cosθ * u[1]
    """
    pass


class ControlAffineUnicycleApprox(
        control_affine_system.ControlPiecewiseAffineSystem):
    """
    The state is [x, y, theta]
    The control is [vel, thetadot]
    The dynamics is [xdot, ydot] = ϕ(θ)*u[0]
    θ̇dot= u[1]
    """
    def __init__(self, phi, x_lo, x_up, u_lo, u_up,
                 method: mip_utils.PropagateBoundsMethod):
        super(ControlAffineUnicycleApprox,
              self).__init__(x_lo, x_up, u_lo, u_up)
        assert (phi[0].in_features == 1)
        assert (phi[-1].out_features == 2)
        self.phi = phi
        self.method = method
        self.relu_free_pattern = relu_to_optimization.ReLUFreePattern(
            self.phi, self.dtype)

    def f(self, x):
        return torch.zeros_like(x, dtype=self.dtype)

    def G(self, x):
        if len(x.shape) == 1:
            G_val = torch.zeros((3, 2), dtype=self.dtype)
            G_val[:2, 0] = self.phi(x[2:])
            G_val[2, 1] = 1.
        elif len(x.shape) == 2:
            G_val = torch.zeros((x.shape[0], 3, 2), dtype=self.dtype)
            G_val[:, :2, 0] = self.phi(x[:, 2:])
            G_val[:, 2, 1] = 1.
        return G_val

    def mixed_integer_constraints(
            self) -> control_affine_system.ControlAffineSystemConstraintReturn:
        ret = control_affine_system.ControlAffineSystemConstraintReturn()
        ret.mip_cnstr_f = gurobi_torch_mip.MixedIntegerConstraintsReturn()
        ret.mip_cnstr_f.Cout = torch.zeros((3, ), dtype=self.dtype)
        mip_cnstr_phi = self.relu_free_pattern.output_constraint(
            self.x_lo[2:], self.x_up[2:], self.method)
        mip_cnstr_phi.transform_input(
            torch.tensor([[0, 0, 1]], dtype=self.dtype),
            torch.tensor([0], dtype=self.dtype))
        assert (mip_cnstr_phi.Aout_input is None)
        assert (mip_cnstr_phi.Aout_binary is None)
        num_phi_slack = mip_cnstr_phi.num_slack()
        ret.mip_cnstr_G = mip_cnstr_phi
        ret.mip_cnstr_G.Aout_slack = torch.cat((torch.vstack(
            (mip_cnstr_phi.Aout_slack[0, :],
             torch.zeros((num_phi_slack, ),
                         dtype=self.dtype), mip_cnstr_phi.Aout_slack[1, :])),
                                                torch.zeros((3, num_phi_slack),
                                                            dtype=self.dtype)))
        ret.mip_cnstr_G.Cout = torch.cat((torch.stack(
            (mip_cnstr_phi.Cout[0], torch.tensor(0, dtype=self.dtype),
             mip_cnstr_phi.Cout[1])), torch.tensor([0, 0, 1],
                                                   dtype=self.dtype)))
        ret.f_lo = torch.zeros((3, ), dtype=self.dtype)
        ret.f_up = torch.zeros((3, ), dtype=self.dtype)
        ret.G_flat_lo = torch.cat((torch.stack(
            (mip_cnstr_phi.nn_output_lo[0], torch.tensor(0, dtype=self.dtype),
             mip_cnstr_phi.nn_output_lo[1])),
                                   torch.tensor([0, 0, 1], dtype=self.dtype)))
        ret.G_flat_up = torch.cat((torch.stack(
            (mip_cnstr_phi.nn_output_up[0], torch.tensor(0, dtype=self.dtype),
             mip_cnstr_phi.nn_output_up[1])),
                                   torch.tensor([0, 0, 1], dtype=self.dtype)))
        return ret

    def can_be_equilibrium_state(self, x: torch.Tensor) -> bool:
        return True
