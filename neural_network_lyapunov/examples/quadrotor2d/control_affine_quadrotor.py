import neural_network_lyapunov.control_affine_system as control_affine_system
import neural_network_lyapunov.mip_utils as mip_utils
import neural_network_lyapunov.relu_to_optimization as relu_to_optimization
import neural_network_lyapunov.gurobi_torch_mip as gurobi_torch_mip

import torch


class ControlAffineQuadrotor2d(
        control_affine_system.SecondOrderControlAffineSystem):
    """
    The dynamics is
    vdot = -ϕ_b(q*[2]) * u* + ϕ_b(q[2])*u
    """
    def __init__(self, x_lo, x_up, u_lo, u_up, phi_b,
                 u_equilibrium: torch.Tensor,
                 method: mip_utils.PropagateBoundsMethod):
        super(ControlAffineQuadrotor2d, self).__init__(x_lo, x_up, u_lo, u_up)
        assert (phi_b[0].in_features == 1)
        assert (phi_b[-1].out_features == 6)
        self.theta_equilibrium = 0.
        self.u_equilibrium = u_equilibrium
        self.phi_b = phi_b
        self.method = method
        self.relu_free_pattern_b = relu_to_optimization.ReLUFreePattern(
            self.phi_b, self.dtype)

    @property
    def a_val(self):
        return -self.phi_b(
            torch.tensor([self.theta_equilibrium], dtype=self.dtype)).reshape(
                (self.nq, self.u_dim)) @ self.u_equilibrium

    def a(self, x):
        return self.a_val

    def b(self, x):
        return self.phi_b(x[2].unsqueeze(0)).reshape((self.nq, self.u_dim))

    def _mixed_integer_constraints_v(self):
        mip_cnstr_a = gurobi_torch_mip.MixedIntegerConstraintsReturn()
        mip_cnstr_a.Cout = self.a_val
        a_lo = self.a_val
        a_up = self.a_val
        mip_cnstr_b_flat = self.relu_free_pattern_b.output_constraint(
            self.x_lo[2].unsqueeze(0), self.x_up[2].unsqueeze(0), self.method)
        # The input to the network is just x[2], but for mip_cnstr_b_flat we
        # want the input to be x, hence we take the transform
        # x[2] = [0, 0, 1, 0, 0, 0] * x + 0
        mip_cnstr_b_flat.transform_input(
            torch.tensor([[0, 0, 1, 0, 0, 0]], dtype=self.dtype),
            torch.tensor([0], dtype=self.dtype))
        b_flat_lo = mip_cnstr_b_flat.nn_output_lo
        b_flat_up = mip_cnstr_b_flat.nn_output_up
        return mip_cnstr_a, mip_cnstr_b_flat, a_lo, a_up, b_flat_lo, b_flat_up
