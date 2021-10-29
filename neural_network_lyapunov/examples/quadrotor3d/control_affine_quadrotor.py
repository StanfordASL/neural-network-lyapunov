import neural_network_lyapunov.control_affine_system as control_affine_system
import neural_network_lyapunov.mip_utils as mip_utils
import neural_network_lyapunov.relu_to_optimization as relu_to_optimization
import neural_network_lyapunov.gurobi_torch_mip as gurobi_torch_mip

import torch


class ControlAffineQuadrotor(control_affine_system.ControlPiecewiseAffineSystem
                             ):
    """
    The dynamics is
    rpy_dot = ϕ_a(rpy, omega) - ϕ_a(rpy, 0)
    pos_ddot = ϕ_b(rpy) * u - ϕ_b(0) * u*
    omega_dot = ϕ_c(omega) - ϕ_c(0) - C*u* + C*u
    """
    def __init__(self, x_lo, x_up, u_lo, u_up, phi_a, phi_b, phi_c,
                 C: torch.Tensor, u_equilibrium: torch.Tensor,
                 method: mip_utils.PropagateBoundsMethod):
        super(ControlAffineQuadrotor, self).__init__(x_lo, x_up, u_lo, u_up)
        assert (self.x_dim == 12)
        assert (self.u_dim == 4)
        assert (phi_a[0].in_features == 6)
        assert (phi_a[-1].out_features == 3)
        self.phi_a = phi_a
        assert (phi_b[0].in_features == 3)
        assert (phi_b[-1].out_features == 12)
        self.phi_b = phi_b
        assert (phi_c[0].in_features == 3)
        assert (phi_c[-1].out_features == 3)
        self.phi_c = phi_c
        assert (isinstance(C, torch.Tensor))
        assert (C.shape == (3, 4))
        self.C = C
        self.u_equilibrium = u_equilibrium
        self.method = method
        self.relu_free_pattern_a = relu_to_optimization.ReLUFreePattern(
            self.phi_a, self.dtype)
        self.relu_free_pattern_b = relu_to_optimization.ReLUFreePattern(
            self.phi_b, self.dtype)
        self.relu_free_pattern_c = relu_to_optimization.ReLUFreePattern(
            self.phi_c, self.dtype)

    def f(self, x):
        rpy = x[3:6]
        omega = x[9:12]

        pos_dot = x[6:9]
        rpy_dot = self.phi_a(torch.cat((rpy, omega))) - self.phi_a(
            torch.cat((rpy, torch.zeros((3, ), dtype=self.dtype))))
        f_val = torch.empty((12, ), dtype=self.dtype)
        f_val[:3] = pos_dot
        f_val[3:6] = rpy_dot
        f_val[6:9] = -self.phi_b(torch.zeros((3, ), dtype=self.dtype)).reshape(
            (3, 4)) @ self.u_equilibrium
        f_val[9:12] = self.phi_c(omega) - self.phi_c(
            torch.zeros((3, ), dtype=self.dtype)) - self.C @ self.u_equilibrium
        return f_val

    def G(self, x):
        rpy = x[3:6]
        G_val = torch.zeros((self.x_dim, self.u_dim), dtype=self.dtype)
        G_val[6:9, :] = self.phi_b(rpy).reshape((3, 4))
        G_val[9:12, :] = self.C
        return G_val

    def mixed_integer_constraints(self):
        ret = control_affine_system.ControlAffineSystemConstraintReturn()

        rpy_lo = self.x_lo[3:6]
        rpy_up = self.x_up[3:6]
        omega_lo = self.x_lo[9:12]
        omega_up = self.x_up[9:12]

        # pos_dot = pos_dot
        mip_cnstr_posdot_f = gurobi_torch_mip.MixedIntegerConstraintsReturn()
        mip_cnstr_posdot_f.Aout_input = torch.zeros((3, 12), dtype=self.dtype)
        mip_cnstr_posdot_f.Aout_input[:, 6:9] = torch.eye(3, dtype=self.dtype)
        mip_cnstr_posdot_G = gurobi_torch_mip.MixedIntegerConstraintsReturn()
        mip_cnstr_posdot_G.Cout = torch.zeros((12, ), dtype=self.dtype)
        mip_cnstr_posdot_G.Aout_input = torch.zeros((12, 12), dtype=self.dtype)

        # rpy_dot = ϕ_a(rpy, omega) - ϕ_a(rpy, 0)
        mip_cnstr_ret_a1 = self.relu_free_pattern_a.output_constraint(
            torch.cat((rpy_lo, omega_lo)), torch.cat((rpy_up, omega_up)),
            self.method)
        x_to_rpyomega_transform1 = torch.zeros((6, 12), dtype=self.dtype)
        x_to_rpyomega_transform1[:3, 3:6] = torch.eye(3, dtype=self.dtype)
        x_to_rpyomega_transform1[3:6, 9:12] = torch.eye(3, dtype=self.dtype)
        mip_cnstr_ret_a1.transform_input(x_to_rpyomega_transform1,
                                         torch.zeros(6, dtype=self.dtype))
        mip_cnstr_ret_a2 = self.relu_free_pattern_a.output_constraint(
            torch.cat((rpy_lo, torch.zeros(3, dtype=self.dtype))),
            torch.cat((rpy_up, torch.zeros(3, dtype=self.dtype))), self.method)
        x_to_rpyomega_transform2 = torch.zeros((6, 12), dtype=self.dtype)
        x_to_rpyomega_transform2[:3, 3:6] = torch.eye(3, dtype=self.dtype)
        mip_cnstr_ret_a2.transform_input(x_to_rpyomega_transform2,
                                         torch.zeros(6, dtype=self.dtype))

        mip_cnstr_rpydot_f = \
            gurobi_torch_mip.concatenate_mixed_integer_constraints(
                mip_cnstr_ret_a1,
                mip_cnstr_ret_a2,
                same_slack=False,
                same_binary=False,
                stack_output=False)
        assert (mip_cnstr_ret_a1.Aout_input is None)
        assert (mip_cnstr_ret_a2.Aout_input is None)
        assert (mip_cnstr_ret_a1.Aout_binary is None)
        assert (mip_cnstr_ret_a2.Aout_binary is None)
        mip_cnstr_rpydot_f.Aout_slack = torch.cat(
            (mip_cnstr_ret_a1.Aout_slack, -mip_cnstr_ret_a2.Aout_slack), dim=1)
        mip_cnstr_rpydot_f.Cout = mip_cnstr_ret_a1.Cout - mip_cnstr_ret_a2.Cout
        mip_cnstr_rpydot_G = gurobi_torch_mip.MixedIntegerConstraintsReturn()
        mip_cnstr_rpydot_G.Cout = torch.zeros((12, ), dtype=self.dtype)
        mip_cnstr_rpydot_G.Aout_input = torch.zeros((12, 12), dtype=self.dtype)

        # pos_ddot = ϕ_b(rpy) * u - ϕ_b(0) * u*
        mip_cnstr_posddot_f = gurobi_torch_mip.MixedIntegerConstraintsReturn()
        mip_cnstr_posddot_f.Cout = -self.phi_b(
            torch.zeros((3, ), dtype=self.dtype)).reshape(
                (3, 4)) @ self.u_equilibrium
        mip_cnstr_posddot_f.Aout_input = torch.zeros((3, 3), dtype=self.dtype)
        x_to_rpy_transform = torch.zeros((3, 12), dtype=self.dtype)
        x_to_rpy_transform[:, 3:6] = torch.eye(3, dtype=self.dtype)
        mip_cnstr_posddot_f.transform_input(
            x_to_rpy_transform, torch.zeros((3, ), dtype=self.dtype))
        mip_cnstr_posddot_G = self.relu_free_pattern_b.output_constraint(
            rpy_lo, rpy_up, self.method)
        mip_cnstr_posddot_G.transform_input(
            x_to_rpy_transform, torch.zeros((3, ), dtype=self.dtype))

        # omega_dot = ϕ_c(omega) - ϕ_c(0) - C*u* + C*u
        mip_cnstr_omegadot_f = self.relu_free_pattern_c.output_constraint(
            omega_lo, omega_up, self.method)
        x_to_omega_transform = torch.zeros((3, 12), dtype=self.dtype)
        x_to_omega_transform[:, 9:12] = torch.eye(3, dtype=self.dtype)
        mip_cnstr_omegadot_f.transform_input(
            x_to_omega_transform, torch.zeros((3, ), dtype=self.dtype))
        if mip_cnstr_omegadot_f.Cout is None:
            mip_cnstr_omegadot_f.Cout = -self.phi_c(
                torch.zeros(
                    (3, ), dtype=self.dtype)) - self.C @ self.u_equilibrium
        else:
            mip_cnstr_omegadot_f.Cout += -self.phi_c(
                torch.zeros(
                    (3, ), dtype=self.dtype)) - self.C @ self.u_equilibrium
        mip_cnstr_omegadot_G = gurobi_torch_mip.MixedIntegerConstraintsReturn()
        mip_cnstr_omegadot_G.Cout = self.C.reshape((-1, ))
        mip_cnstr_omegadot_G.Aout_input = torch.zeros((12, 12),
                                                      dtype=self.dtype)

        mip_cnstr_qdot_f = \
            gurobi_torch_mip.concatenate_mixed_integer_constraints(
                mip_cnstr_posdot_f,
                mip_cnstr_rpydot_f,
                same_slack=False,
                same_binary=False,
                stack_output=True)
        mip_cnstr_qdot_G = \
            gurobi_torch_mip.concatenate_mixed_integer_constraints(
                mip_cnstr_posdot_G,
                mip_cnstr_rpydot_G,
                same_slack=False,
                same_binary=False,
                stack_output=True)
        mip_cnstr_vdot_f = \
            gurobi_torch_mip.concatenate_mixed_integer_constraints(
                mip_cnstr_posddot_f,
                mip_cnstr_omegadot_f,
                same_slack=False,
                same_binary=False,
                stack_output=True)
        mip_cnstr_vdot_G = \
            gurobi_torch_mip.concatenate_mixed_integer_constraints(
                mip_cnstr_posddot_G,
                mip_cnstr_omegadot_G,
                same_slack=False,
                same_binary=False,
                stack_output=True)

        ret.mip_cnstr_f = \
            gurobi_torch_mip.concatenate_mixed_integer_constraints(
                mip_cnstr_qdot_f,
                mip_cnstr_vdot_f,
                same_slack=False,
                same_binary=False,
                stack_output=True)
        ret.mip_cnstr_G = \
            gurobi_torch_mip.concatenate_mixed_integer_constraints(
                mip_cnstr_qdot_G,
                mip_cnstr_vdot_G,
                same_slack=False,
                same_binary=False,
                stack_output=True)
        posdot_lo = self.x_lo[6:9]
        posdot_up = self.x_up[6:9]
        # TODO(hongkai.dai): when method = LP or MIP, compute rpydot_lo and
        # rpydot_up through another optimization problem.
        rpydot_lo = mip_cnstr_ret_a1.nn_output_lo - \
            mip_cnstr_ret_a2.nn_output_up
        rpydot_up = mip_cnstr_ret_a1.nn_output_up - \
            mip_cnstr_ret_a2.nn_output_lo
        posddot_f_lo = -self.phi_b(torch.zeros(
            (3, ), dtype=self.dtype)).reshape((3, 4)) @ self.u_equilibrium
        posddot_f_up = -self.phi_b(torch.zeros(
            (3, ), dtype=self.dtype)).reshape((3, 4)) @ self.u_equilibrium
        posddot_G_lo = mip_cnstr_posddot_G.nn_output_lo
        posddot_G_up = mip_cnstr_posddot_G.nn_output_up

        omegadot_f_lo = mip_cnstr_omegadot_f.nn_output_lo - self.phi_c(
            torch.zeros((3, ), dtype=self.dtype)) - self.C @ self.u_equilibrium
        omegadot_f_up = mip_cnstr_omegadot_f.nn_output_up - self.phi_c(
            torch.zeros((3, ), dtype=self.dtype)) - self.C @ self.u_equilibrium
        omegadot_G_lo = self.C.reshape((-1, ))
        omegadot_G_up = self.C.reshape((-1, ))
        ret.f_lo = torch.cat(
            (posdot_lo, rpydot_lo, posddot_f_lo, omegadot_f_lo))
        ret.f_up = torch.cat(
            (posdot_up, rpydot_up, posddot_f_up, omegadot_f_up))
        ret.G_flat_lo = torch.cat((torch.zeros(
            (24, ), dtype=self.dtype), posddot_G_lo, omegadot_G_lo))
        ret.G_flat_up = torch.cat((torch.zeros(
            (24, ), dtype=self.dtype), posddot_G_up, omegadot_G_up))

        return ret
