import robust_value_approx.relu_to_optimization as relu_to_optimization


class AutonomousReLUSystem:
    """
    This system models a discrete time autonomous hybrid linear system
    (piecewise affine system) using a feedforward neural network with ReLU
    activations
    x[n+1] = relu(x[n])
    """

    def __init__(self, x_dim, dtype, x_lo, x_up, relu_model):
        """
        @param x_dim The dimension of x.
        @param dtype The torch datatype
        """
        self.dtype = dtype
        self.x_dim = x_dim
        self.x_lo = x_lo
        self.x_up = x_up
        assert(relu_model[0].in_features == relu_model[-1].out_features)
        self.relu_free_pattern = relu_to_optimization.ReLUFreePattern(
            relu_model, dtype)

    @property
    def x_lo_all(self):
        return self.x_lo.detach().numpy()

    @property
    def x_up_all(self):
        return self.x_up.detach().numpy()

    def mixed_integer_constraints(self, relu_model):
        """
        @return Aout_s, Cout,
                Ain_x, Ain_s, Ain_gamma, rhs_in,
                Aeq_x, Aeq_s, Aeq_gamma, rhs_eq
                such that
                x[n+1] = Aout_s @ s + Cout
                s.t.
                Ain_x @ x + Ain_s @ s + Ain_gamma @ gamma <= rhs_in
                Aeq_x @ x + Aeq_s @ s + Aeq_gamma @ gamma == rhs_eq
        """
        (Ain_x, Ain_s, Ain_gamma, rhs_in, Aeq_x, Aeq_s, Aeq_gamma, rhs_eq,
         Aout_s, Cout, z_pre_relu_lo, z_pre_relu_up, z_post_relu_lo,
         z_post_relu_up) = self.relu_free_pattern.output_constraint(
            relu_model, self.x_lo, self.x_up)

        return (Aout_s, Cout,
                Ain_x, Ain_s, Ain_gamma, rhs_in,
                Aeq_x, Aeq_s, Aeq_gamma, rhs_eq)
