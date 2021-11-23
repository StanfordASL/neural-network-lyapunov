"""
When we train the control Lyapunov and control Barrier function, we can also
train a nominal controller. This controller should be smooth, and we will
penalize the violation of the Lyapunov/barrier condition for this nominal
controller at sampled states. This penalization will encourage the control
Lyapunov/barrier function to induce a smooth controller. Note that we DON'T
require the nominal controller and the CLF/CBF to satisfy the Lyapunov/barrier
condition on all of the inifinitely many states.
We need to compute the gradient of the controller.
"""
import torch


class NominalController:
    def __init__(self):
        pass

    def output(self, x: torch.Tensor):
        """
        Compute the output action of the the controller given state.
        """
        assert (isinstance(x, torch.Tensor))
        return self._do_output(x)

    def _do_output(self, x: torch.Tensor):
        raise NotImplementedError

    def parameters(self):
        """
        Return all the trainable parameters of the controller.
        """
        raise NotImplementedError


class NominalNNController(NominalController):
    """
    A controller of the form u = clamp(ϕ(x) − ϕ(x*)+u*, u_lo, u_up)
    If x* is None, then u = clamp(ϕ(x)+u*, u_lo, u_up)
    If u* is None, then u = clamp(ϕ(x)−ϕ(x*), u_lo, u_up)
    If both x* and u* are None, then u = clamp(ϕ(x), u_lo, u_up)
    """
    def __init__(self, network, x_star: torch.Tensor, u_star: torch.Tensor,
                 u_lo: torch.Tensor, u_up: torch.Tensor):
        super(NominalNNController, self).__init__()
        self.network = network
        self.x_star = x_star
        self.u_star = u_star
        self.u_lo = u_lo
        self.u_up = u_up

    def _do_output(self, x):
        u_before_clamp = self.network(x)
        if self.x_star is not None:
            phi_x_star = self.network(self.x_star)
            u_before_clamp -= phi_x_star
        if self.u_star is not None:
            u_before_clamp += self.u_star
        if len(u_before_clamp.shape) == 1:
            return torch.clamp(u_before_clamp, min=self.u_lo, max=self.u_up)
        else:
            return torch.clamp(u_before_clamp,
                               min=self.u_lo.repeat(u_before_clamp.shape[0],
                                                    1),
                               max=self.u_up.repeat(u_before_clamp.shape[0],
                                                    1))

    def parameters(self):
        return self.network.parameters()
