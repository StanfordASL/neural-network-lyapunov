import torch
import numpy as np
import wandb

from enum import Enum
import neural_network_lyapunov.utils as utils


class ProjectGradientMode(Enum):
    LOSS1 = 1
    LOSS2 = 2
    BOTH = 3
    EMPHASIZE_LOSS1 = 4
    EMPHASIZE_LOSS2 = 5


def project_gradient(network, loss1, loss2, mode, retain_graph=False):
    """
    We want both loss1 and loss2 to decrease in each iteration. We denote the
    gradient of the loss as n₁ = ∂loss₁/∂θ, n₂=∂loss₂/∂θ. If the optimizer
    descends on the direction -n₁ - n₂, then this direction doesn't guarantee
    the summed loss to decrease. We adopt the idea in
    Gradient Surgery for Multi-Task Learning by Yu et.al., that we select
    the component of n₁ that is perpendicular to n₂, (so that if we descend
    along this direction, it doesn't increase loss2), and similarly we
    select the component of n₂ that is perpendicular to n₁.
    If n₁ᵀn₂ < 0
    We denote n1_perp = n₁ - (n₁ᵀn₂/(n₂ᵀn₂))n₂
              n2_perp = n₂ - (n₂ᵀn₁/(n₁ᵀn₁))n₁
    We will descend on -n1_perp - n2_perp direction.
    If n₁ᵀn₂ >= 0, we descend on -n₁ - n₂ as usual.
    @param network The network.
    @param loss1 A torch tensor.
    @param loss2 A torch tensor.
    @param mode If mode = ProjectGradientMode.LOSS1, we only descend
    on n1_perp (i.e., decrease the positivity loss, but don't affect the
    derivative loss).
    If mode = ProjectGradientMode.LOSS2, we only descend on n2_perp (
    i.e., decrease the derivative loss, but don't affect the positivity
    loss).
    If mode = ProjectGradientMode.BOTH, we descend on n1_perp + n2_perp.
    If mode = ProjectGradientMode.EMPHASIZE_LOSS1, we descend on n1 + n2_perp
    If mode = ProjectGradientMode.EMPHASIZE_LOSS2, we descend on n1_perp + n2
    @return (needs_projection, n1, n2) If n₁ᵀn₂ >= 0, we do not need
    projection return False;otherwise return True.
    """
    assert (isinstance(mode, ProjectGradientMode))
    loss1.backward(retain_graph=True)
    n1 = torch.cat([p.grad.reshape((-1, )) for p in network.parameters()])
    for p in network.parameters():
        if p.grad is not None:
            p.grad.data.zero_()
    loss2.backward(retain_graph=retain_graph)
    n2 = torch.cat([p.grad.reshape((-1, )) for p in network.parameters()])
    with torch.no_grad():
        if n1 @ n2 > 0:
            grad = n1 + n2
            need_projection = False
        else:
            need_projection = True
            grad = torch.zeros(n1.shape, dtype=loss1.dtype)
            if mode == ProjectGradientMode.LOSS1 or\
                mode == ProjectGradientMode.BOTH or\
                    mode == ProjectGradientMode.EMPHASIZE_LOSS2:
                n1_perp = n1 - (n1 @ n2 / (n2 @ n2)) * n2
                grad += n1_perp
            if mode == ProjectGradientMode.EMPHASIZE_LOSS1:
                grad += n1
            if mode == ProjectGradientMode.LOSS2 or\
                    mode == ProjectGradientMode.BOTH or\
                    mode == ProjectGradientMode.EMPHASIZE_LOSS1:
                n2_perp = n2 - (n1 @ n2 / (n1 @ n1)) * n1
                grad += n2_perp
            if mode == ProjectGradientMode.EMPHASIZE_LOSS2:
                grad += n2
        # Now set the network gradient.
        param_count = 0
        for p in network.parameters():
            p_size = np.prod((p.shape))
            p.grad = \
                grad[param_count:param_count+p_size].reshape(p.shape).clone()
            param_count += p_size
        return (need_projection, n1, n2)


def wandb_config_update(args, lyapunov_relu, controller_relu, x_lo, x_up, u_lo,
                        u_up):
    wandb.config.update(args)
    lyapunov_linear_layer_width, _, _ = utils.extract_relu_structure(
        lyapunov_relu)
    if controller_relu is not None:
        controller_linear_layer_width, _, _ = utils.extract_relu_structure(
            controller_relu)
    wandb.config.update({
        "lyapunov_linear_layer_width":
        lyapunov_linear_layer_width
    })

    if controller_relu is not None:
        wandb.config.update({
            "controller_linear_layer_width":
            controller_linear_layer_width
        })
    wandb.config.update({
        "x_lo": x_lo,
        "x_up": x_up,
        "u_lo": u_lo,
        "u_up": u_up
    })
