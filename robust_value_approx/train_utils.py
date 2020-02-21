import torch
import numpy as np

from enum import Enum


class ProjectGradientMode(Enum):
    LOSS1 = 1
    LOSS2 = 2
    BOTH = 3


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
    @return needs_projection. If n₁ᵀn₂ >= 0, we do not need projection,
    return False, and the gradient in the network is not updated. Otherwise
    return True, and update the gradient of the network using n1_perp and
    n2_perp.
    """
    assert(isinstance(mode, ProjectGradientMode))
    loss1.backward(retain_graph=True)
    n1 = torch.cat([
        p.grad.reshape((-1,)) for p in network.parameters()])
    for p in network.parameters():
        if p.grad is not None:
            p.grad.data.zero_()
    loss2.backward(retain_graph=retain_graph)
    n2 = torch.cat([
        p.grad.reshape((-1,)) for p in network.parameters()])
    with torch.no_grad():
        if n1 @ n2 > 0:
            grad = n1 + n2
            need_projection = False
        else:
            need_projection = True
            grad = torch.zeros(n1.shape, dtype=loss1.dtype)
            if mode == ProjectGradientMode.LOSS1 or\
                    mode == ProjectGradientMode.BOTH:
                n1_perp = n1 - (n1 @ n2 / (n2 @ n2)) * n2
                grad += n1_perp
            if mode == ProjectGradientMode.LOSS2 or\
                    mode == ProjectGradientMode.BOTH:
                n2_perp = n2 - (n1 @ n2 / (n1 @ n1)) * n1
                grad += n2_perp
        # Now set the network gradient.
        param_count = 0
        for p in network.parameters():
            p_size = np.prod((p.shape))
            p.grad = \
                grad[param_count:param_count+p_size].reshape(p.shape).clone()
            param_count += p_size
        return need_projection
