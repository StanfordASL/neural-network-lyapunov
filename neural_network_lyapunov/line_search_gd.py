import torch
from torch.optim.optimizer import Optimizer, required


class LineSearchGD(Optimizer):
    r"""Implements gradient descent (optionally with momentum) with line search
    to find the step size. In each iteration, the step size should satisfy
    Armijo's rule
    f(x+αp) ≤ f(x)+c₁αpᵀ∇ₓf
    where α is the step size.

    Nesterov momentum is based on the formula from
    `On the importance of initialization and momentum in deep learning`__.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
        momentum (float, optional): momentum factor (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        dampening (float, optional): dampening for momentum (default: 0)
        nesterov (bool, optional): enables Nesterov momentum (default: False)
        loss_minimal_decrement (float, optional): c1 in Armijo's rule.
        minimal_step_size (float, optional): the minimal step size in each
        iteration.
        step_size_reduction. If step size α doesn't satisfy Armijo's rule, then
        try step_size_reduction * α.

    Example:
        >>> optimizer = torch.optim.LineSearchGD(model.parameters(), lr=0.1,
                momentum=0.9)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()

    __ http://www.cs.toronto.edu/%7Ehinton/absps/momentum.pdf

    .. note::
        The implementation of SGD with Momentum/Nesterov subtly differs from
        Sutskever et. al. and implementations in some other frameworks.

        Considering the specific case of Momentum, the update can be written as

        .. math::
                  v_{t+1} = \mu * v_{t} + g_{t+1} \\
                  p_{t+1} = p_{t} - lr * v_{t+1}

        where p, g, v and :math:`\mu` denote the parameters, gradient,
        velocity, and momentum respectively.

        This is in contrast to Sutskever et. al. and
        other frameworks which employ an update of the form

        .. math::
             v_{t+1} = \mu * v_{t} + lr * g_{t+1} \\
             p_{t+1} = p_{t} - v_{t+1}

        The Nesterov version is analogously modified.
    """
    def __init__(self,
                 params,
                 lr=required,
                 momentum=0,
                 dampening=0,
                 weight_decay=0,
                 nesterov=False,
                 min_step_size_decrease=1e-4,
                 loss_minimal_decrement=1e-4,
                 step_size_reduction=0.2,
                 min_improvement=0.):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(lr=lr,
                        momentum=momentum,
                        dampening=dampening,
                        weight_decay=weight_decay,
                        nesterov=nesterov,
                        min_step_size_decrease=min_step_size_decrease,
                        loss_minimal_decrement=loss_minimal_decrement,
                        step_size_reduction=step_size_reduction,
                        min_improvement=min_improvement)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError(
                "Nesterov momentum requires a momentum and zero dampening")
        super(LineSearchGD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(LineSearchGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def add_grad(self, p, t, d_p):
        p.data.add_(t, d_p)

    def directional_evaluate(self, closure, p, t, d_p):
        for i in range(len(p)):
            self.add_grad(p[i], t, d_p[i])
        loss = closure()
        return loss

    def line_search(self, loss0, closure, p, t, d_p):
        for group in self.param_groups:
            loss_minimal_decrement = group['loss_minimal_decrement']
            step_size_reduction = group['step_size_reduction']
            min_step_size_decrease = group['min_step_size_decrease']
            min_improvement = group['min_improvement']

        increment = loss_minimal_decrement * torch.sum(
            torch.stack([torch.sum(p[i].grad * d_p[i])
                         for i in range(len(p))]))
        alpha_prev = 0
        alpha = 1
        while alpha > min_step_size_decrease:
            loss = self.directional_evaluate(closure, p,
                                             (alpha - alpha_prev) * t, d_p)
            if loss <= loss0 + t * increment and \
                    loss <= loss0 - min_improvement:
                return loss
            alpha_prev = alpha
            alpha *= step_size_reduction

        return loss

    def step(self, closure, loss0):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        assert (isinstance(loss0, float))
        p_all = []
        dp_all = []
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            lr = group['lr']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = \
                            torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf
                p_all.append(p)
                dp_all.append(-d_p.clone())
        return self.line_search(loss0, closure, p_all, lr, dp_all)
