import math
import torch
from torch.optim.optimizer import Optimizer


class LineSearchAdam(Optimizer):
    r"""Implements Adam algorithm with line search to determine the step size.
    The step size should satisfy Armijo's rule
    f(x+αp) ≤ f(x)+c₁αpᵀ∇ₓf
    where α is the step size, p is the descent direction.

    It has been proposed in `Adam: A Method for Stochastic Optimization`_.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False)
        loss_minimal_decrement (float, optional): c1 in Armijo's rule.
        min_step_size_decrease (float, optional): the minimal step size in each
        iteration is lr * min_step_size_decrease.
        step_size_reduction. If step size α doesn't satisfy Armijo's rule, then
        try step_size_reduction * α.

    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """
    def __init__(self,
                 params,
                 lr=1e-3,
                 betas=(0.9, 0.999),
                 eps=1e-8,
                 weight_decay=0,
                 amsgrad=False,
                 loss_minimal_decrement=1e-4,
                 min_step_size_decrease=1e-4,
                 step_size_reduction=0.2,
                 min_improvement=0.):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(
                betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(
                betas[1]))
        defaults = dict(lr=lr,
                        betas=betas,
                        eps=eps,
                        weight_decay=weight_decay,
                        amsgrad=amsgrad,
                        loss_minimal_decrement=loss_minimal_decrement,
                        min_step_size_decrease=min_step_size_decrease,
                        step_size_reduction=step_size_reduction,
                        min_improvement=min_improvement)
        super(LineSearchAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(LineSearchAdam, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    def add_grad(self, p, t, d_p):
        p.data.add_(t, d_p)

    def directional_evaluate(self, closure, p, t, d_p):
        for i in range(len(p)):
            self.add_grad(p[i], t[i], d_p[i])
        loss = closure()
        return loss

    def line_search(self, loss0, closure, p, t, d_p):
        for group in self.param_groups:
            loss_minimal_decrement = group['loss_minimal_decrement']
            step_size_reduction = group['step_size_reduction']
            min_step_size_decrease = group['min_step_size_decrease']
            min_improvement = group['min_improvement']

        if loss_minimal_decrement is not None:
            decrement = loss_minimal_decrement * torch.sum(
                torch.stack([
                    t[i] * torch.sum(p[i].grad * d_p[i]) for i in range(len(p))
                ]))
        else:
            decrement = None

        alpha_prev = 0
        alpha = 1.
        while alpha > min_step_size_decrease:
            loss = self.directional_evaluate(closure, p,
                                             [(alpha - alpha_prev) * ti
                                              for ti in t], d_p)
            if decrement is not None and \
                loss < loss0 + alpha * decrement and \
                    loss < loss0 - min_improvement:
                return loss
            if decrement is None and loss < loss0 - min_improvement:
                return loss
            alpha_prev = alpha
            alpha *= step_size_reduction
        return loss

    def step_direction(self):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """

        p_all = []
        step_size_all = []
        dp_all = []
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError(
                        "Adam does not support sparse gradients, please " +
                        "consider SparseAdam instead")
                amsgrad = group['amsgrad']

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(
                        p.data, memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(
                        p.data, memory_format=torch.preserve_format)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad.
                        # values
                        state['max_exp_avg_sq'] = torch.zeros_like(
                            p.data, memory_format=torch.preserve_format)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1
                bias_correction1 = 1 - beta1**state['step']
                bias_correction2 = 1 - beta2**state['step']

                if group['weight_decay'] != 0:
                    grad.add_(group['weight_decay'], p.data)

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg.
                    # till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = (max_exp_avg_sq.sqrt() /
                             math.sqrt(bias_correction2)).add_(group['eps'])
                else:
                    denom = (exp_avg_sq.sqrt() /
                             math.sqrt(bias_correction2)).add_(group['eps'])

                step_size = group['lr'] / bias_correction1

                p_all.append(p)
                step_size_all.append(step_size)
                dp_all.append(-torch.div(exp_avg, denom).clone())
        return p_all, step_size_all, dp_all

    def step(self, closure, loss0):
        p_all, step_size_all, dp_all = self.step_direction()

        loss = self.line_search(loss0, closure, p_all, step_size_all, dp_all)
        return loss
