import math
import torch


class Adam16(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.997),
        eps=1e-8,
        weight_decay=0,
        amsgrad=False,
        adamw=True,
        scheduler='inverse_sqrt',
        min_lr=1e-9,
        warmup_steps=8000,
    ):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        super(Adam16, self).__init__(params, defaults)

        self.base_lr = lr
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.scheduler = scheduler
        self.current_step = 0

    def __setstate__(self, state):
        super(Adam16, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    def get_lr(self):
        cur_lr = self.base_lr

        if self.scheduler == 'inverse_sqrt':
            if self.current_step < self.warmup_steps:
                cur_lr *= self.current_step / self.warmup_steps
            else:
                cur_lr *= (self.warmup_steps / self.current_step) ** 0.5
        elif self.scheduler == 'cosine':
            raise NotImplementedError

        return max(self.min_lr, cur_lr)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """

        self.current_step += 1
        new_lr = self.get_lr()

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            group['lr'] = new_lr
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                amsgrad = group['amsgrad']

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data, memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data, memory_format=torch.preserve_format)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data, memory_format=torch.preserve_format)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                if group['weight_decay'] != 0:
                    grad.add_(group['weight_decay'], p.data)

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = (max_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                else:
                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])

                step_size = group['lr'] / bias_correction1

                p.data.addcdiv_(-step_size, exp_avg, denom)

        return loss



class AdamDeprecated(object):
    def __init__(
        self,
        parameters,
        lr=5e-4,
        warmup_step=8000,
        betas=(0.9, 0.98),
        weight_decay=1e-4,
        min_lr=1e-9,
        eps=1e-8,
        start_step=0,
        scheduler='inverse_sqrt',
        adamw=True
    ):
        assert scheduler in ['inverse_sqrt', 'cosine']

        self.lr = lr
        self.min_lr = min_lr
        self.warmup_step = warmup_step
        self.scheduler = scheduler
        self.current_step = start_step

        init_lr = self.get_lr()
        if not adamw:
            self.optimizer = torch.optim.Adam(parameters, lr=init_lr, betas=betas,
                                        weight_decay=weight_decay, eps=eps)
        else:
            self.optimizer = torch.optim.AdamW(parameters, lr=init_lr, betas=betas,
                                         weight_decay=weight_decay, eps=eps)

    def step(self):
        self.current_step += 1
        cur_lr = self.get_lr()
        self.set_lr(cur_lr)
        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad()

    def set_lr(self, lr):
        for p in self.optimizer.param_groups:
            p['lr'] = lr

    def get_lr(self):
        cur_lr = self.lr

        if self.scheduler == 'inverse_sqrt':
            cur_lr *= min(1, self.current_step / self.warmup_step)
            cur_lr *= self.warmup_step ** 0.5
            cur_lr *= max(self.current_step, self.warmup_step) ** -0.5
        elif self.scheduler == 'cosine':
            raise NotImplementedError

        return max(self.min_lr, cur_lr)

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)
