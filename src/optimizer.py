import torch.optim as optim

class AdamOptimizer(object):
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
        adamw=False
    ):
        assert scheduler in ['inverse_sqrt', 'cosine']

        self.lr = lr
        self.min_lr = min_lr
        self.warmup_step = warmup_step
        self.scheduler = scheduler
        self.current_step = start_step

        init_lr = self.get_lr()
        if not adamw:
            self.optimizer = optim.Adam(parameters, lr=init_lr, betas=betas,
                                        weight_decay=weight_decay, eps=eps)
        else:
            self.optimizer = optim.AdamW(parameters, lr=init_lr, betas=betas,
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
