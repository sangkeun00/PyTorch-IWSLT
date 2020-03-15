import warnings

from torch.optim.lr_scheduler import _LRScheduler


# TODO: It might not be a good practice to use internal classes
class InverseSqrtScheduler(_LRScheduler):
    def __init__(self,
                 optimizer,
                 warmup_steps=8000,
                 min_lr=1e-9,
                 last_epoch=-1):
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        super().__init__(optimizer, last_epoch)

    def compute_lr(self, base_lr):
        if self._step_count < self.warmup_steps and self.warmup_steps > 0:
            cur_lr = base_lr * self._step_count / self.warmup_steps
        else:
            cur_lr = base_lr * (max(1, self.warmup_steps) /
                                self._step_count)**0.5

        return max(cur_lr, self.min_lr)

    def get_lr(self):
        return [self.compute_lr(base_lr) for base_lr in self.base_lrs]
