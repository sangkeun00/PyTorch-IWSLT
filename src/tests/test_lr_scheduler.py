import unittest

import torch
import torch.optim as optim
import torch.nn as nn

from ..optim.lr_scheduler import InverseSqrtScheduler


class SchedulerTest(unittest.TestCase):
    def test_inverse_lr(self):
        linear = nn.Linear(1, 1)
        opt = torch.optim.Adam(linear.parameters(), lr=0.01)
        scheduler = InverseSqrtScheduler(opt, warmup_steps=10, min_lr=1e-9)

        check_list = {
            1: 0.001,
            2: 0.002,
            3: 0.003,
            4: 0.004,
            5: 0.005,
            6: 0.006,
            7: 0.007,
            8: 0.008,
            9: 0.009,
            10: 0.01,
            11: 0.01 * (10 / 11)**0.5,
            12: 0.01 * (10 / 12)**0.5,
            13: 0.01 * (10 / 13)**0.5,
            14: 0.01 * (10 / 14)**0.5,
        }

        for it in range(1, 15):
            opt.zero_grad()
            loss = linear(torch.zeros(10, 1)).sum()
            loss.backward()
            for group in opt.param_groups:
                curr_lr = group['lr']
                if it in check_list:
                    self.assertAlmostEqual(curr_lr,
                                           check_list[it],
                                           msg='step {} check fail'.format(it))
            opt.step()
            scheduler.step()


if __name__ == '__main__':
    unittest.main()
