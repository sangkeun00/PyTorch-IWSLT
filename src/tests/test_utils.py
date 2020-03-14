import unittest

import torch

from ..model import utils


class UtilsTest(unittest.TestCase):
    def test_mask_query(self):
        mask = utils.create_causual_mask(3)
        truth = [
            [0., float('-inf'), float('-inf')],
            [0., 0., float('-inf')],
            [0., 0., 0.],
        ]
        for row, row_values in enumerate(truth):
            for col, value in enumerate(row_values):
                self.assertEqual(mask[row][col].cpu().item(), truth[row][col],
                                 'check fail at %d %d' % (row, col))


if __name__ == '__main__':
    unittest.main()
