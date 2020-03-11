import unittest

import torch

from ..model import utils


class UtilsTest(unittest.TestCase):
    def test_mask_length(self):
        lengths = torch.tensor([1, 2, 3, 4])
        self.assertEqual(
            utils.create_mask(lengths).size()[-1], 4, 'check empty max_length')
        self.assertEqual(
            utils.create_mask(lengths, max_length=5).size()[-1], 5,
            'check given max_length')

    def test_mask_content(self):
        lengths = torch.tensor([1, 2, 3, 4])
        mask = utils.create_mask(lengths, max_length=5)
        truth = [
            [False, True, True, True, True],
            [False, False, True, True, True],
            [False, False, False, True, True],
            [False, False, False, False, True],
        ]
        for row, row_values in enumerate(truth):
            for col, value in enumerate(row_values):
                self.assertEqual(mask[row][0][col].cpu().item(), value)

    def test_mask_query(self):
        lengths = torch.tensor([1, 2, 3, 4])
        mask = utils.create_mask(lengths, max_length=5, causal=True)
        truth = [
            [False, True, True, True, True],
            [False, False, True, True, True],
            [False, False, False, True, True],
            [False, False, False, False, True],
        ]
        for row, row_values in enumerate(truth):
            for col, value in enumerate(row_values):
                for qidx in range(5):
                    causal_value = value or qidx < col
                    self.assertEqual(mask[row][qidx][col].cpu().item(),
                                     causal_value,
                                     'check fail at %d %d' % (qidx, col))


if __name__ == '__main__':
    unittest.main()
