import unittest

import torch

from ..model import utils


class UtilsTest(unittest.TestCase):
    def test_mask_length(self):
        lengths = torch.tensor([1, 2, 3, 4])
        self.assertEqual(
            utils.create_mask(lengths).size()[1], 4, 'check empty max_length')
        self.assertEqual(
            utils.create_mask(lengths, max_length=5).size()[1], 5,
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
                self.assertEqual(mask[row][col].cpu().item(), value)


if __name__ == '__main__':
    unittest.main()
