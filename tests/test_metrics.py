"""Tests for explanation metrics in the presence of tied saliency values."""

import unittest

import torch

from explanations.metrics import iou_at_k, spearman_r, topk_mask


class SaliencyMetricsTests(unittest.TestCase):
    def test_topk_selects_exact_count_when_boundary_is_tied(self):
        saliency = torch.zeros(1, 1, 4, 4)
        mask = topk_mask(saliency, 0.25)
        self.assertEqual(mask.sum().item(), 4)
        self.assertEqual(mask.reshape(-1).nonzero().reshape(-1).tolist(), [0, 1, 2, 3])

    def test_iou_of_disjoint_topk_regions(self):
        first = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4)
        second = -first
        self.assertEqual(iou_at_k(first, second, 0.25), 0.0)

    def test_spearman_averages_tied_ranks(self):
        first = torch.tensor([0.0, 0.0, 1.0, 1.0]).reshape(1, 1, 2, 2)
        second = first.clone()
        self.assertAlmostEqual(spearman_r(first, second), 1.0)
        self.assertAlmostEqual(spearman_r(first, -second), -1.0)

    def test_constant_saliency_has_documented_zero_correlation(self):
        first = torch.zeros(1, 1, 2, 2)
        second = torch.arange(4, dtype=torch.float32).reshape(1, 1, 2, 2)
        self.assertEqual(spearman_r(first, second), 0.0)


if __name__ == "__main__":
    unittest.main()
