import unittest

import torch

from geotransformer.modules.liver import GeometryAwareFineRefiner
from geotransformer.modules.liver.overlap_selection import select_overlap_region


def _random_rotation():
    rotation = torch.linalg.qr(torch.randn(3, 3)).Q
    if torch.det(rotation) < 0:
        rotation[:, 0] *= -1
    return rotation


def test_a3_is_rigid_invariant():
    torch.manual_seed(23)
    module = GeometryAwareFineRefiner(
        feature_dim=16,
        num_heads=4,
        dropout=0.0,
        residual_init=0.1,
    ).eval()
    ref_points = torch.randn(3, 9, 3)
    src_points = torch.randn(3, 11, 3)
    ref_features = torch.randn(3, 9, 16)
    src_features = torch.randn(3, 11, 16)
    ref_masks = torch.ones(3, 9, dtype=torch.bool)
    src_masks = torch.ones(3, 11, dtype=torch.bool)
    rotation = _random_rotation()
    translation = torch.randn(3)

    first = module(
        ref_features,
        src_features,
        ref_points,
        src_points,
        ref_masks,
        src_masks,
    )
    second = module(
        ref_features,
        src_features,
        ref_points @ rotation.T + translation,
        src_points @ rotation.T + translation,
        ref_masks,
        src_masks,
    )
    torch.testing.assert_close(first[0], second[0], atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(first[1], second[1], atol=2e-5, rtol=2e-5)


def test_a3_masks_invalid_points():
    module = GeometryAwareFineRefiner(feature_dim=8, num_heads=2).eval()
    ref_masks = torch.tensor([[True, True, False]])
    src_masks = torch.tensor([[True, False, False, False]])
    ref_output, src_output = module(
        torch.randn(1, 3, 8),
        torch.randn(1, 4, 8),
        torch.randn(1, 3, 3),
        torch.randn(1, 4, 3),
        ref_masks,
        src_masks,
    )
    assert torch.count_nonzero(ref_output[:, 2:]) == 0
    assert torch.count_nonzero(src_output[:, 1:]) == 0
    assert torch.isfinite(ref_output).all()
    assert torch.isfinite(src_output).all()


def test_source_overlap_selection_falls_back_on_flat_probabilities():
    valid_mask = torch.ones(20, dtype=torch.bool)
    selected = select_overlap_region(
        torch.full((20,), 0.5),
        valid_mask,
        threshold=0.5,
        min_superpoints=4,
        max_ratio=0.5,
        min_spread=0.05,
    )
    torch.testing.assert_close(selected, valid_mask)


def load_tests(loader, tests, pattern):
    suite = unittest.TestSuite()
    for name, value in sorted(globals().items()):
        if name.startswith('test_') and callable(value):
            suite.addTest(unittest.FunctionTestCase(value))
    return suite
