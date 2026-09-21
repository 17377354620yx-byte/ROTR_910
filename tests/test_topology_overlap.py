import importlib.util
import unittest
from pathlib import Path

import torch
from easydict import EasyDict as edict

from geotransformer.modules.geotransformer import SuperPointMatching
from geotransformer.modules.liver import TopologyOverlapRefiner


ROOT = Path(__file__).resolve().parents[1]


def _load_p2p_loss_module():
    path = ROOT / "experiments" / "geotransformer.p2p_liver" / "loss.py"
    spec = importlib.util.spec_from_file_location("p2p_loss_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_refiner_is_neutral_and_finite_at_initialization():
    torch.manual_seed(7)
    module = TopologyOverlapRefiner(
        feature_dim=16, hidden_dim=12, num_neighbors=4, dropout=0.0
    ).eval()
    ref_points = torch.randn(11, 3)
    src_points = torch.randn(17, 3)
    ref_features = torch.randn(11, 16)
    src_features = torch.randn(17, 16)

    ref_output, src_output, diagnostics = module(
        ref_points, src_points, ref_features, src_features
    )

    torch.testing.assert_close(ref_output, ref_features)
    torch.testing.assert_close(src_output, src_features)
    assert set(diagnostics) == {"ref_overlap_logits", "src_overlap_logits"}
    assert all(torch.isfinite(value).all() for value in diagnostics.values())
    torch.testing.assert_close(diagnostics["ref_overlap_logits"], torch.zeros(11))
    torch.testing.assert_close(diagnostics["src_overlap_logits"], torch.zeros(17))


def test_refiner_without_poincare_uses_four_edge_features():
    torch.manual_seed(29)
    module = TopologyOverlapRefiner(
        feature_dim=8,
        hidden_dim=10,
        num_neighbors=3,
        dropout=0.0,
        use_poincare=False,
    ).eval()
    assert module.edge_gate[0].in_features == 4
    output = module(
        torch.randn(9, 3),
        torch.randn(13, 3),
        torch.randn(9, 8),
        torch.randn(13, 8),
    )
    assert all(torch.isfinite(value).all() for value in output[:2])
    assert all(torch.isfinite(value).all() for value in output[2].values())


def test_refiner_without_descriptor_update_has_no_projection_parameters():
    torch.manual_seed(31)
    module = TopologyOverlapRefiner(
        feature_dim=8,
        hidden_dim=10,
        num_neighbors=3,
        dropout=0.0,
        refine_descriptors=False,
    ).eval()
    assert module.output_proj is None
    ref_features = torch.randn(9, 8)
    src_features = torch.randn(13, 8)
    ref_output, src_output, _ = module(
        torch.randn(9, 3), torch.randn(13, 3), ref_features, src_features
    )
    torch.testing.assert_close(ref_output, ref_features)
    torch.testing.assert_close(src_output, src_features)


def test_rtor_outputs_are_rigid_invariant():
    torch.manual_seed(11)
    module = TopologyOverlapRefiner(
        feature_dim=8, hidden_dim=10, num_neighbors=3, dropout=0.0
    ).eval()
    ref_points = torch.randn(9, 3)
    src_points = torch.randn(13, 3)
    ref_features = torch.randn(9, 8)
    src_features = torch.randn(13, 8)
    rotation = torch.linalg.qr(torch.randn(3, 3)).Q
    if torch.det(rotation) < 0:
        rotation[:, 0] *= -1
    translation = torch.randn(3)

    first = module(ref_points, src_points, ref_features, src_features)
    second = module(
        ref_points @ rotation.T + translation,
        src_points @ rotation.T + translation,
        ref_features,
        src_features,
    )

    torch.testing.assert_close(first[0], second[0], atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(first[1], second[1], atol=2e-5, rtol=2e-5)
    for key in first[2]:
        torch.testing.assert_close(
            first[2][key], second[2][key], atol=2e-5, rtol=2e-5
        )


def test_overlap_weights_calibrate_coarse_ranking_without_hard_masking():
    matcher = SuperPointMatching(num_correspondences=1, dual_normalization=False)
    ref_features = torch.tensor([[1.0, 0.0]])
    src_features = torch.tensor([[1.0, 0.0], [0.99, 0.01]])
    src_features = torch.nn.functional.normalize(src_features, dim=1)
    ref_mask = torch.ones(1, dtype=torch.bool)
    src_mask = torch.ones(2, dtype=torch.bool)

    _, baseline_src, _ = matcher(ref_features, src_features, ref_mask, src_mask)
    _, weighted_src, _ = matcher(
        ref_features,
        src_features,
        ref_mask,
        src_mask,
        ref_weights=torch.ones(1),
        src_weights=torch.tensor([0.05, 1.0]),
    )

    assert baseline_src.item() == 0
    assert weighted_src.item() == 1


def test_uniform_overlap_weights_preserve_coarse_scores():
    torch.manual_seed(19)
    matcher = SuperPointMatching(num_correspondences=12, dual_normalization=True)
    ref_features = torch.nn.functional.normalize(torch.randn(5, 8), dim=1)
    src_features = torch.nn.functional.normalize(torch.randn(7, 8), dim=1)
    baseline = matcher(ref_features, src_features)
    weighted = matcher(
        ref_features,
        src_features,
        ref_weights=torch.full((5,), 0.525),
        src_weights=torch.full((7,), 0.525),
    )
    torch.testing.assert_close(weighted[0], baseline[0])
    torch.testing.assert_close(weighted[1], baseline[1])
    torch.testing.assert_close(weighted[2], baseline[2], atol=1e-8, rtol=1e-6)


def test_overlap_loss_is_finite_and_backpropagates():
    loss_module = _load_p2p_loss_module()
    cfg = edict(topology_overlap=edict(positive_overlap=0.1, focal_gamma=2.0))
    criterion = loss_module.TopologyOverlapLoss(cfg)
    ref_logits = torch.zeros(4, requires_grad=True)
    src_logits = torch.zeros(5, requires_grad=True)
    output = {
        "gt_node_corr_indices": torch.tensor([[0, 1], [2, 3]]),
        "gt_node_corr_overlaps": torch.tensor([0.8, 0.6]),
        "ref_overlap_logits": ref_logits,
        "src_overlap_logits": src_logits,
        "ref_node_masks": torch.ones(4, dtype=torch.bool),
        "src_node_masks": torch.ones(5, dtype=torch.bool),
    }
    loss = criterion(output)
    loss.backward()

    assert torch.isfinite(loss)
    assert ref_logits.grad is not None and torch.isfinite(ref_logits.grad).all()
    assert src_logits.grad is not None and torch.isfinite(src_logits.grad).all()


def load_tests(loader, tests, pattern):
    suite = unittest.TestSuite()
    for name, value in sorted(globals().items()):
        if name.startswith('test_') and callable(value):
            suite.addTest(unittest.FunctionTestCase(value))
    return suite
