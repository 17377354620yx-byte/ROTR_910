import importlib.util
from pathlib import Path
import unittest

import torch


EXPERIMENT = Path(__file__).resolve().parents[1] / "experiments/geotransformer.p2i_lreg"


def _load(filename, name):
    spec = importlib.util.spec_from_file_location(name, EXPERIMENT / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


config = _load("config.py", "p2i_loss_config")
losses = _load("loss.py", "p2i_losses")


class P2ILRegLossTest(unittest.TestCase):
    def _output(self):
        return {
            "ref_feats_c": torch.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True),
            "src_feats_c": torch.tensor([[0.9, 0.1], [0.1, 0.9]], requires_grad=True),
            "gt_node_corr_indices": torch.tensor([[0, 0], [1, 1]], dtype=torch.long),
            "gt_node_corr_overlaps": torch.tensor([1.0, 1.0]),
            "ref_node_corr_knn_points": torch.tensor([[[0.0, 0.0, 0.0], [0.001, 0.0, 0.0]]]),
            "src_node_corr_knn_points": torch.tensor([[[0.0, 0.0, 0.0], [0.001, 0.0, 0.0]]]),
            "ref_node_corr_knn_masks": torch.ones(1, 2, dtype=torch.bool),
            "src_node_corr_knn_masks": torch.ones(1, 2, dtype=torch.bool),
            "matching_scores": torch.zeros(1, 3, 3, requires_grad=True),
            "ref_overlap_logits": torch.zeros(2, requires_grad=True),
            "src_overlap_logits": torch.zeros(2, requires_grad=True),
            "ref_node_masks": torch.ones(2, dtype=torch.bool),
            "src_node_masks": torch.ones(2, dtype=torch.bool),
        }

    def test_overall_loss_is_finite_and_backward_for_both_architectures(self):
        for architecture in ("geotransformer", "rtor"):
            with self.subTest(architecture=architecture):
                output = self._output()
                criterion = losses.OverallLoss(config.make_cfg(architecture, create_dirs=False))
                result = criterion(output, {"transform": torch.eye(4)})
                self.assertTrue(torch.isfinite(result["loss"]))
                result["loss"].backward()
                self.assertIsNotNone(output["matching_scores"].grad)
                if architecture == "rtor":
                    self.assertIsNotNone(output["ref_overlap_logits"].grad)
                else:
                    self.assertEqual(float(result["o_loss"]), 0.0)


if __name__ == "__main__":
    unittest.main()
