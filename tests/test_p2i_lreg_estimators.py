import importlib.util
from pathlib import Path
import sys
import unittest
import warnings

import numpy as np
import torch


EXPERIMENT = Path(__file__).resolve().parents[1] / "experiments/geotransformer.p2i_lreg"


def _load_module(filename, name):
    spec = importlib.util.spec_from_file_location(name, EXPERIMENT / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


estimators = _load_module("estimators.py", "p2i_lreg_estimators")


class P2ILRegEstimatorTest(unittest.TestCase):
    def test_topk_is_stable_and_shared_across_arrays(self):
        ref = torch.arange(12, dtype=torch.float32).reshape(4, 3)
        src = ref + 100
        scores = torch.tensor([0.5, 0.7, 0.7, 0.2])
        selected = estimators.select_topk_correspondences(ref, src, scores, 2)
        torch.testing.assert_close(selected[0], ref[[1, 2]])
        torch.testing.assert_close(selected[1], src[[1, 2]])
        torch.testing.assert_close(selected[2], scores[[1, 2]])

    def test_weighted_svd_recovers_source_to_reference_transform(self):
        source = torch.tensor(
            [[0.0, 0.0, 0.0], [0.02, 0.0, 0.0], [0.0, 0.03, 0.0], [0.0, 0.0, 0.04]]
        )
        rotation = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        translation = torch.tensor([0.1, -0.2, 0.3])
        reference = source @ rotation.T + translation
        transform = estimators.weighted_svd(source, reference, torch.ones(4))
        torch.testing.assert_close(transform[:3, :3], rotation, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(transform[:3, 3], translation, atol=1e-5, rtol=1e-5)

    def test_too_few_correspondences_is_explicit_failure(self):
        output = {
            "ref_corr_points": torch.zeros(2, 3),
            "src_corr_points": torch.zeros(2, 3),
            "corr_scores": torch.ones(2),
        }
        result = estimators.estimate_pose("weighted_svd", output, None, topk=250)
        self.assertFalse(result.valid)
        self.assertIsNone(result.transform)
        self.assertEqual(result.num_correspondences, 2)

    def test_ransac_recovers_known_pose_without_refinement(self):
        rng = np.random.default_rng(8)
        source = torch.from_numpy(rng.normal(size=(12, 3)).astype(np.float32) * 0.02)
        rotation = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        translation = torch.tensor([0.03, -0.01, 0.02])
        reference = source @ rotation.T + translation
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            transform = estimators.ransac_50k(
                source,
                reference,
                distance_threshold_m=0.001,
                iterations=50000,
                confidence=0.999,
                seed=7,
            )
        self.assertEqual(caught, [])
        torch.testing.assert_close(transform[:3, :3], rotation, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(transform[:3, 3], translation, atol=1e-5, rtol=1e-5)

    def test_lgr_required_inputs_do_not_include_ground_truth(self):
        self.assertNotIn("transform", estimators.LGR_REQUIRED_OUTPUTS)
        self.assertNotIn("gt_node_corr_indices", estimators.LGR_REQUIRED_OUTPUTS)


if __name__ == "__main__":
    unittest.main()
