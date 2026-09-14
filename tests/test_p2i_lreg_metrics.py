import importlib.util
from pathlib import Path
import unittest

import numpy as np
import torch


EXPERIMENT = Path(__file__).resolve().parents[1] / "experiments/geotransformer.p2i_lreg"


def _load_module(filename, name):
    spec = importlib.util.spec_from_file_location(name, EXPERIMENT / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


metrics = _load_module("metrics.py", "p2i_lreg_metrics")


class P2ILRegMetricTest(unittest.TestCase):
    def test_ir_uses_euclidean_ten_millimetre_threshold(self):
        source = torch.zeros(10, 3)
        reference = source.clone()
        reference[-1, 0] = 0.01001
        transform = torch.eye(4)
        ir = metrics.correspondence_inlier_ratio(reference, source, transform, 0.01)
        self.assertAlmostEqual(float(ir), 0.9, places=6)

    def test_fmr_boundary_is_strictly_greater_than_five_percent(self):
        self.assertEqual(metrics.feature_matching_recall([0.05], 0.05), 0.0)
        self.assertEqual(metrics.feature_matching_recall([0.050001], 0.05), 1.0)

    def test_registration_errors_report_degrees_and_millimetres(self):
        gt = torch.eye(4)
        estimate = torch.eye(4)
        estimate[:3, :3] = torch.tensor(
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
        )
        estimate[0, 3] = 0.01
        rre, rte = metrics.registration_errors(gt, estimate)
        self.assertAlmostEqual(float(rre), 90.0, places=5)
        self.assertAlmostEqual(float(rte), 10.0, places=5)

    def test_rr_uses_mean_source_displacement_and_strict_boundary(self):
        source = torch.tensor([[0.0, 0.0, 0.0], [0.02, 0.0, 0.0]])
        gt = torch.eye(4)
        estimate = torch.eye(4)
        estimate[0, 3] = 0.009
        self.assertEqual(float(metrics.registration_recall(source, gt, estimate, 0.01)), 1.0)
        estimate[0, 3] = 0.01
        self.assertEqual(float(metrics.registration_recall(source, gt, estimate, 0.01)), 0.0)

    def test_accumulator_excludes_failed_pose_from_rre_rte(self):
        accumulator = metrics.MetricAccumulator(fmr_threshold=0.05)
        accumulator.add(ir=0.1, rr=1.0, rre_deg=2.0, rte_mm=3.0, pose_valid=True)
        accumulator.add(ir=0.0, rr=0.0, rre_deg=np.nan, rte_mm=np.nan, pose_valid=False)
        summary = accumulator.summary()
        self.assertEqual(summary["count"], 2)
        self.assertEqual(summary["valid_pose_count"], 1)
        self.assertAlmostEqual(summary["IR"], 0.05)
        self.assertAlmostEqual(summary["FMR"], 0.5)
        self.assertAlmostEqual(summary["RR"], 0.5)
        self.assertAlmostEqual(summary["RRE_deg"], 2.0)
        self.assertAlmostEqual(summary["RTE_mm"], 3.0)


if __name__ == "__main__":
    unittest.main()
