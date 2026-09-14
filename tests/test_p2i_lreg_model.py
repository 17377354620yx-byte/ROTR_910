import importlib.util
from pathlib import Path
import unittest

import torch


EXPERIMENT = Path(__file__).resolve().parents[1] / "experiments/geotransformer.p2i_lreg"


def _load_module(filename, name):
    spec = importlib.util.spec_from_file_location(name, EXPERIMENT / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


config = _load_module("config.py", "p2i_lreg_config_model_test")
model_entry = _load_module("model.py", "p2i_lreg_model")


class _Recorder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.seen = None

    def forward(self, data):
        self.seen = data
        return {"ok": True}


class _AutocastProbe(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.proj = torch.nn.Linear(channels, channels, bias=False)
        self.autocast_enabled = None
        self.output_dtype = None

    def forward(self, ref_points, src_points, ref_feats, src_feats):
        self.autocast_enabled = torch.is_autocast_enabled(ref_feats.device.type)
        ref_output = self.proj(ref_feats)
        src_output = self.proj(src_feats)
        self.output_dtype = ref_output.dtype
        return ref_output, src_output


class P2ILRegModelTest(unittest.TestCase):
    def test_architecture_switches_exactly_requested_modules(self):
        baseline = model_entry.create_model(config.make_cfg("geotransformer", create_dirs=False))
        rtor = model_entry.create_model(config.make_cfg("rtor", create_dirs=False))
        self.assertIsNone(baseline.topology_overlap_refiner)
        self.assertIsNone(baseline.fine_local_refiner)
        self.assertIsNotNone(rtor.topology_overlap_refiner)
        self.assertIsNone(rtor.fine_local_refiner)

    def test_test_forward_removes_gt_without_mutating_batch(self):
        recorder = _Recorder().eval()
        batch = {"features": torch.ones(2, 1), "transform": torch.eye(4)}
        result = model_entry.forward_without_ground_truth(recorder, batch)
        self.assertTrue(result["ok"])
        self.assertNotIn("transform", recorder.seen)
        self.assertIn("transform", batch)

    def test_effective_batch_two_uses_single_pair_micro_batches(self):
        cfg = config.make_cfg("geotransformer", create_dirs=False)
        self.assertEqual(cfg.train.batch_size, 1)
        self.assertEqual(cfg.optim.grad_acc_steps, 2)
        self.assertEqual(cfg.protocol.effective_batch_size, 2)

    def test_selective_bf16_is_confined_to_geometric_transformer(self):
        cfg = config.make_cfg("rtor", create_dirs=False)
        self.assertTrue(cfg.precision.selective_bf16)
        model = model_entry.create_model(cfg)
        probe = _AutocastProbe(cfg.geotransformer.input_dim)
        model.transformer = probe
        points = torch.randn(1, 6, 3)
        features = torch.randn(1, 6, cfg.geotransformer.input_dim)

        ref_features, src_features = model._encode_coarse_features(
            points, points, features, features
        )

        self.assertTrue(probe.autocast_enabled)
        self.assertEqual(probe.output_dtype, torch.bfloat16)
        self.assertEqual(ref_features.dtype, torch.float32)
        self.assertEqual(src_features.dtype, torch.float32)


if __name__ == "__main__":
    unittest.main()
