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


if __name__ == "__main__":
    unittest.main()
