import importlib.util
from pathlib import Path
import sys
import unittest
import numpy as np
import torch

EXPERIMENT = Path(__file__).resolve().parents[1] / 'experiments/geotransformer.p2p_liver'
sys.path.insert(0, str(EXPERIMENT))
from config import make_cfg
from dataset import LiverTask3TrainDataset, DeterministicValidationDataset
from model import create_model


class ProtocolTest(unittest.TestCase):
    def test_disabled_modules_have_no_parameters(self):
        for architecture, rtor, a3 in [('geotransformer', False, False), ('rtor_only', True, False),
                                      ('a3_only', False, True), ('rtor_a3', True, True)]:
            model = create_model(make_cfg(architecture))
            names = list(model.state_dict())
            self.assertEqual(any(n.startswith('topology_overlap_refiner.') for n in names), rtor)
            self.assertEqual(any(n.startswith('fine_local_refiner.') for n in names), a3)

    def test_disabled_overlap_weight_returns_uniform_matching_weights(self):
        cfg = make_cfg(
            interaction_profile='cooperative',
            ablation_profile='abl2_no_soft_weight',
        )
        model = create_model(cfg).eval()
        result = model._apply_rtor(
            torch.randn(7, 3),
            torch.randn(11, 3),
            torch.nn.functional.normalize(torch.randn(7, 256), dim=1),
            torch.nn.functional.normalize(torch.randn(11, 256), dim=1),
        )
        ref_probability, src_probability = result[2:4]
        ref_weight, src_weight = result[4:6]
        torch.testing.assert_close(ref_weight, torch.ones_like(ref_probability))
        torch.testing.assert_close(src_weight, torch.ones_like(src_probability))

    def test_final_ablation_removes_rtor_descriptor_projection(self):
        cfg = make_cfg(
            interaction_profile='cooperative',
            ablation_profile='abl5_no_rtor_descriptor',
        )
        model = create_model(cfg)
        self.assertIsNone(model.topology_overlap_refiner.output_proj)
        self.assertFalse(
            any(
                name.startswith('topology_overlap_refiner.output_proj.')
                for name in model.state_dict()
            )
        )

    def test_compact_profile_removes_proposal_and_geometry_branches(self):
        cfg = make_cfg(
            interaction_profile='cooperative',
            ablation_profile='compact_no_proposal_poincare_geometry',
        )
        self.assertFalse(cfg.ablation.predicted_proposal_exposure)
        self.assertTrue(cfg.ablation.overlap_soft_weight)
        self.assertFalse(cfg.ablation.rtor_poincare)
        self.assertFalse(cfg.ablation.a3_geometry_bias)
        self.assertTrue(cfg.ablation.rtor_descriptor_update)
        self.assertEqual(cfg.coarse_matching.predicted_ratio_max, 0.0)

        model = create_model(cfg)
        self.assertFalse(model.topology_overlap_refiner.use_poincare)
        self.assertTrue(model.topology_overlap_refiner.refine_descriptors)
        self.assertIsNotNone(model.topology_overlap_refiner.output_proj)
        self.assertFalse(model.fine_local_refiner.use_geometry_bias)

    def test_visibility_edges_are_counted_once(self):
        spec = importlib.util.spec_from_file_location('p2p_eval', EXPERIMENT / 'test.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        values = np.ones(9)
        rows = module._visibility_summary(values, np.array([i/10 for i in range(2, 11)]))
        self.assertEqual([r['count'] for r in rows], [1]*7 + [2])

    def test_validation_restores_random_state(self):
        import random
        class RandomDataset:
            def __getitem__(self, index):
                return random.random(), np.random.rand()
        data = DeterministicValidationDataset(RandomDataset(), 2, 123)
        random.seed(7)
        np.random.seed(7)
        before = random.getstate(), np.random.get_state()
        first = data[0]
        self.assertEqual(first, data[0])
        self.assertEqual(before[0], random.getstate())
        np.testing.assert_array_equal(before[1][1], np.random.get_state()[1])


if __name__ == '__main__':
    unittest.main()
