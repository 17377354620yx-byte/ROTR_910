import sys
from pathlib import Path
import unittest
import torch
from geotransformer.modules.liver.cooperative_matching import mix_coarse_proposals, predicted_ratio_at_epoch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments/geotransformer.p2p_liver'))
from config import ABLATION_PROFILES, make_cfg
from loss import FineMatchingLoss


class CooperativeTest(unittest.TestCase):
    def test_progressive_ablation_profiles_are_cumulative(self):
        expected = {
            'abl1_no_proposal': (False, True, True, True, True),
            'abl2_no_soft_weight': (False, False, True, True, True),
            'abl3_no_poincare': (False, False, False, True, True),
            'abl4_no_a3_geometry': (False, False, False, False, True),
            'abl5_no_rtor_descriptor': (False, False, False, False, False),
        }
        self.assertEqual(
            tuple(ABLATION_PROFILES)[:len(expected)],
            tuple(expected),
        )
        for profile, switches in expected.items():
            cfg = make_cfg(
                architecture='rtor_a3',
                interaction_profile='cooperative',
                ablation_profile=profile,
            )
            actual = (
                cfg.ablation.predicted_proposal_exposure,
                cfg.ablation.overlap_soft_weight,
                cfg.ablation.rtor_poincare,
                cfg.ablation.a3_geometry_bias,
                cfg.ablation.rtor_descriptor_update,
            )
            self.assertEqual(actual, switches)
            self.assertEqual(cfg.ablation.profile, profile)
            self.assertFalse(cfg.overlap_selection.enabled)
            self.assertEqual(
                cfg.coarse_matching.predicted_ratio_max,
                0.25 if switches[0] else 0.0,
            )

    def test_no_ablation_preserves_cooperative_defaults(self):
        cfg = make_cfg(interaction_profile='cooperative')
        self.assertEqual(cfg.ablation.profile, 'none')
        self.assertEqual(
            (
                cfg.ablation.predicted_proposal_exposure,
                cfg.ablation.overlap_soft_weight,
                cfg.ablation.rtor_poincare,
                cfg.ablation.a3_geometry_bias,
                cfg.ablation.rtor_descriptor_update,
            ),
            (True, True, True, True, True),
        )
        self.assertEqual(cfg.coarse_matching.predicted_ratio_max, 0.25)

    def test_ablation_profiles_reject_non_cooperative_models(self):
        with self.assertRaisesRegex(ValueError, 'Ablation profiles require'):
            make_cfg(ablation_profile='abl1_no_proposal')
        with self.assertRaisesRegex(ValueError, 'Ablation profiles require'):
            make_cfg(
                architecture='a3_only',
                interaction_profile='cooperative',
                ablation_profile='abl1_no_proposal',
            )

    def test_schedule_keeps_warmup_and_bounds(self):
        self.assertEqual([predicted_ratio_at_epoch(x) for x in (1,5,20,150)],[0.,0.,.25,.25])

    def test_mixed_proposals_keep_budget_and_unique_pairs(self):
        gt=torch.arange(4);pr=torch.tensor([9,0,7,6]);scores=torch.ones(4)
        r,s,w=mix_coarse_proposals(gt,gt,scores,pr,pr,scores*.1,.5)
        self.assertEqual(len(r),4)
        self.assertEqual(len(set(zip(r.tolist(),s.tolist()))),4)
        self.assertIn(9,r.tolist())
        self.assertAlmostEqual(float(w[r==9]),.1,places=6)

    def test_zero_ratio_is_exact_legacy(self):
        gt=torch.arange(4);scores=torch.rand(4)
        result=mix_coarse_proposals(gt,gt,scores,gt.flip(0),gt,scores,.0)
        for a,b in zip(result,(gt,gt,scores)):torch.testing.assert_close(a,b)

    def test_profile_preserves_lgr_and_single_encoder(self):
        cfg=make_cfg(interaction_profile='cooperative')
        self.assertEqual(cfg.fine_matching.acceptance_radius,.1)
        self.assertFalse(cfg.model.dual_encoder)
        self.assertFalse(cfg.overlap_selection.enabled)

    def test_soft_overlap_preserves_gt_curriculum_and_objectives(self):
        baseline = make_cfg()
        cfg = make_cfg(interaction_profile='soft_overlap')
        self.assertFalse(cfg.overlap_selection.enabled)
        self.assertEqual(cfg.coarse_matching, baseline.coarse_matching)
        self.assertEqual(cfg.loss, baseline.loss)
        self.assertEqual(cfg.fine_matching, baseline.fine_matching)
        self.assertEqual(cfg.model.dual_encoder, baseline.model.dual_encoder)
        for epoch in (1, 20, 150):
            self.assertEqual(predicted_ratio_at_epoch(epoch, cfg.coarse_matching.exposure_start_epoch,
                             cfg.coarse_matching.exposure_end_epoch, cfg.coarse_matching.predicted_ratio_max), 0.)

    def test_fine_loss_finite_without_positive_pairs(self):
        cfg=make_cfg(interaction_profile='cooperative')
        scores=torch.randn(1,3,3,requires_grad=True)
        out=dict(ref_node_corr_knn_points=torch.zeros(1,2,3),
                 src_node_corr_knn_points=torch.ones(1,2,3)*100,
                 ref_node_corr_knn_masks=torch.ones(1,2,dtype=torch.bool),
                 src_node_corr_knn_masks=torch.ones(1,2,dtype=torch.bool),matching_scores=scores)
        value=FineMatchingLoss(cfg)(out,dict(transform=torch.eye(4)))
        self.assertTrue(torch.isfinite(value));value.backward();self.assertTrue(torch.isfinite(scores.grad).all())


if __name__=='__main__':unittest.main()
