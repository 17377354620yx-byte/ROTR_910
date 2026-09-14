import unittest
import torch
from diagnostics.interaction.metrics import gradient_metrics, score_metrics


class InteractionMetricsTest(unittest.TestCase):
    def test_missing_path_is_not_zero_cosine(self):
        r = gradient_metrics([None], [torch.ones(3)])
        self.assertEqual(r['status'], 'NO_GRADIENT_PATH')
        self.assertIsNone(r['cosine'])

    def test_gradient_coordinates_include_disconnected_parameters(self):
        r = gradient_metrics([torch.ones(1), torch.ones(3)], [torch.ones(1), None])
        self.assertAlmostEqual(r['cosine'], .5)

    def test_removed_gt_candidate_counts_as_missed_not_best_rank(self):
        r = score_metrics(torch.tensor([[.8,.2]]), torch.tensor([[True,False]]),
                          torch.tensor([[False,True]]))
        self.assertEqual(r['mrr'], 0.)
        self.assertEqual(r['gt_pair_survival'], 0.)

    def test_padding_and_gt_free_queries_do_not_improve_rank(self):
        r = score_metrics(torch.tensor([[3.,2.,100.],[1.,4.,100.]]),
                          torch.tensor([[False,True,False],[False,False,False]]),
                          torch.tensor([[True,True,False],[True,True,False]]), 'logit')
        self.assertEqual(r['gt_queries'], 1)
        self.assertAlmostEqual(r['mrr'], .5)


if __name__ == '__main__':
    unittest.main()
