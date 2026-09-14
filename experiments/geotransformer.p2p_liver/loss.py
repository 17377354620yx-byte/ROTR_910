"""Training losses and validation metrics for RTOR+A3."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from geotransformer.modules.loss import WeightedCircleLoss
from geotransformer.modules.ops import pairwise_distance
from geotransformer.modules.ops.transformation import apply_transform
from geotransformer.modules.registration.metrics import isotropic_transform_error


class CoarseMatchingLoss(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.weighted_circle_loss = WeightedCircleLoss(
            cfg.coarse_loss.positive_margin,
            cfg.coarse_loss.negative_margin,
            cfg.coarse_loss.positive_optimal,
            cfg.coarse_loss.negative_optimal,
            cfg.coarse_loss.log_scale,
        )
        self.positive_overlap = float(cfg.coarse_loss.positive_overlap)

    def forward(self, output_dict):
        feature_distances = torch.sqrt(
            pairwise_distance(
                output_dict['ref_feats_c'],
                output_dict['src_feats_c'],
                normalized=True,
            ).clamp_min(1e-12)
        )
        overlaps = torch.zeros_like(feature_distances)
        corr_indices = output_dict['gt_node_corr_indices']
        overlaps[corr_indices[:, 0], corr_indices[:, 1]] = output_dict[
            'gt_node_corr_overlaps'
        ]
        positive_masks = overlaps > self.positive_overlap
        negative_masks = overlaps == 0
        positive_scales = torch.sqrt(overlaps * positive_masks.float())
        return self.weighted_circle_loss(
            positive_masks,
            negative_masks,
            feature_distances,
            positive_scales,
        )


class FineMatchingLoss(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.positive_radius = float(cfg.fine_loss.positive_radius)

    def forward(self, output_dict, data_dict):
        ref_points = output_dict['ref_node_corr_knn_points']
        src_points = apply_transform(
            output_dict['src_node_corr_knn_points'],
            data_dict['transform'],
        )
        ref_masks = output_dict['ref_node_corr_knn_masks']
        src_masks = output_dict['src_node_corr_knn_masks']
        matching_scores = output_dict['matching_scores']

        distances = pairwise_distance(ref_points, src_points)
        valid_pairs = ref_masks.unsqueeze(2) & src_masks.unsqueeze(1)
        gt_corr_map = (distances < self.positive_radius**2) & valid_pairs
        slack_row_labels = (gt_corr_map.sum(2) == 0) & ref_masks
        slack_col_labels = (gt_corr_map.sum(1) == 0) & src_masks

        labels = torch.zeros_like(matching_scores, dtype=torch.bool)
        labels[:, :-1, :-1] = gt_corr_map
        labels[:, :-1, -1] = slack_row_labels
        labels[:, -1, :-1] = slack_col_labels
        selected = matching_scores[labels]
        return -selected.mean() if selected.numel() else selected.sum()


class TopologyOverlapLoss(nn.Module):
    """Balanced focal supervision for bilateral superpoint overlap logits."""

    def __init__(self, cfg) -> None:
        super().__init__()
        self.positive_overlap = float(cfg.topology_overlap.positive_overlap)
        self.gamma = float(cfg.topology_overlap.focal_gamma)

    @staticmethod
    def _node_targets(count, indices, overlaps, device, dtype):
        targets = torch.zeros(count, device=device, dtype=dtype)
        if indices.numel():
            targets.scatter_reduce_(
                0,
                indices.to(device=device),
                overlaps.to(device=device, dtype=dtype),
                reduce='amax',
            )
        return targets

    def _focal_loss(self, logits, targets, masks):
        logits = logits[masks]
        targets = targets[masks]
        if logits.numel() == 0:
            return logits.sum()
        labels = (targets > self.positive_overlap).to(logits.dtype)
        probabilities = torch.sigmoid(logits)
        pt = torch.where(labels > 0, probabilities, 1.0 - probabilities)
        negative_count = labels.numel() - labels.sum()
        positive_weight = (negative_count / max(labels.numel(), 1)).clamp(
            0.05, 0.95
        )
        alpha = torch.where(labels > 0, positive_weight, 1.0 - positive_weight)
        bce = F.binary_cross_entropy_with_logits(logits, labels, reduction='none')
        return (alpha * (1.0 - pt).pow(self.gamma) * bce).mean()

    def forward(self, output_dict):
        corr_indices = output_dict['gt_node_corr_indices'].detach()
        corr_overlaps = output_dict['gt_node_corr_overlaps'].detach()
        ref_logits = output_dict['ref_overlap_logits']
        src_logits = output_dict['src_overlap_logits']
        ref_targets = self._node_targets(
            len(ref_logits),
            corr_indices[:, 0],
            corr_overlaps,
            ref_logits.device,
            ref_logits.dtype,
        )
        src_targets = self._node_targets(
            len(src_logits),
            corr_indices[:, 1],
            corr_overlaps,
            src_logits.device,
            src_logits.dtype,
        )
        ref_loss = self._focal_loss(
            ref_logits,
            ref_targets,
            output_dict['ref_node_masks'],
        )
        src_loss = self._focal_loss(
            src_logits,
            src_targets,
            output_dict['src_node_masks'],
        )
        return ref_loss + src_loss


class OverallLoss(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.coarse_loss = CoarseMatchingLoss(cfg)
        self.fine_loss = FineMatchingLoss(cfg)
        self.overlap_loss = TopologyOverlapLoss(cfg)
        self.weight_coarse = float(cfg.loss.weight_coarse_loss)
        self.weight_fine = float(cfg.loss.weight_fine_loss)
        self.weight_overlap = float(cfg.topology_overlap.weight_loss)
        self.rtor_enabled = bool(cfg.ablation.rtor_enabled)

    def forward(self, output_dict, data_dict):
        coarse_loss = self.coarse_loss(output_dict)
        fine_loss = self.fine_loss(output_dict, data_dict)
        overlap_loss = (self.overlap_loss(output_dict)
                        if self.rtor_enabled
                        else coarse_loss.new_zeros(()))
        total_loss = (
            self.weight_coarse * coarse_loss
            + self.weight_fine * fine_loss
            + self.weight_overlap * overlap_loss
        )
        return {
            'loss': total_loss,
            'c_loss': coarse_loss,
            'f_loss': fine_loss,
            'o_loss': overlap_loss,
        }


class Evaluator(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.acceptance_overlap = float(cfg.eval.acceptance_overlap)
        self.acceptance_radius = float(cfg.eval.acceptance_radius)
        self.acceptance_rmse = float(cfg.eval.rmse_threshold)

    @torch.no_grad()
    def evaluate_coarse(self, output_dict):
        ref_count = output_dict['ref_points_c'].shape[0]
        src_count = output_dict['src_points_c'].shape[0]
        gt_mask = output_dict['gt_node_corr_overlaps'] > self.acceptance_overlap
        gt_indices = output_dict['gt_node_corr_indices'][gt_mask]
        gt_map = torch.zeros(
            ref_count,
            src_count,
            device=output_dict['ref_points_c'].device,
        )
        gt_map[gt_indices[:, 0], gt_indices[:, 1]] = 1.0
        return gt_map[
            output_dict['ref_node_corr_indices'],
            output_dict['src_node_corr_indices'],
        ].mean()

    @torch.no_grad()
    def evaluate_fine(self, output_dict, data_dict):
        src_corr_points = apply_transform(
            output_dict['src_corr_points'],
            data_dict['transform'],
        )
        distances = torch.linalg.norm(
            output_dict['ref_corr_points'] - src_corr_points,
            dim=1,
        )
        return (distances < self.acceptance_radius).float().mean()

    @torch.no_grad()
    def evaluate_registration(self, output_dict, data_dict):
        transform = data_dict['transform']
        estimated_transform = output_dict['estimated_transform']
        rre, rte = isotropic_transform_error(transform, estimated_transform)
        realignment = torch.inverse(transform) @ estimated_transform
        realigned_source = apply_transform(output_dict['src_points'], realignment)
        rmse = torch.linalg.norm(
            realigned_source - output_dict['src_points'], dim=1
        ).mean()
        recall = (rmse < self.acceptance_rmse).float()
        return rre, rte, rmse, recall

    def forward(self, output_dict, data_dict):
        rre, rte, rmse, recall = self.evaluate_registration(
            output_dict, data_dict
        )
        return {
            'PIR': self.evaluate_coarse(output_dict),
            'IR': self.evaluate_fine(output_dict, data_dict),
            'RRE': rre,
            'RTE': rte,
            'RMSE': rmse,
            'RR': recall,
        }


__all__ = ['Evaluator', 'OverallLoss', 'TopologyOverlapLoss']
