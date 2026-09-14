"""Training losses for P2I-LReg rigid registration."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from geotransformer.modules.loss import WeightedCircleLoss
from geotransformer.modules.ops import pairwise_distance
from geotransformer.modules.ops.transformation import apply_transform


class CoarseMatchingLoss(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.circle = WeightedCircleLoss(
            cfg.coarse_loss.positive_margin,
            cfg.coarse_loss.negative_margin,
            cfg.coarse_loss.positive_optimal,
            cfg.coarse_loss.negative_optimal,
            cfg.coarse_loss.log_scale,
        )
        self.positive_overlap = float(cfg.coarse_loss.positive_overlap)

    def forward(self, output):
        distances = torch.sqrt(
            pairwise_distance(output["ref_feats_c"], output["src_feats_c"], normalized=True)
            .clamp_min(1e-12)
        )
        overlaps = torch.zeros_like(distances)
        indices = output["gt_node_corr_indices"]
        overlaps[indices[:, 0], indices[:, 1]] = output["gt_node_corr_overlaps"]
        positive = overlaps > self.positive_overlap
        negative = overlaps == 0
        scales = torch.sqrt(overlaps * positive.float())
        return self.circle(positive, negative, distances, scales)


class FineMatchingLoss(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.positive_radius = float(cfg.fine_loss.positive_radius)

    def forward(self, output, data):
        reference = output["ref_node_corr_knn_points"]
        source = apply_transform(output["src_node_corr_knn_points"], data["transform"])
        valid = output["ref_node_corr_knn_masks"].unsqueeze(2) & output[
            "src_node_corr_knn_masks"
        ].unsqueeze(1)
        matches = (pairwise_distance(reference, source) < self.positive_radius**2) & valid
        slack_rows = (matches.sum(2) == 0) & output["ref_node_corr_knn_masks"]
        slack_cols = (matches.sum(1) == 0) & output["src_node_corr_knn_masks"]
        labels = torch.zeros_like(output["matching_scores"], dtype=torch.bool)
        labels[:, :-1, :-1] = matches
        labels[:, :-1, -1] = slack_rows
        labels[:, -1, :-1] = slack_cols
        selected = output["matching_scores"][labels]
        return -selected.mean() if selected.numel() else selected.sum()


class TopologyOverlapLoss(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.positive_overlap = float(cfg.topology_overlap.positive_overlap)
        self.gamma = float(cfg.topology_overlap.focal_gamma)

    @staticmethod
    def _targets(count, indices, overlaps, template):
        targets = template.new_zeros(count)
        if indices.numel():
            targets.scatter_reduce_(0, indices, overlaps.to(template), reduce="amax")
        return targets

    def _one_side(self, logits, targets, masks):
        logits, targets = logits[masks], targets[masks]
        if not logits.numel():
            return logits.sum()
        labels = (targets > self.positive_overlap).to(logits.dtype)
        probabilities = torch.sigmoid(logits)
        pt = torch.where(labels > 0, probabilities, 1.0 - probabilities)
        negatives = labels.numel() - labels.sum()
        positive_weight = (negatives / max(labels.numel(), 1)).clamp(0.05, 0.95)
        alpha = torch.where(labels > 0, positive_weight, 1.0 - positive_weight)
        bce = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
        return (alpha * (1.0 - pt).pow(self.gamma) * bce).mean()

    def forward(self, output):
        indices = output["gt_node_corr_indices"].detach()
        overlaps = output["gt_node_corr_overlaps"].detach()
        ref_logits, src_logits = output["ref_overlap_logits"], output["src_overlap_logits"]
        ref_targets = self._targets(len(ref_logits), indices[:, 0], overlaps, ref_logits)
        src_targets = self._targets(len(src_logits), indices[:, 1], overlaps, src_logits)
        return self._one_side(ref_logits, ref_targets, output["ref_node_masks"]) + self._one_side(
            src_logits, src_targets, output["src_node_masks"]
        )


class OverallLoss(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.coarse = CoarseMatchingLoss(cfg)
        self.fine = FineMatchingLoss(cfg)
        self.overlap = TopologyOverlapLoss(cfg)
        self.weight_coarse = float(cfg.loss.weight_coarse_loss)
        self.weight_fine = float(cfg.loss.weight_fine_loss)
        self.weight_overlap = float(cfg.topology_overlap.weight_loss)
        self.rtor_enabled = bool(cfg.ablation.rtor_enabled)

    def forward(self, output, data):
        coarse = self.coarse(output)
        fine = self.fine(output, data)
        overlap = self.overlap(output) if self.rtor_enabled else coarse.new_zeros(())
        loss = self.weight_coarse * coarse + self.weight_fine * fine + self.weight_overlap * overlap
        return {"loss": loss, "c_loss": coarse, "f_loss": fine, "o_loss": overlap}


__all__ = ["CoarseMatchingLoss", "FineMatchingLoss", "OverallLoss", "TopologyOverlapLoss"]
