"""Geometry-aware fine correspondence refinement used by RTOR+A3."""

import math
from typing import Tuple

import torch
import torch.nn as nn


def _masked_softmax(
    logits: torch.Tensor, key_mask: torch.Tensor, dim: int = -1
) -> torch.Tensor:
    """Apply a numerically safe softmax over valid support points."""
    mask = key_mask[:, None, None, :]
    logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
    probabilities = torch.softmax(logits, dim=dim)
    probabilities = probabilities * mask.to(probabilities.dtype)
    normalizer = probabilities.sum(dim=dim, keepdim=True).clamp_min(1e-12)
    return probabilities / normalizer


class MaskedGeometryAttention(nn.Module):
    """Multi-head cross-attention with an additive rigid-invariant bias."""

    def __init__(self, feature_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        if feature_dim % num_heads != 0:
            raise ValueError("feature_dim must be divisible by num_heads")

        self.feature_dim = int(feature_dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.feature_dim // self.num_heads
        self.query = nn.Linear(feature_dim, feature_dim, bias=False)
        self.key = nn.Linear(feature_dim, feature_dim, bias=False)
        self.value = nn.Linear(feature_dim, feature_dim, bias=False)
        self.output = nn.Linear(feature_dim, feature_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query_features: torch.Tensor,
        support_features: torch.Tensor,
        query_mask: torch.Tensor,
        support_mask: torch.Tensor,
        geometry_bias: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, query_count, _ = query_features.shape
        support_count = support_features.shape[1]
        query = self.query(query_features).reshape(
            batch_size, query_count, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key = self.key(support_features).reshape(
            batch_size, support_count, self.num_heads, self.head_dim
        ).transpose(1, 2)
        value = self.value(support_features).reshape(
            batch_size, support_count, self.num_heads, self.head_dim
        ).transpose(1, 2)

        logits = torch.matmul(query, key.transpose(-1, -2))
        logits = logits / math.sqrt(self.head_dim)
        logits = logits + geometry_bias[:, None]
        probabilities = self.dropout(_masked_softmax(logits, support_mask))
        message = torch.matmul(probabilities, value).transpose(1, 2).reshape(
            batch_size, query_count, self.feature_dim
        )
        message = self.output(message)
        return message * query_mask[..., None].to(message.dtype)


class LocalInteractionBlock(nn.Module):
    """Pre-normalized cross-attention block with a learned residual gate."""

    def __init__(
        self, feature_dim: int, num_heads: int, dropout: float, residual_init: float
    ) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(feature_dim)
        self.support_norm = nn.LayerNorm(feature_dim)
        self.attention = MaskedGeometryAttention(feature_dim, num_heads, dropout)
        self.message_norm = nn.LayerNorm(feature_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(feature_dim, 2 * feature_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * feature_dim, feature_dim),
        )
        self.dropout = nn.Dropout(dropout)
        self.residual_scale = nn.Parameter(torch.tensor(float(residual_init)))

    def forward(
        self,
        query_features: torch.Tensor,
        support_features: torch.Tensor,
        query_mask: torch.Tensor,
        support_mask: torch.Tensor,
        geometry_bias: torch.Tensor,
    ) -> torch.Tensor:
        message = self.attention(
            self.query_norm(query_features),
            self.support_norm(support_features),
            query_mask,
            support_mask,
            geometry_bias,
        )
        message = message + self.feed_forward(self.message_norm(message))
        output = query_features + self.residual_scale * self.dropout(message)
        return output * query_mask[..., None].to(output.dtype)


class GeometryAwareFineRefiner(nn.Module):
    """A3 bidirectional cross-patch refinement before Sinkhorn.

    The geometry signature contains only centered, scale-normalized intra-patch
    measurements. It is therefore invariant to the unknown global rigid pose.
    Both directions read the same pre-attention features, avoiding update-order
    dependence.

    The module hierarchy intentionally retains ``layers.0.*`` so checkpoints
    trained with the original A3 implementation remain strictly loadable.
    """

    def __init__(
        self,
        feature_dim: int,
        num_heads: int = 4,
        dropout: float = 0.0,
        geometry_sigma: float = 0.25,
        geometry_weight: float = 1.0,
        residual_init: float = 0.0,
    ) -> None:
        super().__init__()
        if geometry_sigma <= 0:
            raise ValueError("geometry_sigma must be positive")

        self.feature_dim = int(feature_dim)
        self.geometry_sigma = float(geometry_sigma)
        self.geometry_weight = float(geometry_weight)
        self.layers = nn.ModuleList(
            [
                LocalInteractionBlock(
                    feature_dim,
                    num_heads,
                    dropout,
                    residual_init=residual_init,
                )
            ]
        )

    @staticmethod
    def _validate_inputs(
        ref_features: torch.Tensor,
        src_features: torch.Tensor,
        ref_points: torch.Tensor,
        src_points: torch.Tensor,
        ref_masks: torch.Tensor,
        src_masks: torch.Tensor,
    ) -> None:
        if ref_features.ndim != 3 or src_features.ndim != 3:
            raise ValueError("fine patch features must have shape (P, K, C)")
        if ref_points.shape != (*ref_features.shape[:2], 3):
            raise ValueError("ref_points must have shape (P, K, 3)")
        if src_points.shape != (*src_features.shape[:2], 3):
            raise ValueError("src_points must have shape (P, K, 3)")
        if ref_masks.shape != ref_features.shape[:2]:
            raise ValueError("ref_masks must have shape (P, K)")
        if src_masks.shape != src_features.shape[:2]:
            raise ValueError("src_masks must have shape (P, K)")
        if ref_features.shape[0] != src_features.shape[0]:
            raise ValueError("ref and src must contain the same number of patches")

    @staticmethod
    def _geometry_signature(
        points: torch.Tensor, masks: torch.Tensor
    ) -> torch.Tensor:
        weights = masks.to(points.dtype)
        counts = weights.sum(dim=1, keepdim=True).clamp_min(1.0)
        centers = (points * weights[..., None]).sum(dim=1) / counts
        centered = points - centers[:, None]
        distances = torch.cdist(centered, centered)
        pair_masks = masks[:, :, None] & masks[:, None, :]
        off_diagonal = ~torch.eye(
            points.shape[1], dtype=torch.bool, device=points.device
        )[None]
        neighbour_masks = pair_masks & off_diagonal
        neighbour_counts = neighbour_masks.sum(dim=2).clamp_min(1)
        scale_counts = neighbour_masks.sum(dim=(1, 2)).clamp_min(1)
        scales = (
            (distances * neighbour_masks.to(distances.dtype)).sum(dim=(1, 2))
            / scale_counts
        ).clamp_min(1e-6)
        normalized = distances / scales[:, None, None]
        means = (
            normalized * neighbour_masks.to(normalized.dtype)
        ).sum(dim=2) / neighbour_counts
        variances = (
            (normalized - means[..., None]).square()
            * neighbour_masks.to(normalized.dtype)
        ).sum(dim=2) / neighbour_counts
        radial = torch.linalg.norm(centered, dim=-1) / scales[:, None]
        signatures = torch.stack([radial, means, variances.sqrt()], dim=-1)
        return signatures * weights[..., None]

    def _cross_geometry_bias(
        self,
        ref_points: torch.Tensor,
        src_points: torch.Tensor,
        ref_masks: torch.Tensor,
        src_masks: torch.Tensor,
    ) -> torch.Tensor:
        ref_signature = self._geometry_signature(ref_points, ref_masks)
        src_signature = self._geometry_signature(src_points, src_masks)
        difference = (
            ref_signature[:, :, None, :] - src_signature[:, None, :, :]
        ).abs().mean(dim=-1)
        return -self.geometry_weight * difference / self.geometry_sigma

    def forward(
        self,
        ref_features: torch.Tensor,
        src_features: torch.Tensor,
        ref_points: torch.Tensor,
        src_points: torch.Tensor,
        ref_masks: torch.Tensor,
        src_masks: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self._validate_inputs(
            ref_features, src_features, ref_points, src_points, ref_masks, src_masks
        )
        if ref_features.shape[-1] != self.feature_dim:
            raise ValueError(
                f"Expected {self.feature_dim}D features, got {ref_features.shape[-1]}"
            )

        geometry_bias = self._cross_geometry_bias(
            ref_points, src_points, ref_masks, src_masks
        )
        ref_features = ref_features * ref_masks[..., None].to(ref_features.dtype)
        src_features = src_features * src_masks[..., None].to(src_features.dtype)
        old_ref, old_src = ref_features, src_features
        layer = self.layers[0]
        ref_features = layer(
            old_ref, old_src, ref_masks, src_masks, geometry_bias
        )
        src_features = layer(
            old_src,
            old_ref,
            src_masks,
            ref_masks,
            geometry_bias.transpose(1, 2),
        )
        return ref_features, src_features


__all__ = ["GeometryAwareFineRefiner"]
