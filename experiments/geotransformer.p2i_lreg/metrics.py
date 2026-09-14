"""Shared rigid-registration metrics for all P2I-LReg architectures."""

import math

import numpy as np
import torch

from geotransformer.modules.ops.transformation import apply_transform


def correspondence_inlier_ratio(ref_corr_points, src_corr_points, transform, distance_threshold_m):
    """Fraction of predicted pairs whose GT-aligned Euclidean error is small."""
    if ref_corr_points.shape != src_corr_points.shape:
        raise ValueError("reference/source correspondence shapes differ")
    if ref_corr_points.numel() == 0:
        return transform.new_zeros(())
    aligned_source = apply_transform(src_corr_points, transform)
    distances = torch.linalg.norm(ref_corr_points - aligned_source, dim=-1)
    return (distances < float(distance_threshold_m)).float().mean()


def feature_matching_recall(inlier_ratios, inlier_ratio_threshold):
    values = np.asarray(inlier_ratios, dtype=np.float64)
    if values.size == 0:
        return float("nan")
    return float(np.mean(values > float(inlier_ratio_threshold)))


def registration_errors(gt_transform, estimated_transform):
    """Return geodesic RRE in degrees and translation L2 error in millimetres."""
    relative_rotation = gt_transform[:3, :3].transpose(-1, -2) @ estimated_transform[:3, :3]
    cosine = ((torch.trace(relative_rotation) - 1.0) * 0.5).clamp(-1.0, 1.0)
    rre_deg = torch.acos(cosine) * (180.0 / math.pi)
    rte_mm = torch.linalg.norm(gt_transform[:3, 3] - estimated_transform[:3, 3]) * 1000.0
    return rre_deg, rte_mm


def registration_recall(src_points, gt_transform, estimated_transform, mean_displacement_threshold_m):
    gt_points = apply_transform(src_points, gt_transform)
    estimated_points = apply_transform(src_points, estimated_transform)
    mean_displacement = torch.linalg.norm(gt_points - estimated_points, dim=-1).mean()
    return (mean_displacement < float(mean_displacement_threshold_m)).float()


class MetricAccumulator:
    """Macro-average sample metrics while preserving pose-estimator failures."""

    def __init__(self, fmr_threshold):
        self.fmr_threshold = float(fmr_threshold)
        self.rows = []

    def add(self, *, ir, rr, rre_deg, rte_mm, pose_valid):
        self.rows.append(
            {
                "ir": float(ir),
                "rr": float(rr),
                "rre_deg": float(rre_deg),
                "rte_mm": float(rte_mm),
                "pose_valid": bool(pose_valid),
            }
        )

    def summary(self):
        if not self.rows:
            return {
                "count": 0,
                "valid_pose_count": 0,
                "IR": float("nan"),
                "FMR": float("nan"),
                "RR": float("nan"),
                "RRE_deg": float("nan"),
                "RTE_mm": float("nan"),
            }
        ir = np.asarray([row["ir"] for row in self.rows], dtype=np.float64)
        rr = np.asarray([row["rr"] for row in self.rows], dtype=np.float64)
        valid = np.asarray([row["pose_valid"] for row in self.rows], dtype=bool)
        rre = np.asarray([row["rre_deg"] for row in self.rows], dtype=np.float64)
        rte = np.asarray([row["rte_mm"] for row in self.rows], dtype=np.float64)
        return {
            "count": int(len(self.rows)),
            "valid_pose_count": int(valid.sum()),
            "IR": float(ir.mean()),
            "FMR": feature_matching_recall(ir, self.fmr_threshold),
            "RR": float(rr.mean()),
            "RRE_deg": float(np.nanmean(rre)) if valid.any() else float("nan"),
            "RTE_mm": float(np.nanmean(rte)) if valid.any() else float("nan"),
        }


__all__ = [
    "MetricAccumulator",
    "correspondence_inlier_ratio",
    "feature_matching_recall",
    "registration_errors",
    "registration_recall",
]
