"""Shared top-k pose estimators; no estimator consumes ground truth."""

from dataclasses import dataclass

import numpy as np
import open3d as o3d
import torch

from geotransformer.modules.geotransformer import LocalGlobalRegistration


LGR_REQUIRED_OUTPUTS = (
    "ref_node_corr_knn_points",
    "src_node_corr_knn_points",
    "ref_node_corr_knn_masks",
    "src_node_corr_knn_masks",
    "matching_scores",
    "node_corr_scores",
)


@dataclass(frozen=True)
class PoseEstimate:
    transform: torch.Tensor | None
    valid: bool
    num_correspondences: int
    method: str
    reason: str = ""


def select_topk_correspondences(ref_points, src_points, scores, topk):
    if ref_points.shape != src_points.shape or ref_points.ndim != 2 or ref_points.shape[1] != 3:
        raise ValueError("correspondence point arrays must both have shape (N, 3)")
    if scores.ndim != 1 or len(scores) != len(ref_points):
        raise ValueError("scores must have shape (N,)")
    if topk <= 0:
        raise ValueError("topk must be positive")
    count = min(int(topk), len(scores))
    score_values = scores.detach().cpu().numpy()
    order = np.lexsort((np.arange(len(score_values)), -score_values))[:count]
    indices = torch.as_tensor(order, dtype=torch.long, device=scores.device)
    return ref_points[indices], src_points[indices], scores[indices]


def weighted_svd(src_points, ref_points, weights=None):
    """Estimate an SE(3) matrix mapping source points to reference points."""
    if src_points.shape != ref_points.shape or src_points.ndim != 2 or src_points.shape[1] != 3:
        raise ValueError("point arrays must both have shape (N, 3)")
    if len(src_points) < 3:
        raise ValueError("at least three correspondences are required")
    if weights is None:
        weights = torch.ones(len(src_points), dtype=src_points.dtype, device=src_points.device)
    weights = weights.to(device=src_points.device, dtype=src_points.dtype).reshape(-1)
    if len(weights) != len(src_points):
        raise ValueError("weights must have shape (N,)")
    weights = torch.clamp(weights, min=0)
    weight_sum = weights.sum()
    if not torch.isfinite(weight_sum) or float(weight_sum) <= 0:
        raise ValueError("correspondence weights have zero or invalid mass")
    weights = weights / weight_sum
    src_center = (src_points * weights[:, None]).sum(dim=0)
    ref_center = (ref_points * weights[:, None]).sum(dim=0)
    src_centered = src_points - src_center
    ref_centered = ref_points - ref_center
    covariance = src_centered.transpose(0, 1) @ (weights[:, None] * ref_centered)
    u, _, vh = torch.linalg.svd(covariance)
    correction = torch.eye(3, dtype=src_points.dtype, device=src_points.device)
    correction[-1, -1] = torch.sign(torch.det(vh.transpose(0, 1) @ u.transpose(0, 1)))
    rotation = vh.transpose(0, 1) @ correction @ u.transpose(0, 1)
    translation = ref_center - rotation @ src_center
    transform = torch.eye(4, dtype=src_points.dtype, device=src_points.device)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    return transform


def ransac_50k(
    src_points,
    ref_points,
    *,
    distance_threshold_m,
    iterations,
    confidence,
    seed,
):
    """Open3D correspondence RANSAC. Deliberately performs no ICP refinement."""
    if len(src_points) < 3:
        raise ValueError("at least three correspondences are required")
    source = np.asarray(src_points.detach().cpu(), dtype=np.float64)
    reference = np.asarray(ref_points.detach().cpu(), dtype=np.float64)
    source_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(source))
    reference_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(reference))
    indices = np.column_stack([np.arange(len(source)), np.arange(len(source))]).astype(np.int32)
    o3d.utility.random.seed(int(seed))
    criteria = o3d.pipelines.registration.RANSACConvergenceCriteria(
        int(iterations), float(confidence)
    )
    result = o3d.pipelines.registration.registration_ransac_based_on_correspondence(
        source_cloud,
        reference_cloud,
        o3d.utility.Vector2iVector(indices),
        float(distance_threshold_m),
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3,
        [],
        criteria,
    )
    transformation = np.array(result.transformation, copy=True)
    return torch.as_tensor(transformation, dtype=src_points.dtype, device=src_points.device)


def _lgr(output, cfg, topk):
    missing = [key for key in LGR_REQUIRED_OUTPUTS if key not in output]
    if missing:
        raise KeyError(f"model output lacks LGR tensors: {missing}")
    matcher = LocalGlobalRegistration(
        cfg.fine_matching.topk,
        cfg.fine_matching.acceptance_radius,
        mutual=cfg.fine_matching.mutual,
        confidence_threshold=cfg.fine_matching.confidence_threshold,
        use_dustbin=cfg.fine_matching.use_dustbin,
        use_global_score=cfg.fine_matching.use_global_score,
        correspondence_threshold=cfg.fine_matching.correspondence_threshold,
        correspondence_limit=int(topk),
        num_refinement_steps=cfg.fine_matching.num_refinement_steps,
    ).to(output["matching_scores"].device)
    scores = output["matching_scores"]
    if not matcher.use_dustbin:
        scores = scores[:, :-1, :-1]
    _, _, _, transform = matcher(
        output["ref_node_corr_knn_points"],
        output["src_node_corr_knn_points"],
        output["ref_node_corr_knn_masks"],
        output["src_node_corr_knn_masks"],
        scores,
        output["node_corr_scores"],
    )
    return transform


def _valid_transform(transform):
    if transform is None or transform.shape != (4, 4) or not torch.isfinite(transform).all():
        return False
    return bool(torch.det(transform[:3, :3]).abs() > 1e-6)


def estimate_pose(method, output, cfg, topk, seed=7351):
    """Estimate src-to-ref pose from prediction tensors only."""
    if method not in ("lgr", "weighted_svd", "ransac_50k"):
        raise ValueError(f"unknown estimator: {method}")
    ref_points, src_points, scores = select_topk_correspondences(
        output["ref_corr_points"], output["src_corr_points"], output["corr_scores"], topk
    )
    count = len(scores)
    if count < 3:
        return PoseEstimate(None, False, count, method, "fewer than three correspondences")
    try:
        if method == "weighted_svd":
            transform = weighted_svd(src_points, ref_points, scores)
        elif method == "ransac_50k":
            transform = ransac_50k(
                src_points,
                ref_points,
                distance_threshold_m=float(cfg.protocol.ransac_distance_threshold_m),
                iterations=int(cfg.protocol.ransac_iterations),
                confidence=float(cfg.protocol.ransac_confidence),
                seed=seed,
            )
        else:
            transform = _lgr(output, cfg, topk)
    except (RuntimeError, ValueError) as error:
        return PoseEstimate(None, False, count, method, str(error))
    if not _valid_transform(transform):
        return PoseEstimate(None, False, count, method, "invalid transformation")
    return PoseEstimate(transform, True, count, method)


__all__ = [
    "LGR_REQUIRED_OUTPUTS",
    "PoseEstimate",
    "estimate_pose",
    "ransac_50k",
    "select_topk_correspondences",
    "weighted_svd",
]
