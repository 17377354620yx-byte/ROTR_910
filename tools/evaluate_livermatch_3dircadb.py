#!/usr/bin/env python3
"""Evaluate LiverMatch and LiverMatch+P2P on canonical 3D-IRCADb pairs."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import importlib
from pathlib import Path
import random
import sys
import time

import numpy as np

BENCHMARK_PROJECT = Path(__file__).resolve().parents[1]
if str(BENCHMARK_PROJECT) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_PROJECT))

from tools.ircadb_benchmark import (
    NormalizedPair,
    PredictionWriter,
    SampleRecord,
    load_samples,
    normalize_pair,
    transform_to_mm,
    transform_to_network,
)
from tools.p2p_benchmark_metrics import correspondence_inlier_ratio


PROJECT = Path("/home/yangx/code/point_clould/LiverMatch")


def method_name(mode: str) -> str:
    try:
        return {"base": "livermatch", "p2p": "livermatch_p2p"}[mode]
    except KeyError as exc:
        raise ValueError(f"Unknown LiverMatch mode: {mode}") from exc


def _voxel_down_sample(points: np.ndarray, voxel_size: float) -> np.ndarray:
    if not len(points):
        return points.copy()
    minimum = points.min(axis=0) - voxel_size * 0.5
    voxels = np.floor((points - minimum) / voxel_size).astype(np.int64)
    _, inverse = np.unique(voxels, axis=0, return_inverse=True)
    counts = np.bincount(inverse)
    return np.column_stack([
        np.bincount(inverse, weights=points[:, axis]) / counts for axis in range(3)
    ])


def _native_voxel(points: np.ndarray, voxel_size: float) -> np.ndarray:
    center = points.mean(axis=0)
    radius = float(np.linalg.norm(points - center, axis=1).max())
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError("Point cloud radius must be positive and finite")
    normalized = (points - center) / radius
    return _voxel_down_sample(normalized, voxel_size) * radius + center


@dataclass(frozen=True)
class LiverContext:
    record: SampleRecord
    source_mm: np.ndarray
    target_mm: np.ndarray
    gt_transform_mm: np.ndarray
    normalized: NormalizedPair
    network_transform: np.ndarray
    source_markers: np.ndarray
    target_markers: np.ndarray


class _LazyContexts:
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset.load_context(index)


class CanonicalLiverDataset:
    """Dataset surface expected by the released LiverMatch evaluation functions."""

    def __init__(self, root: Path, voxel_size: float = 0.04, visibility="all", limit: int = 0):
        self.records = load_samples(root, visibility=visibility, limit=limit)
        self.voxel_size = float(voxel_size)
        self.test_list = np.asarray([record.sample_id for record in self.records])
        self.contexts = _LazyContexts(self)

    def __len__(self):
        return len(self.records)

    def load_context(self, index: int) -> LiverContext:
        record = self.records[index]
        sample = record.load()
        source_raw = np.asarray(sample["src_points"], dtype=np.float64)
        target_raw = np.asarray(sample["ref_points"], dtype=np.float64)
        source_mm = _native_voxel(source_raw, self.voxel_size)
        target_mm = _native_voxel(target_raw, self.voxel_size)
        normalized = normalize_pair(source_mm, target_mm)
        gt = np.asarray(sample["transform"], dtype=np.float64)
        source_markers = (source_raw - normalized.source_center) / normalized.scale
        target_markers_mm = source_raw @ gt[:3, :3].T + gt[:3, 3]
        target_markers = (target_markers_mm - normalized.target_center) / normalized.scale
        return LiverContext(
            record=record,
            source_mm=source_raw,
            target_mm=target_raw,
            gt_transform_mm=gt,
            normalized=normalized,
            network_transform=transform_to_network(gt, normalized),
            source_markers=np.ascontiguousarray(source_markers, dtype=np.float64),
            target_markers=np.ascontiguousarray(target_markers, dtype=np.float64),
        )

    def get_data_np(self, index: int):
        context = self.load_context(index)
        pair = context.normalized
        return (
            np.ascontiguousarray(pair.source, dtype=np.float64),
            np.ascontiguousarray(pair.target, dtype=np.float64),
            context.source_markers,
            context.target_markers,
            pair.source_center,
            pair.target_center,
            pair.scale,
            context.gt_transform_mm[:3, :3],
            context.gt_transform_mm[:3, 3, None],
        )

    def __getitem__(self, index: int):
        source, target, *_ = self.get_data_np(index)
        source_features = np.ones_like(source[:, :1], dtype=np.float32)
        target_features = np.ones_like(target[:, :1], dtype=np.float32)
        import torch

        return (
            source,
            target,
            source_features,
            target_features,
            np.eye(3, dtype=np.float32),
            np.ones((3, 1), dtype=np.float32),
            torch.ones(1, 2).long(),
            source,
            target,
            torch.ones(1),
        )


def persist_details(
    writer: PredictionWriter,
    context: LiverContext,
    details: dict,
    runtime_seconds: float,
    mode: str,
) -> dict:
    estimate = transform_to_mm(np.asarray(details["estimated_transform"]), context.normalized)
    matches = np.asarray(details.get("matches", np.empty((0, 2))), dtype=np.int64)
    ir = None
    source = np.asarray(details.get("source", context.normalized.source))
    target = np.asarray(details.get("target", context.normalized.target))
    if len(matches):
        valid = (
            (matches[:, 0] >= 0) & (matches[:, 0] < len(source))
            & (matches[:, 1] >= 0) & (matches[:, 1] < len(target))
        )
        matches = matches[valid]
        if len(matches):
            ir = correspondence_inlier_ratio(
                source[matches[:, 0]], target[matches[:, 1]], context.network_transform, 0.1
            )
    solver = "svd" if mode == "base" else "p2p_cluster_k5"
    return writer.write_success(
        context.record,
        estimate,
        context.gt_transform_mm,
        runtime_seconds,
        context.source_mm,
        context.target_mm,
        ir=ir,
        metadata={"mode": mode, "solver": solver, "correspondence_count": len(matches)},
    )


def _load_module(mode: str):
    if str(PROJECT) not in sys.path:
        sys.path.insert(0, str(PROJECT))
    if mode == "p2p":
        sys.modules["lib.utils"] = importlib.import_module("p2p_util")
    return importlib.import_module("eval_in_silico" if mode == "base" else "eval_in_silico_p2p")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("base", "p2p"), required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=PROJECT / "configs/new_task3_004_002.yaml")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--visibility", choices=("all", "0.20", "0.30"), default="all")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--k", type=int, default=5)
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    if args.mode == "p2p" and args.k != 5:
        raise ValueError("The approved LiverMatch+P2P protocol requires --k 5")
    random.seed(args.seed)
    np.random.seed(args.seed)

    import torch

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    module = _load_module(args.mode)
    config = module.edict(module.load_config(str(args.config)))
    config.architecture = module.architectures[config.model_name]
    config.device = torch.device(args.device)
    module.config = config
    model_kwargs = {"K": args.k} if args.mode == "p2p" else {}
    module.model = module.KPFCNN(config, **model_kwargs).to(config.device).eval()
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    module.model.load_state_dict(state["state_dict"], strict=True)
    dataset = CanonicalLiverDataset(
        args.data_root, voxel_size=0.04, visibility=args.visibility, limit=args.limit
    )
    if not len(dataset):
        raise ValueError("No 3D-IRCADb samples match the requested selection")
    module.demo_set = dataset
    writer = PredictionWriter(method_name(args.mode), args.output, dataset.test_list)

    for index, context in enumerate(dataset.contexts):
        started = time.perf_counter()
        kwargs = {"debug": False, "use_SVD": True, "return_details": True}
        if args.mode == "p2p":
            kwargs.update(cluster=True, use_corr_cl=True, use_all_matches=True)
        try:
            details = module.eva_one(index, **kwargs)
            row = persist_details(
                writer, context, details, time.perf_counter() - started, args.mode
            )
            message = f"RMSE={row['RMSE']:.4f} mm"
        except Exception as exc:
            writer.write_failure(
                context.record,
                context.gt_transform_mm,
                time.perf_counter() - started,
                error=f"{type(exc).__name__}: {exc}",
                metadata={"mode": args.mode},
            )
            message = f"ERROR={type(exc).__name__}: {exc}"
        if index == 0 or (index + 1) % args.log_every == 0 or index + 1 == len(dataset):
            print(f"[{index + 1}/{len(dataset)}] {context.record.sample_id} {message}", flush=True)
    payload = writer.finalize()
    print(f"Wrote {payload['counts']['total']} predictions to {args.output}")


if __name__ == "__main__":
    main()
