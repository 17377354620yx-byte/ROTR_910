"""Canonical 3D-IRCADb adapter for stack-mode GeoTransformer models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset

from geotransformer.utils.data import (
    build_dataloader_stack_mode,
    calibrate_neighbors_stack_mode,
    registration_collate_fn_stack_mode,
)
from dataset import _norm_vox
from tools.ircadb_benchmark import (
    NormalizedPair,
    SampleRecord,
    load_samples,
    normalize_pair,
    transform_to_mm,
    transform_to_network,
)


@dataclass(frozen=True)
class IRCADbContext:
    record: SampleRecord
    source_mm: np.ndarray
    target_mm: np.ndarray
    gt_transform_mm: np.ndarray
    normalized: NormalizedPair


class IRCADbStackDataset(Dataset):
    def __init__(self, root: Path, voxel_size: float, visibility="all", limit: int = 0):
        self.root = Path(root)
        self.voxel_size = float(voxel_size)
        self.records = load_samples(self.root, visibility=visibility, limit=limit)

    def __len__(self):
        return len(self.records)

    def load_context(self, index: int) -> IRCADbContext:
        record = self.records[index]
        sample = record.load()
        source_raw = np.asarray(sample["src_points"], dtype=np.float64)
        target_raw = np.asarray(sample["ref_points"], dtype=np.float64)
        source_mm, _ = _norm_vox(source_raw, self.voxel_size)
        target_mm, _ = _norm_vox(target_raw, self.voxel_size)
        normalized = normalize_pair(source_mm, target_mm)
        return IRCADbContext(
            record=record,
            source_mm=source_raw,
            target_mm=target_raw,
            gt_transform_mm=np.asarray(sample["transform"], dtype=np.float64),
            normalized=normalized,
        )

    def __getitem__(self, index: int) -> dict:
        context = self.load_context(index)
        pair = context.normalized
        transform = transform_to_network(context.gt_transform_mm, pair)
        reference = np.ascontiguousarray(pair.target, dtype=np.float32)
        source = np.ascontiguousarray(pair.source, dtype=np.float32)
        return {
            "scene_name": "3D-IRCADb",
            "ref_frame": context.record.sample_id,
            "src_frame": "complete_liver",
            "sample_name": context.record.sample_id,
            "index": int(index),
            "overlap": np.float32(context.record.visibility),
            "ref_points": reference,
            "src_points": source,
            "ref_feats": np.ones((len(reference), 1), dtype=np.float32),
            "src_feats": np.ones((len(source), 1), dtype=np.float32),
            "transform": np.asarray(transform, dtype=np.float32),
        }


def estimate_to_mm(estimated_transform: np.ndarray, context: IRCADbContext) -> np.ndarray:
    return transform_to_mm(estimated_transform, context.normalized)


def build_ircadb_loader(cfg, root: Path, visibility="all", limit: int = 0, num_workers=None,
                         neighbor_limits=None):
    dataset = IRCADbStackDataset(root, cfg.data.voxel_size, visibility, limit)
    if not len(dataset):
        raise ValueError("No 3D-IRCADb samples match the requested selection")
    if neighbor_limits is None:
        neighbor_limits = calibrate_neighbors_stack_mode(
            dataset,
            registration_collate_fn_stack_mode,
            cfg.backbone.num_stages,
            cfg.backbone.init_voxel_size,
            cfg.backbone.init_radius,
            sample_threshold=min(int(cfg.data.neighbor_calibration_samples), 200),
        )
    workers = cfg.test.num_workers if num_workers is None else int(num_workers)
    loader = build_dataloader_stack_mode(
        dataset,
        registration_collate_fn_stack_mode,
        cfg.backbone.num_stages,
        cfg.backbone.init_voxel_size,
        cfg.backbone.init_radius,
        neighbor_limits,
        batch_size=1,
        num_workers=workers,
        shuffle=False,
        drop_last=False,
        distributed=False,
    )
    return dataset, loader, list(neighbor_limits)

