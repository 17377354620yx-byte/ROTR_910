"""P2I-LReg synthetic complete-to-partial rigid-registration adapter."""

from functools import lru_cache, partial
import hashlib
import os
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
import torch
from torch.utils.data import Dataset
import yaml

from geotransformer.utils.data import (
    build_dataloader_stack_mode,
    calibrate_neighbors_stack_mode,
    registration_collate_fn_stack_mode,
)


_SMALL_PLY_INSPECTION_BYTES = 512


def _normalise_root(root):
    root = Path(root).expanduser().resolve()
    if (root / "Liver_regis").is_dir() and not (root / "01").is_dir():
        root = root / "Liver_regis"
    return root


def voxel_downsample(points, voxel_size):
    """Voxel centroid downsampling in the input coordinate unit (metres here)."""
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not len(points):
        raise ValueError("points must be a nonempty (N, 3) array")
    if voxel_size <= 0:
        raise ValueError("voxel_size must be positive")
    minimum = points.min(axis=0) - voxel_size * 0.5
    indices = np.floor((points - minimum) / voxel_size).astype(np.int64)
    _, inverse = np.unique(indices, axis=0, return_inverse=True)
    counts = np.bincount(inverse)
    return np.column_stack(
        [np.bincount(inverse, weights=points[:, i]) / counts for i in range(3)]
    )


def sample_fixed_points(points, count, rng):
    """Return exactly ``count`` real points; short clouds are wrap-sampled."""
    points = np.asarray(points)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
        raise ValueError("cannot sample an empty or malformed point cloud")
    if count <= 0:
        raise ValueError("count must be positive")
    replace = len(points) < count
    indices = rng.choice(len(points), size=count, replace=replace)
    return np.ascontiguousarray(points[indices])


def load_source_to_reference_transform(pose_metadata, frame_id):
    """Convert released Blender C2W pose (translation in mm) to W2C in metres."""
    frame_id = str(frame_id)[-5:]
    if frame_id not in pose_metadata:
        raise KeyError(f"frame {frame_id} is absent from camPose.yml")
    record = pose_metadata[frame_id][0]
    rotation = np.asarray(record["cam_R_c2w"], dtype=np.float64).reshape(3, 3)
    translation = np.asarray(record["cam_t_c2w"], dtype=np.float64).reshape(3) / 1000.0
    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, :3] = rotation
    c2w[:3, 3] = translation
    transform = np.linalg.inv(c2w)
    if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-8):
        raise ValueError("invalid homogeneous camera pose")
    return transform


def apply_transform(points, transform):
    points = np.asarray(points)
    transform = np.asarray(transform)
    return points @ transform[:3, :3].T + transform[:3, 3]


def chamfer_before_after(reference, source, transform):
    """Measure asymmetric and symmetric nearest-neighbour distances in metres."""
    reference = np.asarray(reference, dtype=np.float64)
    source = np.asarray(source, dtype=np.float64)
    aligned = apply_transform(source, transform)

    def directed(first, second):
        return float(cKDTree(second).query(first, k=1, workers=-1)[0].mean())

    ref_before = directed(reference, source)
    src_before = directed(source, reference)
    ref_after = directed(reference, aligned)
    src_after = directed(aligned, reference)
    return {
        "ref_to_src_before_m": ref_before,
        "ref_to_src_after_m": ref_after,
        "symmetric_before_m": ref_before + src_before,
        "symmetric_after_m": ref_after + src_after,
    }


def _stable_seed(seed, text):
    digest = hashlib.blake2b(str(text).encode("utf-8"), digest_size=8).digest()
    return (int.from_bytes(digest, "little") + int(seed)) % (2**63 - 1)


def _read_split(path):
    with open(path, encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def _all_records(root, list_name):
    records = []
    for patient in range(1, 22):
        patient_id = f"{patient:02d}"
        split_path = root / patient_id / list_name
        if not split_path.is_file():
            raise FileNotFoundError(split_path)
        for frame_name in _read_split(split_path):
            records.append((patient_id, frame_name))
    return records


def _ply_vertex_count(path):
    with open(path, "rb") as handle:
        if handle.readline().strip() != b"ply":
            raise ValueError(f"invalid PLY header: {path}")
        for _ in range(256):
            line = handle.readline()
            if not line:
                break
            fields = line.strip().split()
            if len(fields) == 3 and fields[:2] == [b"element", b"vertex"]:
                return int(fields[2])
            if line.strip() == b"end_header":
                break
    raise ValueError(f"PLY vertex count is absent: {path}")


def filter_unusable_reference_records(root, records):
    """Exclude released synthetic placeholders without a usable observation.

    P2I-LReg represents missing synthetic observations as a one-vertex PLY at
    the origin. The filter uses only the observed point cloud, never GT pose or
    GT correspondences, and runs before the deterministic train/val split.
    """
    root = Path(root)
    usable = []
    excluded = []
    for patient_id, frame_name in records:
        path = root / patient_id / "syn" / "liverPcds" / f"{frame_name}.ply"
        if not path.is_file():
            raise FileNotFoundError(path)
        # Valid released clouds are at least several KiB. Inspect only tiny
        # candidates so dataset construction does not read ~49k PLY headers.
        valid = True
        if path.stat().st_size <= _SMALL_PLY_INSPECTION_BYTES:
            vertex_count = _ply_vertex_count(path)
            valid = vertex_count > 0
            if vertex_count > 0:
                points = np.asarray(o3d.io.read_point_cloud(str(path)).points)
                valid = len(points) > 0 and bool(np.any(points != 0.0))
        if valid:
            usable.append((patient_id, frame_name))
        else:
            excluded.append((patient_id, frame_name))
    return usable, excluded


def reference_filter_manifest(excluded_records):
    return {
        "policy": "exclude_empty_observation_without_gt",
        "excluded_count": len(excluded_records),
        "excluded_case_ids": [f"{patient}/{frame}" for patient, frame in excluded_records],
    }


@lru_cache(maxsize=4)
def _filtered_split_records(root, list_name):
    records = _all_records(Path(root), list_name)
    usable, excluded = filter_unusable_reference_records(Path(root), records)
    return tuple(usable), tuple(excluded)


def _partition_train_records(records, validation_size, seed):
    validation_size = int(validation_size)
    if validation_size <= 0 or validation_size >= len(records):
        raise ValueError("validation_size must be between zero and train-set size")
    order = np.random.default_rng(seed).permutation(len(records))
    validation = {int(index) for index in order[:validation_size]}
    train_records = [record for index, record in enumerate(records) if index not in validation]
    val_records = [records[int(index)] for index in order[:validation_size]]
    return train_records, val_records


class P2ILRegDataset(Dataset):
    """Official synthetic source/ref pairs with a strict src-to-ref transform."""

    def __init__(self, root, split, cfg, limit=None):
        if split not in ("train", "val", "test"):
            raise ValueError(f"unknown split: {split}")
        self.root = _normalise_root(root)
        self.split = split
        self.cfg = cfg
        self.seed = int(cfg.seed)
        self.voxel_size = float(cfg.data.voxel_size)
        self.num_points = int(cfg.data.num_points)
        self.source_surface_samples = int(cfg.data.source_surface_samples)
        train_records = None
        if split in ("train", "val"):
            released_train, excluded_records = _filtered_split_records(
                str(self.root), "train_syn.txt"
            )
            train_records, val_records = _partition_train_records(
                released_train, cfg.data.validation_size, self.seed
            )
            self.records = train_records if split == "train" else val_records
        else:
            self.records, excluded_records = _filtered_split_records(
                str(self.root), "test_syn.txt"
            )
            self.records = list(self.records)
        excluded_records = list(excluded_records)
        self.excluded_records = excluded_records
        self.num_excluded_records = len(excluded_records)
        if limit is not None and int(limit) > 0:
            self.records = self.records[: int(limit)]
        self._source_cache = {}
        self._pose_cache = {}

    def __len__(self):
        return len(self.records)

    def _rng(self, case_id):
        if self.split == "train":
            return np.random.default_rng(np.random.randint(0, 2**31 - 1))
        return np.random.default_rng(_stable_seed(self.seed, case_id))

    def _load_source(self, patient_id):
        if patient_id not in self._source_cache:
            path = self.root / patient_id / "model" / "reconstructed_mesh_world_m.obj"
            mesh = o3d.io.read_triangle_mesh(str(path))
            if mesh.is_empty():
                raise ValueError(f"empty source mesh: {path}")
            o3d.utility.random.seed(_stable_seed(self.seed, patient_id) % (2**31 - 1))
            cloud = mesh.sample_points_uniformly(self.source_surface_samples)
            points = voxel_downsample(np.asarray(cloud.points), self.voxel_size)
            self._source_cache[patient_id] = np.ascontiguousarray(points, dtype=np.float32)
        return self._source_cache[patient_id]

    def _load_pose(self, patient_id):
        if patient_id not in self._pose_cache:
            path = self.root / patient_id / "syn" / "camPose.yml"
            with open(path, encoding="utf-8") as handle:
                self._pose_cache[patient_id] = yaml.safe_load(handle)
        return self._pose_cache[patient_id]

    def _load_reference(self, patient_id, frame_name):
        path = self.root / patient_id / "syn" / "liverPcds" / f"{frame_name}.ply"
        cloud = o3d.io.read_point_cloud(str(path))
        if cloud.is_empty():
            raise ValueError(f"empty reference point cloud: {path}")
        stride = int(self.cfg.data.target_uniform_stride)
        if stride > 1:
            cloud = cloud.uniform_down_sample(every_k_points=stride)
        cloud = cloud.remove_duplicated_points()
        if len(cloud.points) >= int(self.cfg.data.target_outlier_neighbors):
            cloud, _ = cloud.remove_statistical_outlier(
                nb_neighbors=int(self.cfg.data.target_outlier_neighbors),
                std_ratio=float(self.cfg.data.target_outlier_std_ratio),
            )
        points = np.asarray(cloud.points, dtype=np.float64) / 1000.0
        points = points[~np.all(points == 0.0, axis=1)]
        if len(points) == 0:
            raise ValueError(f"reference has no valid nonzero points: {path}")
        return voxel_downsample(points, self.voxel_size)

    def __getitem__(self, index):
        patient_id, frame_name = self.records[index]
        frame_id = frame_name[-5:]
        case_id = f"{patient_id}/{frame_name}"
        rng = self._rng(case_id)
        source = sample_fixed_points(self._load_source(patient_id), self.num_points, rng)
        reference = sample_fixed_points(
            self._load_reference(patient_id, frame_name), self.num_points, rng
        )
        transform = load_source_to_reference_transform(
            self._load_pose(patient_id), frame_id
        ).astype(np.float32)
        rotation = transform[:3, :3]
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-4) or not np.isclose(
            np.linalg.det(rotation), 1.0, atol=1e-4
        ):
            raise ValueError(f"non-SE(3) transform in clean rigid sample {case_id}")
        return {
            "scene_name": "P2I-LReg-syn",
            "ref_frame": frame_name,
            "src_frame": "complete_preoperative_liver",
            "sample_name": case_id,
            "case_id": case_id,
            "patient_id": patient_id,
            "frame_id": frame_id,
            "index": int(index),
            "ref_points": np.ascontiguousarray(reference, dtype=np.float32),
            "src_points": np.ascontiguousarray(source, dtype=np.float32),
            "ref_feats": np.ones((self.num_points, 1), dtype=np.float32),
            "src_feats": np.ones((self.num_points, 1), dtype=np.float32),
            "transform": transform,
        }


def _neighbor_limits(cfg, dataset):
    if cfg.data.neighbor_limits is not None:
        return np.asarray(cfg.data.neighbor_limits, dtype=np.int64)
    return calibrate_neighbors_stack_mode(
        dataset,
        registration_collate_fn_stack_mode,
        cfg.backbone.num_stages,
        cfg.backbone.init_voxel_size,
        cfg.backbone.init_radius,
        sample_threshold=int(cfg.data.neighbor_calibration_samples),
    )


def _loader(cfg, dataset, limits, train, distributed=False):
    section = cfg.train if train else cfg.test
    return build_dataloader_stack_mode(
        dataset,
        registration_collate_fn_stack_mode,
        cfg.backbone.num_stages,
        cfg.backbone.init_voxel_size,
        cfg.backbone.init_radius,
        limits,
        batch_size=section.batch_size,
        num_workers=section.num_workers,
        shuffle=train,
        drop_last=train,
        distributed=distributed,
    )


def train_valid_data_loader(cfg, distributed=False, train_limit=None, validation_limit=None):
    train_set = P2ILRegDataset(cfg.data.root, "train", cfg, limit=train_limit)
    val_set = P2ILRegDataset(cfg.data.root, "val", cfg, limit=validation_limit)
    limits = _neighbor_limits(cfg, train_set)
    return (
        _loader(cfg, train_set, limits, True, distributed),
        _loader(cfg, val_set, limits, False),
        limits,
    )


def test_data_loader(cfg, limit=None):
    dataset = P2ILRegDataset(cfg.data.root, "test", cfg, limit=limit)
    limits = _neighbor_limits(cfg, dataset)
    return _loader(cfg, dataset, limits, False), limits


__all__ = [
    "P2ILRegDataset",
    "apply_transform",
    "chamfer_before_after",
    "filter_unusable_reference_records",
    "load_source_to_reference_transform",
    "reference_filter_manifest",
    "sample_fixed_points",
    "test_data_loader",
    "train_valid_data_loader",
    "voxel_downsample",
]
