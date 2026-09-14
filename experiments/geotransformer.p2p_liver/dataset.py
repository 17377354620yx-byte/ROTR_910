"""LiverMatch Task3-compatible data adapter for GeoTransformer."""

import json
import os
import random

import numpy as np
from scipy.spatial.transform import Rotation
from torch.utils.data import Dataset

from geotransformer.utils.data import (
    build_dataloader_stack_mode,
    calibrate_neighbors_stack_mode,
    registration_collate_fn_stack_mode,
)
from geotransformer.utils.pointcloud import get_transform_from_rotation_translation


def _voxel_down_sample(points, voxel_size):
    """Open3D ``voxel_down_sample`` equivalent without its NumPy 2 crash."""
    min_bound = points.min(axis=0) - voxel_size * 0.5
    voxel_indices = np.floor((points - min_bound) / voxel_size).astype(np.int64)
    _, inverse = np.unique(voxel_indices, axis=0, return_inverse=True)
    counts = np.bincount(inverse)
    return np.column_stack(
        [np.bincount(inverse, weights=points[:, axis]) / counts for axis in range(3)]
    )


def _normalize(points, centroid=None, radius=None):
    if centroid is None:
        centroid = np.mean(points, axis=0)
    normalized = points - centroid
    if radius is None:
        radius = np.max(np.sqrt(np.sum(normalized ** 2, axis=1)))
    return normalized / radius, centroid, radius


def _norm_vox(points, voxel_size):
    normalized, centroid, radius = _normalize(points)
    down = _voxel_down_sample(normalized, voxel_size)
    return down * radius + centroid, radius


def _crop(points, keep_ratio):
    phi = np.random.uniform(0.0, 2 * np.pi)
    cos_theta = np.random.uniform(-1.0, 1.0)
    theta = np.arccos(cos_theta)
    direction = np.asarray(
        [np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)]
    )
    centered = points[:, :3] - np.mean(points[:, :3], axis=0)
    distances = np.dot(centered, direction)
    if keep_ratio == 0.5:
        mask = distances > 0
    else:
        mask = distances > np.percentile(distances, (1.0 - keep_ratio) * 100)
    return points[mask, :]


def _rigid_transform(source, target):
    """Return the least-squares rigid transform mapping paired source to target."""
    source_center = np.mean(source, axis=0)
    target_center = np.mean(target, axis=0)
    covariance = (source - source_center).T @ (target - target_center)
    u, _, vh = np.linalg.svd(covariance)
    rotation = vh.T @ u.T
    if np.linalg.det(rotation) < 0:
        vh[-1] *= -1
        rotation = vh.T @ u.T
    translation = target_center - rotation @ source_center
    return get_transform_from_rotation_translation(rotation, translation)


class LiverTask3TrainDataset(Dataset):
    """Dynamic pairs matching ``liverTask3.get_input_train``."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.root = cfg.data.train_root
        with open(cfg.data.train_list, encoding="utf-8") as handle:
            self.groups = json.load(handle)
        self.voxel_size = float(cfg.data.voxel_size)
        self.min_visibility = float(cfg.data.min_visibility)
        self.max_visibility = float(cfg.data.max_visibility)
        self.max_noise = float(cfg.data.max_noise_mm)
        limit = int(os.environ.get("P2P_TRAIN_LIMIT", "0"))
        self.length = min(len(self.groups), limit) if limit > 0 else len(self.groups)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        group = self.groups[str(index)]
        source_name, target_name = random.sample(group, 2)
        with np.load(os.path.join(self.root, source_name), allow_pickle=True) as entry:
            source = np.asarray(entry["vs_vox"])
        with np.load(os.path.join(self.root, target_name), allow_pickle=True) as entry:
            target_full = np.asarray(entry["vs_vox"])

        source, _ = _norm_vox(source, self.voxel_size)
        target_full, _ = _norm_vox(target_full, self.voxel_size)
        visibility = self.min_visibility + (
            self.max_visibility - self.min_visibility
        ) * np.random.rand(1)[0]
        visibility = min(visibility, 1.0)
        target = _crop(target_full, visibility)

        sigma = np.random.rand(1)[0] * self.max_noise
        target += (np.random.rand(target.shape[0], 3) - 0.5) * sigma

        source_center = np.mean(source[:, :3], axis=0)
        scale = np.max(np.sqrt(np.sum((source - source_center) ** 2, axis=1)))
        source = (source - source_center) / scale
        target = (target - source_center) / scale
        translation = -np.mean(target[:, :3], axis=0)
        target += translation
        rotation = np.eye(3)

        euler = np.random.rand(3) * np.pi * 2
        augmentation_rotation = Rotation.from_euler("zyx", euler).as_matrix()
        if np.random.rand(1)[0] > 0.5:
            source = source @ augmentation_rotation.T
            rotation = rotation @ augmentation_rotation.T
        else:
            target = target @ augmentation_rotation.T
            rotation = augmentation_rotation @ rotation
            translation = augmentation_rotation @ translation

        transform = get_transform_from_rotation_translation(rotation, translation)
        return _as_sample(
            index,
            reference=target,
            source=source,
            transform=transform,
            overlap=visibility,
            sample_name=f"train/{index}:{source_name}->{target_name}",
        )


class LiverTask3TestDataset(Dataset):
    """Fixed in-silico test set used by the P2P paper."""

    def __init__(self, cfg, noise_mm=None, rotate_target=True):
        self.cfg = cfg
        self.root = cfg.data.test_root
        self.samples = np.load(cfg.data.test_list)["test"]
        self.voxel_size = float(cfg.data.voxel_size)
        self.noise_mm = noise_mm
        self.rotate_target = rotate_target
        self.sample_indices = np.arange(len(self.samples), dtype=np.int64)
        requested_indices = os.environ.get("P2P_TEST_INDICES")
        if requested_indices:
            if ":" in requested_indices:
                start, stop = (int(value) for value in requested_indices.split(":"))
                selected = np.arange(start, stop, dtype=np.int64)
            else:
                selected = np.asarray(
                    [int(value) for value in requested_indices.split(",") if value.strip()],
                    dtype=np.int64,
                )
            if np.any(selected < 0) or np.any(selected >= len(self.samples)):
                raise IndexError("P2P_TEST_INDICES contains an out-of-range index")
            self.samples = self.samples[selected]
            self.sample_indices = self.sample_indices[selected]
        limit = int(os.environ.get("P2P_TEST_LIMIT", "0"))
        if limit > 0:
            self.samples = self.samples[:limit]
            self.sample_indices = self.sample_indices[:limit]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        original_index = int(self.sample_indices[index])
        sample_name = str(self.samples[index])
        with np.load(os.path.join(self.root, sample_name), allow_pickle=True) as entry:
            source = np.asarray(entry["src_pcd"])
            target = np.asarray(entry["tgt_pcd"])
            source_markers = np.asarray(entry["src_vol"])
            target_markers = np.asarray(entry["tgt_vol"])
            if self.noise_mm is not None:
                target = target + entry[str(self.noise_mm)]
            if self.rotate_target:
                target_rotation = np.asarray(entry["rot_tgt"])
                target = target @ target_rotation.T
                target_markers = target_markers @ target_rotation.T

        source, scale = _norm_vox(source, self.voxel_size)
        target, _ = _norm_vox(target, self.voxel_size)
        source_center = np.mean(source[:, :3], axis=0)
        target_center = np.mean(target[:, :3], axis=0)
        source = (source - source_center) / scale
        target = (target - target_center) / scale
        source_markers = (source_markers - source_center) / scale
        target_markers = (target_markers - target_center) / scale

        # The released target is non-rigidly deformed, so there is no exact SE(3)
        # ground truth.  This marker-paired least-squares transform is the rigid
        # task optimum and is used only for diagnostics / oracle decomposition.
        # Inference does not consume ``transform`` when the model is in eval mode.
        oracle_transform = _rigid_transform(source_markers, target_markers)
        oracle_aligned_markers = (
            source_markers @ oracle_transform[:3, :3].T
            + oracle_transform[:3, 3]
        )
        oracle_rms_tre = np.sqrt(
            np.mean(np.sum((oracle_aligned_markers - target_markers) ** 2, axis=1))
        ) * scale

        sample = _as_sample(
            original_index,
            reference=target,
            source=source,
            transform=oracle_transform,
            overlap=np.nan,
            sample_name=sample_name,
        )
        sample.update(
            source_markers=source_markers.astype(np.float32),
            target_markers=target_markers.astype(np.float32),
            physical_scale=np.float32(scale),
            oracle_rms_tre_mm=np.float32(oracle_rms_tre),
        )
        return sample


class LiverTask3InVitroTestDataset(Dataset):
    """Official 800-case in-vitro phantom test set used by paper Table VII."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.root = cfg.data.in_vitro_root
        if not os.path.isfile(cfg.data.in_vitro_list):
            raise FileNotFoundError(f"Missing in-vitro split: {cfg.data.in_vitro_list}")
        self.samples = np.load(cfg.data.in_vitro_list).astype(str)
        self.voxel_size = float(cfg.data.voxel_size)
        self.sample_indices = np.arange(len(self.samples), dtype=np.int64)
        requested_indices = os.environ.get("P2P_TEST_INDICES")
        if requested_indices:
            if ":" in requested_indices:
                start, stop = (int(value) for value in requested_indices.split(":"))
                selected = np.arange(start, stop, dtype=np.int64)
            else:
                selected = np.asarray(
                    [int(value) for value in requested_indices.split(",") if value.strip()],
                    dtype=np.int64,
                )
            if np.any(selected < 0) or np.any(selected >= len(self.samples)):
                raise IndexError("P2P_TEST_INDICES contains an out-of-range in-vitro index")
            self.samples = self.samples[selected]
            self.sample_indices = self.sample_indices[selected]
        limit = int(os.environ.get("P2P_TEST_LIMIT", "0"))
        if limit > 0:
            self.samples = self.samples[:limit]
            self.sample_indices = self.sample_indices[:limit]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        original_index = int(self.sample_indices[index])
        sample_name = str(self.samples[index])
        path = os.path.join(self.root, "Rigid_test_data", sample_name)
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        with np.load(path, allow_pickle=False) as entry:
            source = np.asarray(entry["src_vs"], dtype=np.float64)
            target = np.asarray(entry["tgt_vs"], dtype=np.float64)
            source_markers = np.asarray(entry["src_marker"], dtype=np.float64)
            target_markers = np.asarray(entry["tgt_marker"], dtype=np.float64)
            visibility = float(entry["vis"]) if "vis" in entry else len(target) / len(source)

        # Match the public LiverMatch in-vitro adapter: normalized-space voxel
        # size 0.04, independent centering, and the complete-liver radius as
        # the physical scale used to restore marker TRE in millimetres.
        source, scale = _norm_vox(source, self.voxel_size)
        target, _ = _norm_vox(target, self.voxel_size)
        source_center = np.mean(source[:, :3], axis=0)
        target_center = np.mean(target[:, :3], axis=0)
        source = (source - source_center) / scale
        target = (target - target_center) / scale
        source_markers = (source_markers - source_center) / scale
        target_markers = (target_markers - target_center) / scale

        oracle_transform = _rigid_transform(source_markers, target_markers)
        oracle_aligned_markers = (
            source_markers @ oracle_transform[:3, :3].T + oracle_transform[:3, 3]
        )
        oracle_rms_tre = np.sqrt(
            np.mean(np.sum((oracle_aligned_markers - target_markers) ** 2, axis=1))
        ) * scale
        sample = _as_sample(
            original_index,
            reference=target,
            source=source,
            transform=oracle_transform,
            overlap=visibility,
            sample_name=sample_name,
        )
        sample.update(
            source_markers=source_markers.astype(np.float32),
            target_markers=target_markers.astype(np.float32),
            physical_scale=np.float32(scale),
            oracle_rms_tre_mm=np.float32(oracle_rms_tre),
        )
        return sample


class DeterministicValidationDataset(Dataset):
    """Small validation view that does not perturb training RNG state."""

    def __init__(self, dataset, size, seed):
        self.dataset = dataset
        self.size = size
        self.seed = seed

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        random.seed(self.seed + index)
        np.random.seed(self.seed + index)
        try:
            return self.dataset[index]
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)


def _as_sample(index, reference, source, transform, overlap, sample_name):
    reference = np.ascontiguousarray(reference, dtype=np.float32)
    source = np.ascontiguousarray(source, dtype=np.float32)
    return {
        "scene_name": "P2P_in_silico",
        "ref_frame": sample_name,
        "src_frame": "complete_liver",
        "sample_name": sample_name,
        "index": int(index),
        "overlap": np.float32(overlap),
        "ref_points": reference,
        "src_points": source,
        "ref_feats": np.ones((len(reference), 1), dtype=np.float32),
        "src_feats": np.ones((len(source), 1), dtype=np.float32),
        "transform": np.asarray(transform, dtype=np.float32),
    }


def _calibrate(cfg, dataset):
    return calibrate_neighbors_stack_mode(
        dataset,
        registration_collate_fn_stack_mode,
        cfg.backbone.num_stages,
        cfg.backbone.init_voxel_size,
        cfg.backbone.init_radius,
        sample_threshold=int(cfg.data.neighbor_calibration_samples),
    )


def _build_loader(cfg, dataset, limits, train, distributed=False):
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
        drop_last=False,
        distributed=distributed,
    )


def train_valid_data_loader(cfg, distributed=False):
    train_dataset = LiverTask3TrainDataset(cfg)
    limits = _calibrate(cfg, train_dataset)
    train_loader = _build_loader(cfg, train_dataset, limits, True, distributed)
    validation_size = min(int(cfg.data.validation_size), len(train_dataset))
    validation_dataset = DeterministicValidationDataset(
        train_dataset, validation_size, int(cfg.seed) + 1000003
    )
    valid_loader = _build_loader(cfg, validation_dataset, limits, False)
    return train_loader, valid_loader, limits


def test_data_loader(cfg, noise_mm=None, dataset_name="in_silico"):
    calibration_dataset = LiverTask3TrainDataset(cfg)
    limits = _calibrate(cfg, calibration_dataset)
    if dataset_name == "in_vitro":
        if noise_mm is not None:
            raise ValueError("The official in-vitro Table VII protocol has no synthetic noise option")
        dataset = LiverTask3InVitroTestDataset(cfg)
    elif dataset_name == "in_silico":
        dataset = LiverTask3TestDataset(cfg, noise_mm=noise_mm, rotate_target=True)
    else:
        raise ValueError(f"Unknown test dataset: {dataset_name}")
    return _build_loader(cfg, dataset, limits, False), limits