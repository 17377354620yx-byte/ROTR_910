"""Shared contracts for the deterministic 3D-IRCADb rigid benchmark."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np

from tools.p2p_benchmark_metrics import pose_metrics, strict_success_metrics


@dataclass(frozen=True)
class SampleRecord:
    root: Path
    sample_id: str
    path: Path
    case_id: int
    pair_id: int
    visibility: float
    dataset_id: str

    def load(self) -> dict[str, np.ndarray]:
        with np.load(self.path, allow_pickle=False) as data:
            return {key: np.asarray(data[key]) for key in data.files}


@dataclass(frozen=True)
class NormalizedPair:
    source: np.ndarray
    target: np.ndarray
    source_center: np.ndarray
    target_center: np.ndarray
    scale: float


def _manifest(root: Path) -> tuple[dict, str]:
    path = Path(root) / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    raw = path.read_bytes()
    manifest = json.loads(raw)
    return manifest, hashlib.sha256(raw).hexdigest()


def load_samples(
    root: Path,
    visibility: float | str | None = None,
    limit: int = 0,
    sample_ids: Iterable[str] | None = None,
) -> list[SampleRecord]:
    """Load and validate a stable ordered selection from the canonical manifest."""
    root = Path(root).resolve()
    manifest, dataset_id = _manifest(root)
    rows = manifest.get("samples")
    if not isinstance(rows, list) or manifest.get("sample_count") != len(rows):
        raise ValueError("Manifest sample count does not match samples")
    ids = [str(row["sample_id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate sample ID in manifest")
    selected_ids = None if sample_ids is None else set(map(str, sample_ids))
    if selected_ids is not None:
        unavailable = selected_ids - set(ids)
        if unavailable:
            raise ValueError(f"Unknown sample IDs: {sorted(unavailable)}")
    requested_visibility = None
    if visibility not in (None, "all"):
        requested_visibility = float(visibility)
        if requested_visibility not in (0.2, 0.3):
            raise ValueError("visibility must be 0.2, 0.3, or all")
    records = []
    for row in sorted(rows, key=lambda item: str(item["sample_id"])):
        sample_id = str(row["sample_id"])
        value = float(row["visibility"])
        if selected_ids is not None and sample_id not in selected_ids:
            continue
        if requested_visibility is not None and not np.isclose(value, requested_visibility):
            continue
        path = root / row["path"]
        if not path.is_file():
            raise FileNotFoundError(path)
        records.append(SampleRecord(
            root=root,
            sample_id=sample_id,
            path=path,
            case_id=int(row["case_id"]),
            pair_id=int(row["pair_id"]),
            visibility=value,
            dataset_id=dataset_id,
        ))
    if limit < 0:
        raise ValueError("limit must be nonnegative")
    return records[:limit] if limit else records


def normalize_pair(source_mm: np.ndarray, target_mm: np.ndarray) -> NormalizedPair:
    """Independently center two clouds and scale both by complete-source radius."""
    source = np.asarray(source_mm, dtype=np.float64)
    target = np.asarray(target_mm, dtype=np.float64)
    for name, points in (("source", source), ("target", target)):
        if points.ndim != 2 or points.shape[1] != 3 or not len(points):
            raise ValueError(f"{name} must be a non-empty (N,3) array")
        if not np.isfinite(points).all():
            raise ValueError(f"{name} contains non-finite coordinates")
    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    scale = float(np.linalg.norm(source - source_center, axis=1).max())
    if not np.isfinite(scale) or scale <= np.finfo(np.float64).eps:
        raise ValueError("source point cloud has zero radius")
    return NormalizedPair(
        source=np.ascontiguousarray((source - source_center) / scale),
        target=np.ascontiguousarray((target - target_center) / scale),
        source_center=source_center,
        target_center=target_center,
        scale=scale,
    )


def validate_transform(transform: np.ndarray, *, atol: float = 1e-5) -> np.ndarray:
    transform = np.asarray(transform, dtype=np.float64)
    if transform.shape != (4, 4):
        raise ValueError("transform must have shape (4,4)")
    if not np.isfinite(transform).all():
        raise ValueError("transform must contain finite values")
    if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=atol):
        raise ValueError("transform has an invalid homogeneous row")
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=atol) or not np.isclose(
        np.linalg.det(rotation), 1.0, atol=atol
    ):
        raise ValueError("transform rotation is not in SO(3)")
    return transform


def transform_to_network(transform_mm: np.ndarray, pair: NormalizedPair) -> np.ndarray:
    transform_mm = validate_transform(transform_mm)
    output = transform_mm.copy()
    rotation = transform_mm[:3, :3]
    output[:3, 3] = (
        rotation @ pair.source_center + transform_mm[:3, 3] - pair.target_center
    ) / pair.scale
    return output


def transform_to_mm(transform_network: np.ndarray, pair: NormalizedPair) -> np.ndarray:
    transform_network = validate_transform(transform_network)
    output = transform_network.copy()
    rotation = transform_network[:3, :3]
    output[:3, 3] = (
        pair.scale * transform_network[:3, 3]
        + pair.target_center
        - rotation @ pair.source_center
    )
    return output


def rigid_metrics_mm(
    source_mm: np.ndarray,
    target_mm: np.ndarray,
    estimated_transform_mm: np.ndarray,
    gt_transform_mm: np.ndarray,
) -> dict[str, float]:
    estimate = validate_transform(estimated_transform_mm)
    ground_truth = validate_transform(gt_transform_mm)
    result = pose_metrics(source_mm, target_mm, estimate, ground_truth, 1.0)
    strict = strict_success_metrics(result["RMSE"], result["RRE"], result["RTE"])
    result.update(strict)
    result["RR"] = strict["RR_5deg_5mm"]
    return {key: float(value) for key, value in result.items()}


def _sample_metadata(sample: SampleRecord | Mapping) -> dict:
    metadata = {
        "sample_id": str(sample.sample_id if isinstance(sample, SampleRecord) else sample["sample_id"]),
        "case_id": int(sample.case_id if isinstance(sample, SampleRecord) else sample["case_id"]),
        "pair_id": int(sample.pair_id if isinstance(sample, SampleRecord) else sample["pair_id"]),
        "visibility": float(sample.visibility if isinstance(sample, SampleRecord) else sample["visibility"]),
    }
    dataset_id = sample.dataset_id if isinstance(sample, SampleRecord) else sample.get("dataset_id")
    if dataset_id is not None:
        metadata["dataset_id"] = str(dataset_id)
    return metadata


def _json_safe(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


class PredictionWriter:
    """Write per-sample physical transforms and a complete method summary."""

    def __init__(self, method: str, output_dir: Path, expected_ids: Iterable[str]):
        self.method = str(method)
        self.output_dir = Path(output_dir)
        self.prediction_dir = self.output_dir / "predictions"
        self.prediction_dir.mkdir(parents=True, exist_ok=True)
        self.expected_ids = list(map(str, expected_ids))
        if len(self.expected_ids) != len(set(self.expected_ids)):
            raise ValueError("expected_ids contains duplicates")
        self.records: dict[str, dict] = {}

    def _reserve(self, metadata: dict) -> None:
        sample_id = metadata["sample_id"]
        if sample_id not in set(self.expected_ids):
            raise ValueError(f"Unexpected sample ID: {sample_id}")
        if sample_id in self.records:
            raise ValueError(f"Duplicate prediction for sample ID: {sample_id}")

    def write_success(
        self,
        sample: SampleRecord | Mapping,
        estimated_transform: np.ndarray,
        gt_transform: np.ndarray,
        runtime_seconds: float,
        source_mm: np.ndarray,
        target_mm: np.ndarray,
        *,
        pir: float | None = None,
        ir: float | None = None,
        metadata: Mapping | None = None,
    ) -> dict:
        sample_meta = _sample_metadata(sample)
        self._reserve(sample_meta)
        estimate = validate_transform(estimated_transform)
        gt = validate_transform(gt_transform)
        metrics = rigid_metrics_mm(source_mm, target_mm, estimate, gt)
        row = {
            **sample_meta,
            "method": self.method,
            "status": "ok",
            "error": None,
            "runtime_seconds": float(runtime_seconds),
            "estimated_transform": estimate.tolist(),
            "gt_transform": gt.tolist(),
            "PIR": None if pir is None else float(pir),
            "IR": None if ir is None else float(ir),
            **metrics,
            "metadata": _json_safe(dict(metadata or {})),
        }
        path = self.prediction_dir / f"{sample_meta['sample_id']}.npz"
        np.savez_compressed(
            path,
            estimated_transform=estimate,
            gt_transform=gt,
            sample_id=np.asarray(sample_meta["sample_id"]),
            case_id=np.int64(sample_meta["case_id"]),
            pair_id=np.int64(sample_meta["pair_id"]),
            visibility=np.float64(sample_meta["visibility"]),
            runtime_seconds=np.float64(runtime_seconds),
            status=np.asarray("ok"),
        )
        row["prediction_path"] = str(path.resolve())
        self.records[sample_meta["sample_id"]] = row
        return row

    def write_failure(
        self,
        sample: SampleRecord | Mapping,
        gt_transform: np.ndarray,
        runtime_seconds: float,
        *,
        status: str = "error",
        error: str,
        metadata: Mapping | None = None,
    ) -> dict:
        sample_meta = _sample_metadata(sample)
        self._reserve(sample_meta)
        if status == "ok":
            raise ValueError("failure status cannot be 'ok'")
        gt = validate_transform(gt_transform)
        row = {
            **sample_meta,
            "method": self.method,
            "status": str(status),
            "error": str(error),
            "runtime_seconds": float(runtime_seconds),
            "estimated_transform": None,
            "gt_transform": gt.tolist(),
            "PIR": None,
            "IR": None,
            "SR_5mm": 0.0,
            "RR": 0.0,
            "RR_5deg_5mm": 0.0,
            "metadata": _json_safe(dict(metadata or {})),
            "prediction_path": None,
        }
        self.records[sample_meta["sample_id"]] = row
        return row

    def finalize(self) -> dict:
        expected, actual = set(self.expected_ids), set(self.records)
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        if missing:
            raise ValueError(f"Missing predictions: {missing[:10]}")
        if extra:
            raise ValueError(f"Unexpected predictions: {extra[:10]}")
        rows = [self.records[sample_id] for sample_id in self.expected_ids]
        dataset_ids = {row.get("dataset_id") for row in rows if row.get("dataset_id")}
        if len(dataset_ids) > 1:
            raise ValueError(f"Predictions mix dataset manifests: {sorted(dataset_ids)}")
        ok = sum(row["status"] == "ok" for row in rows)
        payload = {
            "schema_version": 1,
            "method": self.method,
            "dataset_id": next(iter(dataset_ids)) if dataset_ids else None,
            "counts": {"total": len(rows), "ok": int(ok), "failed": int(len(rows) - ok)},
            "samples": rows,
        }
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "summary.json").write_text(
            json.dumps(_json_safe(payload), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        csv_fields = [
            "sample_id", "dataset_id", "case_id", "pair_id", "visibility", "method", "status",
            "error", "runtime_seconds", "PIR", "IR", "RRE", "RTE", "RMSE",
            "SR_5mm", "RR", "RR_5deg_5mm", "r_rmse", "r_mae", "t_rmse",
            "t_mae", "chamfer_dist", "prediction_path",
        ]
        with (self.output_dir / "samples.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=csv_fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        return payload
