#!/usr/bin/env python3
"""Normalize heterogeneous selected-case predictions into one cache schema."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


def normalize_transform(value: Any, index: int | None = None) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.ndim == 3:
        if index is None:
            raise ValueError("A batch transform requires an index")
        if index < 0 or index >= len(matrix):
            raise ValueError("Transform batch index is out of range")
        matrix = matrix[index]
    if matrix.shape == (3, 4):
        matrix = np.vstack([matrix, [0.0, 0.0, 0.0, 1.0]])
    if matrix.shape != (4, 4):
        raise ValueError(f"Transform shape must be 3x4 or 4x4, got {matrix.shape}")
    if not np.isfinite(matrix).all():
        raise ValueError("Transform must contain only finite values")
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-8):
        raise ValueError("Transform must have a homogeneous [0,0,0,1] last row")
    return matrix


def _load_json(value: Any) -> dict:
    if isinstance(value, Mapping):
        return dict(value)
    return json.loads(Path(value).read_text(encoding="utf-8"))


def _canonical_path(value: str) -> str:
    return str(Path(value).expanduser().resolve(strict=False))


def _safe_sample(value: str) -> str:
    stem = Path(value).stem
    safe = "".join(char if char.isalnum() or char in "-_" else "_" for char in stem)
    return safe or "sample"


def _row_for_case(rows: Sequence[Mapping], case: Mapping) -> Mapping:
    sample = str(case["sample"])
    matches = [
        row for row in rows
        if str(row.get("sample", row.get("sample_name", ""))) == sample
    ]
    if len(matches) == 1:
        return matches[0]
    subset_index = int(case["subset_index"])
    matches = [row for row in rows if int(row.get("index", -1)) == subset_index]
    if len(matches) != 1:
        raise ValueError(f"Expected one summary row for {sample}, found {len(matches)}")
    return matches[0]


def _artifact_transform(
    row: Mapping, artifacts: Mapping, subset_index: int
) -> np.ndarray | None:
    embedded = row.get("estimated_transform")
    if embedded is not None:
        return normalize_transform(embedded)
    if "transforms" in artifacts or "transforms_path" in artifacts:
        value = artifacts.get("transforms")
        if value is None:
            value = np.load(Path(artifacts["transforms_path"]), allow_pickle=False)
        batch = np.asarray(value)
        expected = int(artifacts.get("expected_count", len(artifacts.get("selection", []))))
        if expected and len(batch) != expected:
            raise ValueError(f"Transform count {len(batch)} does not match expected count {expected}")
        return normalize_transform(batch, index=subset_index)
    if "prediction_dir" in artifacts:
        path = Path(artifacts["prediction_dir"]) / f"{subset_index:05d}.npz"
        if not path.is_file():
            raise ValueError(f"Missing prediction file: {path}")
        with np.load(path, allow_pickle=False) as payload:
            return normalize_transform(payload["estimated_transform"])
    return None


def collect_method_predictions(
    method: str,
    dataset: str,
    selection: Sequence[Mapping],
    artifacts: Mapping,
    output_root: Path,
) -> list[dict]:
    summary = _load_json(artifacts["summary"])
    checkpoint = str(summary.get("checkpoint", artifacts.get("checkpoint", "")))
    expected_checkpoint = str(artifacts.get("expected_checkpoint", ""))
    if expected_checkpoint and _canonical_path(checkpoint) != _canonical_path(expected_checkpoint):
        raise ValueError(
            f"Summary checkpoint {checkpoint!r} does not match expected checkpoint "
            f"{expected_checkpoint!r}"
        )
    rows = list(summary.get("samples", []))
    if len(rows) != len(selection):
        raise ValueError(
            f"Summary sample count {len(rows)} does not match selection count {len(selection)}"
        )
    artifact_map = dict(artifacts)
    artifact_map["expected_count"] = len(selection)
    artifact_map["selection"] = selection
    records = []
    for case in selection:
        subset_index = int(case["subset_index"])
        sample = str(case["sample"])
        row = _row_for_case(rows, case)
        status = str(row.get("status", row.get("solver_status", "ok")))
        transform = _artifact_transform(row, artifact_map, subset_index)
        if status == "ok" and transform is None:
            raise ValueError(f"Successful prediction has no transform: {method}/{sample}")
        rms = float(row.get("rms_tre_mm", row.get("RMS_TRE_mm")))
        if not math.isfinite(rms):
            raise ValueError(f"Non-finite RMS-TRE: {method}/{sample}")
        metadata = {
            "method": method,
            "dataset": dataset,
            "sample": sample,
            "subset_index": subset_index,
            "original_index": case.get("original_index"),
            "status": status,
            "error": row.get("error", row.get("solver_error")),
            "checkpoint": checkpoint,
            "rms_tre_mm": rms,
        }
        directory = Path(output_root) / "predictions" / dataset / _safe_sample(sample)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{method}.npz"
        arrays = {
            "rms_tre_mm": np.asarray(rms, dtype=np.float64),
            "status": np.asarray(status),
            "metadata_json": np.asarray(json.dumps(metadata, ensure_ascii=False)),
        }
        if status == "ok":
            arrays["estimated_transform"] = transform
        np.savez_compressed(path, **arrays)
        records.append({**metadata, "prediction_path": str(path.resolve())})

    summary_path = Path(output_root) / "predictions" / dataset / f"{method}_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps({"method": method, "dataset": dataset, "samples": records}, indent=2,
                   ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", required=True)
    parser.add_argument("--dataset", choices=("in_silico", "in_vitro"), required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-checkpoint", default="")
    parser.add_argument("--prediction-dir", type=Path)
    parser.add_argument("--transforms-path", type=Path)
    args = parser.parse_args()
    selection_payload = json.loads(args.selection.read_text(encoding="utf-8"))
    selection = selection_payload["datasets"][args.dataset]["cases"]
    artifacts = {
        "summary": args.summary,
        "expected_checkpoint": args.expected_checkpoint,
    }
    if args.prediction_dir:
        artifacts["prediction_dir"] = args.prediction_dir
    if args.transforms_path:
        artifacts["transforms_path"] = args.transforms_path
    records = collect_method_predictions(
        args.method, args.dataset, selection, artifacts, args.output_root
    )
    print(f"Collected {len(records)} predictions for {args.method}/{args.dataset}")


if __name__ == "__main__":
    main()
