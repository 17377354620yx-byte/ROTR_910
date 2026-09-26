import json
from pathlib import Path

import numpy as np
import pytest

from tools.ircadb_benchmark import (
    PredictionWriter,
    load_samples,
    normalize_pair,
    rigid_metrics_mm,
    transform_to_mm,
    transform_to_network,
    validate_transform,
)
from tools.p2p_benchmark_metrics import strict_success_metrics


def _transform(angle_deg=31.0, translation=(4.0, -7.0, 2.5)):
    angle = np.deg2rad(angle_deg)
    rotation = np.asarray(
        [[np.cos(angle), -np.sin(angle), 0.0],
         [np.sin(angle), np.cos(angle), 0.0],
         [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    return transform


def _sample_tree(root: Path):
    folder = root / "visibility_0.20"
    folder.mkdir(parents=True)
    source = np.asarray([[0, 0, 0], [10, 0, 0], [0, 20, 0], [0, 0, 30]], dtype=np.float32)
    transform = _transform()
    target = source @ transform[:3, :3].T + transform[:3, 3]
    sample_id = "case01_pair00_vis020"
    path = folder / f"{sample_id}.npz"
    np.savez_compressed(
        path,
        src_points=source,
        ref_points=target,
        clean_ref_points=target,
        transform=transform,
        case_id=np.int64(1),
        pair_id=np.int64(0),
        visibility=np.float64(0.2),
        sample_id=np.asarray(sample_id),
        units=np.asarray("mm"),
    )
    manifest = {
        "parameters": {"schema_version": 1, "seed": 20260822},
        "sample_count": 1,
        "samples": [{
            "sample_id": sample_id,
            "path": path.relative_to(root).as_posix(),
            "case_id": 1,
            "pair_id": 0,
            "visibility": 0.2,
        }],
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return source.astype(np.float64), target.astype(np.float64), transform


def test_load_samples_filters_and_rejects_duplicate_ids(tmp_path):
    _sample_tree(tmp_path)
    records = load_samples(tmp_path, visibility=0.2, limit=1)
    assert [record.sample_id for record in records] == ["case01_pair00_vis020"]
    assert load_samples(tmp_path, visibility=0.3) == []
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["samples"].append(dict(manifest["samples"][0]))
    manifest["sample_count"] = 2
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Duplicate sample ID"):
        load_samples(tmp_path)


def test_independent_centering_transform_round_trip(tmp_path):
    source_mm, target_mm, gt_mm = _sample_tree(tmp_path)
    pair = normalize_pair(source_mm, target_mm)
    network_gt = transform_to_network(gt_mm, pair)
    recovered = transform_to_mm(network_gt, pair)
    np.testing.assert_allclose(recovered, gt_mm, atol=1e-8)
    np.testing.assert_allclose(pair.source.mean(axis=0), 0.0, atol=1e-8)
    np.testing.assert_allclose(pair.target.mean(axis=0), 0.0, atol=1e-8)


def test_transform_validation_rejects_nonrigid_and_nonfinite():
    invalid = np.eye(4)
    invalid[0, 0] = 2.0
    with pytest.raises(ValueError, match="rotation"):
        validate_transform(invalid)
    invalid = np.eye(4)
    invalid[0, 3] = np.nan
    with pytest.raises(ValueError, match="finite"):
        validate_transform(invalid)


def test_strict_thresholds_are_not_inclusive():
    assert strict_success_metrics(5.0, 0.0, 0.0)["SR_5mm"] == 0.0
    assert strict_success_metrics(0.0, 5.0, 0.0)["RR_5deg_5mm"] == 0.0
    assert strict_success_metrics(0.0, 0.0, 5.0)["RR_5deg_5mm"] == 0.0


def test_rigid_metrics_are_zero_for_ground_truth(tmp_path):
    source, target, gt = _sample_tree(tmp_path)
    metrics = rigid_metrics_mm(source, target, gt, gt)
    assert metrics["RRE"] == pytest.approx(0.0)
    assert metrics["RTE"] == pytest.approx(0.0)
    assert metrics["RMSE"] == pytest.approx(0.0)
    assert metrics["SR_5mm"] == 1.0
    assert metrics["RR_5deg_5mm"] == 1.0


def test_prediction_writer_requires_exactly_one_record_per_expected_id(tmp_path):
    source, target, gt = _sample_tree(tmp_path / "data")
    records = load_samples(tmp_path / "data")
    writer = PredictionWriter("demo", tmp_path / "predictions", [records[0].sample_id, "missing"])
    writer.write_success(records[0], gt, gt, 0.25, source, target)
    with pytest.raises(ValueError, match="Missing predictions"):
        writer.finalize()


def test_prediction_writer_serializes_success_and_explicit_failure(tmp_path):
    source, target, gt = _sample_tree(tmp_path / "data")
    record = load_samples(tmp_path / "data")[0]
    failed_id = "case01_pair01_vis020"
    writer = PredictionWriter("demo", tmp_path / "predictions", [record.sample_id, failed_id])
    success = writer.write_success(record, gt, gt, 0.25, source, target)
    writer.write_failure(
        {"sample_id": failed_id, "case_id": 1, "pair_id": 1, "visibility": 0.2},
        gt,
        1.0,
        status="timeout",
        error="Exceeded 1 second",
    )
    summary = writer.finalize()

    assert success["PIR"] is None and success["IR"] is None
    assert summary["dataset_id"] == record.dataset_id
    assert summary["counts"] == {"total": 2, "ok": 1, "failed": 1}
    failed = next(row for row in summary["samples"] if row["sample_id"] == failed_id)
    assert failed["status"] == "timeout"
    assert failed["estimated_transform"] is None
    saved = np.load(tmp_path / "predictions" / "predictions" / f"{record.sample_id}.npz")
    np.testing.assert_allclose(saved["estimated_transform"], gt)
