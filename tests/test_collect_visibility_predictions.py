import json
from pathlib import Path

import numpy as np
import pytest

from tools.collect_visibility_predictions import (
    collect_method_predictions,
    normalize_transform,
)
from tools.evaluate_livermatch_visibility import prediction_export_fields


def transform(tx=0.0):
    value = np.eye(4)
    value[0, 3] = tx
    return value


def test_normalize_transform_accepts_rigid_matrix_shapes_and_batch_index():
    np.testing.assert_allclose(normalize_transform(transform(1.0)), transform(1.0))
    np.testing.assert_allclose(normalize_transform(transform(2.0)[:3]), transform(2.0))
    batch = np.stack([transform(3.0), transform(4.0)])
    np.testing.assert_allclose(normalize_transform(batch, index=1), transform(4.0))


@pytest.mark.parametrize(
    "value,match",
    [
        (np.full((4, 4), np.nan), "finite"),
        (np.diag([1.0, 1.0, 1.0, 2.0]), "homogeneous"),
        (np.zeros((2, 2)), "shape"),
    ],
)
def test_normalize_transform_rejects_invalid_matrices(value, match):
    with pytest.raises(ValueError, match=match):
        normalize_transform(value)


def test_collect_rejects_count_and_checkpoint_mismatch(tmp_path: Path):
    selection = [{"subset_index": 0, "sample": "a.npz"}, {"subset_index": 1, "sample": "b.npz"}]
    summary = {
        "checkpoint": "/actual/model.pth",
        "samples": [
            {"index": 0, "sample": "a.npz", "rms_tre_mm": 1.0},
            {"index": 1, "sample": "b.npz", "rms_tre_mm": 2.0},
        ],
    }
    with pytest.raises(ValueError, match="checkpoint"):
        collect_method_predictions(
            "ours",
            "in_silico",
            selection,
            {"summary": summary, "transforms": np.stack([transform(), transform()]), "expected_checkpoint": "/expected/model.pth"},
            tmp_path,
        )
    with pytest.raises(ValueError, match="count"):
        collect_method_predictions(
            "ours",
            "in_silico",
            selection,
            {"summary": {**summary, "checkpoint": "/expected/model.pth"}, "transforms": np.stack([transform()]), "expected_checkpoint": "/expected/model.pth"},
            tmp_path,
        )


def test_collect_writes_success_and_failure_without_identity_fallback(tmp_path: Path):
    selection = [{"subset_index": 0, "sample": "ok.npz"}, {"subset_index": 1, "sample": "bad.npz"}]
    summary = {
        "checkpoint": "/model.pth",
        "samples": [
            {"index": 0, "sample": "ok.npz", "rms_tre_mm": 1.25, "status": "ok", "estimated_transform": transform(2.0).tolist()},
            {"index": 1, "sample": "bad.npz", "rms_tre_mm": 99.0, "status": "timeout", "estimated_transform": None, "error": "slow"},
        ],
    }

    records = collect_method_predictions(
        "goicp",
        "in_vitro",
        selection,
        {"summary": summary, "expected_checkpoint": "/model.pth"},
        tmp_path,
    )

    assert [row["status"] for row in records] == ["ok", "timeout"]
    success = np.load(Path(records[0]["prediction_path"]), allow_pickle=False)
    failed = np.load(Path(records[1]["prediction_path"]), allow_pickle=False)
    np.testing.assert_allclose(success["estimated_transform"], transform(2.0))
    assert "estimated_transform" not in failed.files
    metadata = json.loads(str(success["metadata_json"]))
    assert metadata["checkpoint"] == "/model.pth"


def test_livermatch_prediction_export_serializes_success_and_failure():
    success = prediction_export_fields({"estimated_transform": transform(3.0)})
    failed = prediction_export_fields({}, status="error", error="model failed")
    assert success == {"status": "ok", "error": None, "estimated_transform": transform(3.0).tolist()}
    assert failed == {"status": "error", "error": "model failed", "estimated_transform": None}


def test_collect_accepts_lepard_metric_and_sample_field_names(tmp_path: Path):
    selection = [{"subset_index": 0, "sample": "nested/case.npz"}]
    summary = {
        "checkpoint": "/lepard.pth",
        "samples": [{
            "index": 0,
            "sample_name": "nested/case.npz",
            "RMS_TRE_mm": 1.5,
            "estimated_transform": transform(1.0).tolist(),
        }],
    }

    records = collect_method_predictions(
        "lepard",
        "in_silico",
        selection,
        {"summary": summary, "expected_checkpoint": "/lepard.pth"},
        tmp_path,
    )

    assert records[0]["sample"] == "nested/case.npz"
    assert records[0]["rms_tre_mm"] == 1.5
