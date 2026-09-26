import json
from pathlib import Path

import numpy as np
import pytest

from visualization.ircadb_method_comparison import (
    LABELS,
    METHODS,
    load_case_transforms,
    pointwise_gt_error_mm,
    render_sample,
)
from tools.ircadb_benchmark import load_samples


def test_twelve_labels_in_approved_order():
    assert len(LABELS) == 12
    assert LABELS == ["Initial", *[label for _, label in METHODS], "Ground Truth"]


def test_complete_figure_requires_ten_predictions(tmp_path):
    with pytest.raises(ValueError, match="Missing prediction"):
        load_case_transforms("case01_pair00_vis020", tmp_path)


def test_error_range_is_in_physical_mm():
    source = np.asarray([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    estimate = np.eye(4)
    estimate[:3, 3] = [3.0, 0.0, 0.0]
    ground_truth = np.eye(4)
    ground_truth[:3, 3] = [1.0, 0.0, 0.0]
    error = pointwise_gt_error_mm(source, estimate, ground_truth)
    np.testing.assert_allclose(error, [2.0, 2.0])


def _synthetic_dataset(root: Path):
    rng = np.random.default_rng(9)
    source = rng.normal(size=(100, 3)) * 10
    gt = np.eye(4)
    gt[:3, 3] = [2, -1, 3]
    target = source[:30] + gt[:3, 3]
    sample_id = "case01_pair00_vis020"
    relative = Path("visibility_0.20") / f"{sample_id}.npz"
    (root / relative).parent.mkdir(parents=True)
    np.savez_compressed(root / relative, src_points=source, ref_points=target,
                        transform=gt, case_id=1, pair_id=0, visibility=0.2,
                        sample_id=np.asarray(sample_id), units=np.asarray("mm"))
    (root / "manifest.json").write_text(json.dumps({
        "parameters": {"schema_version": 1}, "sample_count": 1,
        "samples": [{"sample_id": sample_id, "path": relative.as_posix(),
                     "case_id": 1, "pair_id": 0, "visibility": 0.2}],
    }))
    return sample_id, gt


def test_synthetic_render_uses_explicit_viridis_and_shared_range(tmp_path):
    data_root = tmp_path / "data"
    sample_id, gt = _synthetic_dataset(data_root)
    result_root = tmp_path / "results"
    for method, _ in METHODS:
        directory = result_root / method
        directory.mkdir(parents=True)
        (directory / "summary.json").write_text(json.dumps({
            "samples": [{"sample_id": sample_id, "status": "ok",
                         "estimated_transform": gt.tolist()}],
        }))
    transforms = load_case_transforms(sample_id, result_root)
    record = load_samples(data_root)[0]
    payload = render_sample(record, transforms, tmp_path / "figures", dpi=40,
                            colormap="viridis")
    assert Path(payload["png"]).is_file()
    assert Path(payload["pdf"]).is_file()
    assert payload["colormap"] == "viridis"
    assert payload["shared_color_range_mm"][0] == 0.0
