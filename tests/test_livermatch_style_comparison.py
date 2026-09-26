import json
from pathlib import Path

import numpy as np

from visualization.livermatch_style_comparison import (
    COLUMN_LABELS,
    METHOD_KEYS,
    SOURCE_COLOR,
    TARGET_COLOR,
    apply_transform,
    build_panel_specs,
    marker_rms_tre,
    render_dataset_comparison,
)


def transform(tx=0.0):
    matrix = np.eye(4)
    matrix[0, 3] = tx
    return matrix


def test_layout_colors_transform_metric_and_shared_camera():
    source = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    target = source + np.array([1.0, 0.0, 0.0])
    predictions = {method: {"status": "ok", "transform": transform(1.0)} for method in METHOD_KEYS}
    specs, camera = build_panel_specs(
        source, target, source, target, transform(1.0), predictions, physical_scale=2.0
    )

    assert COLUMN_LABELS == (
        "Initial Position", "Ours", "GeoTransformer", "CASTv2", "DFAT",
        "Lepard", "Lepard+P2P", "PARENet", "LiverMatch",
        "LiverMatch+P2P", "Go-ICP", "Ground Truth",
    )
    assert len(specs) == 12
    np.testing.assert_allclose(SOURCE_COLOR * 255, [0, 150, 255])
    np.testing.assert_allclose(TARGET_COLOR * 255, [254, 92, 92])
    np.testing.assert_allclose(apply_transform(source, transform(1.0)), target)
    assert marker_rms_tre(source, target, transform(0.0), 2.0) == 2.0
    assert all(spec["camera"] is camera for spec in specs)
    assert specs[1]["title"] == "Ours\nRMS-TRE: 0.00 mm"


def _write_vitro_case(root: Path, name: str):
    data = root / "subsets/in_vitro/Rigid_test_data"
    data.mkdir(parents=True, exist_ok=True)
    source = np.array(
        [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]]
    )
    target = source + np.array([1.0, 0.0, 0.0])
    np.savez(
        data / name,
        src_vs=source,
        tgt_vs=target,
        src_marker=source,
        tgt_marker=target,
        vis=np.asarray(0.3),
    )


def _write_prediction(root: Path, sample: str, method: str, *, failed=False):
    directory = root / "predictions/in_vitro" / Path(sample).stem
    directory.mkdir(parents=True, exist_ok=True)
    metadata = {
        "method": method,
        "dataset": "in_vitro",
        "sample": sample,
        "status": "timeout" if failed else "ok",
        "error": "slow" if failed else None,
        "checkpoint": f"/{method}.pth",
    }
    arrays = {
        "rms_tre_mm": np.asarray(0.0),
        "status": np.asarray(metadata["status"]),
        "metadata_json": np.asarray(json.dumps(metadata)),
    }
    if not failed:
        arrays["estimated_transform"] = transform(0.5)
    np.savez_compressed(directory / f"{method}.npz", **arrays)


def test_failed_method_and_pyvista_error_fall_back_to_complete_software_figure(
    tmp_path: Path, monkeypatch
):
    cases = []
    for index in range(3):
        sample = f"case_{index}.npz"
        _write_vitro_case(tmp_path, sample)
        cases.append({"subset_index": index, "sample": sample, "original_index": index})
        for method in METHOD_KEYS:
            _write_prediction(
                tmp_path, sample, method, failed=(index == 1 and method == "dfat")
            )

    def renderer_failure(*args, **kwargs):
        raise RuntimeError("VTK unavailable")

    monkeypatch.setattr(
        "visualization.livermatch_style_comparison._render_panel_pyvista",
        renderer_failure,
    )
    result = render_dataset_comparison(
        "in_vitro", cases, tmp_path / "predictions", tmp_path, backend="pyvista", dpi=50
    )

    assert result["panel_count"] == 36
    assert result["backend"] == "software"
    assert Path(result["png"]).is_file()
    assert Path(result["pdf"]).is_file()
    failed = [panel for panel in result["panels"] if panel["status"] != "ok"]
    assert len(failed) == 1
    assert "FAILED" in failed[0]["title"]
    assert failed[0]["error"] == "slow"
