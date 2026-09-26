from pathlib import Path

import numpy as np
import pytest

from tools.prepare_visibility_best3 import build_subset, select_best_cases


def test_select_best_cases_filters_invalid_and_breaks_ties_stably():
    summary = {
        "samples": [
            {"sample": "z.npz", "original_index": 9, "index": 0, "visibility": 0.2, "rms_tre_mm": 0.4},
            {"sample": "b.npz", "original_index": 4, "index": 1, "visibility": 0.3, "rms_tre_mm": 0.2},
            {"sample": "a.npz", "original_index": 4, "index": 2, "visibility": 0.3, "rms_tre_mm": 0.2},
            {"sample": "c.npz", "original_index": 2, "index": 3, "visibility": 0.3, "rms_tre_mm": 0.2},
            {"sample": "bad.npz", "original_index": 0, "index": 4, "visibility": 0.2, "rms_tre_mm": float("nan")},
            {"sample": "failed.npz", "original_index": 1, "index": 5, "visibility": 0.2, "rms_tre_mm": 0.1, "status": "failed"},
            {"sample": "c.npz", "original_index": 2, "index": 6, "visibility": 0.3, "rms_tre_mm": 0.1},
        ]
    }

    selected = select_best_cases(summary, top_k=3)

    assert [(row["sample"], row["original_index"]) for row in selected] == [
        ("c.npz", 2),
        ("a.npz", 4),
        ("b.npz", 4),
    ]
    assert [row["subset_index"] for row in selected] == [0, 1, 2]


@pytest.mark.parametrize("dataset", ["in_silico", "in_vitro"])
def test_build_subset_preserves_name_statistics_and_original_index(tmp_path: Path, dataset: str):
    source = tmp_path / "source"
    output = tmp_path / "subset"
    names = np.asarray(["one.npz", "two.npz", "three.npz"])
    if dataset == "in_silico":
        metadata = source / "Deform_mesh_npz_test"
        data_dir = metadata / "Test"
        metadata.mkdir(parents=True)
        data_dir.mkdir()
        np.savez(metadata / "list.npz", test=names)
        np.savez(metadata / "stat_svd.npz", vis=[0.21, 0.31, 0.22], deform=[1.0, 2.0, 3.0])
    else:
        data_dir = source / "Rigid_test_data"
        data_dir.mkdir(parents=True)
        np.save(source / "rigid_list.npy", names)
        np.savez(source / "stat.npz", vis=[0.21, 0.31, 0.22], deform=[1.0, 2.0, 3.0])
    np.save(source / "original_indices.npy", [10, 11, 12])
    cases = [
        {"sample": "three.npz", "original_index": 12},
        {"sample": "one.npz", "original_index": 10},
    ]

    manifest = build_subset(dataset, source, cases, output)

    if dataset == "in_silico":
        written_names = np.load(output / "Deform_mesh_npz_test/list.npz")["test"]
        stats = np.load(output / "Deform_mesh_npz_test/stat_svd.npz")
        linked_data = output / "Deform_mesh_npz_test/Test"
    else:
        written_names = np.load(output / "rigid_list.npy")
        stats = np.load(output / "stat.npz")
        linked_data = output / "Rigid_test_data"
    assert written_names.tolist() == ["three.npz", "one.npz"]
    assert stats["vis"].tolist() == [0.22, 0.21]
    assert stats["deform"].tolist() == [3.0, 1.0]
    assert np.load(output / "original_indices.npy").tolist() == [12, 10]
    assert linked_data.is_symlink()
    assert manifest["samples"] == ["three.npz", "one.npz"]


def test_build_subset_rejects_unknown_sample(tmp_path: Path):
    source = tmp_path / "source"
    metadata = source / "Deform_mesh_npz_test"
    (metadata / "Test").mkdir(parents=True)
    np.savez(metadata / "list.npz", test=["one.npz"])
    np.savez(metadata / "stat_svd.npz", vis=[0.2], deform=[1.0])
    np.save(source / "original_indices.npy", [10])

    with pytest.raises(ValueError, match="missing.npz"):
        build_subset(
            "in_silico",
            source,
            [{"sample": "missing.npz", "original_index": 99}],
            tmp_path / "subset",
        )
