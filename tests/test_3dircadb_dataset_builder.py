import json
from pathlib import Path

import numpy as np

from tools.build_3dircadb_low_overlap import (
    build_dataset,
    make_nested_pair,
    read_ascii_stl,
    sample_mesh_surface,
    validate_dataset,
)


def _write_stl(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        """solid liver
facet normal 0 0 1
 outer loop
  vertex 0 0 0
  vertex 10 0 0
  vertex 0 10 0
 endloop
endfacet
facet normal 0 0 0
 outer loop
  vertex 1 1 1
  vertex 1 1 1
  vertex 1 1 1
 endloop
endfacet
endsolid liver
""",
        encoding="ascii",
    )


def test_ascii_stl_sampling_ignores_degenerate_facets_and_is_repeatable(tmp_path):
    path = tmp_path / "liver.stl"
    _write_stl(path)
    triangles = read_ascii_stl(path)
    assert triangles.shape == (2, 3, 3)

    first = sample_mesh_surface(triangles, 50, np.random.default_rng(17))
    second = sample_mesh_surface(triangles, 50, np.random.default_rng(17))

    assert np.array_equal(first, second)
    assert np.isfinite(first).all()
    assert np.all(first[:, 2] == 0.0)


def test_nested_pair_is_exact_and_repeatable():
    rng = np.random.default_rng(5)
    source = rng.normal(size=(1000, 3))
    first = make_nested_pair(source, case_id=3, pair_id=7, seed=20260822)
    second = make_nested_pair(source, case_id=3, pair_id=7, seed=20260822)

    assert np.array_equal(first[0.20]["crop_indices"], second[0.20]["crop_indices"])
    assert set(first[0.20]["crop_indices"]).issubset(first[0.30]["crop_indices"])
    assert np.array_equal(first[0.20]["transform"], first[0.30]["transform"])
    positions = {int(value): i for i, value in enumerate(first[0.30]["crop_indices"])}
    selected = [positions[int(value)] for value in first[0.20]["crop_indices"]]
    assert np.array_equal(first[0.20]["noise"], first[0.30]["noise"][selected])
    assert len(first[0.20]["crop_indices"]) == 200
    assert len(first[0.30]["crop_indices"]) == 300
    rotation = first[0.20]["rotation"]
    np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-12)
    np.testing.assert_allclose(np.linalg.det(rotation), 1.0, atol=1e-12)
    assert 25.0 <= first[0.20]["rotation_angle_deg"] <= 90.0
    assert np.all(np.abs(first[0.20]["translation"]) <= 20.0)


def test_small_build_is_reproducible_and_valid(tmp_path):
    input_root = tmp_path / "input"
    _write_stl(input_root / "3Dircadb1.1" / "MESHES_VTK" / "liver.stl")
    output_a = tmp_path / "output-a"
    output_b = tmp_path / "output-b"
    kwargs = dict(
        input_root=input_root,
        seed=20260822,
        surface_points=100,
        pairs_per_case=1,
        case_ids=[1],
    )

    build_dataset(output_root=output_a, **kwargs)
    build_dataset(output_root=output_b, **kwargs)
    report = validate_dataset(output_a)

    assert report == {"cases": 1, "pairs": 1, "samples": 2, "vis020": 1, "vis030": 1}
    assert json.loads(json.dumps(report)) == report
    manifest_a = json.loads((output_a / "manifest.json").read_text())
    manifest_b = json.loads((output_b / "manifest.json").read_text())
    assert manifest_a == manifest_b
    for row in manifest_a["samples"]:
        with np.load(output_a / row["path"], allow_pickle=False) as a, np.load(
            output_b / row["path"], allow_pickle=False
        ) as b:
            assert a.files == b.files
            for key in a.files:
                assert np.array_equal(a[key], b[key])


def test_default_build_creates_one_nested_pair_per_case(tmp_path):
    input_root = tmp_path / "input"
    for case_id in (1, 2):
        _write_stl(input_root / f"3Dircadb1.{case_id}" / "MESHES_VTK" / "liver.stl")

    output = tmp_path / "output"
    build_dataset(
        input_root=input_root,
        output_root=output,
        surface_points=100,
        case_ids=[1, 2],
    )

    assert validate_dataset(output) == {
        "cases": 2, "pairs": 2, "samples": 4, "vis020": 2, "vis030": 2,
    }
