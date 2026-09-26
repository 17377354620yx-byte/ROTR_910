from pathlib import Path
import subprocess
import sys

import numpy as np

from scripts.summarize_visibility_02_04 import MODELS
from tools.evaluate_goicp_visibility import (
    deterministic_subsample,
    load_p2p_sample,
    normalize_for_goicp,
    recover_source_to_target_transform,
    visualization_transform,
)


def test_deterministic_subsample_is_bounded_and_reproducible():
    points = np.arange(60, dtype=np.float64).reshape(20, 3)
    first = deterministic_subsample(points, max_points=7, seed=7351)
    second = deterministic_subsample(points, max_points=7, seed=7351)
    assert first.shape == (7, 3)
    np.testing.assert_array_equal(first, second)
    assert len(np.unique(first, axis=0)) == 7


def test_goicp_normalization_and_transform_recovery():
    source = np.array([[10.0, 0.0, 0.0], [12.0, 0.0, 0.0], [10.0, 2.0, 0.0]])
    target = np.array([[30.0, 4.0, 0.0], [30.0, 6.0, 0.0], [28.0, 4.0, 0.0]])
    normalized_source, normalized_target, source_center, target_center, scale = (
        normalize_for_goicp(source, target)
    )
    assert np.max(np.abs(normalized_source)) <= 1.0
    assert np.max(np.abs(normalized_target)) <= 1.0

    rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    translation = np.array([0.1, -0.2, 0.3])
    transform = recover_source_to_target_transform(
        rotation, translation, source_center, target_center, scale
    )
    expected_translation = scale * translation + target_center - rotation @ source_center
    np.testing.assert_allclose(transform[:3, :3], rotation)
    np.testing.assert_allclose(transform[:3, 3], expected_translation)


def test_in_silico_loader_adds_saved_noise_before_target_rotation(tmp_path: Path):
    rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    source = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    target = np.array([[1.0, 0.0, 0.0], [3.0, 0.0, 0.0], [1.0, 2.0, 0.0]])
    noise = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    source_markers = source.copy()
    target_markers = target.copy()
    path = tmp_path / "sample.npz"
    np.savez(
        path,
        src_pcd=source,
        tgt_pcd=target,
        src_vol=source_markers,
        tgt_vol=target_markers,
        rot_tgt=rotation,
        **{"4": noise},
    )

    sample = load_p2p_sample(path, dataset="in_silico", noise_mm=4, voxel_size=0.0)
    expected_target = (target + noise) @ rotation.T
    expected_markers = target_markers @ rotation.T

    np.testing.assert_allclose(
        sample["target"] * sample["physical_scale"] + sample["target_center"],
        expected_target,
    )
    np.testing.assert_allclose(
        sample["target_markers"] * sample["physical_scale"] + sample["target_center"],
        expected_markers,
    )


def test_new_methods_are_part_of_unified_summary():
    assert "rtorv8_no_proposal" in MODELS
    assert "goicp" in MODELS


def test_goicp_project_exposes_the_p2p_evaluation_cli():
    entrypoint = Path(
        "/home/yangx/code/new_deform/go-icp_cython/evaluate_p2p_visibility.py"
    )
    completed = subprocess.run(
        [sys.executable, str(entrypoint), "--help"],
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "Evaluate trimmed Go-ICP on the P2P liver visibility benchmark" in completed.stdout


def test_visualization_transform_never_uses_identity_for_solver_failure():
    estimate = np.eye(4)
    assert visualization_transform("timeout", estimate) is None
    assert visualization_transform("error", estimate) is None
    np.testing.assert_array_equal(visualization_transform("ok", estimate), estimate)
