import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np


EXPERIMENT = Path(__file__).resolve().parents[1] / "experiments/geotransformer.p2i_lreg"


def _load_module(filename, name):
    spec = importlib.util.spec_from_file_location(name, EXPERIMENT / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


dataset = _load_module("dataset.py", "p2i_lreg_dataset")


class P2ILRegDatasetUnitTest(unittest.TestCase):
    def test_cam_pose_is_inverted_and_millimetres_become_metres(self):
        pose = {
            "00007": [
                {
                    "cam_R_c2w": np.eye(3).reshape(-1).tolist(),
                    "cam_t_c2w": [100.0, -200.0, 300.0],
                }
            ]
        }
        transform = dataset.load_source_to_reference_transform(pose, "00007")
        np.testing.assert_allclose(transform[:3, :3], np.eye(3), atol=1e-7)
        np.testing.assert_allclose(transform[:3, 3], [-0.1, 0.2, -0.3], atol=1e-7)

    def test_voxel_downsample_uses_metre_voxel(self):
        points = np.array(
            [[0.0, 0.0, 0.0], [0.0002, 0.0, 0.0], [0.002, 0.0, 0.0]],
            dtype=np.float64,
        )
        down = dataset.voxel_downsample(points, 0.001)
        self.assertEqual(down.shape, (2, 3))
        np.testing.assert_allclose(np.sort(down[:, 0]), [0.0001, 0.002], atol=1e-7)

    def test_fixed_sampling_is_exact_and_eval_is_deterministic(self):
        points = np.arange(9000 * 3, dtype=np.float64).reshape(9000, 3)
        first = dataset.sample_fixed_points(points, 8192, np.random.default_rng(91))
        second = dataset.sample_fixed_points(points, 8192, np.random.default_rng(91))
        self.assertEqual(first.shape, (8192, 3))
        np.testing.assert_array_equal(first, second)

    def test_fixed_sampling_wraps_without_creating_zero_padding(self):
        points = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        sampled = dataset.sample_fixed_points(points, 5, np.random.default_rng(3))
        self.assertEqual(sampled.shape, (5, 3))
        self.assertTrue(np.all(np.isin(sampled[:, 0], points[:, 0])))

    def test_chamfer_direction_check_prefers_source_to_reference(self):
        source = np.array(
            [[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [0.0, 0.01, 0.0]],
            dtype=np.float64,
        )
        transform = np.eye(4)
        transform[:3, 3] = [0.1, -0.2, 0.3]
        reference = source + transform[:3, 3]
        values = dataset.chamfer_before_after(reference, source, transform)
        self.assertGreater(values["ref_to_src_before_m"], 0.1)
        self.assertLess(values["ref_to_src_after_m"], 1e-10)
        self.assertLess(values["symmetric_after_m"], values["symmetric_before_m"])

    def test_placeholder_reference_is_filtered_before_split(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cloud_dir = root / "01" / "syn" / "liverPcds"
            cloud_dir.mkdir(parents=True)
            placeholder = cloud_dir / "ptsLiverCam00001.ply"
            placeholder.write_text(
                "ply\n"
                "format ascii 1.0\n"
                "element vertex 1\n"
                "property float x\n"
                "property float y\n"
                "property float z\n"
                "end_header\n"
                "0 0 0\n",
                encoding="ascii",
            )
            valid = cloud_dir / "ptsLiverCam00002.ply"
            valid.write_text(
                "ply\n"
                "format ascii 1.0\n"
                "element vertex 2\n"
                "property float x\n"
                "property float y\n"
                "property float z\n"
                "end_header\n"
                "0 0 0\n"
                "1 2 3\n",
                encoding="ascii",
            )
            records = [
                ("01", "ptsLiverCam00001"),
                ("01", "ptsLiverCam00002"),
            ]

            usable, excluded = dataset.filter_unusable_reference_records(root, records)

            self.assertEqual(usable, [("01", "ptsLiverCam00002")])
            self.assertEqual(excluded, [("01", "ptsLiverCam00001")])

            manifest = dataset.reference_filter_manifest(excluded)
            self.assertEqual(manifest["policy"], "exclude_empty_observation_without_gt")
            self.assertEqual(manifest["excluded_count"], 1)
            self.assertEqual(manifest["excluded_case_ids"], ["01/ptsLiverCam00001"])


if __name__ == "__main__":
    unittest.main()
