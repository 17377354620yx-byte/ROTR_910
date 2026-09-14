"""Independent checks for units, transform direction, marker association and TRE."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
import numpy as np

MODULE = Path(__file__).resolve().parents[1] / 'experiments/geotransformer.p2p_liver/depoll_protocol.py'
spec = importlib.util.spec_from_file_location('depoll_protocol', MODULE)
depoll = importlib.util.module_from_spec(spec)
spec.loader.exec_module(depoll)


class TestDePollProtocol(unittest.TestCase):
    def test_normalized_to_physical_with_rotation_and_distinct_centers(self):
        # Hand-computed 90-degree rotation, scale 10 and unequal centroids.
        transform = np.array([[0., -1., 0., .2], [1., 0., 0., -.3],
                              [0., 0., 1., .4], [0., 0., 0., 1.]])
        source_center, target_center = np.array([100., 20., -50.]), np.array([-7., 80., 60.])
        physical = depoll.physical_transform(transform, source_center, target_center, 10.)
        np.testing.assert_allclose(depoll.apply(np.array([[110., 40., -20.]]), physical),
                                   [[-25., 87., 94.]])

    def test_mean_tre_is_not_rms(self):
        source = np.zeros((2, 3))
        target = np.array([[3., 0., 0.], [0., 4., 0.]])
        scores = depoll.marker_errors(source, target, np.eye(4))
        self.assertEqual(scores['tre_mm'], 3.5)
        self.assertAlmostEqual(scores['rms_tre_mm'], np.sqrt(12.5))

    def test_ply_ascii_and_binary_agree(self):
        with tempfile.TemporaryDirectory() as tmp:
            for fmt in ['ascii', 'binary_little_endian', 'binary_big_endian']:
                path = Path(tmp) / f'{fmt}.ply'
                header = f'ply\nformat {fmt} 1.0\nelement vertex 2\nproperty float x\nproperty float y\nproperty float z\nend_header\n'
                with path.open('wb') as h:
                    h.write(header.encode())
                    if fmt == 'ascii':
                        h.write(b'1 2 3\n4 5 6\n')
                    else:
                        np.array([[1,2,3],[4,5,6]], dtype='<f4' if 'little' in fmt else '>f4').tofile(h)
                np.testing.assert_array_equal(depoll.read_ply(path), [[1,2,3],[4,5,6]])

    @unittest.skipUnless(Path('/home/yangx/code/new_deform/DEPOLL').is_dir(), 'Local DEPOLL not present')
    def test_published_ground_truth_all_13_cases(self):
        values = {'clips': [], 'balls': []}
        for case in range(1, 14):
            data = depoll.load_case('/home/yangx/code/new_deform/DEPOLL', f'{case:02d}', 'preop')
            for group in values:
                values[group].append(depoll.marker_errors(data['source_markers'][group],
                                     data['target_markers'][group], data['released_transform'])['tre_mm'])
            intra = depoll.load_case('/home/yangx/code/new_deform/DEPOLL', f'{case:02d}', 'intraop')
            for group in values:
                error = depoll.marker_errors(intra['source_markers'][group], intra['target_markers'][group],
                                             intra['released_transform'])['tre_mm']
                self.assertLess(error, 1e-9)
        # Published Table 4 independent reference, not constants from the implementation.
        np.testing.assert_allclose([np.mean(values['clips']), np.std(values['clips']),
                                    np.mean(values['balls']), np.std(values['balls'])],
                                   [28.89, 8.98, 27.85, 10.34], atol=.005, rtol=0)


if __name__ == '__main__':
    unittest.main()
