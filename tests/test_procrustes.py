import unittest
import torch
from geotransformer.modules.registration.procrustes import weighted_procrustes


class ProcrustesTest(unittest.TestCase):
    def test_pose_is_independent_of_confidence_units(self):
        torch.manual_seed(9)
        source = torch.randn(40, 3, dtype=torch.float64) + 2
        rotation = torch.linalg.qr(torch.randn(3, 3, dtype=torch.float64)).Q
        if torch.det(rotation) < 0:
            rotation[:, -1] *= -1
        translation = torch.tensor([0.4, -0.8, 0.5], dtype=torch.float64)
        reference = source @ rotation.T + translation
        weights = torch.rand(40, dtype=torch.float64)
        for scale in (1.0, 1e-5, 1e-10):
            r, t = weighted_procrustes(source, reference, weights * scale)
            torch.testing.assert_close(r, rotation, atol=1e-10, rtol=1e-10)
            torch.testing.assert_close(t, translation, atol=1e-10, rtol=1e-10)

    def test_zero_mass_is_finite_identity(self):
        points = torch.randn(12, 3)
        result = weighted_procrustes(points, points + 1, torch.zeros(12), return_transform=True)
        torch.testing.assert_close(result, torch.eye(4))


if __name__ == '__main__':
    unittest.main()
