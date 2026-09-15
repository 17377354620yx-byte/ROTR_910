import unittest

import torch

from geotransformer.modules.ops.pointcloud_partition import point_to_node_partition


class PointToNodePartitionTest(unittest.TestCase):
    def test_fewer_points_than_patch_size_are_sentinel_padded(self):
        points = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            dtype=torch.float32,
        )
        nodes = torch.tensor(
            [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=torch.float32
        )

        _, node_masks, indices, masks = point_to_node_partition(
            points, nodes, point_limit=5
        )

        self.assertEqual(indices.shape, (2, 5))
        self.assertEqual(masks.shape, (2, 5))
        self.assertTrue(torch.all(indices[~masks] == len(points)))
        self.assertTrue(torch.all(node_masks))
        self.assertLessEqual(int(masks.sum(dim=1).max()), len(points))


if __name__ == "__main__":
    unittest.main()
