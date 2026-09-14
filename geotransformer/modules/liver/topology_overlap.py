"""Topology and overlap refinement for rigid liver registration."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _gather_rows(values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    return values[indices.reshape(-1)].reshape(*indices.shape, values.shape[-1])


class TopologyOverlapRefiner(nn.Module):
    """Refine coarse descriptors and predict bilateral visible-overlap logits.

    Local graph geometry is rigid invariant. Poincare distance augments the
    edge attention, while a zero-initialized residual projection preserves the
    input descriptor at initialization.
    """

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 128,
        num_neighbors: int = 8,
        poincare_curvature: float = 1.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if feature_dim <= 0 or hidden_dim <= 0:
            raise ValueError("feature_dim and hidden_dim must be positive")
        if num_neighbors <= 0:
            raise ValueError("num_neighbors must be positive")
        if poincare_curvature <= 0:
            raise ValueError("poincare_curvature must be positive")
        self.feature_dim = int(feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_neighbors = int(num_neighbors)
        self.poincare_curvature = float(poincare_curvature)

        self.input_proj = nn.Linear(self.feature_dim, self.hidden_dim)
        # normalized distance, curvature, linearity, planarity, Poincare distance
        self.edge_gate = nn.Sequential(
            nn.Linear(5, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, 1),
        )
        self.graph_update = nn.Sequential(
            nn.Linear(2 * self.hidden_dim + 3, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.cross_query = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.cross_key = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.cross_value = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.cross_update = nn.Sequential(
            nn.Linear(3 * self.hidden_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.overlap_head = nn.Sequential(
            nn.Linear(2 * self.hidden_dim + 5, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, 1),
        )
        self.output_proj = nn.Linear(self.hidden_dim, self.feature_dim)

        # Old descriptors and a uniform overlap prior are reproduced at step 0.
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)
        nn.init.zeros_(self.overlap_head[-1].weight)
        nn.init.zeros_(self.overlap_head[-1].bias)

    @staticmethod
    @torch.no_grad()
    def _local_geometry(points: torch.Tensor, k: int):
        count = points.shape[0]
        if count == 0:
            raise ValueError("RTOR requires at least one superpoint per cloud")
        if count < 2:
            indices = torch.zeros((count, 1), dtype=torch.long, device=points.device)
            distances = points.new_zeros((count, 1))
            geometry = points.new_zeros((count, 3))
            return indices, distances, geometry

        k = min(max(1, k), count - 1)
        distance_map = torch.cdist(points, points)
        indices = distance_map.topk(k=k + 1, dim=1, largest=False).indices[:, 1:]
        distances = distance_map.gather(1, indices)
        neighbours = _gather_rows(points, indices)
        offsets = neighbours - points[:, None, :]
        covariance = offsets.transpose(1, 2) @ offsets / float(k)
        eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0)
        l0, l1, l2 = eigenvalues.unbind(dim=1)
        trace = (l0 + l1 + l2).clamp_min(1e-12)
        curvature = l0 / trace
        linearity = (l2 - l1) / l2.clamp_min(1e-12)
        planarity = (l1 - l0) / l2.clamp_min(1e-12)
        geometry = torch.stack([curvature, linearity, planarity], dim=1)
        median_nn = distances[:, 0].median().clamp_min(1e-12)
        return indices, distances / median_nn, geometry

    def _poincare_map(self, features: torch.Tensor) -> torch.Tensor:
        curvature = features.new_tensor(self.poincare_curvature)
        sqrt_c = curvature.sqrt()
        norm = torch.linalg.norm(features, dim=-1, keepdim=True).clamp_min(1e-12)
        mapped = torch.tanh(sqrt_c * norm) * features / (sqrt_c * norm)
        max_norm = (1.0 - 1e-4) / sqrt_c
        mapped_norm = torch.linalg.norm(mapped, dim=-1, keepdim=True).clamp_min(1e-12)
        return mapped * (max_norm / mapped_norm).clamp_max(1.0)

    def _poincare_distance(
        self, center: torch.Tensor, neighbours: torch.Tensor
    ) -> torch.Tensor:
        curvature = center.new_tensor(self.poincare_curvature)
        delta2 = ((center[:, None, :] - neighbours) ** 2).sum(dim=-1)
        center2 = (center**2).sum(dim=-1, keepdim=True)
        neighbour2 = (neighbours**2).sum(dim=-1)
        denominator = (
            (1.0 - curvature * center2) * (1.0 - curvature * neighbour2)
        ).clamp_min(1e-8)
        argument = (
            1.0 + 2.0 * curvature * delta2 / denominator
        ).clamp_min(1.0 + 1e-7)
        return torch.acosh(argument) / curvature.sqrt()

    def _graph_refine(self, points: torch.Tensor, features: torch.Tensor):
        hidden = self.input_proj(features)
        indices, relative_distances, geometry = self._local_geometry(
            points, self.num_neighbors
        )
        neighbour_hidden = _gather_rows(hidden, indices)
        poincare = self._poincare_map(F.normalize(hidden, p=2, dim=-1))
        neighbour_poincare = _gather_rows(poincare, indices)
        hyperbolic_distance = self._poincare_distance(
            poincare, neighbour_poincare
        )
        neighbour_geometry = _gather_rows(geometry, indices)
        edge_geometry = torch.cat(
            [
                relative_distances[..., None],
                geometry[:, None, :].expand_as(neighbour_geometry),
                hyperbolic_distance[..., None],
            ],
            dim=-1,
        )
        weights = torch.softmax(self.edge_gate(edge_geometry).squeeze(-1), dim=1)
        message = (weights[..., None] * neighbour_hidden).sum(dim=1)
        update = self.graph_update(torch.cat([hidden, message, geometry], dim=-1))
        return hidden + update, geometry

    def _cross_support(self, query: torch.Tensor, support: torch.Tensor):
        logits = self.cross_query(query) @ self.cross_key(support).transpose(0, 1)
        logits = logits / math.sqrt(self.hidden_dim)
        probabilities = torch.softmax(logits, dim=1)
        aggregate = probabilities @ self.cross_value(support)
        maximum = probabilities.max(dim=1).values
        entropy = -(
            probabilities * probabilities.clamp_min(1e-12).log()
        ).sum(dim=1)
        if support.shape[0] > 1:
            entropy = entropy / math.log(support.shape[0])
        else:
            entropy = entropy.new_zeros(entropy.shape)
        return aggregate, maximum, entropy

    def forward(
        self,
        ref_points: torch.Tensor,
        src_points: torch.Tensor,
        ref_features: torch.Tensor,
        src_features: torch.Tensor,
    ):
        if ref_features.shape[1] != self.feature_dim:
            raise ValueError(
                f"Expected {self.feature_dim}D ref features, got {ref_features.shape[1]}"
            )
        if src_features.shape[1] != self.feature_dim:
            raise ValueError(
                f"Expected {self.feature_dim}D src features, got {src_features.shape[1]}"
            )
        ref_graph, ref_geometry = self._graph_refine(ref_points, ref_features)
        src_graph, src_geometry = self._graph_refine(src_points, src_features)
        ref_cross, ref_max, ref_entropy = self._cross_support(
            ref_graph, src_graph
        )
        src_cross, src_max, src_entropy = self._cross_support(
            src_graph, ref_graph
        )

        ref_delta = self.cross_update(
            torch.cat([ref_graph, ref_cross, ref_graph - ref_cross], dim=-1)
        )
        src_delta = self.cross_update(
            torch.cat([src_graph, src_cross, src_graph - src_cross], dim=-1)
        )
        ref_hidden = ref_graph + ref_delta
        src_hidden = src_graph + src_delta

        ref_overlap_input = torch.cat(
            [
                ref_hidden,
                ref_cross,
                ref_geometry,
                ref_max[:, None],
                ref_entropy[:, None],
            ],
            dim=-1,
        )
        src_overlap_input = torch.cat(
            [
                src_hidden,
                src_cross,
                src_geometry,
                src_max[:, None],
                src_entropy[:, None],
            ],
            dim=-1,
        )
        ref_logits = self.overlap_head(ref_overlap_input).squeeze(-1)
        src_logits = self.overlap_head(src_overlap_input).squeeze(-1)

        overlap_output = {
            "ref_overlap_logits": ref_logits,
            "src_overlap_logits": src_logits,
        }
        return (
            ref_features + self.output_proj(ref_hidden),
            src_features + self.output_proj(src_hidden),
            overlap_output,
        )
