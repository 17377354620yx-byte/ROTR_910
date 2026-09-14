"""Inference-time overlap-region selection utilities."""

import math

import torch


@torch.no_grad()
def select_overlap_region(
    probabilities,
    valid_mask,
    threshold,
    min_superpoints,
    max_ratio,
    min_spread,
):
    """Select confident nodes while retaining a safe low-overlap fallback."""
    if probabilities.ndim != 1 or valid_mask.ndim != 1:
        raise ValueError('probabilities and valid_mask must be one-dimensional')
    if probabilities.shape != valid_mask.shape:
        raise ValueError('probabilities and valid_mask must have identical shapes')

    valid_indices = torch.nonzero(valid_mask, as_tuple=True)[0]
    num_valid = valid_indices.numel()
    if num_valid == 0:
        return valid_mask.clone()

    min_keep = min(max(1, int(min_superpoints)), num_valid)
    max_keep = min(
        num_valid,
        max(min_keep, int(math.ceil(float(max_ratio) * num_valid))),
    )
    valid_probabilities = probabilities[valid_indices]
    probability_spread = valid_probabilities.max() - valid_probabilities.min()
    if probability_spread < float(min_spread):
        return valid_mask.clone()

    candidate_indices = valid_indices[valid_probabilities >= float(threshold)]
    if candidate_indices.numel() < min_keep:
        selected = valid_indices[valid_probabilities.topk(min_keep).indices]
    elif candidate_indices.numel() > max_keep:
        candidate_probabilities = probabilities[candidate_indices]
        selected = candidate_indices[candidate_probabilities.topk(max_keep).indices]
    else:
        selected = candidate_indices

    region_mask = torch.zeros_like(valid_mask)
    region_mask[selected] = True
    return region_mask


__all__ = ['select_overlap_region']
