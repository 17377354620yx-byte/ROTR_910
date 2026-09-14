"""Expose fine matching to predicted proposals without discarding positive supervision."""
import torch


@torch.no_grad()
def mix_coarse_proposals(gt_ref, gt_src, gt_scores, pred_ref, pred_src, pred_scores, ratio):
    """Keep a fixed GT-sized budget, include predictions, and remove duplicate pairs.

    Predicted scores are never replaced by GT overlaps. Ground truth determines
    fine supervision only, as in the original target generator.
    """
    if not 0 <= ratio <= 1:
        raise ValueError('predicted ratio must be in [0, 1]')
    budget = len(gt_ref)
    if ratio == 0 or budget == 0:
        return gt_ref, gt_src, gt_scores
    predicted_count = min(len(pred_ref), int(round(budget*ratio)))
    gt_count = budget-predicted_count
    pairs = torch.stack([torch.cat([gt_ref[:gt_count], pred_ref[:predicted_count], gt_ref[gt_count:]]),
                         torch.cat([gt_src[:gt_count], pred_src[:predicted_count], gt_src[gt_count:]])], -1)
    scores = torch.cat([gt_scores[:gt_count], pred_scores[:predicted_count], gt_scores[gt_count:]])
    seen, keep = set(), []
    for i,pair in enumerate(pairs.cpu().tolist()):
        key = tuple(pair)
        if key not in seen:
            keep.append(i);seen.add(key)
        if len(keep) == budget:
            break
    indices = torch.as_tensor(keep, device=pairs.device)
    return pairs[indices,0], pairs[indices,1], scores[indices]


def predicted_ratio_at_epoch(epoch, start=5, end=20, maximum=.25):
    if end <= start or not 0 <= maximum <= 1:
        raise ValueError('Invalid candidate-exposure schedule')
    return float(maximum)*min(1., max(0., (epoch-start)/(end-start)))
