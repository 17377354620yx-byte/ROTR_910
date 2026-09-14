"""Mask-aware, same-universe ranking and aligned gradient diagnostics."""
import math
import numpy as np
import torch


def correlation(x, y):
    x, y = x.detach().double().reshape(-1), y.detach().double().reshape(-1)
    valid = torch.isfinite(x) & torch.isfinite(y)
    x, y = x[valid], y[valid]
    if len(x) < 3:
        return None
    x, y = x-x.mean(), y-y.mean()
    denominator = x.norm()*y.norm()
    return float((x@y)/denominator) if denominator > 1e-15 else None


@torch.no_grad()
def score_metrics(scores, positives, valid, kind='probability'):
    """Scores end in (queries, candidates); GT-negative rows do not invent ranks."""
    scores = scores.detach().float().reshape(-1, scores.shape[-1])
    positives = positives.detach().reshape_as(scores).bool()
    valid = valid.detach().reshape_as(scores).bool()
    eligible = positives.any(-1)
    masked = scores.masked_fill(~valid, -torch.inf)
    best = masked.masked_fill(~positives, -torch.inf).max(-1).values
    found = torch.isfinite(best) & eligible
    ranks = (masked > best[:, None]).sum(-1).float()+1
    reciprocal = torch.where(found, ranks.reciprocal(), 0.)
    count = valid.sum(-1)
    probability = torch.softmax(masked, -1).nan_to_num() if kind == 'logit' else scores.clamp_min(0)*valid
    probability = probability / probability.sum(-1, keepdim=True).clamp_min(1e-30)
    entropy = -(probability*probability.clamp_min(1e-30).log()).sum(-1)
    entropy = entropy / count.clamp_min(2).float().log()
    pos, neg = scores[positives & valid], scores[~positives & valid]
    result = dict(gt_queries=int(eligible.sum()), gt_pairs=int(positives.sum()),
                  gt_pair_survival=float((positives & valid).sum()/positives.sum().clamp_min(1)),
                  mrr=float(reciprocal[eligible].mean()) if eligible.any() else None,
                  rank_mean_found=float(ranks[found].mean()) if found.any() else None,
                  recall1=float((found & (ranks <= 1))[eligible].float().mean()) if eligible.any() else None,
                  recall5=float((found & (ranks <= 5))[eligible].float().mean()) if eligible.any() else None,
                  entropy=float(entropy[count > 1].mean()) if (count > 1).any() else None,
                  positive_mean=float(pos.mean()) if pos.numel() else None,
                  negative_mean=float(neg.mean()) if neg.numel() else None)
    # Coarse probabilities span orders of magnitude; fine logits use a common
    # fixed range. Out-of-range counts remain explicit.
    edges = np.linspace(-12, 0, 49) if kind == 'probability' else np.linspace(-20, 40, 61)
    for label, values in [('positive', pos), ('negative', neg)]:
        values = values.cpu().numpy()
        if kind == 'probability':
            values = np.log10(np.maximum(values, 1e-30))
        result[label+'_hist'] = np.histogram(values, edges)[0].tolist()
        result[label+'_underflow'] = int((values < edges[0]).sum())
        result[label+'_overflow'] = int((values > edges[-1]).sum())
    result['hist_edges'] = edges.tolist()
    return result


@torch.no_grad()
def feature_metrics(before, after, masks=None):
    before, after = before.detach().float(), after.detach().float()
    if masks is not None:
        before, after = before[masks], after[masks]
    delta = after-before
    return dict(before_norm=float(before.norm(dim=-1).mean()),
                after_norm=float(after.norm(dim=-1).mean()),
                residual_norm=float(delta.norm(dim=-1).mean()),
                relative_residual=float(delta.norm()/before.norm().clamp_min(1e-15)),
                feature_cosine=float(torch.nn.functional.cosine_similarity(before, after, dim=-1).mean()))


def gradient_metrics(first, second):
    """Use the same full parameter coordinate system, padding disconnected parts with zero."""
    dot = norm_a = norm_b = 0.
    path_a = path_b = False
    shared_count = 0
    for a, b in zip(first, second):
        if a is not None:
            path_a = True
            norm_a += float(a.double().square().sum())
        if b is not None:
            path_b = True
            norm_b += float(b.double().square().sum())
        if a is not None and b is not None:
            dot += float((a.double()*b.double()).sum())
            shared_count += a.numel()
    na, nb = math.sqrt(norm_a), math.sqrt(norm_b)
    status = ('NO_GRADIENT_PATH' if not (path_a and path_b) else
              'NONFINITE' if not (math.isfinite(na) and math.isfinite(nb)) else
              'ZERO_NORM' if min(na, nb) <= 1e-12 else 'OK')
    return dict(norm_a=na, norm_b=nb, cosine=dot/(na*nb) if status == 'OK' else None,
                dot=dot, status=status, shared_parameter_count=shared_count)
