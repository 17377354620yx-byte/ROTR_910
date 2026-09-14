"""Fixed TRAINING-only checkpoint probes. Never selects on released test TRE."""
import argparse
import contextlib
import hashlib
import itertools
import json
import os
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/'experiments/geotransformer.p2p_liver')]
import numpy as np
import torch
import torch.nn.functional as F
from config import make_cfg
from dataset import LiverTask3TrainDataset, DeterministicValidationDataset
from model import create_model
from loss import OverallLoss, FineMatchingLoss
from diagnostics.interaction.metrics import correlation, feature_metrics, gradient_metrics, score_metrics
from geotransformer.modules.ops import apply_transform, pairwise_distance
from geotransformer.utils.data import registration_collate_fn_stack_mode
from geotransformer.utils.torch import initialize, to_cuda

HISTORY = Path('/home/yangx/code/new_deform/RTORv6/output')


class Capture:
    def __init__(self, model):
        self.values = {}
        self.handles = []
        for name, module in [('rtor', model.topology_overlap_refiner), ('a3', model.fine_local_refiner)]:
            if module is not None:
                self.handles.append(module.register_forward_hook(self.hook(name)))

    def hook(self, name):
        def capture(module, inputs, outputs):
            self.values[name] = (inputs, outputs)
        return capture

    def close(self):
        self.values.clear()
        for h in self.handles:
            h.remove()


def coarse_scores(ref, src):
    score = torch.exp(-pairwise_distance(ref, src, normalized=True))
    return (score/score.sum(-1, keepdim=True).clamp_min(1e-12)
            * score/score.sum(-2, keepdim=True).clamp_min(1e-12))


@torch.no_grad()
def structure(model, capture, output, data):
    result = {}
    ref, src = output['ref_feats_c'], output['src_feats_c']
    valid = output['ref_node_masks'][:, None] & output['src_node_masks'][None, :]
    gt = torch.zeros_like(valid)
    pairs = output['gt_node_corr_indices'][output['gt_node_corr_overlaps'] > .1]
    gt[pairs[:, 0], pairs[:, 1]] = True
    after = coarse_scores(ref, src)
    before = after
    if 'rtor' in capture.values:
        inputs, outputs = capture.values['rtor']
        before = coarse_scores(inputs[2], inputs[3])
        result['rtor_ref_features'] = feature_metrics(inputs[2], outputs[0])
        result['rtor_src_features'] = feature_metrics(inputs[3], outputs[1])
    prior = output['ref_overlap_weight'][:, None]*output['src_overlap_weight'][None, :]
    calibrated = model.coarse_matching._calibrate_scores(after, output['ref_overlap_weight'], output['src_overlap_weight'])
    result['coarse_before_rtor'] = score_metrics(before, gt, valid)
    result['coarse_after_rtor_descriptor'] = score_metrics(after, gt, valid)
    result['coarse_after_overlap_weight'] = score_metrics(calibrated, gt, valid)
    focused = output['focused_ref_node_masks'][:, None] & output['focused_src_node_masks'][None, :]
    result['coarse_after_focus'] = score_metrics(calibrated, gt, focused)
    result['overlap_descriptor_correlation'] = correlation(prior[valid], after[valid])
    result['rtor_descriptor_increment_prior_correlation'] = correlation((after-before)[valid], prior[valid])
    pred_ref, pred_src = output['ref_node_corr_indices'], output['src_node_corr_indices']
    result['PIR'] = float(gt[pred_ref, pred_src].float().mean())
    if 'a3' in capture.values:
        inputs, outputs = capture.values['a3']
        rf, sf, rp, sp, rm, sm = inputs
        before_f = torch.einsum('bnd,bmd->bnm', rf, sf)/rf.shape[-1]**.5
        after_f = torch.einsum('bnd,bmd->bnm', outputs[0], outputs[1])/rf.shape[-1]**.5
        distances = pairwise_distance(rp, apply_transform(sp, data['transform']))
        valid_f = rm[:, :, None] & sm[:, None, :]
        positive_f = (distances < .04**2) & valid_f
        result['fine_before_a3'] = score_metrics(before_f, positive_f, valid_f, 'logit')
        result['fine_after_a3'] = score_metrics(after_f, positive_f, valid_f, 'logit')
        result['fine_after_rtor_a3_sinkhorn'] = score_metrics(output['matching_scores'][:, :-1, :-1].exp(), positive_f, valid_f)
        result['a3_ref_features'] = feature_metrics(rf, outputs[0], rm)
        result['a3_src_features'] = feature_metrics(sf, outputs[1], sm)
        # Coarse and fine outputs live on different grids. Align by the exact
        # predicted patch pair; never correlate unrelated flattened tensors.
        if len(pred_ref) == len(rf):
            delta = ((after_f-before_f)*valid_f).sum((1, 2))/valid_f.sum((1, 2)).clamp_min(1)
            confidence = prior[pred_ref, pred_src]
            descriptor = after[pred_ref, pred_src]
            quality = positive_f.sum((1, 2)).float()/valid_f.sum((1, 2)).clamp_min(1)
            result['rtor_a3_aligned_output_correlation'] = correlation(confidence, delta)
            result['descriptor_a3_aligned_output_correlation'] = correlation(descriptor, delta)
            result['overlap_patch_gt_quality_correlation'] = correlation(confidence, quality)
            result['a3_increment_patch_gt_quality_correlation'] = correlation(delta, quality)
            result['patch_positive_fraction'] = float(quality.mean())
        result['a3_gate'] = float(model.fine_local_refiner.layers[0].residual_scale)
    return result


def gradient_probe(model, capture, output, data, objective):
    losses = objective(output, data)
    components = {k: losses[k] for k in ('c_loss', 'f_loss', 'o_loss')}
    if 'a3' in capture.values:
        (rf, sf, _, _, rm, sm), _ = capture.values['a3']
        logits = torch.einsum('bnd,bmd->bnm', rf, sf)/rf.shape[-1]**.5
        off_output = dict(output, matching_scores=model.optimal_transport(logits, rm, sm))
        components['f_without_a3'] = objective.fine_loss(off_output, data)
    named = list(model.named_parameters())
    parameters = [p for _, p in named]
    groups = {name: [i for i,(n,_) in enumerate(named) if n.startswith(prefix)]
              for name, prefix in [('backbone', 'backbone.'), ('rtor', 'topology_overlap_refiner.'),
                                   ('a3', 'fine_local_refiner.'), ('transformer', 'transformer.') ]}
    gradients = {}
    for name, loss in components.items():
        if loss.requires_grad:
            grads = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
            gradients[name] = [g.detach().cpu() if g is not None else None for g in grads]
            del grads
        else:
            gradients[name] = [None]*len(parameters)
    if 'f_without_a3' in gradients:
        gradients['a3_increment'] = [(a-b if b is not None else a) if a is not None else
                                     (-b if b is not None else None)
                                     for a,b in zip(gradients['f_loss'], gradients['f_without_a3'])]
    rows = []
    for group, indices in groups.items():
        for a,b in [('o_loss', 'f_loss'), ('o_loss', 'c_loss'), ('f_loss', 'c_loss'),
                    ('o_loss', 'a3_increment'), ('c_loss', 'a3_increment')]:
            if b not in gradients:
                continue
            metrics = gradient_metrics([gradients[a][i] for i in indices], [gradients[b][i] for i in indices])
            rows.append(dict(module=group, loss_a=a, loss_b=b, **metrics))
    # First-order weighting sensitivity, explicitly not a completed training experiment.
    idx = groups['backbone']
    vectors = {}
    for name in ('c_loss', 'f_loss', 'o_loss'):
        vectors[name] = torch.cat([gradients[name][i].reshape(-1) if gradients[name][i] is not None
                                  else torch.zeros(parameters[i].numel()) for i in idx])
    weights = []
    for wc, wf, wo in [(1,1,0), (1,1,.1), (1,1,.5), (1,1,1), (1,.5,.5), (1,2,.5)]:
        direction = wc*vectors['c_loss']+wf*vectors['f_loss']+wo*vectors['o_loss']
        norm = direction.norm().clamp_min(1e-30)
        weights.append(dict(weights=[wc,wf,wo], total_norm=float(norm),
                            predicted_loss_change_per_unit_step={k:-float(v@direction/norm) for k,v in vectors.items()},
                            status='FIRST_ORDER_ONLY_NOT_OPTIMIZER_REPLAY'))
    return dict(losses={k:float(v.detach()) for k,v in components.items()}, gradients=rows, weight_sensitivity=weights)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--architectures', nargs='+', default=['rtor_a3'])
    parser.add_argument('--epochs', nargs='+', type=int, default=[1,5,10,50,75,100,130,140,150])
    parser.add_argument('--samples', type=int, default=16)
    parser.add_argument('--output', required=True)
    parser.add_argument('--no_gradients', action='store_true')
    args = parser.parse_args()
    root = Path(args.output).resolve()
    if not root.is_relative_to(ROOT):
        parser.error('Only project-local writes are allowed')
    root.mkdir(parents=True, exist_ok=True)
    if (root/'records.jsonl').exists():
        parser.error('Use a fresh output directory; never append duplicate evidence')
    initialize(7351)
    torch.set_num_threads(4)
    cfg = make_cfg('rtor_a3', 'legacy', False)
    train = LiverTask3TrainDataset(cfg)
    indices = np.random.default_rng(20260909).choice(len(train), args.samples, replace=False).tolist()
    deterministic = DeterministicValidationDataset(train, len(train), 19000003)
    manifest = dict(arguments=vars(args), training_root=cfg.data.train_root, indices=indices,
                    augmentation_seed=19000003, seed=7351, neighbor_limits=[7,22,32,39],
                    evaluation_mode='eval for structure; train/dropout for gradients; no optimizer updates',
                    restrictions='No released test files/metrics. Original single-encoder architecture, weights and LGR threshold frozen.',
                    a3_loss_definition='No independent A3 loss exists. f_loss is a pathway proxy; a3_increment is grad(f_on - f_off) on the same GT patches.',
                    interpretation='Checkpoint replay on fixed training inputs, not historical per-step gradients or unseen-anatomy validation',
                    checkpoints={}, code_sha256={})
    for path in [Path(__file__), ROOT/'diagnostics/interaction/metrics.py',
                 ROOT/'geotransformer/modules/liver/registration_model.py',
                 ROOT/'experiments/geotransformer.p2p_liver/loss.py']:
        manifest['code_sha256'][str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    (root/'manifest.json').write_text(json.dumps(manifest, indent=2))
    samples = []
    for index in indices:
        data = deterministic[index]
        samples.append(registration_collate_fn_stack_mode([data], 4, .02, .05, [7,22,32,39]))
    (root/'sample_manifest.json').write_text(json.dumps([dict(index=i, name=s['sample_name'],
         points=[int(x) for x in s['lengths'][0]], visibility=float(s['overlap'])) for i,s in zip(indices,samples)], indent=2))
    start = time.monotonic()
    with (root/'records.jsonl').open('w') as handle:
        for architecture, epoch in itertools.product(args.architectures, args.epochs):
            cfg = make_cfg(architecture, 'legacy', False)
            model = create_model(cfg).cuda()
            checkpoint = HISTORY/f'geotransformer.p2p_liver.ablation.scratch_v2.{architecture}.seed7351/snapshots/epoch-{epoch}.pth.tar'
            state = torch.load(checkpoint, map_location='cpu', weights_only=False)
            model.load_state_dict(state['model'], strict=True)
            del state
            manifest['checkpoints'][f'{architecture}/{epoch}'] = dict(path=str(checkpoint), sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest())
            (root/'manifest.json').write_text(json.dumps(manifest, indent=2))
            capture = Capture(model)
            objective = OverallLoss(cfg).cuda()
            for position, sample in enumerate(samples):
                data = to_cuda(sample)
                row = dict(architecture=architecture, epoch=epoch, sample_position=position,
                           sample=sample['sample_name'], phase='early' if epoch<=10 else 'middle' if epoch<=100 else 'late')
                model.eval()
                with torch.no_grad():
                    output = model(data)
                    row['structure'] = structure(model, capture, output, data)
                    loss_values = objective(output, data)
                    row['predicted_patch_losses'] = {k:float(v) for k,v in loss_values.items()}
                del output, loss_values
                capture.values.clear()
                if not args.no_gradients:
                    model.train()
                    initialize(7351+position)
                    # Preserve the exact training computation while moving saved
                    # activations to CPU to fit alongside other users' GPU jobs.
                    with torch.autograd.graph.save_on_cpu(pin_memory=True):
                        output = model(data)
                        row.update(gradient_probe(model, capture, output, data, objective))
                    del output
                    capture.values.clear()
                handle.write(json.dumps(row, allow_nan=False)+'\n')
                handle.flush()
                del data
                torch.cuda.empty_cache()
                print(f'{architecture} epoch {epoch}: {position+1}/{len(samples)} ({time.monotonic()-start:.1f}s)', flush=True)
            capture.close()
            del model, objective, capture
            torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
