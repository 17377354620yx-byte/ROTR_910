"""Same-checkpoint interventions; test-set runs are diagnostic, not tuning data."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'experiments/geotransformer.p2p_liver'))

import numpy as np
import torch
from config import make_cfg
from dataset import (LiverTask3InVitroTestDataset, LiverTask3TestDataset,
                     LiverTask3TrainDataset, DeterministicValidationDataset, _build_loader)
from model import create_model
from geotransformer.utils.torch import initialize, to_cuda
from geotransformer.modules.ops import apply_transform
from surface_refinement import refine_surface


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', required=True)
    parser.add_argument('--architecture', choices=['geotransformer', 'rtor_only', 'a3_only', 'rtor_a3'], default='rtor_a3')
    parser.add_argument('--dual_encoder', action='store_true')
    parser.add_argument('--dataset', choices=['in_vitro', 'in_silico', 'validation'], default='in_vitro')
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--variants', nargs='+', default=['legacy', 'uncapped', 'no_focus'])
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if not output.is_relative_to(ROOT):
        parser.error('Output must be inside this project')
    output.mkdir(parents=True, exist_ok=True)
    initialize(7351)
    torch.set_num_threads(4)
    cfg = make_cfg(args.architecture)
    cfg.model.dual_encoder = args.dual_encoder
    dataset = (LiverTask3InVitroTestDataset(cfg) if args.dataset == 'in_vitro' else
               LiverTask3TrainDataset(cfg) if args.dataset == 'validation' else LiverTask3TestDataset(cfg))
    indices = np.arange(0, len(dataset), args.stride)
    if args.limit:
        indices = indices[:args.limit]
    if args.dataset == 'validation':
        dataset = torch.utils.data.Subset(DeterministicValidationDataset(dataset, len(dataset), 1007354), indices.tolist())
    else:
        dataset.samples = dataset.samples[indices]
        dataset.sample_indices = dataset.sample_indices[indices]
    # Frozen limits recorded by the original scratch_v2 training run.
    limits = [7, 22, 32, 39]
    loader = _build_loader(cfg, dataset, limits, False)
    model = create_model(cfg).cuda().eval()
    state = torch.load(args.snapshot, map_location='cpu', weights_only=False)
    model.load_state_dict(state['model'], strict=True)
    del state
    manifest = dict(arguments=vars(args), seed=7351, neighbor_limits=limits,
                    torch=torch.__version__, checkpoint_sha256=hashlib.sha256(
                        Path(args.snapshot).read_bytes()).hexdigest(),
                    purpose='paired causal diagnosis on already exposed test data; not a held-out claim',
                    indices=indices.tolist())
    manifest['code_sha256'] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in [Path(__file__), ROOT/'geotransformer/modules/liver/registration_model.py',
                                        ROOT/'geotransformer/modules/registration/procrustes.py',
                                        ROOT/'geotransformer/modules/geotransformer/local_global_registration.py']}
    if args.dataset == 'validation':
        manifest['purpose'] = 'Fixed training-distribution augmentation validation; pretrained checkpoint has seen these anatomy groups'
        manifest['metric'] = 'RMS displacement relative to augmentation transform in normalized units, not volumetric TRE'
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    stats = np.load(cfg.data.in_vitro_statistics if args.dataset == 'in_vitro' else cfg.data.statistics)
    records = []
    start = time.monotonic()
    with (output / 'samples.csv').open('w') as handle, torch.inference_mode():
        writer = None
        for i, data in enumerate(loader):
            data = to_cuda(data)
            for variant in args.variants:
                if variant not in ('legacy', 'uncapped', 'no_focus', 'surface', 'fine_only_weights',
                                   'radius04', 'radius06', 'top1', 'robust', 'half_rtor', 'no_rtor_residual'):
                    raise ValueError(variant)
                model.focus_enabled = variant != 'no_focus'
                model.focus_max_ratio = 1.0 if variant in ('uncapped', 'no_focus') else 0.5
                model.fine_matching.use_global_score = variant != 'fine_only_weights'
                model.fine_matching.acceptance_radius = {'radius04': .04, 'radius06': .06}.get(variant, .1)
                model.fine_matching.k = 1 if variant == 'top1' else 3
                model.fine_matching.robust_refinement_radius = .04 if variant == 'robust' else None
                model.rtor_residual_scale = {'half_rtor': .5, 'no_rtor_residual': 0.}.get(variant, 1.)
                pred = model(data)
                if variant == 'surface':
                    refined = refine_surface(pred['src_points'].cpu().numpy(),
                                             pred['ref_points'].cpu().numpy(),
                                             pred['estimated_transform'].cpu().numpy())
                    pred['estimated_transform'] = torch.as_tensor(refined, device='cuda', dtype=torch.float32)
                if args.dataset == 'validation':
                    source = pred['src_points']
                    error = apply_transform(source, pred['estimated_transform']) - apply_transform(source, data['transform'])
                    tre = error.square().sum(-1).mean().sqrt()
                else:
                    error = apply_transform(data['source_markers'], pred['estimated_transform']) - data['target_markers']
                    tre = error.square().sum(-1).mean().sqrt() * float(data['physical_scale'])
                index = int(data['index'])
                row = dict(index=index, sample=data['sample_name'], variant=variant,
                           visibility=float(data['overlap']) if args.dataset == 'validation' else float(stats['vis'][index]),
                           rms_tre_mm=float(tre) if args.dataset != 'validation' else None,
                           validation_rms_displacement=float(tre) if args.dataset == 'validation' else None,
                           oracle_rms_tre_mm=float(data['oracle_rms_tre_mm']) if args.dataset != 'validation' else None,
                           focused_src=int(pred['num_focused_src_nodes']),
                           valid_src=int(pred['src_node_masks'].sum()),
                           correspondences=len(pred['corr_scores']))
                if writer is None:
                    writer = csv.DictWriter(handle, fieldnames=list(row))
                    writer.writeheader()
                writer.writerow(row)
                records.append(row)
                del pred
            handle.flush()
            if (i+1) % 10 == 0 or i == 0:
                print(f'{i+1}/{len(loader)} cases, {time.monotonic()-start:.1f}s', flush=True)
    summary = {}
    for variant in args.variants:
        rows = [r for r in records if r['variant'] == variant]
        key = 'validation_rms_displacement' if args.dataset == 'validation' else 'rms_tre_mm'
        values = np.array([r[key] for r in rows])
        visibility = np.array([r['visibility'] for r in rows])
        summary[variant] = dict(n=len(rows), mean=float(values.mean()), median=float(np.median(values)),
                               failures_20mm=int((values >= 20).sum()) if args.dataset != 'validation' else None, bins=[])
        for j in range(8):
            lo, hi = (j+2)/10, (j+3)/10
            mask = (visibility >= lo) & ((visibility <= hi) if j == 7 else (visibility < hi))
            summary[variant]['bins'].append(dict(lower=lo, upper=hi, n=int(mask.sum()),
                 mean=float(values[mask].mean()) if mask.any() else None))
    (output / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
