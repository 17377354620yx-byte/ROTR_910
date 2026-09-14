"""Evaluate the trainval.py checkpoint on both published DePoLL tasks.

No markers, associations, or ground-truth transforms enter model inference.
See DEPOLL_EVALUATION.md for coordinate conventions and comparability limits.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
from depoll_protocol import load_case, marker_errors, physical_transform


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('/home/yangx/code/new_deform/DEPOLL'))
    parser.add_argument('--snapshot', type=Path)
    parser.add_argument('--protocol', choices=['intraop', 'preop', 'both'], default='both')
    parser.add_argument('--output', type=Path, required=True, help='New output directory (refuses overwrite)')
    parser.add_argument('--audit_only', action='store_true', help='Verify all geometry/metrics without a model')
    parser.add_argument('--cases', nargs='+', default=[f'{i:02d}' for i in range(1, 14)])
    parser.add_argument('--seed', type=int, default=7351)
    parser.add_argument('--input_voxel', type=float, default=0.04,
                        help='Per-cloud normalized voxel size used in training; 0 keeps every vertex')
    args = parser.parse_args()
    if args.input_voxel < 0 or not np.isfinite(args.input_voxel):
        parser.error('--input_voxel must be finite and nonnegative')
    if len(set(args.cases)) != len(args.cases) or any(c not in [f'{i:02d}' for i in range(1,14)] for c in args.cases):
        parser.error('--cases must contain unique IDs from 01 to 13')
    if not args.audit_only and args.snapshot is None:
        parser.error('--snapshot is required for inference')
    if args.output.exists():
        parser.error(f'Output already exists; choose a new directory: {args.output}')
    args.output.mkdir(parents=True)
    protocols = ['intraop', 'preop'] if args.protocol == 'both' else [args.protocol]
    manifest = dict(command=sys.argv, root=str(args.root.resolve()), seed=args.seed,
                    input_voxel=args.input_voxel, point_limit=None,
                    metric='mean Euclidean marker distance per case; mean/std(ddof=0) across cases',
                    marker_frame='CT YAML -> M -> video',
                    association='reference index -> intraoperative index, 1-based; inverted for source order',
                    source_frame='inverse(M) applied to released source and source markers',
                    audit_only=args.audit_only)
    if not args.audit_only:
        import random
        import torch
        from config import make_cfg
        from dataset import _norm_vox
        from model import create_model
        from geotransformer.utils.data import registration_collate_fn_stack_mode
        from geotransformer.utils.torch import to_cuda
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', '4')))
        checkpoint = torch.load(args.snapshot, map_location='cpu', weights_only=False)
        protocol = checkpoint.get('metadata', {}).get('p2p_protocol', {})
        required = ['architecture', 'registration_profile', 'interaction_profile', 'dual_encoder', 'neighbor_limits']
        if any(k not in protocol for k in required):
            raise ValueError('Checkpoint lacks training protocol metadata; cannot silently guess inference configuration')
        cfg = make_cfg(protocol['architecture'], protocol['registration_profile'],
                       protocol['dual_encoder'], protocol['interaction_profile'])
        limits = protocol['neighbor_limits']
        if len(limits) != cfg.backbone.num_stages or any(int(n) <= 0 for n in limits):
            raise ValueError(f'Invalid checkpoint neighbor limits: {limits}')
        model = create_model(cfg).cuda().eval()
        model.load_state_dict(checkpoint['model'], strict=True)
        manifest.update(snapshot=str(args.snapshot.resolve()),
                        snapshot_sha256=sha256(args.snapshot),
                        checkpoint_epoch=checkpoint.get('epoch'), training_protocol=protocol,
                        config=cfg, torch_version=torch.__version__, gpu=torch.cuda.get_device_name())
        del checkpoint
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    rows = []
    for task in protocols:
        for case in args.cases:
            data = load_case(args.root, case, task)
            row = dict(protocol=task, case=case, raw_source_points=len(data['source']), raw_target_points=len(data['target']))
            for group in ['clips', 'balls']:
                baseline = marker_errors(data['source_markers'][group], data['target_markers'][group], data['released_transform'])
                row[f'released_{group}_tre_mm'] = baseline['tre_mm']
            if not args.audit_only:
                source, target = data['source'], data['target']
                scale = np.linalg.norm(source - source.mean(0), axis=1).max()
                if args.input_voxel > 0:
                    source, scale = _norm_vox(source, args.input_voxel)
                    target, _ = _norm_vox(target, args.input_voxel)
                source_center, target_center = source.mean(0), target.mean(0)
                source = ((source - source_center) / scale).astype(np.float32)
                target = ((target - target_center) / scale).astype(np.float32)
                # Deliberately no transform/marker fields in this input dict.
                sample = dict(src_points=source, ref_points=target,
                              src_feats=np.ones((len(source), 1), np.float32),
                              ref_feats=np.ones((len(target), 1), np.float32))
                start = time.perf_counter()
                batch = registration_collate_fn_stack_mode([sample], cfg.backbone.num_stages,
                          cfg.backbone.init_voxel_size, cfg.backbone.init_radius, limits)
                with torch.inference_mode():
                    result = model(to_cuda(batch))
                torch.cuda.synchronize()
                row['preprocess_and_inference_seconds'] = time.perf_counter() - start
                normalized = result['estimated_transform'].cpu().numpy()
                if not np.isfinite(normalized).all():
                    raise ValueError(f'Nonfinite prediction: {task}/{case}')
                transform = physical_transform(normalized, source_center, target_center, scale)
                saved = dict(estimated_transform_mm=transform,
                             released_source_to_video_mm=transform @ np.linalg.inv(data['released_transform']),
                             normalized_transform=normalized,
                             source_center_mm=source_center, target_center_mm=target_center,
                             physical_scale_mm=scale, released_transform_mm=data['released_transform'])
                for group in ['clips', 'balls']:
                    scores = marker_errors(data['source_markers'][group], data['target_markers'][group], transform)
                    row[f'{group}_tre_mm'] = scores['tre_mm']
                    row[f'{group}_rms_tre_mm'] = scores['rms_tre_mm']
                    saved[f'{group}_distances_mm'] = np.asarray(scores['distances_mm'])
                    saved[f'source_{group}_mm'] = data['source_markers'][group]
                    saved[f'target_{group}_mm'] = data['target_markers'][group]
                row.update(input_source_points=len(source), input_target_points=len(target),
                           fine_correspondences=int(result['corr_scores'].numel()))
                np.savez_compressed(args.output / f'{task}_{case}.npz', **saved)
                del result, batch
            rows.append(row)
            with (args.output / 'cases.csv').open('w', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            print(json.dumps(row), flush=True)
    summary = {}
    for task in protocols:
        selected = [r for r in rows if r['protocol'] == task]
        summary[task] = dict(count=len(selected))
        keys = [k for k in selected[0] if k.endswith('_mm')]
        for key in keys:
            values = np.asarray([r[key] for r in selected])
            summary[task][key] = dict(mean=float(values.mean()), std=float(values.std()),
                                     std_ddof1=float(values.std(ddof=1)) if len(values)>1 else None)
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == '__main__':
    main()
