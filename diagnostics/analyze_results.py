"""Align historical ablations and interventions by exact sample identity."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
HISTORY = Path('/home/yangx/code/new_deform/RTORv6/output')


def read_rows(path):
    rows = list(csv.DictReader(path.open()))
    keys = [(int(r['index']), r['sample']) for r in rows]
    if len(keys) != len(set(keys)):
        raise ValueError(f'Duplicate samples in {path}')
    return dict(zip(keys, rows))


def paired_stats(delta, groups):
    delta = np.asarray(delta)
    rng = np.random.default_rng(7351)
    # Filename groups are protocol groups, not established independent patients.
    unique = np.unique(groups)
    sums = np.array([delta[np.array(groups) == g].sum() for g in unique])
    counts = np.array([np.sum(np.array(groups) == g) for g in unique])
    resamples = rng.integers(len(unique), size=(5000, len(unique)))
    cluster_means = sums[resamples].sum(1) / counts[resamples].sum(1)
    return dict(mean_delta_mm=float(delta.mean()), median_delta_mm=float(np.median(delta)),
                improved_fraction=float((delta < 0).mean()),
                protocol_groups=len(unique),
                protocol_group_bootstrap_ci95=np.quantile(cluster_means, [.025, .975]).tolist(),
                inference_limit='Descriptive, single training seed; filename groups are not verified patient identities')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', choices=['in_vitro', 'in_silico'], default='in_vitro')
    p.add_argument('--interventions', nargs='*', default=[])
    p.add_argument('--output', required=True)
    args = p.parse_args()
    output = Path(args.output).resolve()
    if not output.is_relative_to(ROOT):
        p.error('Output must be inside this project')
    datasets = {}
    paths = {}
    for arch in ('geotransformer', 'rtor_only', 'a3_only', 'rtor_a3'):
        path = HISTORY / f'geotransformer.p2p_liver.ablation.scratch_v2.{arch}.seed7351' / f'evaluation_epoch150/{args.dataset}_noise_none.csv'
        if not path.exists():
            path = HISTORY / f'geotransformer.p2p_liver.ablation.scratch_v2_eval.{arch}.seed7351' / f'evaluation_epoch150/{args.dataset}_noise_none.csv'
        datasets[arch] = read_rows(path)
        paths[arch] = str(path)
    for path_string in args.interventions:
        path = Path(path_string)
        rows = list(csv.DictReader(path.open()))
        for variant in sorted(set(r['variant'] for r in rows)):
            key = path.parent.name + '/' + variant
            datasets[key] = {(int(r['index']), r['sample']): r for r in rows if r['variant'] == variant}
            paths[key] = str(path)
    result = dict(dataset=args.dataset, sources=paths, comparisons={})
    baseline = datasets['geotransformer']
    for label, data in datasets.items():
        keys = sorted(data)
        if not set(keys).issubset(baseline):
            raise ValueError(f'Unmatched samples: {label}')
        values = np.array([float(data[k]['rms_tre_mm']) for k in keys])
        ref = np.array([float(baseline[k]['rms_tre_mm']) for k in keys])
        vis = np.array([float(baseline[k]['visibility']) for k in keys])
        groups = [k[1].split('_')[0] for k in keys]
        rows = dict(n=len(keys), mean_tre_mm=float(values.mean()),
                    std_tre_mm=float(values.std()), failures_20mm=int((values >= 20).sum()),
                    versus_baseline=paired_stats(values-ref, groups), visibility_bins=[])
        for j in range(8):
            lo, hi = (j+2)/10, (j+3)/10
            mask = (vis >= lo) & ((vis <= hi) if j == 7 else (vis < hi))
            rows['visibility_bins'].append(dict(lower=lo, upper=hi, n=int(mask.sum()),
                mean_tre_mm=float(values[mask].mean()) if mask.any() else None,
                delta_baseline_mm=float((values-ref)[mask].mean()) if mask.any() else None))
        result['comparisons'][label] = rows
    output.write_text(json.dumps(result, indent=2))
    for name, value in result['comparisons'].items():
        print(name, value['n'], round(value['mean_tre_mm'], 6), value['failures_20mm'])


if __name__ == '__main__':
    main()
