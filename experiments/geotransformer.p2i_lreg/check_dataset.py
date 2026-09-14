"""Numerically verify P2I-LReg units and src-to-ref transform direction."""

import argparse
import json
import os.path as osp
import sys

import numpy as np


_ROOT_DIR = osp.realpath(osp.join(osp.dirname(__file__), "..", ".."))
if _ROOT_DIR in sys.path:
    sys.path.remove(_ROOT_DIR)
sys.path.insert(0, _ROOT_DIR)

from config import make_cfg
from dataset import P2ILRegDataset, chamfer_before_after


def run_check(dataset, num_samples, seed, min_improved_fraction, max_median_ratio):
    if num_samples < 100:
        raise ValueError("sanity check requires at least 100 samples")
    if num_samples > len(dataset):
        raise ValueError("num_samples exceeds dataset size")
    indices = np.random.default_rng(seed).choice(len(dataset), num_samples, replace=False)
    records = []
    for index in indices:
        sample = dataset[int(index)]
        values = chamfer_before_after(
            sample["ref_points"], sample["src_points"], sample["transform"]
        )
        records.append({"case_id": sample["case_id"], **values})
    asymmetric = np.asarray(
        [row["ref_to_src_after_m"] < row["ref_to_src_before_m"] for row in records]
    )
    symmetric = np.asarray(
        [row["symmetric_after_m"] < row["symmetric_before_m"] for row in records]
    )
    before = np.asarray([row["ref_to_src_before_m"] for row in records])
    after = np.asarray([row["ref_to_src_after_m"] for row in records])
    ratios = after / np.maximum(before, np.finfo(np.float64).eps)
    improved = asymmetric & symmetric
    summary = {
        "num_samples": int(num_samples),
        "seed": int(seed),
        "improved_count": int(improved.sum()),
        "improved_fraction": float(improved.mean()),
        "median_ref_to_src_before_m": float(np.median(before)),
        "median_ref_to_src_after_m": float(np.median(after)),
        "median_after_before_ratio": float(np.median(ratios)),
        "passed": bool(
            improved.mean() >= min_improved_fraction
            and np.median(ratios) <= max_median_ratio
        ),
        "samples": records,
    }
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--architecture", choices=["geotransformer", "rtor"], default="rtor")
    parser.add_argument("--data_root", default=None)
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=7351)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    cfg = make_cfg(args.architecture)
    if args.data_root:
        cfg.data.root = args.data_root
    dataset = P2ILRegDataset(cfg.data.root, "test", cfg)
    summary = run_check(
        dataset,
        args.num_samples,
        args.seed,
        float(cfg.protocol.sanity_min_improved_fraction),
        float(cfg.protocol.sanity_max_median_ratio),
    )
    output = args.output or osp.join(cfg.output_dir, "dataset_sanity.json")
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps({key: value for key, value in summary.items() if key != "samples"}, indent=2))
    print(f"details: {output}")
    if not summary["passed"]:
        raise SystemExit("P2I-LReg transform sanity check failed")


if __name__ == "__main__":
    main()
