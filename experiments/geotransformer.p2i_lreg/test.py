"""Fair P2I-LReg rigid evaluation with shared top-k, estimators and metrics."""

import argparse
import csv
import json
import os.path as osp
import sys


ROOT = osp.realpath(osp.join(osp.dirname(__file__), "..", ".."))
if ROOT in sys.path:
    sys.path.remove(ROOT)
sys.path.insert(0, ROOT)

import numpy as np
import torch

from geotransformer.utils.torch import to_cuda

from config import make_cfg
from dataset import reference_filter_manifest, test_data_loader
from estimators import estimate_pose, select_topk_correspondences
from metrics import (
    MetricAccumulator,
    correspondence_inlier_ratio,
    registration_errors,
    registration_recall,
)
from model import create_model, forward_without_ground_truth


def _load_checkpoint(model, path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    return checkpoint.get("metadata", {}).get("p2i_lreg_protocol", {})


@torch.no_grad()
def evaluate(cfg, snapshot, methods, topks, limit, output_path):
    model = create_model(cfg).cuda().eval()
    metadata = _load_checkpoint(model, snapshot)
    if metadata.get("architecture", cfg.ablation.architecture) != cfg.ablation.architecture:
        raise ValueError("checkpoint architecture does not match --architecture")
    if "neighbor_limits" in metadata:
        cfg.data.neighbor_limits = metadata["neighbor_limits"]
    loader, limits = test_data_loader(cfg, limit=limit)
    accumulators = {
        (method, topk): MetricAccumulator(cfg.protocol.fmr_inlier_ratio_threshold)
        for method in methods
        for topk in topks
    }
    rows = []
    for sample_index, cpu_data in enumerate(loader):
        data = to_cuda(cpu_data)
        output = forward_without_ground_truth(model, data)
        for topk in topks:
            ref_corr, src_corr, _ = select_topk_correspondences(
                output["ref_corr_points"], output["src_corr_points"], output["corr_scores"], topk
            )
            ir = correspondence_inlier_ratio(
                ref_corr, src_corr, data["transform"], cfg.protocol.ir_distance_threshold_m
            )
            for method in methods:
                pose = estimate_pose(method, output, cfg, topk, seed=cfg.seed + sample_index)
                if pose.valid:
                    rre, rte = registration_errors(data["transform"], pose.transform)
                    rr = registration_recall(
                        output["src_points"],
                        data["transform"],
                        pose.transform,
                        cfg.protocol.rr_mean_displacement_threshold_m,
                    )
                    rre_value, rte_value, rr_value = float(rre), float(rte), float(rr)
                else:
                    rre_value, rte_value, rr_value = float("nan"), float("nan"), 0.0
                accumulators[(method, topk)].add(
                    ir=float(ir), rr=rr_value, rre_deg=rre_value, rte_mm=rte_value, pose_valid=pose.valid
                )
                rows.append(
                    {
                        "case_id": cpu_data["case_id"],
                        "architecture": cfg.ablation.architecture,
                        "method": method,
                        "topk": int(topk),
                        "IR": float(ir),
                        "RR": rr_value,
                        "RRE_deg": rre_value,
                        "RTE_mm": rte_value,
                        "pose_valid": int(pose.valid),
                    }
                )
    summaries = []
    for (method, topk), accumulator in accumulators.items():
        summaries.append({"architecture": cfg.ablation.architecture, "method": method, "topk": topk, **accumulator.summary()})
    payload = {
        "protocol": dict(cfg.protocol),
        "checkpoint": osp.abspath(snapshot),
        "neighbor_limits": [int(value) for value in limits],
        "reference_filter": reference_filter_manifest(loader.dataset.excluded_records),
        "summaries": summaries,
        "samples": rows,
    }
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=True)
    csv_path = osp.splitext(output_path)[0] + ".csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--architecture", choices=["geotransformer", "rtor"], required=True)
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--estimators", nargs="+", choices=["lgr", "weighted_svd", "ransac_50k"], default=["lgr", "weighted_svd", "ransac_50k"])
    parser.add_argument("--topk", nargs="+", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    cfg = make_cfg(args.architecture)
    topks = args.topk or list(cfg.protocol.topk)
    output = args.output or osp.join(cfg.output_dir, "test_summary.json")
    payload = evaluate(cfg, args.snapshot, args.estimators, topks, args.limit, output)
    print(json.dumps(payload["summaries"], indent=2))


if __name__ == "__main__":
    main()
