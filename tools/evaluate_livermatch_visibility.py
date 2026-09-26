#!/usr/bin/env python3
"""Evaluate LiverMatch or LiverMatch+P2P on a standalone P2P subset."""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch

BENCHMARK_PROJECT = Path(__file__).resolve().parents[1]
if str(BENCHMARK_PROJECT) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_PROJECT))

from tools.p2p_benchmark_metrics import (
    METRIC_DEFINITIONS,
    correspondence_inlier_ratio,
    pose_metrics,
    summarize_metrics,
)


PROJECT = Path("/home/yangx/code/point_clould/LiverMatch")


def prediction_export_fields(
    details: dict, *, status: str = "ok", error: str | None = None
) -> dict:
    transform = details.get("estimated_transform") if status == "ok" else None
    return {
        "status": status,
        "error": error,
        "estimated_transform": (
            None if transform is None else np.asarray(transform, dtype=np.float64).tolist()
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("base", "p2p"), required=True)
    parser.add_argument("--dataset", choices=("in_silico", "in_vitro"), required=True)
    parser.add_argument("--noise", choices=("none", "4"), default="none")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT / "configs/new_task3_004_002.yaml",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--inlier-radius", type=float, default=0.1)
    return parser.parse_args()


def load_legacy_module(mode: str, dataset: str):
    sys.path.insert(0, str(PROJECT))
    if mode == "p2p":
        # The released scripts use an unshipped lib.utils module. The required
        # implementations are shipped as p2p_util.py in the same project.
        sys.modules["lib.utils"] = importlib.import_module("p2p_util")
    module_name = {
        ("base", "in_silico"): "eval_in_silico",
        ("p2p", "in_silico"): "eval_in_silico_p2p",
        ("base", "in_vitro"): "eva_in_vitro",
        ("p2p", "in_vitro"): "eva_in_vitro_p2p",
    }[(mode, dataset)]
    return importlib.import_module(module_name)


def dataset_paths(dataset: str, root: Path) -> tuple[str, Path, Path, Path]:
    root = root.resolve()
    if dataset == "in_silico":
        data = root / "Deform_mesh_npz_test/Test"
        sample_list = root / "Deform_mesh_npz_test/list.npz"
        statistics = root / "Deform_mesh_npz_test/stat_svd.npz"
    else:
        data = root / "Rigid_test_data"
        sample_list = root / "rigid_list.npy"
        statistics = root / "stat.npz"
    original_indices = root / "original_indices.npy"
    for path in (data, sample_list, statistics, original_indices):
        if not path.exists():
            raise FileNotFoundError(path)
    return str(data) + os.sep, sample_list, statistics, original_indices


def summarize(values: np.ndarray) -> dict[str, float | int]:
    return {
        "count": int(len(values)),
        "mean_rms_tre_mm": float(values.mean()),
        "std_rms_tre_mm": float(values.std()),
        "success_rate_20mm": float((values < 20.0).mean()),
        "success_rate_5mm": float((values < 5.0).mean()),
    }


def fit_rigid(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Fit the source-to-target rigid oracle from paired volume markers."""
    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    u, _, vt = np.linalg.svd(
        (source - source_center).T @ (target - target_center)
    )
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1] *= -1
        rotation = vt.T @ u.T
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = target_center - rotation @ source_center
    return transform


def main() -> None:
    args = parse_args()
    if args.dataset == "in_vitro" and args.noise != "none":
        raise ValueError("in_vitro does not contain synthetic noise fields")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    if args.k <= 0:
        raise ValueError("--k must be positive")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    module = load_legacy_module(args.mode, args.dataset)
    data_root, sample_list, statistics_path, original_indices_path = dataset_paths(
        args.dataset, args.data_root
    )
    config = module.edict(module.load_config(str(args.config)))
    config.architecture = module.architectures[config.model_name]
    config.device = torch.device(args.device)
    module.config = config
    model_kwargs = {"K": args.k} if args.mode == "p2p" else {}
    module.model = module.KPFCNN(config, **model_kwargs).to(config.device).eval()
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    module.model.load_state_dict(state["state_dict"], strict=True)

    noise = None if args.noise == "none" else int(args.noise)
    module.demo_set = module.LiverDemo(
        config,
        data_root,
        str(sample_list),
        voxel_size=0.04,
        sigma=noise,
        rot=True,
    )
    count = len(module.demo_set) if args.limit <= 0 else min(args.limit, len(module.demo_set))
    with np.load(statistics_path, allow_pickle=False) as statistics:
        visibility = np.asarray(statistics["vis"])
        deformation = np.asarray(statistics["deform"])
    original_indices = np.load(original_indices_path, allow_pickle=False)
    if not (len(module.demo_set) == len(visibility) == len(deformation) == len(original_indices)):
        raise ValueError("Subset list/statistics/original-index lengths do not match")

    rows = []
    started = time.perf_counter()
    for index in range(count):
        kwargs = {"debug": False, "use_SVD": True, "return_details": True}
        if args.mode == "p2p":
            kwargs.update(cluster=True, use_corr_cl=True, use_all_matches=True)
        details = module.eva_one(index, **kwargs)
        rms_tre = float(details["rms_tre"])
        oracle = fit_rigid(details["source_markers"], details["target_markers"])
        matches = details["matches"]
        if len(matches):
            source_corr = details["source"][matches[:, 0]]
            target_corr = details["target"][matches[:, 1]]
            ir = correspondence_inlier_ratio(
                source_corr, target_corr, oracle, args.inlier_radius
            )
        else:
            ir = float("nan")
        extended = pose_metrics(
            details["source"], details["target"],
            details["estimated_transform"], oracle, details["physical_scale"],
        )
        row = {
            "index": index,
            "original_index": int(original_indices[index]),
            "sample": str(module.demo_set.test_list[index]),
            "visibility": float(visibility[index]),
            "deformation_mm": float(deformation[index]),
            "rms_tre_mm": float(rms_tre),
            "success_20mm": int(rms_tre < 20.0),
            "success_5mm": int(rms_tre < 5.0),
            "SR_5mm": float(rms_tre < 5.0),
            "SR_20mm": float(rms_tre < 20.0),
            "PIR": None,
            "IR": ir,
            **extended,
            **prediction_export_fields(details),
        }
        rows.append(row)
        if index == 0 or (index + 1) % args.log_every == 0 or index + 1 == count:
            print(
                f"[{index + 1}/{count}] {row['sample']} RMS-TRE={rms_tre:.4f} mm "
                f"IR={ir:.4f}",
                flush=True,
            )

    values = np.asarray([row["rms_tre_mm"] for row in rows], dtype=np.float64)
    method = "LiverMatch+P2P" if args.mode == "p2p" else "LiverMatch"
    payload = {
        "method": method,
        "dataset": args.dataset,
        "noise_mm": noise,
        "checkpoint": str(args.checkpoint.resolve()),
        "seed": args.seed,
        "p2p_k": args.k if args.mode == "p2p" else None,
        "metric": "RMS-TRE on paired volumetric fiducials, millimetres",
        "metric_note": "PIR is N/A because LiverMatch has no separate coarse stage. Primary success is SR@5mm; RR requires RRE < 5 degrees and RTE < 5 mm. SR@20mm is legacy.",
        "metric_definitions": METRIC_DEFINITIONS,
        "evaluated_samples": len(rows),
        "elapsed_seconds": time.perf_counter() - started,
        "summary": summarize(values),
        "test_metrics": summarize_metrics(rows),
        "samples": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    csv_path = args.output.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            csv_row = dict(row)
            csv_row["estimated_transform"] = json.dumps(row["estimated_transform"])
            writer.writerow(csv_row)
    np.save(args.output.with_name(args.output.stem + "_rms_tre_mm.npy"), values)
    summary = payload["summary"]
    print(
        f"{method}: RMS-TRE={summary['mean_rms_tre_mm']:.4f} +/- "
        f"{summary['std_rms_tre_mm']:.4f} mm, "
        f"SR@5mm={summary['success_rate_5mm']:.4f}, "
        f"legacy SR@20mm={summary['success_rate_20mm']:.4f}"
    )
    print(f"Results written to {args.output} and {csv_path}")


if __name__ == "__main__":
    main()
