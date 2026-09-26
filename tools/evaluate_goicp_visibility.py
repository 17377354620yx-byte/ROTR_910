#!/usr/bin/env python3
"""Go-ICP P2P evaluator that retains successful transforms for visualization."""

import csv
import json
from pathlib import Path
import sys
import time

import numpy as np


GOICP_ROOT = Path("/home/yangx/code/new_deform/go-icp_cython")
if str(GOICP_ROOT) not in sys.path:
    sys.path.insert(0, str(GOICP_ROOT))

import evaluate_p2p_visibility as _impl  # noqa: E402
from evaluate_p2p_visibility import *  # noqa: F401,F403,E402


def visualization_transform(status: str, estimated_transform):
    """Return a real successful estimate; never visualize a failure fallback."""
    if status != "ok":
        return None
    return np.asarray(estimated_transform, dtype=np.float64)


def main() -> None:
    args = _impl.parse_args()
    if args.dataset == "in_vitro" and args.noise != "none":
        raise ValueError("in_vitro does not contain synthetic noise fields")
    if not (0.0 <= args.trim_fraction < 1.0):
        raise ValueError("--trim-fraction must be in [0, 1)")
    if args.dt_size <= 0 or args.timeout <= 0:
        raise ValueError("--dt-size and --timeout must be positive")
    if not args.goicp_root.is_dir():
        raise FileNotFoundError(args.goicp_root)

    names, visibility, deformation, original_indices = _impl.dataset_manifest(
        args.dataset, args.data_root
    )
    count = len(names) if args.limit <= 0 else min(args.limit, len(names))
    noise_mm = None if args.noise == "none" else int(args.noise)
    rows = []
    started = time.perf_counter()
    for index in range(count):
        case_started = time.perf_counter()
        case = _impl.load_p2p_sample(
            _impl.sample_path(args.dataset, args.data_root, names[index]),
            dataset=args.dataset,
            noise_mm=noise_mm,
            voxel_size=args.voxel_size,
        )
        source_cube, target_cube, source_center, target_center, cube_scale = (
            _impl.normalize_for_goicp(case["source"], case["target"])
        )
        source_input = _impl.deterministic_subsample_by_id(
            source_cube, args.max_points, names[index], "source", args.seed
        )
        target_input = _impl.deterministic_subsample_by_id(
            target_cube, args.max_points, names[index], "target", args.seed
        )
        result = _impl.run_goicp_with_timeout(
            args.goicp_root,
            source_input,
            target_input,
            trim_fraction=args.trim_fraction,
            dt_size=args.dt_size,
            mse_threshold=args.mse_threshold,
            timeout=args.timeout,
        )
        estimated = None
        if result["status"] == "ok":
            estimated = _impl.recover_source_to_target_transform(
                result["rotation"], result["translation"],
                source_center, target_center, cube_scale,
            )
        metric_transform = estimated if estimated is not None else np.eye(4)
        residuals = (
            _impl.apply_transform(case["source_markers"], metric_transform)
            - case["target_markers"]
        )
        rms_tre = float(
            np.sqrt(np.mean(np.sum(residuals ** 2, axis=1))) * case["physical_scale"]
        )
        extended = _impl.pose_metrics(
            case["source"], case["target"], metric_transform,
            case["oracle_transform"], case["physical_scale"],
        )
        strict = _impl.strict_success_metrics(rms_tre, extended["RRE"], extended["RTE"])
        row = {
            "index": index,
            "original_index": int(original_indices[index]),
            "sample": names[index],
            "visibility": float(visibility[index]),
            "deformation_mm": float(deformation[index]),
            "rms_tre_mm": rms_tre,
            "success_5mm": int(strict["SR_5mm"]),
            "success_20mm": int(rms_tre < 20.0),
            "SR_5mm": strict["SR_5mm"],
            "SR_20mm": float(rms_tre < 20.0),
            "PIR": None,
            "IR": None,
            **extended,
            "status": result["status"],
            "error": result.get("error"),
            "solver_status": result["status"],
            "solver_error": result.get("error"),
            "solver_objective": result.get("objective"),
            "fallback_identity": result["status"] != "ok",
            "estimated_transform": (
                None if visualization_transform(result["status"], estimated) is None
                else estimated.tolist()
            ),
            "source_points": len(source_input),
            "target_points": len(target_input),
            "elapsed_seconds": time.perf_counter() - case_started,
        }
        rows.append(row)
        print(
            f"[{index + 1}/{count}] {names[index]} status={result['status']} "
            f"RMS-TRE={rms_tre:.4f} mm",
            flush=True,
        )

    values = np.asarray([row["rms_tre_mm"] for row in rows], dtype=np.float64)
    payload = {
        "method": "Go-ICP",
        "dataset": args.dataset,
        "noise_mm": noise_mm,
        "goicp_root": str(args.goicp_root.resolve()),
        "parameters": {
            "max_points_per_cloud": args.max_points,
            "trim_fraction": args.trim_fraction,
            "dt_size": args.dt_size,
            "dt_factor": 2.0,
            "mse_threshold": args.mse_threshold,
            "timeout_seconds": args.timeout,
            "seed": args.seed,
        },
        "evaluated_samples": len(rows),
        "elapsed_seconds": time.perf_counter() - started,
        "samples": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    with args.output.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            csv_row = dict(row)
            csv_row["estimated_transform"] = json.dumps(row["estimated_transform"])
            writer.writerow(csv_row)
    np.save(args.output.with_name(args.output.stem + "_rms_tre_mm.npy"), values)
    print(f"Results written to {args.output}")


if __name__ == "__main__":
    main()
