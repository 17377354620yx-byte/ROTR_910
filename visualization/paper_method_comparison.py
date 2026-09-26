#!/usr/bin/env python3
"""Compose per-case SCI figures comparing nine P2P liver methods."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from visualization.error_heatmap import apply_transform
from visualization.paper_registration_visualizer import (
    SOURCE_COLOR,
    TARGET_COLOR,
    _fit_camera,
    _native_dataset,
    compute_partial_registration_error,
)
from visualization.render import render_point_clouds


OUTPUT_ROOT = ROOT / "output/visualization/paper_comparison"
CASE_IDS = [164, 2355, 360, 3390, 413, 5816, 2912, 4342, 524, 5897]
METHODS = [
    "Ours", "GeoTransformer", "CASTv2", "DFAT", "Lepard",
    "Lepard+P2P", "PARENet", "LiverMatch", "LiverMatch+P2P",
]


def comparison_labels(method_labels) -> list[str]:
    """Return the fixed paper panel order for an arbitrary method sequence."""
    return ["Initial", *list(method_labels), "Ground Truth"]


def pointwise_gt_error_mm(source_mm: np.ndarray, estimate_mm: np.ndarray,
                          ground_truth_mm: np.ndarray) -> np.ndarray:
    """Physical displacement between an estimated and GT transformed source."""
    estimated = apply_transform(np.asarray(source_mm), np.asarray(estimate_mm))
    expected = apply_transform(np.asarray(source_mm), np.asarray(ground_truth_mm))
    return np.linalg.norm(estimated - expected, axis=1)


def shared_color_limit(error_arrays, percentile: float = 95.0) -> float:
    values = np.concatenate([np.asarray(values).reshape(-1) for values in error_arrays])
    finite = values[np.isfinite(values)]
    if not len(finite):
        raise ValueError("Cannot derive a color range from empty/non-finite errors")
    return max(float(np.percentile(finite, percentile)), np.finfo(np.float64).eps)


def _npz_transform(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as data:
        return np.asarray(data["estimated_transform"], dtype=np.float64)


def load_transforms() -> dict[int, dict[str, np.ndarray]]:
    ours = OUTPUT_ROOT / "predictions/ours"
    geo = OUTPUT_ROOT / "predictions/geotransformer"
    dfat = OUTPUT_ROOT / "predictions/dfat"
    parenet = OUTPUT_ROOT / "predictions/parenet/summary_predictions"
    liver = OUTPUT_ROOT / "predictions/livermatch"
    liver_p2p = OUTPUT_ROOT / "predictions/livermatch_p2p"
    cast_summary = json.loads((OUTPUT_ROOT / "predictions/castv2/summary.json").read_text())
    cast_array = np.load(OUTPUT_ROOT / "predictions/castv2/summary_transforms.npy")
    cast = {
        int(record["index"]): np.asarray(transform, dtype=np.float64)
        for record, transform in zip(cast_summary["samples"], cast_array)
    }
    lepard_summary = json.loads((OUTPUT_ROOT / "predictions/lepard/summary.json").read_text())
    lepard = {int(row["index"]): row for row in lepard_summary["samples"]}
    output = {}
    for index in CASE_IDS:
        filename = f"{index:05d}.npz"
        row = lepard[index]
        p2p = row["p2p_transform"]
        if p2p is None:
            p2p = row["ransac_transform"] or row["estimated_transform"]
        output[index] = {
            "Ours": _npz_transform(ours / filename),
            "GeoTransformer": _npz_transform(geo / filename),
            "CASTv2": cast[index],
            "DFAT": _npz_transform(dfat / filename),
            "Lepard": np.asarray(row["estimated_transform"], dtype=np.float64),
            "Lepard+P2P": np.asarray(p2p, dtype=np.float64),
            "PARENet": _npz_transform(parenet / filename),
            "LiverMatch": _npz_transform(liver / filename),
            "LiverMatch+P2P": _npz_transform(liver_p2p / filename),
        }
    return output


def marker_rms(sample: dict, transform: np.ndarray) -> float:
    delta = apply_transform(sample["source_markers"], transform) - sample["target_markers"]
    return float(np.sqrt(np.mean(np.sum(delta * delta, axis=1))) * sample["physical_scale"])


def render_case(index: int, sample: dict, transforms: dict[str, np.ndarray],
                output_dir: Path, dpi: int) -> dict:
    source = np.asarray(sample["src_points"])
    target = np.asarray(sample["ref_points"])
    gt = np.asarray(sample["transform"])
    scale = float(sample["physical_scale"])
    labels = comparison_labels(METHODS)
    all_transforms = [np.eye(4), *(transforms[name] for name in METHODS), gt]

    partial_indices = None
    errors = []
    heat_points = []
    for transform in all_transforms:
        pred, _, error, indices, _ = compute_partial_registration_error(
            source, target, transform, gt, scale_factor=scale,
            partial_indices=partial_indices,
        )
        if partial_indices is None:
            partial_indices = indices
        heat_points.append(pred)
        errors.append(error)

    # Derive the shared scale from registered methods only.  Initial alignment
    # can contain very large errors and otherwise compress all method colors.
    registered_errors = np.concatenate(errors[1:-1])
    color_max = max(1.0, shared_color_limit([registered_errors]))
    norm = Normalize(0.0, color_max, clip=True)
    cmap = matplotlib.colormaps["viridis"]
    panel_size = (520, 430)
    registered_sets = [apply_transform(source, transform) for transform in all_transforms]
    camera = _fit_camera(
        [source, target, *registered_sets], panel_size, None, fill_fraction=0.80
    )
    heat_camera = _fit_camera(
        [apply_transform(source[partial_indices], gt)], panel_size, None,
        fill_fraction=0.74,
    )

    top_images = []
    bottom_images = []
    for registered, points, error in zip(registered_sets, heat_points, errors):
        top_images.append(render_point_clouds(
            [(registered, SOURCE_COLOR), (target, TARGET_COLOR)], camera,
            width=panel_size[0], height=panel_size[1], point_size=2.4,
            backend="software",
        ))
        bottom_images.append(render_point_clouds(
            [(points, cmap(norm(error))[:, :3])], heat_camera,
            width=panel_size[0], height=panel_size[1], point_size=3.0,
            backend="software",
        ))

    fig = plt.figure(figsize=(20.5, 5.05), facecolor="white")
    grid = fig.add_gridspec(
        2, len(labels), left=0.025, right=0.995, top=0.90, bottom=0.17,
        wspace=0.012, hspace=0.055,
    )
    for column, label in enumerate(labels):
        top = fig.add_subplot(grid[0, column])
        bottom = fig.add_subplot(grid[1, column])
        top.imshow(top_images[column])
        bottom.imshow(bottom_images[column])
        top.set_title(label, fontsize=8.2, fontweight="semibold", pad=3)
        for axis in (top, bottom):
            axis.set_axis_off()
        error = errors[column]
        bottom.text(
            0.5, 0.015,
            f"[{error.min():.1f}, {error.max():.1f}] mm",
            transform=bottom.transAxes, ha="center", va="bottom", fontsize=6.7,
            color="white", bbox={"facecolor": "black", "edgecolor": "none",
                                  "alpha": 0.68, "pad": 1.2},
        )
    fig.text(0.008, 0.64, "Registration", rotation=90, va="center", ha="center",
             fontsize=8.5, fontweight="semibold")
    fig.text(0.008, 0.31, "Distance", rotation=90, va="center", ha="center",
             fontsize=8.5, fontweight="semibold")
    color_axis = fig.add_axes([0.37, 0.072, 0.27, 0.025])
    colorbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=color_axis,
                            orientation="horizontal")
    colorbar.set_label(
        f"Surface distance (mm; values ≥ {color_max:.1f} clipped)", fontsize=7.5,
        labelpad=1,
    )
    colorbar.ax.tick_params(labelsize=6.5, length=2, pad=1)
    fig.suptitle(f"Case {index:05d}", y=0.975, fontsize=10, fontweight="semibold")

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / f"case_{index:05d}_comparison"
    fig.savefig(stem.with_suffix(".png"), dpi=dpi, facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), dpi=dpi, facecolor="white")
    plt.close(fig)

    stats = {}
    for label, transform, error in zip(labels, all_transforms, errors):
        stats[label] = {
            "marker_rms_tre_mm": marker_rms(sample, transform),
            "surface_min_mm": float(error.min()),
            "surface_mean_mm": float(error.mean()),
            "surface_p95_mm": float(np.percentile(error, 95)),
            "surface_max_mm": float(error.max()),
        }
    payload = {
        "case_index": index,
        "overlap": float(sample["overlap"]),
        "png": str(stem.with_suffix(".png").resolve()),
        "pdf": str(stem.with_suffix(".pdf").resolve()),
        "shared_color_range_mm": [0.0, color_max],
        "statistics": stats,
    }
    stem.with_suffix(".json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT / "figures")
    parser.add_argument("--dpi", type=int, default=400)
    parser.add_argument("--indices", help="Comma-separated subset of available case indices")
    parser.add_argument("--max-overlap", type=float,
                        help="Keep cases whose overlap is strictly below this value")
    args = parser.parse_args()
    transforms = load_transforms()
    dataset = _native_dataset("in_silico", None)
    selection_summary = json.loads((
        ROOT / "output/geotransformer.p2p_liver.rtor_a3_cooperative_v1_epoch150"
        / "in_silico_noise_none.json"
    ).read_text(encoding="utf-8"))
    overlap_by_index = {
        int(row["index"]): float(row["visibility"])
        for row in selection_summary["samples"]
    }
    case_ids = CASE_IDS
    if args.indices:
        requested = [int(value) for value in args.indices.split(",") if value.strip()]
        unavailable = sorted(set(requested) - set(CASE_IDS))
        if unavailable:
            raise ValueError(f"No saved predictions for case indices: {unavailable}")
        case_ids = requested
    if args.max_overlap is not None:
        case_ids = [index for index in case_ids
                    if overlap_by_index[index] < args.max_overlap]
    if not case_ids:
        raise ValueError("No cases satisfy the requested selection")
    outputs = []
    for position, index in enumerate(case_ids):
        if args.max_overlap is not None:
            group = "low_overlap"
        else:
            group = "best" if CASE_IDS.index(index) < 5 else "failure"
        sample = dict(dataset[index])
        sample["overlap"] = overlap_by_index[index]
        output = render_case(index, sample, transforms[index],
                             args.output_dir / group, args.dpi)
        outputs.append(output)
        print(f"Rendered {group} case {index:05d}", flush=True)
    manifest = args.output_dir / "comparison_manifest.json"
    manifest.write_text(
        json.dumps({"methods": METHODS, "cases": outputs}, indent=2,
                   ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    csv_path = args.output_dir / "marker_rms_tre_mm.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["case_index", "selection", "overlap", *METHODS])
        for position, output in enumerate(outputs):
            writer.writerow([
                output["case_index"],
                "low_overlap" if args.max_overlap is not None else
                ("best" if CASE_IDS.index(output["case_index"]) < 5 else "failure"),
                f"{output['overlap']:.6f}",
                *(f"{output['statistics'][name]['marker_rms_tre_mm']:.6f}"
                  for name in METHODS),
            ])
    print(f"Wrote {manifest}")


if __name__ == "__main__":
    main()
