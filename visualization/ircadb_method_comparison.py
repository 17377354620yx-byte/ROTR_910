#!/usr/bin/env python3
"""Render two-row, twelve-column 3D-IRCADb method comparisons."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.ircadb_benchmark import SampleRecord, load_samples
from visualization.error_heatmap import apply_transform
from visualization.paper_method_comparison import (
    comparison_labels,
    pointwise_gt_error_mm,
    shared_color_limit,
)
from visualization.paper_registration_visualizer import SOURCE_COLOR, TARGET_COLOR, _fit_camera
from visualization.render import render_point_clouds


METHODS = [
    ("ours", "Ours"),
    ("geotransformer", "GeoTransformer"),
    ("castv2", "CASTv2"),
    ("dfat", "DFAT"),
    ("lepard", "Lepard"),
    ("lepard_p2p", "Lepard+P2P"),
    ("parenet", "PARENet"),
    ("livermatch", "LiverMatch"),
    ("livermatch_p2p", "LiverMatch+P2P"),
    ("goicp", "Go-ICP"),
]
LABELS = comparison_labels(label for _, label in METHODS)


def artifact_paths(output_dir: Path, sample_id: str) -> dict[str, Path]:
    base = Path(output_dir) / str(sample_id)
    return {
        "png": Path(f"{base}.png"),
        "pdf": Path(f"{base}.pdf"),
        "json": Path(f"{base}.json"),
    }


def panel_artifact_paths(output_dir: Path, sample_id: str,
                         panel_keys) -> dict[str, dict[str, Path]]:
    directory = Path(output_dir) / "individual_panels" / str(sample_id)
    return {
        key: {
            kind: directory / f"{index:02d}_{key}_{kind}.png"
            for kind in ("registration", "heatmap")
        }
        for index, key in enumerate(panel_keys)
    }


def pointwise_gt_error_mm(source_mm: np.ndarray, estimate_mm: np.ndarray,
                          ground_truth_mm: np.ndarray) -> np.ndarray:
    # Public re-export keeps the 3D-IRCADb visualization API self-contained.
    from visualization.paper_method_comparison import pointwise_gt_error_mm as compute
    return compute(source_mm, estimate_mm, ground_truth_mm)


def partial_source_points(sample: dict) -> np.ndarray:
    source = np.asarray(sample["src_points"], dtype=np.float64)
    if "crop_indices" not in sample:
        raise ValueError("Sample has no crop_indices for Partial heatmap rendering")
    indices = np.asarray(sample["crop_indices"], dtype=np.int64).reshape(-1)
    if not len(indices) or np.any(indices < 0) or np.any(indices >= len(source)):
        raise ValueError("crop_indices are empty or out of source bounds")
    return np.ascontiguousarray(source[indices])


def fit_heatmap_cameras(point_sets, panel_size):
    """Tightly frame every registered Partial cloud with one fixed view direction."""
    return [
        _fit_camera([points], panel_size, None, fill_fraction=0.85)
        for points in point_sets
    ]


def select_ours_best_sample(sample_ids, result_root: Path, methods=METHODS) -> dict:
    """Select the Ours RMSE win with the largest margin over second place."""
    requested = set(map(str, sample_ids))
    values = {}
    for method, _ in methods:
        path = Path(result_root) / method / "summary.json"
        if not path.is_file():
            raise ValueError(f"Missing prediction summary for {method}: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        values[method] = {
            str(row["sample_id"]): float(row["RMSE"])
            for row in payload.get("samples", [])
            if row.get("sample_id") in requested and row.get("status") == "ok"
            and row.get("RMSE") is not None
        }
    candidates = []
    for sample_id in sorted(requested):
        ranked = sorted(
            (scores[sample_id], method)
            for method, scores in values.items() if sample_id in scores
        )
        if len(ranked) >= 2 and ranked[0][1] == "ours":
            candidates.append((ranked[1][0] - ranked[0][0], sample_id, ranked))
    if not candidates:
        raise ValueError("Ours is not the lowest-RMSE method on the selected samples")
    margin, sample_id, ranked = max(candidates)
    return {
        "sample_id": sample_id,
        "ours_rmse_mm": ranked[0][0],
        "second_rmse_mm": ranked[1][0],
        "margin_mm": margin,
    }


def load_case_transforms(sample_id: str, result_root: Path,
                         allow_failed: bool = False) -> dict[str, np.ndarray | None]:
    output = {}
    for method, _ in METHODS:
        path = Path(result_root) / method / "summary.json"
        if not path.is_file():
            raise ValueError(f"Missing prediction summary for {method}: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = [row for row in payload.get("samples", []) if row.get("sample_id") == sample_id]
        if len(rows) != 1:
            raise ValueError(
                f"Missing prediction or duplicate prediction for {method}/{sample_id}"
            )
        row = rows[0]
        if row.get("status") != "ok" or row.get("estimated_transform") is None:
            if not allow_failed:
                raise ValueError(f"Missing prediction for failed method {method}/{sample_id}")
            output[method] = None
        else:
            transform = np.asarray(row["estimated_transform"], dtype=np.float64)
            if transform.shape != (4, 4) or not np.isfinite(transform).all():
                raise ValueError(f"Invalid prediction for {method}/{sample_id}")
            output[method] = transform
    return output


def render_sample(record: SampleRecord, transforms: dict[str, np.ndarray | None],
                  output_dir: Path, dpi: int = 500, colormap: str = "viridis",
                  export_panels: bool = False) -> dict:
    sample = record.load()
    source = np.asarray(sample["src_points"], dtype=np.float64)
    partial_source = partial_source_points(sample)
    target = np.asarray(sample["ref_points"], dtype=np.float64)
    gt = np.asarray(sample["transform"], dtype=np.float64)
    method_transforms = [transforms[method] for method, _ in METHODS]
    display_transforms = [np.eye(4), *[
        np.eye(4) if transform is None else transform for transform in method_transforms
    ], gt]
    failed = [False, *[transform is None for transform in method_transforms], False]
    registered = [apply_transform(source, transform) for transform in display_transforms]
    heat_registered = [
        apply_transform(partial_source, transform) for transform in display_transforms
    ]
    errors = [
        pointwise_gt_error_mm(partial_source, transform, gt)
        for transform in display_transforms
    ]
    registered_method_errors = [
        error for error, is_failed in zip(errors[1:-1], failed[1:-1]) if not is_failed
    ]
    if not registered_method_errors:
        raise ValueError(f"No successful method predictions for {record.sample_id}")
    color_max = shared_color_limit(registered_method_errors)
    norm = Normalize(0.0, color_max, clip=True)
    cmap = matplotlib.colormaps[colormap]
    # Render every registration pair at a separately fitted scale so that
    # outlier poses do not crop the cloud in their own panel. The view direction
    # is kept consistent by _fit_camera; magnification may differ by panel.
    panel_size = (900, 750)
    registration_cameras = [
        _fit_camera([points, target], panel_size, None, fill_fraction=0.78)
        for points in registered
    ]
    heat_cameras = fit_heatmap_cameras(heat_registered, panel_size)

    top_images = [
        render_point_clouds(
            [(points, SOURCE_COLOR), (target, TARGET_COLOR)],
            registration_camera,
            width=panel_size[0], height=panel_size[1],
            point_size=3.0, backend="software",
        )
        for points, registration_camera in zip(registered, registration_cameras)
    ]
    bottom_images = [
        render_point_clouds(
            [(points, cmap(norm(error))[:, :3])],
            heat_camera,
            width=panel_size[0], height=panel_size[1],
            point_size=3.4, backend="software",
        )
        for points, error, heat_camera in zip(heat_registered, errors, heat_cameras)
    ]

    fig = plt.figure(figsize=(2.35 * len(LABELS), 7.4), facecolor="white")
    grid = fig.add_gridspec(
        2, len(LABELS), left=0.022, right=0.997, top=0.90, bottom=0.17,
        wspace=0.012, hspace=0.055,
    )
    for column, label in enumerate(LABELS):
        top = fig.add_subplot(grid[0, column])
        bottom = fig.add_subplot(grid[1, column])
        top.imshow(top_images[column])
        bottom.imshow(bottom_images[column])
        title = f"{label}\nFAILED" if failed[column] else label
        top.set_title(title, fontsize=8.0, fontweight="semibold", pad=3,
                      color="crimson" if failed[column] else "black")
        for axis in (top, bottom):
            axis.set_axis_off()
    fig.text(0.007, 0.64, "Registration", rotation=90, va="center", ha="center",
             fontsize=8.5, fontweight="semibold")
    fig.text(0.007, 0.31, "Distance to GT", rotation=90, va="center", ha="center",
             fontsize=8.5, fontweight="semibold")
    color_axis = fig.add_axes([0.38, 0.070, 0.25, 0.025])
    colorbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=color_axis,
                            orientation="horizontal")
    colorbar.set_label(f"Pointwise displacement (mm; p95 clip={color_max:.2f})",
                       fontsize=7.5, labelpad=1)
    colorbar.ax.tick_params(labelsize=6.5, length=2, pad=1)
    fig.suptitle(
        f"3D-IRCADb case {record.case_id:02d}, pair {record.pair_id:02d}, "
        f"visibility {record.visibility:.0%}",
        y=0.975, fontsize=10, fontweight="semibold",
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = artifact_paths(output_dir, record.sample_id)
    png, pdf = paths["png"], paths["pdf"]
    fig.savefig(png, dpi=dpi, facecolor="white")
    fig.savefig(pdf, dpi=dpi, facecolor="white")
    plt.close(fig)
    individual_panels = {}
    if export_panels:
        panel_keys = ["initial", *[method for method, _ in METHODS], "ground_truth"]
        panel_paths = panel_artifact_paths(output_dir, record.sample_id, panel_keys)
        for key, registration, heatmap in zip(panel_keys, top_images, bottom_images):
            registration_path = panel_paths[key]["registration"]
            heatmap_path = panel_paths[key]["heatmap"]
            registration_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(np.asarray(registration, dtype=np.uint8)).save(registration_path)
            Image.fromarray(np.asarray(heatmap, dtype=np.uint8)).save(heatmap_path)
            individual_panels[key] = {
                "registration": str(registration_path.resolve()),
                "heatmap": str(heatmap_path.resolve()),
            }
    statistics = {}
    for label, error, is_failed in zip(LABELS, errors, failed):
        statistics[label] = {
            "status": "failed" if is_failed else "ok",
            "min_mm": float(error.min()),
            "mean_mm": float(error.mean()),
            "p95_mm": float(np.percentile(error, 95)),
            "max_mm": float(error.max()),
        }
    payload = {
        "sample_id": record.sample_id,
        "case_id": record.case_id,
        "pair_id": record.pair_id,
        "visibility": record.visibility,
        "colormap": colormap,
        "shared_color_range_mm": [0.0, color_max],
        "heatmap_scope": "partial_crop_indices",
        "heatmap_point_count": int(len(partial_source)),
        "individual_panels": individual_panels,
        "png": str(png.resolve()),
        "pdf": str(pdf.resolve()),
        "statistics": statistics,
    }
    paths["json"].write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return payload


def _integers(value: str | None) -> set[int] | None:
    return None if not value else {int(item) for item in value.split(",") if item.strip()}


def main():
    global METHODS, LABELS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--case-ids")
    parser.add_argument("--pair-ids")
    parser.add_argument("--sample-ids")
    parser.add_argument("--visibility", choices=("all", "0.20", "0.30"), default="all")
    parser.add_argument("--dpi", type=int, default=500)
    parser.add_argument("--colormap", default="viridis")
    parser.add_argument("--allow-failed", action="store_true")
    parser.add_argument("--exclude-methods", default="")
    parser.add_argument("--select-best-ours", action="store_true")
    parser.add_argument("--export-panels", action="store_true")
    args = parser.parse_args()
    excluded = {item.strip() for item in args.exclude_methods.split(",") if item.strip()}
    unknown = excluded - {method for method, _ in METHODS}
    if unknown:
        raise ValueError(f"Unknown excluded methods: {sorted(unknown)}")
    METHODS = [item for item in METHODS if item[0] not in excluded]
    LABELS = comparison_labels(label for _, label in METHODS)
    sample_ids = None if not args.sample_ids else [
        item for item in args.sample_ids.split(",") if item.strip()
    ]
    records = load_samples(args.data_root, visibility=args.visibility, sample_ids=sample_ids)
    case_ids = _integers(args.case_ids)
    pair_ids = _integers(args.pair_ids)
    records = [record for record in records
               if (case_ids is None or record.case_id in case_ids)
               and (pair_ids is None or record.pair_id in pair_ids)]
    if not records:
        raise ValueError("No samples satisfy the requested visualization filters")
    selection = None
    if args.select_best_ours:
        selection = select_ours_best_sample(
            [record.sample_id for record in records], args.result_root, METHODS
        )
        records = [record for record in records if record.sample_id == selection["sample_id"]]
        print(
            f"Selected {selection['sample_id']}: Ours RMSE={selection['ours_rmse_mm']:.4f} mm, "
            f"margin={selection['margin_mm']:.4f} mm",
            flush=True,
        )
    outputs = []
    for record in records:
        transforms = load_case_transforms(record.sample_id, args.result_root, args.allow_failed)
        outputs.append(render_sample(
            record, transforms, args.output_dir, args.dpi, args.colormap,
            export_panels=args.export_panels,
        ))
        print(f"Rendered {record.sample_id}", flush=True)
    manifest = args.output_dir / "comparison_manifest.json"
    manifest.write_text(json.dumps({
        "methods": [method for method, _ in METHODS],
        "labels": LABELS,
        "colormap": args.colormap,
        "ours_best_selection": selection,
        "export_panels": args.export_panels,
        "samples": outputs,
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {manifest}")


if __name__ == "__main__":
    main()
