#!/usr/bin/env python3
"""Render two LiverMatch-style 3x12 best-case registration comparisons."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tools.evaluate_goicp_visibility import load_p2p_sample
from visualization.error_heatmap import apply_transform
from visualization.render import make_camera, render_point_clouds


METHODS = (
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
)
METHOD_KEYS = tuple(method for method, _ in METHODS)
COLUMN_LABELS = ("Initial Position", *(label for _, label in METHODS), "Ground Truth")
SOURCE_COLOR = np.asarray([0, 150, 255], dtype=np.float64) / 255.0
TARGET_COLOR = np.asarray([254, 92, 92], dtype=np.float64) / 255.0


def marker_rms_tre(
    source_markers: np.ndarray,
    target_markers: np.ndarray,
    transform: np.ndarray,
    physical_scale: float,
) -> float:
    residuals = apply_transform(source_markers, transform) - target_markers
    return float(np.sqrt(np.mean(np.sum(residuals * residuals, axis=1))) * physical_scale)


def _title(label: str, rms_tre_mm: float, status: str) -> str:
    if status != "ok":
        return f"{label}\nFAILED"
    return f"{label}\nRMS-TRE: {rms_tre_mm:.2f} mm"


def build_panel_specs(
    source: np.ndarray,
    target: np.ndarray,
    source_markers: np.ndarray,
    target_markers: np.ndarray,
    ground_truth: np.ndarray,
    predictions: Mapping[str, Mapping],
    *,
    physical_scale: float,
) -> tuple[list[dict], dict]:
    transforms = [np.eye(4)]
    statuses = ["ok"]
    errors = [None]
    paths = [None]
    checkpoints = [None]
    for method in METHOD_KEYS:
        prediction = predictions[method]
        status = str(prediction.get("status", "ok"))
        transforms.append(
            np.eye(4) if status != "ok" else np.asarray(prediction["transform"], dtype=np.float64)
        )
        statuses.append(status)
        errors.append(prediction.get("error"))
        paths.append(prediction.get("prediction_path"))
        checkpoints.append(prediction.get("checkpoint"))
    transforms.append(np.asarray(ground_truth, dtype=np.float64))
    statuses.append("ok")
    errors.append(None)
    paths.append(None)
    checkpoints.append(None)
    registered = [apply_transform(source, transform) for transform in transforms]
    camera = make_camera([target, *registered])
    specs = []
    for label, transform, points, status, error, path, checkpoint in zip(
        COLUMN_LABELS, transforms, registered, statuses, errors, paths, checkpoints
    ):
        rms = marker_rms_tre(
            source_markers, target_markers, transform, physical_scale
        )
        specs.append(
            {
                "label": label,
                "title": _title(label, rms, status),
                "status": status,
                "error": error,
                "rms_tre_mm": rms,
                "registered_source": points,
                "target": target,
                "camera": camera,
                "prediction_path": path,
                "checkpoint": checkpoint,
            }
        )
    return specs, camera


def _safe_sample(value: str) -> str:
    stem = Path(value).stem
    return "".join(char if char.isalnum() or char in "-_" else "_" for char in stem)


def _load_prediction(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as payload:
        metadata = json.loads(str(payload["metadata_json"]))
        result = {
            "status": str(payload["status"]),
            "error": metadata.get("error"),
            "checkpoint": metadata.get("checkpoint"),
            "prediction_path": str(path.resolve()),
        }
        if "estimated_transform" in payload.files:
            result["transform"] = np.asarray(payload["estimated_transform"])
    return result


def _sample_path(root: Path, dataset: str, sample: str) -> Path:
    folder = "Deform_mesh_npz_test/Test" if dataset == "in_silico" else "Rigid_test_data"
    return root / "subsets" / dataset / folder / sample


def _load_case(root: Path, dataset: str, sample: str) -> dict:
    return load_p2p_sample(
        _sample_path(root, dataset, sample),
        dataset=dataset,
        noise_mm=None,
        voxel_size=0.04,
    )


def _render_panel_pyvista(spec: Mapping, *, width: int, height: int, point_size: float):
    import pyvista as pv

    plotter = pv.Plotter(off_screen=True, window_size=(width, height))
    plotter.set_background("white")
    plotter.add_points(
        spec["registered_source"], color=[0, 150, 255], point_size=point_size,
        render_points_as_spheres=True, opacity=1.0,
    )
    plotter.add_points(
        spec["target"], color=[254, 92, 92], point_size=point_size,
        render_points_as_spheres=True, opacity=1.0,
    )
    camera = spec["camera"]
    plotter.camera_position = [camera["eye"], camera["center"], camera["up"]]
    image = np.asarray(plotter.screenshot(return_img=True, transparent_background=False))
    plotter.close()
    return image[:, :, :3]


def _render_panel_software(spec: Mapping, *, width: int, height: int, point_size: float):
    return render_point_clouds(
        [
            (spec["registered_source"], SOURCE_COLOR),
            (spec["target"], TARGET_COLOR),
        ],
        spec["camera"],
        width=width,
        height=height,
        point_size=point_size,
        backend="software",
    )


def _resolve_backend(backend: str) -> str:
    if backend not in ("auto", "pyvista", "software"):
        raise ValueError("backend must be auto, pyvista, or software")
    if backend == "software":
        return backend
    try:
        import pyvista  # noqa: F401
    except Exception:
        return "software"
    return "pyvista"


def _public_panel(panel: Mapping, row: int, column: int, sample: str) -> dict:
    return {
        "row": row,
        "column": column,
        "sample": sample,
        "label": panel["label"],
        "title": panel["title"],
        "status": panel["status"],
        "error": panel["error"],
        "rms_tre_mm": panel["rms_tre_mm"],
        "prediction_path": panel["prediction_path"],
        "checkpoint": panel["checkpoint"],
    }


def render_dataset_comparison(
    dataset: str,
    cases: Sequence[Mapping],
    predictions_root: Path,
    output_root: Path,
    backend: str = "auto",
    dpi: int = 500,
) -> dict:
    if dataset not in ("in_silico", "in_vitro"):
        raise ValueError(f"Unknown dataset: {dataset}")
    if len(cases) != 3:
        raise ValueError(f"Expected exactly 3 cases, got {len(cases)}")
    output_root = Path(output_root)
    predictions_root = Path(predictions_root)
    all_specs = []
    for case in cases:
        sample_name = str(case["sample"])
        sample = _load_case(output_root, dataset, sample_name)
        predictions = {
            method: _load_prediction(
                predictions_root / dataset / _safe_sample(sample_name) / f"{method}.npz"
            )
            for method in METHOD_KEYS
        }
        specs, _ = build_panel_specs(
            sample["source"],
            sample["target"],
            sample["source_markers"],
            sample["target_markers"],
            sample["oracle_transform"],
            predictions,
            physical_scale=float(sample["physical_scale"]),
        )
        all_specs.append((case, specs))

    selected_backend = _resolve_backend(backend)
    width, height, point_size = 560, 440, 9.0

    def render_all(name: str):
        renderer = _render_panel_pyvista if name == "pyvista" else _render_panel_software
        return [
            [renderer(spec, width=width, height=height, point_size=point_size) for spec in specs]
            for _, specs in all_specs
        ]

    try:
        images = render_all(selected_backend)
    except Exception:
        if selected_backend != "pyvista":
            raise
        selected_backend = "software"
        images = render_all(selected_backend)

    fig = plt.figure(figsize=(31.2, 12.0), facecolor="white")
    grid = fig.add_gridspec(
        3, 12, left=0.024, right=0.998, top=0.925, bottom=0.025,
        wspace=0.006, hspace=0.06,
    )
    public_panels = []
    for row, ((case, specs), row_images) in enumerate(zip(all_specs, images)):
        for column, (spec, panel_image) in enumerate(zip(specs, row_images)):
            axis = fig.add_subplot(grid[row, column])
            axis.imshow(panel_image)
            axis.set_title(
                spec["title"], fontfamily="monospace", fontsize=8.0,
                color="crimson" if spec["status"] != "ok" else "black", pad=3,
            )
            axis.set_axis_off()
            public_panels.append(_public_panel(spec, row, column, str(case["sample"])))
        fig.text(
            0.004, 0.78 - row * 0.30,
            f"Sample {case['sample']}", rotation=90, va="center", ha="left",
            fontfamily="monospace", fontsize=8,
        )
    display_name = "In-silico visibility 20%–40%" if dataset == "in_silico" else "In-vitro visibility 20%–40%"
    fig.suptitle(display_name, y=0.985, fontsize=13, fontfamily="monospace")
    stem = output_root / f"{dataset}_best3_comparison"
    png = stem.with_suffix(".png")
    pdf = stem.with_suffix(".pdf")
    fig.savefig(png, dpi=dpi, facecolor="white")
    fig.savefig(pdf, dpi=dpi, facecolor="white")
    plt.close(fig)

    result = {
        "dataset": dataset,
        "backend": selected_backend,
        "rows": 3,
        "columns": 12,
        "panel_count": len(public_panels),
        "png": str(png.resolve()),
        "pdf": str(pdf.resolve()),
        "panels": public_panels,
    }
    dataset_manifest = output_root / f"{dataset}_manifest.json"
    dataset_manifest.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return result


def _write_audit(output_root: Path, results: Sequence[Mapping]) -> None:
    manifest = {
        "methods": list(METHOD_KEYS),
        "labels": list(COLUMN_LABELS),
        "datasets": list(results),
    }
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    with (output_root / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "dataset", "sample", "label", "rms_tre_mm", "status", "error",
            "checkpoint", "prediction_path",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in results:
            for panel in result["panels"]:
                writer.writerow({"dataset": result["dataset"], **{key: panel[key] for key in fields[1:]}})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--backend", choices=("auto", "pyvista", "software"), default="auto")
    parser.add_argument("--dpi", type=int, default=500)
    args = parser.parse_args()
    selection = json.loads((args.root / "selection.json").read_text(encoding="utf-8"))
    results = [
        render_dataset_comparison(
            dataset,
            selection["datasets"][dataset]["cases"],
            args.root / "predictions",
            args.root,
            backend=args.backend,
            dpi=args.dpi,
        )
        for dataset in ("in_silico", "in_vitro")
    ]
    _write_audit(args.root, results)
    print(f"Wrote {args.root / 'manifest.json'}")


if __name__ == "__main__":
    main()
