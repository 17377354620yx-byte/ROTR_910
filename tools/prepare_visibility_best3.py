#!/usr/bin/env python3
"""Select Ours' best low-overlap cases and build three-case dataset roots."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


DATASETS = {
    "in_silico": "in_silico_none.json",
    "in_vitro": "in_vitro.json",
}


def select_best_cases(summary: Mapping, top_k: int = 3) -> list[dict]:
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    unique: dict[str, dict] = {}
    for raw in summary.get("samples", []):
        row = dict(raw)
        sample = str(row.get("sample", ""))
        status = str(row.get("status", "ok"))
        try:
            rms = float(row["rms_tre_mm"])
            original_index = int(row["original_index"])
        except (KeyError, TypeError, ValueError):
            continue
        if not sample or status != "ok" or not math.isfinite(rms):
            continue
        row["sample"] = sample
        row["rms_tre_mm"] = rms
        row["original_index"] = original_index
        key = (rms, original_index, sample, int(row.get("index", 0)))
        previous = unique.get(sample)
        if previous is None or key < (
            float(previous["rms_tre_mm"]),
            int(previous["original_index"]),
            str(previous["sample"]),
            int(previous.get("index", 0)),
        ):
            unique[sample] = row
    ranked = sorted(
        unique.values(),
        key=lambda row: (
            float(row["rms_tre_mm"]),
            int(row["original_index"]),
            str(row["sample"]),
        ),
    )
    if len(ranked) < top_k:
        raise ValueError(f"Need {top_k} valid unique samples, found {len(ranked)}")
    selected = []
    for subset_index, row in enumerate(ranked[:top_k]):
        selected.append(
            {
                "subset_index": subset_index,
                "source_index": int(row.get("index", subset_index)),
                "original_index": int(row["original_index"]),
                "sample": str(row["sample"]),
                "visibility": float(row["visibility"]),
                "ours_rms_tre_mm": float(row["rms_tre_mm"]),
            }
        )
    return selected


def _dataset_files(dataset: str, root: Path):
    if dataset == "in_silico":
        metadata = root / "Deform_mesh_npz_test"
        with np.load(metadata / "list.npz", allow_pickle=False) as payload:
            names = np.asarray(payload["test"]).astype(str)
        return names, metadata / "stat_svd.npz", metadata / "Test"
    if dataset == "in_vitro":
        names = np.load(root / "rigid_list.npy", allow_pickle=False).astype(str)
        return names, root / "stat.npz", root / "Rigid_test_data"
    raise ValueError(f"Unknown dataset: {dataset}")


def build_subset(
    dataset: str,
    source_root: Path,
    cases: Sequence[Mapping],
    output_root: Path,
) -> dict:
    source_root = Path(source_root).resolve()
    output_root = Path(output_root)
    names, statistics_path, data_dir = _dataset_files(dataset, source_root)
    positions = {name: index for index, name in enumerate(names.tolist())}
    requested = [str(case["sample"]) for case in cases]
    missing = [name for name in requested if name not in positions]
    if missing:
        raise ValueError(f"Samples not found in {source_root}: {', '.join(missing)}")
    indices = np.asarray([positions[name] for name in requested], dtype=np.int64)
    original = np.load(source_root / "original_indices.npy", allow_pickle=False)
    with np.load(statistics_path, allow_pickle=False) as statistics:
        selected_statistics = {key: np.asarray(statistics[key])[indices] for key in statistics.files}

    output_root.mkdir(parents=True, exist_ok=True)
    if dataset == "in_silico":
        metadata = output_root / "Deform_mesh_npz_test"
        metadata.mkdir(parents=True, exist_ok=True)
        np.savez(metadata / "list.npz", test=names[indices])
        np.savez(metadata / "stat_svd.npz", **selected_statistics)
        link = metadata / "Test"
    else:
        np.save(output_root / "rigid_list.npy", names[indices])
        np.savez(output_root / "stat.npz", **selected_statistics)
        link = output_root / "Rigid_test_data"
    if link.exists() or link.is_symlink():
        if not link.is_symlink() or link.resolve() != data_dir.resolve():
            raise FileExistsError(f"Refusing to replace existing data path: {link}")
    else:
        link.symlink_to(data_dir.resolve(), target_is_directory=True)
    np.save(output_root / "original_indices.npy", original[indices])
    return {
        "dataset": dataset,
        "source_root": str(source_root),
        "output_root": str(output_root.resolve()),
        "samples": requested,
        "source_positions": indices.tolist(),
        "original_indices": original[indices].astype(int).tolist(),
    }


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--in-silico-root", type=Path, required=True)
    parser.add_argument("--in-vitro-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=3)
    return parser


def main() -> None:
    args = make_parser().parse_args()
    roots = {"in_silico": args.in_silico_root, "in_vitro": args.in_vitro_root}
    selection = {"datasets": {}}
    for dataset, filename in DATASETS.items():
        summary_path = args.result_root / "ours" / filename
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        cases = select_best_cases(summary, args.top_k)
        subset_root = args.output_root / "subsets" / dataset
        subset = build_subset(dataset, roots[dataset], cases, subset_root)
        selection["datasets"][dataset] = {
            "summary": str(summary_path.resolve()),
            "cases": cases,
            "subset": subset,
        }
    args.output_root.mkdir(parents=True, exist_ok=True)
    output_path = args.output_root / "selection.json"
    output_path.write_text(
        json.dumps(selection, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
