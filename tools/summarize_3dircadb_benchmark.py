#!/usr/bin/env python3
"""Validate and aggregate all 3D-IRCADb benchmark method summaries."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.ircadb_benchmark import load_samples


METHODS = (
    "ours", "geotransformer", "castv2", "dfat", "lepard", "lepard_p2p",
    "parenet", "livermatch", "livermatch_p2p", "goicp",
)
METRICS = (
    "PIR", "IR", "RRE", "RTE", "RMSE", "SR_5mm", "RR_5deg_5mm",
    "r_rmse", "r_mae", "t_rmse", "t_mae", "chamfer_dist", "runtime_seconds",
)


def _mean(rows: list[dict], metric: str) -> float | None:
    if metric in ("SR_5mm", "RR_5deg_5mm"):
        values = [float(row.get(metric, 0.0) or 0.0) for row in rows]
    else:
        values = [row.get(metric) for row in rows if row.get(metric) is not None]
        values = [float(value) for value in values if np.isfinite(float(value))]
    return float(np.mean(values)) if values else None


def aggregate(rows: list[dict], method: str, group: str) -> dict:
    result = {
        "method": method,
        "group": group,
        "count": len(rows),
        "success_count": sum(row.get("status") == "ok" for row in rows),
        "failure_count": sum(row.get("status") != "ok" for row in rows),
    }
    result.update({metric: _mean(rows, metric) for metric in METRICS})
    return result


def summarize(data_root: Path, result_root: Path, allow_incomplete: bool = False) -> dict:
    expected = load_samples(data_root)
    expected_ids = {record.sample_id for record in expected}
    dataset_ids = {record.dataset_id for record in expected}
    if len(dataset_ids) != 1:
        raise ValueError("Selected samples do not share one dataset manifest")
    expected_dataset_id = next(iter(dataset_ids))
    table = []
    by_case = []
    method_payloads = {}
    for method in METHODS:
        path = Path(result_root) / method / "summary.json"
        if not path.is_file():
            if allow_incomplete:
                continue
            raise FileNotFoundError(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("dataset_id") != expected_dataset_id:
            raise ValueError(f"{method} was produced from a different dataset manifest")
        rows = payload.get("samples", [])
        ids = [str(row["sample_id"]) for row in rows]
        if len(ids) != len(set(ids)):
            raise ValueError(f"Duplicate sample IDs for {method}")
        missing = expected_ids - set(ids)
        extra = set(ids) - expected_ids
        if (missing or extra) and not allow_incomplete:
            raise ValueError(f"{method} sample mismatch: missing={len(missing)} extra={len(extra)}")
        method_payloads[method] = payload
        table.append(aggregate(rows, method, "all"))
        for visibility in (0.2, 0.3):
            subset = [row for row in rows if np.isclose(float(row["visibility"]), visibility)]
            table.append(aggregate(subset, method, f"visibility_{visibility:.2f}"))
        for case_id in range(1, 21):
            subset = [row for row in rows if int(row["case_id"]) == case_id]
            if subset:
                by_case.append(aggregate(subset, method, f"case_{case_id:02d}"))
    return {"schema_version": 1, "methods": method_payloads, "summary": table, "by_case": by_case}


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["method", "group", "count", "success_count", "failure_count", *METRICS]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    payload = summarize(args.data_root, args.result_root, args.allow_incomplete)
    args.result_root.mkdir(parents=True, exist_ok=True)
    (args.result_root / "summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    _write_csv(args.result_root / "summary.csv", payload["summary"])
    _write_csv(args.result_root / "summary_by_case.csv", payload["by_case"])
    print(args.result_root / "summary.csv")


if __name__ == "__main__":
    main()
