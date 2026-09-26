#!/usr/bin/env python3
"""Build the deterministic 3D-IRCADb low-overlap rigid benchmark."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np


DEFAULT_INPUT = Path("/mnt/data3/yangx/3Dircadb/3Dircadb1")
DEFAULT_OUTPUT = Path("/mnt/data3/yangx/3Dircadb/low_overlap_rigid_40_seed20260822")
SCHEMA_VERSION = 1
VISIBILITIES = (0.20, 0.30)


def read_ascii_stl(path: Path) -> np.ndarray:
    """Return all triangle vertices from an ASCII STL as ``(F, 3, 3)``."""
    vertices = []
    with Path(path).open("r", encoding="ascii", errors="strict") as handle:
        for line in handle:
            fields = line.strip().split()
            if fields and fields[0].lower() == "vertex":
                if len(fields) != 4:
                    raise ValueError(f"Malformed STL vertex in {path}: {line.rstrip()}")
                vertices.append([float(value) for value in fields[1:]])
    if not vertices or len(vertices) % 3:
        raise ValueError(f"ASCII STL has an invalid vertex count: {path}")
    triangles = np.asarray(vertices, dtype=np.float64).reshape(-1, 3, 3)
    if not np.isfinite(triangles).all():
        raise ValueError(f"ASCII STL contains non-finite coordinates: {path}")
    return triangles


def sample_mesh_surface(
    triangles: np.ndarray, count: int, rng: np.random.Generator
) -> np.ndarray:
    """Sample triangle surfaces proportional to area, ignoring zero-area facets."""
    triangles = np.asarray(triangles, dtype=np.float64)
    if triangles.ndim != 3 or triangles.shape[1:] != (3, 3):
        raise ValueError("triangles must have shape (F, 3, 3)")
    if count <= 0:
        raise ValueError("count must be positive")
    areas = 0.5 * np.linalg.norm(
        np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]),
        axis=1,
    )
    valid = np.isfinite(areas) & (areas > np.finfo(np.float64).eps)
    triangles = triangles[valid]
    areas = areas[valid]
    if not len(triangles):
        raise ValueError("mesh has no non-degenerate triangle")
    triangle_ids = np.searchsorted(
        np.cumsum(areas), rng.random(count) * areas.sum(), side="right"
    )
    chosen = triangles[triangle_ids]
    u = np.sqrt(rng.random(count))
    v = rng.random(count)
    points = (
        (1.0 - u)[:, None] * chosen[:, 0]
        + (u * (1.0 - v))[:, None] * chosen[:, 1]
        + (u * v)[:, None] * chosen[:, 2]
    )
    return np.ascontiguousarray(points, dtype=np.float64)


def _unit_vector(rng: np.random.Generator) -> np.ndarray:
    while True:
        vector = rng.normal(size=3)
        norm = np.linalg.norm(vector)
        if norm > np.finfo(np.float64).eps:
            return vector / norm


def _axis_angle(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    x, y, z = axis
    cross = np.asarray([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    identity = np.eye(3, dtype=np.float64)
    return identity + np.sin(angle_rad) * cross + (1.0 - np.cos(angle_rad)) * (cross @ cross)


def _apply_transform(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    return points @ transform[:3, :3].T + transform[:3, 3]


def make_nested_pair(
    source: np.ndarray,
    case_id: int,
    pair_id: int,
    seed: int,
    *,
    noise_sigma_mm: float = 2.0,
    min_angle_deg: float = 25.0,
    max_angle_deg: float = 90.0,
    translation_mm: float = 20.0,
) -> dict[float, dict[str, np.ndarray | float]]:
    """Create nested 20%/30% partial targets sharing pose and noise."""
    source = np.asarray(source, dtype=np.float64)
    if source.ndim != 2 or source.shape[1] != 3 or len(source) < 10:
        raise ValueError("source must have shape (N, 3) with at least 10 points")
    if not np.isfinite(source).all():
        raise ValueError("source contains non-finite coordinates")
    if not (0 <= noise_sigma_mm and 0 <= min_angle_deg <= max_angle_deg):
        raise ValueError("invalid noise or angle bounds")
    rng = np.random.default_rng(np.random.SeedSequence([seed, case_id, pair_id]))
    crop_normal = _unit_vector(rng)
    projections = (source - source.mean(axis=0)) @ crop_normal
    order = np.argsort(-projections, kind="mergesort")
    counts = {visibility: int(round(len(source) * visibility)) for visibility in VISIBILITIES}
    indices_030 = np.ascontiguousarray(order[: counts[0.30]], dtype=np.int64)

    rotation_axis = _unit_vector(rng)
    angle_deg = float(rng.uniform(min_angle_deg, max_angle_deg))
    rotation = _axis_angle(rotation_axis, np.deg2rad(angle_deg))
    translation = rng.uniform(-translation_mm, translation_mm, size=3)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    noise_030 = rng.normal(0.0, noise_sigma_mm, size=(len(indices_030), 3))

    output = {}
    for visibility in VISIBILITIES:
        count = counts[visibility]
        indices = indices_030[:count].copy()
        clean = _apply_transform(source[indices], transform)
        noise = noise_030[:count].copy()
        output[visibility] = {
            "crop_indices": indices,
            "clean_ref_points": clean,
            "ref_points": clean + noise,
            "noise": noise,
            "rotation": rotation.copy(),
            "translation": translation.copy(),
            "transform": transform.copy(),
            "rotation_angle_deg": angle_deg,
            "crop_normal": crop_normal.copy(),
            "crop_threshold": float(projections[indices[-1]]),
        }
    return output


def _sample_id(case_id: int, pair_id: int, visibility: float) -> str:
    return f"case{case_id:02d}_pair{pair_id:02d}_vis{int(round(visibility * 100)):03d}"


def _parameters(
    input_root: Path,
    seed: int,
    surface_points: int,
    pairs_per_case: int,
    case_ids: list[int],
    noise_sigma_mm: float,
    min_angle_deg: float,
    max_angle_deg: float,
    translation_mm: float,
) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "input_root": str(Path(input_root).resolve()),
        "seed": int(seed),
        "surface_points": int(surface_points),
        "pairs_per_case": int(pairs_per_case),
        "case_ids": [int(value) for value in case_ids],
        "visibilities": list(VISIBILITIES),
        "noise_sigma_mm": float(noise_sigma_mm),
        "min_angle_deg": float(min_angle_deg),
        "max_angle_deg": float(max_angle_deg),
        "translation_mm": float(translation_mm),
        "units": "mm",
        "transform_convention": "source_to_target",
    }


def build_dataset(
    *,
    input_root: Path,
    output_root: Path,
    seed: int = 20260822,
    surface_points: int = 10000,
    pairs_per_case: int = 1,
    case_ids: list[int] | None = None,
    noise_sigma_mm: float = 2.0,
    min_angle_deg: float = 25.0,
    max_angle_deg: float = 90.0,
    translation_mm: float = 20.0,
) -> dict:
    """Build atomically, or validate and reuse an identical existing dataset."""
    input_root = Path(input_root)
    output_root = Path(output_root)
    case_ids = list(range(1, 21)) if case_ids is None else [int(v) for v in case_ids]
    if not case_ids or len(set(case_ids)) != len(case_ids):
        raise ValueError("case_ids must be non-empty and unique")
    if surface_points < 10 or pairs_per_case <= 0:
        raise ValueError("surface_points must be >=10 and pairs_per_case positive")
    parameters = _parameters(
        input_root, seed, surface_points, pairs_per_case, case_ids,
        noise_sigma_mm, min_angle_deg, max_angle_deg, translation_mm,
    )
    if output_root.exists():
        manifest_path = output_root / "manifest.json"
        if not manifest_path.is_file():
            raise FileExistsError(f"Output exists without a manifest: {output_root}")
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("parameters") != parameters:
            raise FileExistsError(f"Output parameters differ; choose another --output-root: {output_root}")
        validate_dataset(output_root)
        return existing

    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_root.name}.staging-", dir=output_root.parent))
    try:
        rows = []
        for case_id in case_ids:
            stl_path = input_root / f"3Dircadb1.{case_id}" / "MESHES_VTK" / "liver.stl"
            if not stl_path.is_file():
                raise FileNotFoundError(stl_path)
            mesh_rng = np.random.default_rng(np.random.SeedSequence([seed, case_id, 0x5A17]))
            source = sample_mesh_surface(read_ascii_stl(stl_path), surface_points, mesh_rng)
            for pair_id in range(pairs_per_case):
                pair = make_nested_pair(
                    source, case_id, pair_id, seed,
                    noise_sigma_mm=noise_sigma_mm,
                    min_angle_deg=min_angle_deg,
                    max_angle_deg=max_angle_deg,
                    translation_mm=translation_mm,
                )
                for visibility in VISIBILITIES:
                    sample_id = _sample_id(case_id, pair_id, visibility)
                    relative = Path(f"visibility_{visibility:.2f}") / f"{sample_id}.npz"
                    destination = staging / relative
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    values = pair[visibility]
                    np.savez_compressed(
                        destination,
                        src_points=source.astype(np.float32),
                        ref_points=np.asarray(values["ref_points"], dtype=np.float32),
                        clean_ref_points=np.asarray(values["clean_ref_points"], dtype=np.float32),
                        transform=np.asarray(values["transform"], dtype=np.float64),
                        rotation=np.asarray(values["rotation"], dtype=np.float64),
                        translation=np.asarray(values["translation"], dtype=np.float64),
                        visibility=np.float64(visibility),
                        case_id=np.int64(case_id),
                        pair_id=np.int64(pair_id),
                        sample_id=np.asarray(sample_id),
                        noise_sigma_mm=np.float64(noise_sigma_mm),
                        rotation_angle_deg=np.float64(values["rotation_angle_deg"]),
                        crop_normal=np.asarray(values["crop_normal"], dtype=np.float64),
                        crop_threshold=np.float64(values["crop_threshold"]),
                        crop_indices=np.asarray(values["crop_indices"], dtype=np.int64),
                        units=np.asarray("mm"),
                    )
                    rows.append({
                        "sample_id": sample_id,
                        "path": relative.as_posix(),
                        "case_id": case_id,
                        "pair_id": pair_id,
                        "visibility": visibility,
                        "source_points": int(len(source)),
                        "target_points": int(len(values["crop_indices"])),
                    })
        rows.sort(key=lambda row: row["sample_id"])
        manifest = {"parameters": parameters, "sample_count": len(rows), "samples": rows}
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        np.savez_compressed(
            staging / "split.npz",
            sample_ids=np.asarray([row["sample_id"] for row in rows]),
            paths=np.asarray([row["path"] for row in rows]),
            test=np.asarray([row["path"] for row in rows]),
        )
        validate_dataset(staging)
        os.replace(staging, output_root)
        return manifest
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def validate_dataset(root: Path) -> dict[str, int]:
    """Validate manifest, arrays, pose direction metadata, and nested crops."""
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    rows = manifest.get("samples", [])
    ids = [row["sample_id"] for row in rows]
    if len(ids) != len(set(ids)) or len(rows) != manifest.get("sample_count"):
        raise ValueError("manifest contains duplicate IDs or an invalid sample count")
    grouped: dict[tuple[int, int], dict[float, dict]] = {}
    required = {
        "src_points", "ref_points", "clean_ref_points", "transform", "rotation",
        "translation", "visibility", "case_id", "pair_id", "sample_id",
        "noise_sigma_mm", "rotation_angle_deg", "crop_normal", "crop_threshold",
        "crop_indices", "units",
    }
    for row in rows:
        path = root / row["path"]
        if not path.is_file():
            raise FileNotFoundError(path)
        with np.load(path, allow_pickle=False) as data:
            if not required.issubset(data.files):
                raise ValueError(f"Sample fields missing: {path}")
            arrays = {key: np.asarray(data[key]) for key in data.files}
        for key in ("src_points", "ref_points", "clean_ref_points", "transform"):
            if not np.isfinite(arrays[key]).all():
                raise ValueError(f"Non-finite {key}: {path}")
        rotation = arrays["transform"][:3, :3]
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-8) or not np.isclose(
            np.linalg.det(rotation), 1.0, atol=1e-8
        ):
            raise ValueError(f"Invalid rotation: {path}")
        clean_expected = _apply_transform(
            arrays["src_points"][arrays["crop_indices"]], arrays["transform"]
        )
        if not np.allclose(clean_expected, arrays["clean_ref_points"], atol=2e-5):
            raise ValueError(f"Transform direction or clean target mismatch: {path}")
        key = (int(arrays["case_id"]), int(arrays["pair_id"]))
        grouped.setdefault(key, {})[float(arrays["visibility"])] = arrays
    for key, pair in grouped.items():
        if set(pair) != set(VISIBILITIES):
            raise ValueError(f"Pair lacks both visibility levels: {key}")
        low, high = pair[0.20], pair[0.30]
        if not np.array_equal(low["src_points"], high["src_points"]):
            raise ValueError(f"Pair source mismatch: {key}")
        if not np.array_equal(low["transform"], high["transform"]):
            raise ValueError(f"Pair transform mismatch: {key}")
        high_positions = {int(value): index for index, value in enumerate(high["crop_indices"])}
        if not set(map(int, low["crop_indices"])).issubset(high_positions):
            raise ValueError(f"Crop is not nested: {key}")
        selected = [high_positions[int(value)] for value in low["crop_indices"]]
        low_noise = low["ref_points"] - low["clean_ref_points"]
        high_noise = high["ref_points"] - high["clean_ref_points"]
        if not np.array_equal(low_noise, high_noise[selected]):
            raise ValueError(f"Nested noise mismatch: {key}")
    case_count = len({int(row["case_id"]) for row in rows})
    return {
        "cases": case_count,
        "pairs": len(grouped),
        "samples": len(rows),
        "vis020": int(sum(bool(np.isclose(float(row["visibility"]), 0.20)) for row in rows)),
        "vis030": int(sum(bool(np.isclose(float(row["visibility"]), 0.30)) for row in rows)),
    }


def _parse_case_ids(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument("--surface-points", type=int, default=10000)
    parser.add_argument("--pairs-per-case", type=int, default=1)
    parser.add_argument("--case-ids", type=_parse_case_ids, default=list(range(1, 21)))
    parser.add_argument("--visibilities", default="0.20,0.30")
    parser.add_argument("--noise-sigma-mm", type=float, default=2.0)
    parser.add_argument("--min-angle-deg", type=float, default=25.0)
    parser.add_argument("--max-angle-deg", type=float, default=90.0)
    parser.add_argument("--translation-mm", type=float, default=20.0)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    parsed_visibility = tuple(float(value) for value in args.visibilities.split(","))
    if parsed_visibility != VISIBILITIES:
        raise ValueError("This benchmark requires --visibilities 0.20,0.30")
    if args.validate_only:
        report = validate_dataset(args.output_root)
    else:
        build_dataset(
            input_root=args.input_root,
            output_root=args.output_root,
            seed=args.seed,
            surface_points=args.surface_points,
            pairs_per_case=args.pairs_per_case,
            case_ids=args.case_ids,
            noise_sigma_mm=args.noise_sigma_mm,
            min_angle_deg=args.min_angle_deg,
            max_angle_deg=args.max_angle_deg,
            translation_mm=args.translation_mm,
        )
        report = validate_dataset(args.output_root)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
