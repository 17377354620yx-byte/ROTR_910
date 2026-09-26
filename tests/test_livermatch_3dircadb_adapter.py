import json
from pathlib import Path

import numpy as np

from tools.evaluate_livermatch_3dircadb import (
    CanonicalLiverDataset,
    method_name,
    persist_details,
)
from tools.ircadb_benchmark import PredictionWriter


def _dataset_root(tmp_path: Path) -> Path:
    rng = np.random.default_rng(8)
    source = rng.normal(size=(400, 3)) * 30
    transform = np.eye(4)
    transform[:3, 3] = [3, -4, 7]
    target = source[:120] + transform[:3, 3]
    sample_id = "case01_pair00_vis020"
    relative = Path("visibility_0.20") / f"{sample_id}.npz"
    (tmp_path / relative).parent.mkdir(parents=True)
    np.savez_compressed(
        tmp_path / relative,
        src_points=source,
        ref_points=target,
        transform=transform,
        case_id=1,
        pair_id=0,
        visibility=0.2,
        sample_id=np.asarray(sample_id),
        units=np.asarray("mm"),
    )
    (tmp_path / "manifest.json").write_text(json.dumps({
        "parameters": {"schema_version": 1},
        "sample_count": 1,
        "samples": [{
            "sample_id": sample_id,
            "path": relative.as_posix(),
            "case_id": 1,
            "pair_id": 0,
            "visibility": 0.2,
        }],
    }))
    return tmp_path


def test_mode_to_method():
    assert method_name("base") == "livermatch"
    assert method_name("p2p") == "livermatch_p2p"


def test_adapter_recovers_physical_transform_and_does_not_fabricate_pir(tmp_path):
    dataset = CanonicalLiverDataset(_dataset_root(tmp_path), voxel_size=0.04, limit=1)
    context = dataset.contexts[0]
    details = {
        "estimated_transform": context.network_transform,
        "matches": np.empty((0, 2), dtype=np.int64),
        "source": context.normalized.source,
        "target": context.normalized.target,
    }

    for mode in ("base", "p2p"):
        output = tmp_path / mode
        writer = PredictionWriter(method_name(mode), output, [context.record.sample_id])
        persist_details(writer, context, details, runtime_seconds=0.1, mode=mode)
        payload = writer.finalize()
        row = payload["samples"][0]
        np.testing.assert_allclose(row["estimated_transform"], context.gt_transform_mm, atol=1e-6)
        assert row["PIR"] is None
        assert row["metadata"]["solver"] == ("svd" if mode == "base" else "p2p_cluster_k5")
