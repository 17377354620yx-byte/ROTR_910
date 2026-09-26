import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

from tools.ircadb_benchmark import transform_to_mm


EXPERIMENT = Path(__file__).resolve().parents[1] / "experiments/geotransformer.p2p_liver"


def _load_adapter():
    sys.path.insert(0, str(EXPERIMENT))
    spec = importlib.util.spec_from_file_location("rtor_ircadb_dataset", EXPERIMENT / "ircadb_dataset.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_dataset(root: Path):
    rows = []
    transform = np.eye(4)
    angle = np.deg2rad(30.0)
    transform[:3, :3] = [[np.cos(angle), -np.sin(angle), 0],
                         [np.sin(angle), np.cos(angle), 0], [0, 0, 1]]
    transform[:3, 3] = [5, -2, 3]
    rng = np.random.default_rng(9)
    source = rng.normal(size=(300, 3)) * [30, 40, 50]
    for pair_id, visibility in ((1, 0.3), (0, 0.2)):
        sample_id = f"case01_pair{pair_id:02d}_vis{int(visibility * 100):03d}"
        indices = np.arange(int(len(source) * visibility))
        target = source[indices] @ transform[:3, :3].T + transform[:3, 3]
        relative = Path(f"visibility_{visibility:.2f}") / f"{sample_id}.npz"
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path, src_points=source.astype(np.float32), ref_points=target.astype(np.float32),
            clean_ref_points=target.astype(np.float32), transform=transform,
            case_id=1, pair_id=pair_id, visibility=visibility,
            sample_id=np.asarray(sample_id), units=np.asarray("mm"),
        )
        rows.append({"sample_id": sample_id, "path": relative.as_posix(), "case_id": 1,
                     "pair_id": pair_id, "visibility": visibility})
    (root / "manifest.json").write_text(json.dumps({
        "parameters": {"schema_version": 1}, "sample_count": 2, "samples": rows,
    }))
    return transform


def test_adapter_orders_samples_filters_visibility_and_recovers_transform(tmp_path):
    module = _load_adapter()
    gt = _write_dataset(tmp_path)
    dataset = module.IRCADbStackDataset(tmp_path, voxel_size=0.04, visibility=0.2, limit=1)

    assert len(dataset) == 1
    assert dataset.records[0].sample_id == "case01_pair00_vis020"
    item = dataset[0]
    context = dataset.load_context(0)
    assert len(item["src_points"]) > len(item["ref_points"])
    assert item["sample_name"] == "case01_pair00_vis020"
    recovered = transform_to_mm(item["transform"], context.normalized)
    np.testing.assert_allclose(recovered, gt, atol=1e-5)


def test_adapter_converts_a_network_prediction_back_to_physical_mm(tmp_path):
    module = _load_adapter()
    gt = _write_dataset(tmp_path)
    dataset = module.IRCADbStackDataset(tmp_path, voxel_size=0.04)
    context = dataset.load_context(0)
    network = module.transform_to_network(gt, context.normalized)
    np.testing.assert_allclose(module.estimate_to_mm(network, context), gt, atol=1e-5)
