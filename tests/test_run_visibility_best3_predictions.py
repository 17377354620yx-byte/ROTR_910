import os
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/run_visibility_best3_predictions.sh"


def test_dry_run_contains_all_models_environments_checkpoints_and_two_datasets(tmp_path: Path):
    env = os.environ.copy()
    env.update(P2P_DRY_RUN="1", P2P_BEST3_ROOT=str(tmp_path / "must_not_exist"))
    completed = subprocess.run(
        [str(SCRIPT)], cwd=ROOT, env=env, text=True, capture_output=True, check=False
    )

    assert completed.returncode == 0, completed.stderr
    lines = [line for line in completed.stdout.splitlines() if line.startswith("MODEL_CMD ")]
    assert len(lines) == 20
    methods = {
        "ours", "geotransformer", "castv2", "dfat", "lepard", "lepard_p2p",
        "parenet", "livermatch", "livermatch_p2p", "goicp",
    }
    assert {(line.split()[1], line.split()[2]) for line in lines} == {
        (method, dataset)
        for method in methods
        for dataset in ("in_silico", "in_vitro")
    }
    output = "\n".join(lines)
    for environment in (
        "geo_v2", "geotransformer", "overpredator_py310", "dfat_both",
        "livermatch5090", "pare5090", "goicp_py310",
    ):
        assert f"/envs/{environment}/bin/python" in output
    for checkpoint in (
        "geotransformer.p2p_liver.rtor_a3_cooperative/snapshots/epoch-150.pth.tar",
        "GeoTransformer_base/output/geotransformer.p2p_liver/snapshots/epoch-150.pth.tar",
        "castv2-p2p-liver-epoch-150.pth.tar",
        "DFAT-main/output/geotransformer.p2p_liver/snapshots/epoch-150.pth.tar",
        "p2p_liver_from_scratch/checkpoints/model_best_loss.pth",
        "PARENet/output/P2P/snapshots/20260919-194858/epoch-150.pth.tar",
        "liver_new_task3_004_002/checkpoints/model_best_loss.pth",
    ):
        assert checkpoint in output
    assert "--k 5" in output
    assert "--save_predictions" in output
    assert str(tmp_path / "must_not_exist") in output
    assert not (tmp_path / "must_not_exist").exists()
