import os
from pathlib import Path
import shlex
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/run_3dircadb_benchmark.sh"
METHODS = (
    "ours", "geotransformer", "castv2", "dfat", "lepard", "lepard_p2p",
    "parenet", "livermatch", "livermatch_p2p", "goicp",
)


def _run(method, visibility="all"):
    env = os.environ.copy()
    env.update({
        "IRCADB_DRY_RUN": "1",
        "IRCADB_DATA_ROOT": "/tmp/canonical-ircadb",
        "IRCADB_RESULT_ROOT": "/tmp/ircadb-results",
    })
    return subprocess.run(
        [str(SCRIPT), method, visibility], env=env, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )


@pytest.mark.parametrize("method", METHODS)
def test_each_method_has_exact_environment_data_and_output(method):
    result = _run(method, "020")
    assert result.returncode == 0, result.stderr
    line = result.stdout.strip()
    assert "/tmp/canonical-ircadb" in line
    assert f"/tmp/ircadb-results/{method}/visibility_020" in line
    assert "--visibility 0.20" in line
    assert "--limit 0" in line


def test_method_specific_commands():
    expected = {
        "ours": ("envs/geo_v2/bin/python", "epoch-150.pth.tar"),
        "geotransformer": ("envs/geotransformer/bin/python", "GeoTransformer_base"),
        "castv2": ("envs/overpredator_py310/bin/python", "castv2-p2p-liver-epoch-150"),
        "dfat": ("envs/dfat_both/bin/python", "DFAT-main"),
        "lepard": ("envs/livermatch5090/bin/python", "--solver ransac"),
        "lepard_p2p": ("envs/livermatch5090/bin/python", "--solver p2p"),
        "parenet": ("envs/pare5090/bin/python", "PARENet/output/P2P"),
        "livermatch": ("envs/livermatch5090/bin/python", "--mode base"),
        "livermatch_p2p": ("envs/livermatch5090/bin/python", "--mode p2p"),
        "goicp": ("envs/goicp_py310/bin/python", "--trim-fraction 0.7"),
    }
    for method, tokens in expected.items():
        output = _run(method).stdout
        assert all(token in output for token in tokens), (method, output)


def test_all_expands_every_method_once():
    result = _run("all")
    assert result.returncode == 0, result.stderr
    outputs = []
    for line in result.stdout.splitlines():
        tokens = shlex.split(line)
        if "--output" in tokens:
            outputs.append(tokens[tokens.index("--output") + 1])
    for method in METHODS:
        assert outputs.count(f"/tmp/ircadb-results/{method}") == 1


def test_invalid_method_exits_two():
    result = _run("unknown")
    assert result.returncode == 2
