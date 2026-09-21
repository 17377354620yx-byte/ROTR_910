from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'scripts/run_p2p_ablation.sh'
COMPACT_SCRIPT = ROOT / 'scripts/run_p2p_compact.sh'
DOC = ROOT / 'experiments/geotransformer.p2p_liver/ABLATION.md'
COMPACT_DOC = ROOT / 'experiments/geotransformer.p2p_liver/COMPACT_MODEL.md'


def test_ablation_runner_has_fixed_order_and_protocol():
    text = SCRIPT.read_text()
    expected = [
        'abl1_no_proposal',
        'abl2_no_soft_weight',
        'abl3_no_poincare',
        'abl4_no_a3_geometry',
        'abl5_no_rtor_descriptor',
    ]
    positions = [text.index(profile) for profile in expected]
    assert positions == sorted(positions)
    assert 'geo_py310' in text
    assert '--architecture rtor_a3' in text
    assert '--interaction_profile cooperative' in text
    assert '--ablation_profile "$profile"' in text
    assert '--max_epoch 150' in text
    assert 'epoch-150.pth.tar' in text


def test_ablation_runner_is_valid_bash_and_has_documentation():
    subprocess.run(['bash', '-n', str(SCRIPT)], check=True)
    documentation = DOC.read_text()
    assert 'run_p2p_ablation.sh' in documentation
    assert 'geo_py310' in documentation


def test_compact_runner_has_selected_profile_and_standard_evaluation():
    subprocess.run(['bash', '-n', str(COMPACT_SCRIPT)], check=True)
    text = COMPACT_SCRIPT.read_text()
    assert 'compact_no_proposal_poincare_geometry' in text
    assert 'geo_py310' in text
    assert '--architecture rtor_a3' in text
    assert '--interaction_profile cooperative' in text
    assert '--max_epoch 150' in text
    assert 'for noise in none 2 4' in text
    assert '--dataset in_vitro' in text
    assert 'epoch-150.pth.tar' in text
    assert 'run_p2p_compact.sh all' in COMPACT_DOC.read_text()
