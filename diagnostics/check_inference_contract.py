"""Real P2P sample: annotations cannot affect the predicted rigid transform."""
import json
from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'experiments/geotransformer.p2p_liver'))
from config import make_cfg
from dataset import LiverTask3InVitroTestDataset
from model import create_model
from geotransformer.utils.data import registration_collate_fn_stack_mode
from geotransformer.utils.torch import initialize, to_cuda

initialize(7351)
torch.set_num_threads(4)
cfg = make_cfg()
data = LiverTask3InVitroTestDataset(cfg)[0]
data = to_cuda(registration_collate_fn_stack_mode([data], 4, .02, .05, [7,22,32,39]))
model = create_model(cfg).cuda().eval()
path = '/home/yangx/code/new_deform/RTORv6/output/geotransformer.p2p_liver.ablation.scratch_v2.rtor_a3.seed7351/snapshots/epoch-150.pth.tar'
model.load_state_dict(torch.load(path, map_location='cpu', weights_only=False)['model'], strict=True)
with torch.inference_mode():
    original = model(data)['estimated_transform']
    no_labels = {k:v for k,v in data.items() if k not in (
        'transform', 'source_markers', 'target_markers', 'oracle_rms_tre_mm', 'overlap')}
    unlabelled = model(no_labels)['estimated_transform']
    changed = dict(data, transform=torch.eye(4, device='cuda'))
    changed['transform'][:3, 3] = 100
    wrong_labels = model(changed)['estimated_transform']
    torch.testing.assert_close(original, unlabelled, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(original, wrong_labels, atol=1e-6, rtol=1e-6)
result = dict(status='PASS', unlabelled_max_abs=float((original-unlabelled).abs().max()),
              corrupted_labels_max_abs=float((original-wrong_labels).abs().max()))
(ROOT/'output/diagnosis_v1/inference_contract.json').write_text(json.dumps(result, indent=2))
print(result)
