"""Compose independently trained coarse/fine branches; never label this a trained joint model."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'experiments/geotransformer.p2p_liver'))
from config import make_cfg
from model import create_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--coarse', required=True)
    parser.add_argument('--fine', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if not output.is_relative_to(ROOT):
        parser.error('Output must be in the project')
    coarse = torch.load(args.coarse, map_location='cpu', weights_only=False)['model']
    fine = torch.load(args.fine, map_location='cpu', weights_only=False)['model']
    cfg = make_cfg()
    cfg.model.dual_encoder = True
    model = create_model(cfg)
    composed = {}
    for name, target in model.state_dict().items():
        if name.startswith('fine_backbone.'):
            value = fine['backbone.' + name.removeprefix('fine_backbone.')]
        elif name.startswith(('fine_local_refiner.', 'optimal_transport.')):
            value = fine[name]
        else:
            value = coarse[name]
        if target.shape != value.shape:
            raise ValueError(f'Tensor mismatch: {name}')
        composed[name] = value
    model.load_state_dict(composed, strict=True)
    metadata = dict(architecture='rtor_a3', dual_encoder=True,
                    construction='RTOR-only coarse encoder + A3-only fine encoder; no joint retraining',
                    coarse=str(Path(args.coarse).resolve()), fine=str(Path(args.fine).resolve()),
                    coarse_sha256=hashlib.sha256(Path(args.coarse).read_bytes()).hexdigest(),
                    fine_sha256=hashlib.sha256(Path(args.fine).read_bytes()).hexdigest())
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(model=composed, metadata=metadata), output)
    output.with_suffix('.json').write_text(json.dumps(metadata, indent=2))
    print(output)


if __name__ == '__main__':
    main()
