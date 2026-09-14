"""Training-only same-checkpoint minimal interventions; never touches final test TRE."""
import json
from pathlib import Path
import sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'experiments/geotransformer.p2p_liver')]
from config import make_cfg
from model import create_model
from loss import OverallLoss
from dataset import LiverTask3TrainDataset, DeterministicValidationDataset
from geotransformer.utils.data import registration_collate_fn_stack_mode
from geotransformer.utils.torch import initialize,to_cuda
from geotransformer.modules.ops import pairwise_distance,apply_transform
from diagnostics.interaction.probe import Capture,structure,HISTORY


@torch.no_grad()
def breakdown(out,data):
    distance=pairwise_distance(out['ref_node_corr_knn_points'],apply_transform(out['src_node_corr_knn_points'],data['transform']))
    rm,sm=out['ref_node_corr_knn_masks'],out['src_node_corr_knn_masks']
    valid=rm[:,:,None]&sm[:,None,:]
    pos=(distance<.04**2)&valid
    row=(~pos.any(2))&rm;col=(~pos.any(1))&sm
    scores=out['matching_scores']
    p=-scores[:,:-1,:-1][pos];s=torch.cat([-scores[:,:-1,-1][row],-scores[:,-1,:-1][col]])
    return dict(positive_labels=p.numel(),dustbin_labels=s.numel(),
                positive_nll=float(p.mean()) if p.numel() else None,
                dustbin_nll=float(s.mean()) if s.numel() else None,
                unmatched_patch_fraction=float((~pos.any((1,2))).float().mean()),
                positive_label_fraction=p.numel()/max(p.numel()+s.numel(),1))


def main():
    initialize(7351);torch.set_num_threads(4)
    cfg=make_cfg();ds=LiverTask3TrainDataset(cfg);det=DeterministicValidationDataset(ds,len(ds),19000003)
    indices=np.random.default_rng(20260909).choice(len(ds),16,replace=False).tolist()
    model=create_model(cfg).cuda()
    path=HISTORY/'geotransformer.p2p_liver.ablation.scratch_v2.rtor_a3.seed7351/snapshots/epoch-150.pth.tar'
    model.load_state_dict(torch.load(path,map_location='cpu',weights_only=False)['model'],strict=True)
    capture=Capture(model);objective=OverallLoss(cfg).cuda()
    root=ROOT/'output/interaction_diagnosis/minimal_interventions';root.mkdir(parents=True,exist_ok=True)
    with (root/'records.jsonl').open('w') as handle:
        for i,index in enumerate(indices):
            data=to_cuda(registration_collate_fn_stack_mode([det[index]],4,.02,.05,[7,22,32,39]))
            for condition in ('legacy_predicted','soft_overlap_predicted','gt_training_candidates'):
                model.focus_enabled=condition=='legacy_predicted'
                model.train(condition=='gt_training_candidates')
                # Isolate candidate exposure from dropout effects.
                for module in model.modules():
                    if isinstance(module,torch.nn.Dropout):module.eval()
                initialize(7351+i)
                with torch.no_grad():
                    out=model(data)
                    row=dict(sample_position=i,sample=data['sample_name'],condition=condition,
                             labels=breakdown(out,data),losses={k:float(v) for k,v in objective(out,data).items()})
                    if condition!='gt_training_candidates':row['structure']=structure(model,capture,out,data)
                handle.write(json.dumps(row)+'\n');handle.flush();capture.values.clear();del out
            del data;torch.cuda.empty_cache()
            print(f'mechanism {i+1}/16',flush=True)
    capture.close()


if __name__=='__main__':main()
