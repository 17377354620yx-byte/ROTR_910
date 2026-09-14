"""One-step shared-backbone loss-weight intervention on training-only pairs."""
import json
from pathlib import Path
import sys
import torch
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'experiments/geotransformer.p2p_liver')]
from config import make_cfg
from model import create_model
from loss import OverallLoss,Evaluator
from dataset import LiverTask3TrainDataset,DeterministicValidationDataset
from geotransformer.utils.data import registration_collate_fn_stack_mode
from geotransformer.utils.torch import initialize,to_cuda
from diagnostics.interaction.probe import HISTORY


def main():
    initialize(7351);torch.set_num_threads(4);cfg=make_cfg()
    ds=LiverTask3TrainDataset(cfg);det=DeterministicValidationDataset(ds,len(ds),19000003)
    indices=json.loads((ROOT/'output/interaction_diagnosis/full_phases/manifest.json').read_text())['indices']
    root=ROOT/'output/interaction_diagnosis/loss_weights';root.mkdir(parents=True,exist_ok=True)
    model=create_model(cfg).cuda();objective=OverallLoss(cfg).cuda();evaluator=Evaluator(cfg).cuda()
    path=HISTORY/'geotransformer.p2p_liver.ablation.scratch_v2.rtor_a3.seed7351/snapshots/epoch-150.pth.tar'
    model.load_state_dict(torch.load(path,map_location='cpu',weights_only=False)['model'],strict=True)
    params=list(model.backbone.parameters());original=[p.detach().cpu().clone() for p in params]
    parameter_norm=sum(float(p.square().sum()) for p in original)**.5
    conditions=[('no_step',None),( 'overlap0',(1,1,0)),('overlap01',(1,1,.1)),
                ('original',(1,1,.5)),('overlap1',(1,1,1)),('fine05',(1,.5,.5)),('fine2',(1,2,.5))]
    rows=[]
    for pair in range(3):
        training=to_cuda(registration_collate_fn_stack_mode([det[indices[2*pair]]],4,.02,.05,[7,22,32,39]))
        held=to_cuda(registration_collate_fn_stack_mode([det[indices[2*pair+1]]],4,.02,.05,[7,22,32,39]))
        for p,base in zip(params,original):p.data.copy_(base)
        model.train()
        for m in model.modules():
            if isinstance(m,torch.nn.Dropout):m.eval()
        initialize(7351+pair)
        with torch.autograd.graph.save_on_cpu(pin_memory=True):
            out=model(training);losses=objective(out,training);grads=[]
            for k in ('c_loss','f_loss','o_loss'):
                grad=torch.autograd.grad(losses[k],params,allow_unused=True,retain_graph=True)
                grads.append([g.detach().cpu() if g is not None else torch.zeros_like(base) for g,base in zip(grad,original)])
        del out,losses,grad;model.eval()
        for name,weights in conditions:
            if weights is None:
                for p,base in zip(params,original):p.data.copy_(base)
            else:
                direction=[sum(w*g[i] for w,g in zip(weights,grads)) for i in range(len(params))]
                norm=sum(float(g.square().sum()) for g in direction)**.5
                rate=parameter_norm*1e-4/max(norm,1e-30)
                for p,base,g in zip(params,original,direction):p.data.copy_(base-rate*g)
            with torch.no_grad():
                output=model(held);result=objective(output,held);metrics=evaluator(output,held)
                row=dict(pair=pair,condition=name,weights=weights,training_sample=training['sample_name'],
                         held_sample=held['sample_name'],relative_backbone_step=0 if weights is None else 1e-4,
                         losses={k:float(v) for k,v in result.items()},
                         metrics={k:float(v) for k,v in metrics.items()},
                         scope='One normalized SGD step on shared backbone; fresh training-derived pair; not full retraining or test-set tuning')
            rows.append(row);del output,result,metrics
        del grads,training,held;torch.cuda.empty_cache();print(f'weights {pair+1}/3',flush=True)
    (root/'results.json').write_text(json.dumps(rows,indent=2))


if __name__=='__main__':main()
