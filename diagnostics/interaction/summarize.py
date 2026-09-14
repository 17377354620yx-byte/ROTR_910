"""Rebuild diagnostics and figures from checkpoint records and original logs."""
import argparse
import csv
import json
from pathlib import Path
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
HISTORY = Path('/home/yangx/code/new_deform/RTORv6/output')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--records', nargs='+', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    root = Path(args.output).resolve()
    if not root.is_relative_to(ROOT):
        parser.error('Output must be project-local')
    root.mkdir(parents=True, exist_ok=True)
    records = [json.loads(line) for p in args.records for line in Path(p).read_text().splitlines()]
    result = dict(records=len(records), checkpoint_replay=True, gradient_groups=[], stages=[])
    for architecture in sorted({r['architecture'] for r in records}):
        for phase in ('early','middle','late'):
            subset = [r for r in records if r['architecture']==architecture and r['phase']==phase]
            if not subset:
                continue
            for a,b in [('o_loss','f_loss'),('o_loss','c_loss'),('f_loss','c_loss'),('o_loss','a3_increment')]:
                rows = [g for r in subset for g in r.get('gradients',[]) if g['module']=='backbone' and g['loss_a']==a and g['loss_b']==b and g['status']=='OK']
                if not rows:
                    continue
                values = np.array([g['cosine'] for g in rows])
                result['gradient_groups'].append(dict(architecture=architecture, phase=phase, losses=[a,b], n=len(values),
                    mean=float(values.mean()), median=float(np.median(values)), negative_fraction=float((values<0).mean()),
                    norm_a=float(np.mean([g['norm_a'] for g in rows])), norm_b=float(np.mean([g['norm_b'] for g in rows]))))
            for stage in ('coarse_before_rtor','coarse_after_rtor_descriptor','coarse_after_overlap_weight','coarse_after_focus',
                          'fine_before_a3','fine_after_a3','fine_after_rtor_a3_sinkhorn'):
                values = [r['structure'][stage] for r in subset if stage in r['structure']]
                if values:
                    result['stages'].append(dict(architecture=architecture, phase=phase, stage=stage,
                        **{k:float(np.mean([v[k] for v in values if v[k] is not None])) for k in ('mrr','recall1','recall5','entropy','gt_pair_survival')}))
    (root/'summary.json').write_text(json.dumps(result, indent=2))
    history = []
    fields = ['loss','c_loss','f_loss','o_loss','PIR','IR','RRE','RTE','RMSE']
    for arch in ('geotransformer','rtor_only','a3_only','rtor_a3'):
        folder = HISTORY/f'geotransformer.p2p_liver.ablation.scratch_v2.{arch}.seed7351/logs'
        for path in folder.glob('train-*.log'):
            for line_number,line in enumerate(path.open(),1):
                if '[CRIT]' not in line or 'Epoch:' not in line or 'c_loss:' not in line:
                    continue
                row = dict(architecture=arch, phase='val' if '[Val]' in line else 'train',
                           epoch=int(re.search(r'Epoch: (\d+)',line)[1]), source=str(path), source_line=line_number)
                for field in fields:
                    match = re.search(r'\b'+field+r': ([\d.eE+-]+)',line)
                    row[field] = float(match[1]) if match else None
                history.append(row)
    with (root/'historical_losses.csv').open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0])); writer.writeheader(); writer.writerows(history)
    fig, axes = plt.subplots(2,3,figsize=(14,7))
    for ax,key in zip(axes.flat,['c_loss','f_loss','o_loss','PIR','IR','RMSE']):
        for arch in ('geotransformer','rtor_only','a3_only','rtor_a3'):
            rows = sorted([r for r in history if r['architecture']==arch and r['phase']=='val'],key=lambda x:x['epoch'])
            ax.plot([r['epoch'] for r in rows],[r[key] for r in rows],label=arch)
        ax.set(title=key+' (historical validation)',xlabel='epoch');ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=7)
    fig.suptitle('A3 has no standalone auxiliary loss; f_loss supervises the fine pathway')
    fig.tight_layout();fig.savefig(root/'loss_metrics.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for i,pair in enumerate([['o_loss','f_loss'],['o_loss','c_loss'],['f_loss','c_loss'],['o_loss','a3_increment']]):
        rows=[g for g in result['gradient_groups'] if g['architecture']=='rtor_a3' and g['losses']==pair]
        x=np.arange(len(rows))+i*.15
        axes[0].plot(x,[r['mean'] for r in rows],'o-',label='/'.join(pair))
        axes[1].plot(x,[r['negative_fraction'] for r in rows],'o-',label='/'.join(pair))
    for ax in axes:
        ax.set_xticks([.225,1.225,2.225],['early','middle','late']);ax.grid(alpha=.2)
    axes[0].axhline(0,color='black',lw=.7);axes[0].set_ylabel('mean cosine, shared backbone');axes[1].set_ylabel('fraction cosine < 0')
    axes[0].legend(fontsize=7);fig.tight_layout();fig.savefig(root/'gradient_conflicts.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    coarse=['coarse_before_rtor','coarse_after_rtor_descriptor','coarse_after_overlap_weight','coarse_after_focus']
    fine=['fine_before_a3','fine_after_a3','fine_after_rtor_a3_sinkhorn']
    for ax,stages in zip(axes,[coarse,fine]):
        for phase in ('early','middle','late'):
            rows={r['stage']:r for r in result['stages'] if r['architecture']=='rtor_a3' and r['phase']==phase}
            if all(k in rows for k in stages):ax.plot(range(len(stages)),[rows[k]['mrr'] for k in stages],'o-',label=phase)
        ax.set_xticks(range(len(stages)),[s.replace('coarse_','').replace('fine_','') for s in stages],rotation=20)
        ax.set_ylabel('GT correspondence MRR');ax.grid(alpha=.2);ax.legend()
    fig.tight_layout();fig.savefig(root/'gt_ranks.png',dpi=180);plt.close(fig)
    late=[r for r in records if r['architecture']=='rtor_a3' and r['phase']=='late']
    full=[r for r in records if r['architecture']=='rtor_a3']
    epochs=sorted({r['epoch'] for r in full})
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    for name in ('rtor_ref_features','rtor_src_features','a3_ref_features','a3_src_features'):
        for ax,key in zip(axes,['before_norm','after_norm','relative_residual']):
            values=[np.mean([r['structure'][name][key] for r in full if r['epoch']==e]) for e in epochs]
            ax.plot(epochs,values,'o-',label=name)
            ax.set(xlabel='epoch',title=key);ax.grid(alpha=.2)
    axes[0].legend(fontsize=6);fig.tight_layout();fig.savefig(root/'feature_residuals.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,4))
    for key in ('overlap_descriptor_correlation','rtor_a3_aligned_output_correlation',
                'descriptor_a3_aligned_output_correlation','overlap_patch_gt_quality_correlation'):
        values=[]
        for e in epochs:
            vs=[r['structure'].get(key) for r in full if r['epoch']==e]
            values.append(np.mean([v for v in vs if v is not None]))
        ax.plot(epochs,values,'o-',label=key)
    ax.axhline(0,color='black',lw=.7);ax.set(xlabel='epoch',ylabel='within-sample aligned Pearson correlation')
    ax.legend(fontsize=7);ax.grid(alpha=.2);fig.tight_layout();fig.savefig(root/'output_correlations.png',dpi=180);plt.close(fig)
    if late:
        fig,axes=plt.subplots(1,2,figsize=(11,4))
        for ax,stages in zip(axes,[coarse[:3],fine[:2]]):
            for stage in stages:
                vs=[r['structure'][stage] for r in late]
                edges=np.array(vs[0]['hist_edges']);centers=(edges[:-1]+edges[1:])/2
                for label,ls in [('positive','-'),('negative','--')]:
                    hist=np.sum([v[label+'_hist'] for v in vs],axis=0).astype(float);hist/=max(hist.sum(),1)
                    ax.plot(centers,hist,ls,label=stage+'/'+label)
            ax.legend(fontsize=6);ax.set_ylabel('normalized histogram');ax.grid(alpha=.2)
        axes[0].set_xlabel('log10 coarse probability');axes[1].set_xlabel('fine matching logit')
        fig.tight_layout();fig.savefig(root/'score_histograms.png',dpi=180);plt.close(fig)
    print(root, len(records), 'records', len(history), 'historical epoch rows')


if __name__=='__main__':main()
