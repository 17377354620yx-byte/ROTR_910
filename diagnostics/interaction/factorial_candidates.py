"""Inference switches on one frozen checkpoint; not independently trained ablations."""
import hashlib
import json
from diagnostics.interaction.verify_mechanism import *
from diagnostics.interaction.metrics import score_metrics


@torch.no_grad()
def measure(out, data):
    valid = out['ref_node_corr_knn_masks'][:, :, None] & out['src_node_corr_knn_masks'][:, None, :]
    distances = pairwise_distance(out['ref_node_corr_knn_points'], apply_transform(out['src_node_corr_knn_points'], data['transform']))
    positive = (distances < .04**2) & valid
    rank = score_metrics(out['matching_scores'][:, :-1, :-1].exp(), positive, valid)
    return dict(mrr=rank['mrr'], gt_queries=rank['gt_queries'], **breakdown(out, data))


def main():
    initialize(7351); torch.set_num_threads(4)
    cfg = make_cfg(); ds = LiverTask3TrainDataset(cfg)
    det = DeterministicValidationDataset(ds, len(ds), 19000003)
    indices = np.random.default_rng(20260909).choice(len(ds), 16, replace=False).tolist()
    path = HISTORY/'geotransformer.p2p_liver.ablation.scratch_v2.rtor_a3.seed7351/snapshots/epoch-150.pth.tar'
    model = create_model(cfg).cuda().eval()
    model.load_state_dict(torch.load(path, map_location='cpu', weights_only=False)['model'], strict=True)
    root = ROOT/'output/interaction_diagnosis/factorial_candidates'; root.mkdir(parents=True, exist_ok=True)
    records = []
    with torch.no_grad(), (root/'records.jsonl').open('w') as handle:
        for i, index in enumerate(indices):
            data = to_cuda(registration_collate_fn_stack_mode([det[index]], 4, .02, .05, [7,22,32,39]))
            outputs = {}
            for rtor in (False, True):
                for a3 in (False, True):
                    model.rtor_enabled = rtor; model.a3_enabled = a3
                    out = model(data)
                    outputs[rtor, a3] = out
            row = dict(sample_position=i, index=index, natural={f'rtor{int(r)}_a3{int(a)}':measure(o, data) for (r,a),o in outputs.items()})
            n = row['natural']
            row['a3_gain_without_rtor'] = n['rtor0_a31']['mrr']-n['rtor0_a30']['mrr']
            row['a3_gain_with_rtor'] = n['rtor1_a31']['mrr']-n['rtor1_a30']['mrr']
            row['interaction'] = row['a3_gain_with_rtor']-row['a3_gain_without_rtor']
            # Hold the exact full-model patch IDs and coarse scores fixed.
            fixed = outputs[True, True]
            proposal = (fixed['fine_ref_node_corr_indices'], fixed['fine_src_node_corr_indices'], torch.ones_like(fixed['fine_ref_node_corr_indices'], dtype=torch.float))
            hook = model.coarse_matching.register_forward_hook(lambda module, inputs, output: proposal)
            try:
                model.rtor_enabled = False; model.a3_enabled = True
                fixed_off = model(data)
                row['fixed_candidates_matching_scores_max_abs'] = float((fixed_off['matching_scores']-fixed['matching_scores']).abs().max())
            finally:
                hook.remove()
            handle.write(json.dumps(row)+'\n'); handle.flush(); records.append(row)
            del outputs, fixed, fixed_off, out, data
            torch.cuda.empty_cache()
            print(f'factorial {i+1}/16', flush=True)
    summary = {key:float(np.mean([r[key] for r in records])) for key in ('a3_gain_without_rtor','a3_gain_with_rtor','interaction')}
    summary['fixed_candidates_max_abs'] = max(r['fixed_candidates_matching_scores_max_abs'] for r in records)
    summary['negative_interaction_samples'] = sum(r['interaction'] < 0 for r in records)
    summary['natural_means'] = {c:{k:float(np.mean([r['natural'][c][k] for r in records])) for k in ('mrr','unmatched_patch_fraction')} for c in records[0]['natural']}
    summary['limitations'] = 'Frozen full checkpoint switches, 16 training inputs; different natural candidate universes; not retraining or final registration accuracy.'
    summary['checkpoint'] = str(path)
    summary['script_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (root/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
