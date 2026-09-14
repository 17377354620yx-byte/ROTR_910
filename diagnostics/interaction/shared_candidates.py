"""Compare independently trained features on identical full-model proposals."""
from diagnostics.interaction.factorial_candidates import *


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--candidate_source', choices=('rtor_a3', 'gt'), default='rtor_a3')
    args = parser.parse_args()
    initialize(7351); torch.set_num_threads(4)
    cfg = make_cfg(); ds = LiverTask3TrainDataset(cfg)
    det = DeterministicValidationDataset(ds, len(ds), 19000003)
    indices = np.random.default_rng(20260909).choice(len(ds), 16, replace=False).tolist()
    models = {}
    for architecture in ('geotransformer', 'rtor_only', 'a3_only', 'rtor_a3'):
        model = create_model(make_cfg(architecture=architecture)).cuda().eval()
        path = HISTORY/f'geotransformer.p2p_liver.ablation.scratch_v2.{architecture}.seed7351/snapshots/epoch-150.pth.tar'
        model.load_state_dict(torch.load(path, map_location='cpu', weights_only=False)['model'], strict=True)
        models[architecture] = model
    suffix = '' if args.candidate_source == 'rtor_a3' else '_gt'
    root = ROOT/f'output/interaction_diagnosis/shared_candidates{suffix}'; root.mkdir(parents=True, exist_ok=True)
    records = []
    with torch.no_grad(), (root/'records.jsonl').open('w') as handle:
        for i, index in enumerate(indices):
            data = to_cuda(registration_collate_fn_stack_mode([det[index]], 4, .02, .05, [7,22,32,39]))
            models['rtor_a3'].a3_enabled = True
            full = models['rtor_a3'](data)
            proposal = tuple(full[k] for k in ('fine_ref_node_corr_indices','fine_src_node_corr_indices','node_corr_scores'))
            if args.candidate_source == 'gt':
                initialize(7351+i)
                proposal = models['rtor_a3'].coarse_target(full['gt_node_corr_indices'], full['gt_node_corr_overlaps'])
            row = dict(index=index, sample_position=i, metrics={})
            for architecture, model in models.items():
                hook = model.coarse_matching.register_forward_hook(lambda module, inputs, output: proposal)
                try:
                    for a3 in ((False, True) if model.fine_local_refiner is not None else (False,)):
                        model.a3_enabled = a3
                        out = model(data)
                        row['metrics'][f'{architecture}_a3{int(a3)}'] = measure(out, data)
                finally:
                    hook.remove()
            records.append(row); handle.write(json.dumps(row)+'\n'); handle.flush()
            del full, out, data, proposal
            torch.cuda.empty_cache()
            print(f'shared candidates {i+1}/16', flush=True)
    summary = {key:{metric:float(np.mean([r['metrics'][key][metric] for r in records])) for metric in ('mrr','positive_nll','dustbin_nll')} for key in records[0]['metrics']}
    (root/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    (root/'manifest.json').write_text(json.dumps(dict(indices=indices, epoch=150, seed=7351, augmentation_seed=19000003, candidate_source=args.candidate_source, script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), limitation='Training inputs only; GT candidates test conditional matching only; predicted candidates selected by full model; one seed.'), indent=2)+'\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
