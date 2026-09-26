# 20%–40% overlap best-three visualization

The pipeline selects the three lowest Ours RMS-TRE cases independently from
the in-silico/no-extra-noise and in-vitro benchmark summaries, reruns all ten
methods on those fixed cases, and writes normalized source-to-target transforms.

```bash
cd /home/yangx/code/new_deform/ai_worker/RTORv2
P2P_GPU=1 scripts/run_visibility_best3_predictions.sh
```

Use `P2P_DRY_RUN=1` to print all 20 model/dataset commands without creating
files. Successful normalized caches are reused; set `P2P_FORCE=1` to rerun and
replace only this visualization's prediction artifacts. Override the isolated
output with `P2P_BEST3_ROOT`; the default is:

```text
output/visualization/visibility_0.2_0.4_livermatch_style/
```

`selection.json` records the six fixed samples. Raw per-project results and logs
are under `raw/`; normalized transformations are under `predictions/`. Every
successful Ours rerun is checked against the selection benchmark with a
0.05-mm tolerance. A method timeout or error remains explicit and is rendered
as `FAILED`; no identity transform is presented as a successful prediction.

After inference, render and audit the figures with:

```bash
/home/yangx/miniconda3/envs/geo_v2/bin/python -m \
  visualization.livermatch_style_comparison \
  --root output/visualization/visibility_0.2_0.4_livermatch_style \
  --backend auto --dpi 500
```

The output root then contains the two PNG/PDF comparison figures,
`metrics.csv`, and `manifest.json`.
