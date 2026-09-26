# 20%–40% overlap best-three visualization

The pipeline selects the three lowest Ours RMS-TRE cases independently from
the in-silico/no-extra-noise and in-vitro benchmark summaries, reruns all ten
methods on those fixed cases, and writes normalized source-to-target transforms.
The selection source is the completed benchmark under
`results/visibility_0.2_0.4_benchmark_metrics_v2/` (`summary.csv` and
`summary.md` provide its aggregate tables; selection uses the per-sample Ours
JSON files in the same directory).

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
`metrics.csv`, and `manifest.json`. Every panel is also exported separately
under `individual_panels/<dataset>/<sample>/`: `raw/` contains point clouds
without titles and `titled/` contains the same panels with method/RMS-TRE
titles. Each dataset directory includes an `index.json` path and metric index.

## Completed run (2026-09-26)

The fixed Ours-best cases used by the delivered figures are:

| Dataset | Sample | Visibility | Ours RMS-TRE (mm) |
|---|---|---:|---:|
| in-silico | `00008/00002/00000.npz` | 0.397333 | 0.525337 |
| in-silico | `00049/00000/00001.npz` | 0.206321 | 0.563705 |
| in-silico | `00046/00002/00003.npz` | 0.386601 | 0.574366 |
| in-vitro | `5_00086.npz` | 0.362451 | 2.009561 |
| in-vitro | `5_00085.npz` | 0.294200 | 2.148071 |
| in-vitro | `5_00087.npz` | 0.314000 | 2.327428 |

Both figures contain 36 panels (3 rows x 12 columns) at 15600 x 6000 pixels.
The run produced all 60 normalized method/sample caches and passed the Ours
consistency check. Go-ICP timed out on in-silico sample
`00049/00000/00001.npz`; that panel is intentionally marked `FAILED`.

This host exposes a VTK X11 render window but has no X server or Xvfb. The
completed run therefore used the deterministic software point renderer while
preserving the LiverMatch layout, white background, blue/red palette, shared
per-row camera, and RMS-TRE titles:

```bash
/home/yangx/miniconda3/envs/geo_v2/bin/python -m \
  visualization.livermatch_style_comparison \
  --root output/visualization/visibility_0.2_0.4_livermatch_style \
  --backend software --dpi 500
```
