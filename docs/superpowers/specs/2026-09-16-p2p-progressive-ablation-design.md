# P2P Progressive Ablation Design

## Objective

Starting from the single-encoder RTOR+A3 `cooperative` configuration, train five cumulative ablations in this fixed order:

1. remove the 25% predicted-proposal exposure;
2. additionally remove RTOR overlap soft weighting;
3. additionally remove the RTOR Poincare-distance branch;
4. additionally remove the A3 geometry bias while retaining ordinary cross-attention;
5. additionally remove the RTOR descriptor residual update.

Every stage keeps source hard focus disabled, the original LGR profile, the original loss weights, seed 7351, and the same data and evaluation protocol. Each stage trains from scratch for 150 epochs and writes to an independent run directory.

## Configuration contract

Expose a named `--ablation_profile` in both training and testing. `none` preserves all historical profiles. The five experiment profiles are cumulative and encode explicit booleans in `cfg.ablation` so the resolved configuration and checkpoint metadata are self-describing.

| Profile | Predicted proposals | Soft overlap weight | Poincare edge feature | A3 geometry bias | RTOR descriptor update |
|---|---:|---:|---:|---:|---:|
| `abl1_no_proposal` | off | on | on | on | on |
| `abl2_no_soft_weight` | off | off | on | on | on |
| `abl3_no_poincare` | off | off | off | on | on |
| `abl4_no_a3_geometry` | off | off | off | off | on |
| `abl5_no_rtor_descriptor` | off | off | off | off | off |

All five profiles require `architecture=rtor_a3`, `interaction_profile=cooperative`, a single encoder, and `registration_profile=legacy`.

## Model behavior

- Disabling predicted proposals sets the maximum scheduled predicted ratio to zero.
- Disabling soft overlap weighting passes uniform weights to coarse matching while retaining overlap prediction and its supervised loss.
- Disabling Poincare geometry removes that scalar from the RTOR edge-gate input and skips its computation.
- Disabling A3 geometry bias skips geometry-signature computation and supplies a zero attention bias, leaving bidirectional patch cross-attention active.
- Disabling descriptor refinement removes the RTOR output projection and returns the incoming coarse descriptors unchanged; RTOR graph/cross-support features still drive overlap logits.

## Reproducibility and evaluation

Training manifests and checkpoint metadata record the profile and resolved switches. Testing rejects a checkpoint whose recorded profile differs from the requested profile. A shared shell driver supports `train`, `test`, and `all`, uses `geo_py310`, and runs stages in the approved order. Testing covers in-silico noise `none`, `2`, and `4`, plus in-vitro `none`.
