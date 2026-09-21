# P2P Progressive Ablation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Add five reproducible cumulative RTOR+A3 ablation profiles and commands that train and evaluate them in the user-approved order.

**Architecture:** Resolve named profiles into five explicit model switches in the experiment configuration. Thread those switches into RTOR, A3, training metadata, and strict test-time protocol checks, then provide one ordered shell driver and concise experiment documentation.

**Tech Stack:** Python 3.10, PyTorch, EasyDict, unittest/pytest, Bash, conda environment `geo_py310`.

**Spec:** `docs/superpowers/specs/2026-09-16-p2p-progressive-ablation-design.md`

## Global Constraints

- Stages are cumulative in the exact user-approved order.
- All stages use `architecture=rtor_a3`, `interaction_profile=cooperative`, `registration_profile=legacy`, and one encoder.
- All full experiments start from scratch, run 150 epochs with seed 7351, and use separate output directories.
- Existing `legacy`, `soft_overlap`, and unablated `cooperative` behavior remains backward compatible.
- Commands use conda environment `geo_py310`.

---

### Task 1: Profile resolution and protocol identity

**Files:**
- Modify: `experiments/geotransformer.p2p_liver/config.py`
- Modify: `experiments/geotransformer.p2p_liver/trainval.py`
- Modify: `experiments/geotransformer.p2p_liver/test.py`
- Test: `tests/test_cooperative_matching.py`

**Interfaces:**
- Consumes: `make_cfg(..., interaction_profile, ablation_profile)`.
- Produces: `ABLATION_PROFILES`, resolved `cfg.ablation` booleans, CLI `--ablation_profile`, and checkpoint protocol identity.

- [x] Write tests asserting every profile's cumulative switch vector, invalid combinations, and `none` backward compatibility.
- [x] Run `conda run -n geo_py310 python -m pytest tests/test_cooperative_matching.py -q` and confirm failure because the new argument/profile map is absent.
- [x] Implement profile resolution and CLI/metadata threading.
- [x] Run the focused tests and confirm they pass.

### Task 2: RTOR soft-weight, Poincare, and descriptor switches

**Files:**
- Modify: `geotransformer/modules/liver/registration_model.py`
- Modify: `geotransformer/modules/liver/topology_overlap.py`
- Test: `tests/test_topology_overlap.py`
- Test: `tests/test_p2p_protocol.py`

**Interfaces:**
- Consumes: `cfg.ablation.overlap_soft_weight`, `cfg.ablation.rtor_poincare`, and `cfg.ablation.rtor_descriptor_update`.
- Produces: uniform matching weights, a four-feature non-Poincare edge gate, and a descriptor-neutral RTOR without output-projection parameters.

- [x] Write tests for non-Poincare edge-gate shape/forward behavior, disabled descriptor projection, and uniform weights from `_apply_rtor`.
- [x] Run the focused tests and confirm failure for missing constructor/config behavior.
- [x] Implement the three switches with unchanged defaults.
- [x] Run the focused tests and confirm they pass.

### Task 3: A3 geometry-bias switch

**Files:**
- Modify: `geotransformer/modules/liver/fine_local_refiner.py`
- Modify: `geotransformer/modules/liver/registration_model.py`
- Test: `tests/test_rtor_a3.py`

**Interfaces:**
- Consumes: `cfg.ablation.a3_geometry_bias` and `GeometryAwareFineRefiner(..., use_geometry_bias: bool)`.
- Produces: ordinary bidirectional patch cross-attention with a zero bias when disabled.

- [x] Write a test that replaces `_cross_geometry_bias` with a raising sentinel and verifies disabled geometry completes forward propagation with finite outputs.
- [x] Run the focused test and confirm failure because the constructor does not accept the switch.
- [x] Implement the switch and zero-bias path.
- [x] Run the focused tests and confirm they pass.

### Task 4: Ordered experiment runner and user commands

**Files:**
- Create: `scripts/run_p2p_ablation.sh`
- Create: `experiments/geotransformer.p2p_liver/ABLATION.md`
- Test: `tests/test_p2p_ablation_runner.py`

**Interfaces:**
- Consumes: profile names and existing `trainval.py`/`test.py` entry points.
- Produces: `bash scripts/run_p2p_ablation.sh train|test|all [profile]` using `geo_py310` and independent run directories.

- [x] Write tests that syntax-check the script and verify the ordered profile list, conda environment, and required CLI arguments.
- [x] Run the runner test and confirm failure because the files do not exist.
- [x] Implement the executable runner and command documentation.
- [x] Run the runner tests and confirm they pass.

### Task 5: Integration verification

**Files:**
- Verify all modified files.

**Interfaces:**
- Consumes: all prior tasks.
- Produces: evidence that profile configuration, model construction, gradients, and command syntax work together.

- [x] Run all P2P unit tests under `geo_py310`.
- [x] Instantiate all five profiles and assert their parameter counts follow the intended structural removals.
- [x] Run a one-sample, one-epoch training smoke test for the final cumulative profile and confirm finite RTOR/A3 gradients and a saved checkpoint.
- [x] Run `bash -n scripts/run_p2p_ablation.sh` and inspect `git diff --check` plus `git status --short`.
