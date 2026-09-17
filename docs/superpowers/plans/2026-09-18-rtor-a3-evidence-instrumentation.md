# RTOR/A3 Evidence Instrumentation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a read-only evidence instrumentation layer and run a 20-case smoke test that measures RTOR coarse ranking/candidate survival and fixed-coarse fine-refiner behavior without modifying the registration model.

**Architecture:** Pure metric functions consume detached NumPy arrays; scoped hooks capture RTOR/A3 tensors and replace only the runtime coarse-matcher return value for fixed-candidate interventions. A CLI reuses existing P2P dataset/model/evaluator APIs, writes strict provenance-bearing artifacts, and a separate renderer generates smoke figures.

**Tech Stack:** Python 3, PyTorch 2.9, NumPy, scikit-learn, pandas, matplotlib, unittest, existing GeoTransformer P2P APIs.

**Spec:** `docs/superpowers/specs/2026-09-17-rtor-a3-evidence-instrumentation-design.md`

## Global Constraints

- Do not modify files under `geotransformer/modules/` or `experiments/geotransformer.p2p_liver/`.
- Do not change dataset split, preprocessing, voxel size, candidate budget, Sinkhorn, LGR, GT definition, metrics, or seed protocol.
- Add only evaluation, analysis, visualization, tests, and generated research artifacts.
- Record checkpoint SHA256, git state, resolved config, seed, dataset hashes, sample IDs, neighbor limits, Sinkhorn iterations, module states, environment, command, and timestamp.
- Fixed-coarse OFF/ON must use byte-identical indices, points, masks, and node scores; fail closed on mismatch.
- Historical checkpoints missing metadata may only yield `SMOKE_ONLY_NOT_FOR_CLAIMS` artifacts.
- JSON rejects NaN/Infinity; undefined values use explicit `null` plus a reason.
- Use `conda run -n geo_v2 python -m unittest discover`; do not add pytest.
- Smoke protocol: in-silico noise-none indices `0:20`, seed 7351, neighbor limits `[7,22,32,39]`.

---

### Task 1: Pure coarse, overlap, and fine metrics

**Files:**
- Create: `research/__init__.py`
- Create: `research/scripts/__init__.py`
- Create: `research/scripts/evidence/__init__.py`
- Create: `research/scripts/evidence/metrics.py`
- Create: `tests/test_research_evidence_metrics.py`

**Interfaces:**
- Produces: `gt_pair_matrix`, `retrieval_metrics`, `candidate_survival_metrics`, `node_overlap_targets`, `binary_probability_metrics`, `fine_matrix_metrics`, `selected_correspondence_metrics`, `quantile_strata`.
- Public metric functions consume NumPy arrays and return JSON-compatible Python values.

- [ ] **Step 1: Write the failing hand-calculated retrieval and survival test**

```python
class CoarseMetricTest(unittest.TestCase):
    def test_retrieval_and_survival_match_hand_calculation(self):
        scores = np.array([[0.9, 0.8, 0.1], [0.7, 0.6, 0.5]])
        gt = np.array([[False, True, False], [True, False, False]])
        retrieval = retrieval_metrics(scores, gt, ks=(1, 2))
        self.assertAlmostEqual(retrieval["recall@1"], 0.5)
        self.assertAlmostEqual(retrieval["recall@2"], 1.0)
        self.assertAlmostEqual(retrieval["mrr"], 0.75)
        survival = candidate_survival_metrics(scores, gt, budget=2, preselection_k=2)
        self.assertAlmostEqual(survival["query_recall_before"], 1.0)
        self.assertAlmostEqual(survival["query_recall_after"], 0.5)
        self.assertAlmostEqual(survival["candidate_survival_query"], 0.5)
        self.assertAlmostEqual(survival["topk_precision"], 0.5)
```

- [ ] **Step 2: Run the test and verify RED**

Run: `conda run -n geo_v2 python -m unittest discover -s tests -p 'test_research_evidence_metrics.py' -v`  
Expected: import failure for `research.scripts.evidence.metrics`.

- [ ] **Step 3: Implement deterministic retrieval and candidate survival**

```python
def retrieval_metrics(scores, gt, ks=(1, 3, 5, 10)):
    scores, gt = _validate_pair_matrices(scores, gt)
    valid_rows = np.flatnonzero(gt.any(axis=1))
    if valid_rows.size == 0:
        return _undefined_retrieval(ks, "no_gt_bearing_queries")
    order = np.argsort(-scores[valid_rows], axis=1, kind="stable")
    ranked_gt = np.take_along_axis(gt[valid_rows], order, axis=1)
    best = ranked_gt.argmax(axis=1) + 1
    return {
        "query_count": int(valid_rows.size),
        **{f"recall@{k}": float(ranked_gt[:, :min(k, ranked_gt.shape[1])].any(1).mean()) for k in ks},
        "mrr": float(np.mean(1.0 / best)),
        **_score_distribution(scores, gt),
    }
```

Global top-K uses stable flattened sorting. Preselection is the union of each GT-bearing query's top-`preselection_k` pairs. Return query/pair survival, query/pair recall, top-K precision, ranking percentile, and explicit reasons for zero denominators.

- [ ] **Step 4: Add failing overlap edge-case tests**

```python
def test_single_class_auc_is_null_and_calibration_is_finite(self):
    metrics = binary_probability_metrics(np.array([0.1, 0.2, 0.3]),
                                         np.array([False, False, False]), bins=3)
    self.assertIsNone(metrics["roc_auc"])
    self.assertEqual(metrics["roc_auc_reason"], "single_class")
    self.assertTrue(np.isfinite(metrics["brier"]))
    self.assertEqual(len(metrics["reliability"]), 3)

def test_node_targets_use_max_pair_overlap(self):
    targets = node_overlap_targets(3, np.array([0, 0, 2]), np.array([0.2, 0.8, 0.4]))
    np.testing.assert_allclose(targets, [0.8, 0.0, 0.4])
```

- [ ] **Step 5: Verify RED, implement overlap classification/calibration, then verify GREEN**

Use `roc_auc_score` and `average_precision_score` only when both classes exist. Compute threshold-0.5 confusion counts, precision, recall, F1, IoU, Brier, ECE, reliability bins, and positive/negative summaries.

- [ ] **Step 6: Add failing fine-distance and strata tests**

```python
def test_selected_correspondence_metrics_convert_to_mm(self):
    ref = np.array([[0., 0., 0.], [1., 0., 0.]])
    src = np.array([[0.01, 0., 0.], [1.10, 0., 0.]])
    result = selected_correspondence_metrics(ref, src, np.array([0.9, 0.2]),
                                             np.eye(4), physical_scale=100.0,
                                             thresholds_mm=(1., 2., 5.))
    self.assertAlmostEqual(result["precision@2mm"], 0.5)
    self.assertAlmostEqual(result["error_mm_mean"], 5.5)

def test_quantile_strata_orders_ambiguity(self):
    labels = quantile_strata(np.array([0., 1., 2., 3., 4., 5.]), higher_is_harder=True)
    self.assertEqual(labels.tolist(), ["easy", "easy", "medium", "medium", "hard", "hard"])
```

- [ ] **Step 7: Verify RED, implement fine metrics/strata, run Task 1 GREEN**

`fine_matrix_metrics` handles `(P,K,K)` scores/distances/masks, row ranking, normalized entropy, optional dustbin mass, reciprocal top-1, and top1-top2 gap. `selected_correspondence_metrics` applies a 4×4 transform and reports normalized plus millimetre errors/precision. Quantiles use stable 1/3 and 2/3 cuts.

- [ ] **Step 8: Commit Task 1**

Run: `git add research/__init__.py research/scripts/__init__.py research/scripts/evidence/__init__.py research/scripts/evidence/metrics.py tests/test_research_evidence_metrics.py && git commit -m 'feat: add mechanism evidence metrics'`

---

### Task 2: Scoped capture and fixed-coarse interventions

**Files:**
- Create: `research/scripts/evidence/capture.py`
- Create: `research/scripts/evidence/interventions.py`
- Create: `tests/test_research_evidence_capture.py`

**Interfaces:**
- Produces: `ModuleCapture`, `fixed_coarse_proposals`, `temporary_model_flags`, `tensor_bundle_sha256`, `coarse_score_stages`.
- `ModuleCapture.values` stores detached cloned tensors under `rtor` and `fine_refiner`.

- [ ] **Step 1: Write failing lifecycle/restoration tests**

```python
def test_fixed_proposal_hook_restores_after_exception(self):
    matcher = ToyMatcher()
    original = matcher(torch.tensor([1.]))
    fixed = (torch.tensor([3]), torch.tensor([4]), torch.tensor([.5]))
    with self.assertRaisesRegex(RuntimeError, "boom"):
        with fixed_coarse_proposals(matcher, fixed):
            got = matcher(torch.tensor([1.]))
            self.assertEqual(tuple(x.item() for x in got), (3, 4, .5))
            raise RuntimeError("boom")
    torch.testing.assert_close(matcher(torch.tensor([1.])), original)

def test_temporary_flags_restore_values(self):
    model = SimpleNamespace(a3_enabled=True, focus_enabled=True)
    with temporary_model_flags(model, a3_enabled=False, focus_enabled=False):
        self.assertFalse(model.a3_enabled)
    self.assertTrue(model.a3_enabled)
    self.assertTrue(model.focus_enabled)
```

- [ ] **Step 2: Run test and verify RED**

Run: `conda run -n geo_v2 python -m unittest discover -s tests -p 'test_research_evidence_capture.py' -v`.

- [ ] **Step 3: Implement context managers and deterministic tensor hashes**

```python
@contextmanager
def fixed_coarse_proposals(module, proposal):
    frozen = tuple(t.detach().clone() for t in proposal)
    handle = module.register_forward_hook(lambda _m, _i, _o: tuple(t.clone() for t in frozen))
    try:
        yield frozen
    finally:
        handle.remove()
```

Hash sorted names, dtype, shape, and contiguous CPU bytes. Flags must restore on normal and exceptional exits.

- [ ] **Step 4: Add passive-capture equivalence and hash-sensitivity tests**

Use a toy model with named RTOR/fine modules. Assert outputs before/during/after capture are exactly equal, captured tensors are detached, and one changed proposal score changes the digest.

- [ ] **Step 5: Implement capture and coarse score stages**

Capture only present modules, recursively clone tensors, never mutate outputs, and remove handles on exit. Recompute `baseline_descriptor`, `rtor_descriptor`, `soft_calibrated`, and `hard_focus` using the exact current matching formula and output masks.

- [ ] **Step 6: Run all evidence tests GREEN and commit**

Run: `conda run -n geo_v2 python -m unittest discover -s tests -p 'test_research_evidence_*.py' -v`  
Commit: `git add research/scripts/evidence/capture.py research/scripts/evidence/interventions.py tests/test_research_evidence_capture.py && git commit -m 'feat: add scoped registration interventions'`

---

### Task 3: Provenance and strict serialization

**Files:**
- Create: `research/scripts/evidence/provenance.py`
- Create: `tests/test_research_evidence_provenance.py`

**Interfaces:**
- Produces: `sha256_file`, `to_builtin`, `git_provenance`, `build_manifest`, `validate_manifest`, `write_strict_json`.
- Validation returns `errors`, `warnings`, and `status` without silently repairing data.

- [ ] **Step 1: Write failing JSON/schema tests**

```python
def test_strict_json_rejects_nonfinite(self):
    with tempfile.TemporaryDirectory() as directory:
        with self.assertRaisesRegex(ValueError, "non-finite"):
            write_strict_json(Path(directory) / "bad.json", {"x": float("nan")})

def test_missing_checkpoint_metadata_downgrades_smoke(self):
    manifest = complete_manifest_fixture()
    manifest["checkpoints"][0]["metadata_present"] = False
    verdict = validate_manifest(manifest)
    self.assertEqual(verdict["status"], "SMOKE_ONLY_NOT_FOR_CLAIMS")
    self.assertIn("checkpoint metadata missing", verdict["warnings"])
```

- [ ] **Step 2: Run and verify RED**

Run: `conda run -n geo_v2 python -m unittest discover -s tests -p 'test_research_evidence_provenance.py' -v`.

- [ ] **Step 3: Implement conversion, finite scan, file hashes, and git state**

Convert EasyDict/dict/list/tuple/Path/NumPy/Torch values. Recursively reject nonfinite floats with dotted paths. Write UTF-8 JSON using `allow_nan=False` and atomic replacement. Record HEAD, branch, porcelain status, and diff stat read-only.

- [ ] **Step 4: Implement manifest builder/validator**

Require schema/metrics versions, scientific status, git, command, timestamp, environment, dataset/hash, IDs/count, seed, limits, budgets, modules, profiles, config, and checkpoints. Protocol conflicts are errors; missing historical metadata forces smoke-only warning.

- [ ] **Step 5: Run all evidence tests GREEN and commit**

Commit: `git add research/scripts/evidence/provenance.py tests/test_research_evidence_provenance.py && git commit -m 'feat: add evidence provenance contract'`

---

### Task 4: P2P evidence runner and artifact writer

**Files:**
- Create: `research/scripts/evidence/runner.py`
- Create: `research/run_smoke_evidence.py`
- Create: `tests/test_research_evidence_runner.py`

**Interfaces:**
- Produces: `EvidenceRunConfig`, `load_p2p_runtime`, `analyze_rtor_sample`, `run_fixed_coarse_fine`, `pose_record`, `ArtifactWriter`, `run_evidence`.
- CLI defaults implement the approved 20-case protocol.

- [ ] **Step 1: Write failing writer/identity tests**

```python
def test_writer_emits_re_readable_tables(self):
    with tempfile.TemporaryDirectory() as directory:
        writer = ArtifactWriter(Path(directory))
        writer.add("coarse", {"sample": "a", "stage": "R0", "mrr": .5})
        writer.finalize({"scientific_status": "SMOKE_ONLY_NOT_FOR_CLAIMS"})
        rows = list(csv.DictReader(open(Path(directory) / "coarse_metrics.csv")))
        self.assertEqual(rows[0]["stage"], "R0")

def test_identity_gate_rejects_changed_proposal(self):
    with self.assertRaisesRegex(RuntimeError, "fixed-coarse identity"):
        assert_fixed_coarse_identity({"pairs": "aaa"}, {"pairs": "bbb"})
```

- [ ] **Step 2: Run and verify RED**

Run: `conda run -n geo_v2 python -m unittest discover -s tests -p 'test_research_evidence_runner.py' -v`.

- [ ] **Step 3: Implement runtime import, dataset, and strict loading**

Import existing experiment `make_cfg`, `LiverTask3TestDataset`, `_build_loader`, `create_model`, and `Evaluator`. Build noise-none test data with existing `_build_loader` and supplied limits; do not calibrate. Strict-load each historical epoch-150 checkpoint.

- [ ] **Step 4: Implement per-sample RTOR analysis**

Capture the full forward, build exact GT pairs at overlap >0.1, and calculate bilateral retrieval/candidate survival for R0–R3. Calculate ref/src overlap metrics and retain pooled labels/probabilities. Record hard-focus source-node and pair survival.

- [ ] **Step 5: Implement fixed-coarse OFF/ON**

Freeze `(fine_ref_node_corr_indices, fine_src_node_corr_indices, node_corr_scores)`. Run `a3_enabled=False/True` under the replacement hook. Hash indices, points, masks, and node scores from both outputs; fail closed on mismatch. Compute fine/selected/pose metrics and save compact visualization arrays.

- [ ] **Step 6: Implement endpoint 2×2 smoke**

Run B/R/F/RF on identical IDs and record PIR/IR/RR/RRE/RTE/RMSE/RMS-TRE/SR@20. Keep status smoke-only and do not emit a paper-valid interaction claim.

- [ ] **Step 7: Implement writer, aggregate summary, manifest, and CLI**

Write `samples.jsonl`, normalized coarse/overlap/fine/pose CSVs, strict `summary.json`, and validated `manifest.json`. Summary includes finite audit, identity count, capture-equivalence count, pooled overlap metrics, and disclaimer. Refuse overwrite of non-empty output directories.

- [ ] **Step 8: Run unit tests and CLI help GREEN, then commit**

Run: `conda run -n geo_v2 python -m unittest discover -s tests -p 'test_research_evidence_*.py' -v`  
Run: `conda run -n geo_v2 python research/run_smoke_evidence.py --help`  
Commit: `git add research/scripts/evidence/runner.py research/run_smoke_evidence.py tests/test_research_evidence_runner.py && git commit -m 'feat: add P2P mechanism evidence runner'`

---

### Task 5: Reproducible smoke visualizations

**Files:**
- Create: `research/scripts/evidence/visualize.py`
- Create: `research/render_smoke_evidence.py`
- Create: `tests/test_research_evidence_visualize.py`

**Interfaces:**
- Produces: `render_all(run_dir)` writing PNG/PDF/SVG plus `figures/selection.json`.

- [ ] **Step 1: Write failing renderer test**

```python
def test_renderer_writes_all_three_formats(self):
    with tempfile.TemporaryDirectory() as directory:
        root = build_visual_fixture(Path(directory))
        render_all(root)
        for stem in ("overlap_histogram", "candidate_rank_stages", "fixed_coarse_fine"):
            for suffix in (".png", ".pdf", ".svg"):
                self.assertTrue((root / "figures" / f"{stem}{suffix}").is_file())
        self.assertTrue((root / "figures" / "selection.json").is_file())
```

- [ ] **Step 2: Run and verify RED**

Run: `conda run -n geo_v2 python -m unittest discover -s tests -p 'test_research_evidence_visualize.py' -v`.

- [ ] **Step 3: Implement deterministic plotting and format fan-out**

Use Agg, white background, 9pt font, fixed method/correctness colors, `bbox_inches="tight"`, PNG 300 dpi and vector PDF/SVG. Render overlap histogram, candidate-rank stages, and fixed-coarse fine paired metrics.

- [ ] **Step 4: Implement data-driven case selection and correspondence panel**

Select largest fine-IR improvement, nearest-zero change, and largest degradation using stable `(delta, sample_id)` ordering. Save rules/IDs. Render the improvement OFF/ON panel with identical camera, point size, line cap, and green/red correctness convention.

- [ ] **Step 5: Run all tests GREEN and commit**

Commit: `git add research/scripts/evidence/visualize.py research/render_smoke_evidence.py tests/test_research_evidence_visualize.py && git commit -m 'feat: render mechanism smoke evidence'`

---

### Task 6: Run and audit the 20-case smoke experiment

**Files:**
- Generate: `research/results/smoke_preflight/`
- Generate: `research/results/smoke_20260918_20cases/`
- Create: `research/module_evidence_matrix.md`
- Create: `research/smoke_test_report.md`

**Interfaces:**
- Consumes the runner and historical epoch-150 checkpoints.
- Produces the Stage 4 bundle and explicit pass/fail report; no final paper claims.

- [ ] **Step 1: Run one-case preflight**

Run: `conda run -n geo_v2 python research/run_smoke_evidence.py --output research/results/smoke_preflight --indices 0 --seed 7351 --neighbor-limits 7 22 32 39`  
Expected: exit 0, one sample, smoke-only status, capture and identity 1/1.

- [ ] **Step 2: Audit preflight**

```python
from pathlib import Path
import json
p = Path("research/results/smoke_preflight")
s = json.loads((p / "summary.json").read_text())
assert s["nonfinite_count"] == 0
assert s["fixed_coarse_identity_pass"] == 1
assert s["capture_equivalence_pass"] == 1
```

- [ ] **Step 3: Run fixed 20-case smoke**

Run: `conda run -n geo_v2 python research/run_smoke_evidence.py --output research/results/smoke_20260918_20cases --indices 0:20 --seed 7351 --neighbor-limits 7 22 32 39`.

- [ ] **Step 4: Render smoke figures**

Run: `conda run -n geo_v2 python research/render_smoke_evidence.py research/results/smoke_20260918_20cases`.

- [ ] **Step 5: Audit the complete bundle**

Require 20 unique IDs, four endpoint rows/sample, four RTOR stages/full sample, two fixed-coarse rows/sample, identity and passive capture 20/20, strict JSON parsing, no nonfinite values, and all figure formats.

- [ ] **Step 6: Write smoke report and provisional evidence matrix**

`smoke_test_report.md` reports instrumentation validity and observed directions labeled `NOT FOR SCIENTIFIC CLAIMS`. `module_evidence_matrix.md` marks efficacy claims `PENDING FULL EVALUATION`; only instrumentation contracts may be PASS.

- [ ] **Step 7: Run final verification**

Run all `test_research_evidence_*.py`, `test_topology_overlap.py`, `test_rtor_a3.py`, `test_p2p_protocol.py`, and `test_cooperative_matching.py` via unittest discovery. Then run `git diff --check`.

- [ ] **Step 8: Commit verified Stage 3–4 artifacts**

Run: `git add research tests docs/superpowers/plans/2026-09-18-rtor-a3-evidence-instrumentation.md && git commit -m 'research: validate RTOR A3 evidence instrumentation'`.
