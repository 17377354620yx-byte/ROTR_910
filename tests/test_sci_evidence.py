import json
import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np

from scripts.build_sci_evidence import (
    Evaluation,
    build_tables,
    cooperative_profile_comparisons,
    factorial_interaction,
    holm_adjust,
    load_evaluation,
    paired_comparison,
    summarize_evaluation,
    write_evidence,
)


class SciEvidenceTest(unittest.TestCase):
    def test_load_evaluation_accepts_current_and_legacy_summary_keys(self):
        rows = [
            {"sample": "a/0.npz", "rms_tre_mm": 2.0, "visibility": 0.4},
            {"sample": "b/0.npz", "rms_tre_mm": 4.0, "visibility": 0.6},
        ]
        with tempfile.TemporaryDirectory() as directory:
            current = Path(directory) / "current.json"
            legacy = Path(directory) / "legacy.json"
            current.write_text(json.dumps({"method": {"mean_rms_tre_mm": 3.0}, "samples": rows}))
            legacy.write_text(json.dumps({"GeoTransformer": {"mean_rms_tre_mm": 3.0}, "samples": rows}))

            current_eval = load_evaluation(current, "candidate")
            legacy_eval = load_evaluation(legacy, "baseline")

        self.assertEqual(current_eval.summary["mean_rms_tre_mm"], 3.0)
        self.assertEqual(legacy_eval.summary["mean_rms_tre_mm"], 3.0)
        self.assertEqual(current_eval.samples["sample"].tolist(), ["a/0.npz", "b/0.npz"])

    def test_paired_comparison_uses_candidate_minus_baseline_and_matches_by_name(self):
        baseline = {"a": 5.0, "b": 2.0, "c": 8.0}
        candidate = {"c": 4.0, "a": 3.0, "b": 3.0}

        result = paired_comparison(baseline, candidate, bootstrap_repetitions=200, seed=7)

        self.assertEqual(result["n"], 3)
        self.assertAlmostEqual(result["mean_delta_mm"], -5.0 / 3.0)
        self.assertAlmostEqual(result["median_delta_mm"], -2.0)
        self.assertAlmostEqual(result["win_rate"], 2.0 / 3.0)
        self.assertLessEqual(result["mean_ci_low_mm"], result["mean_delta_mm"])
        self.assertGreaterEqual(result["mean_ci_high_mm"], result["mean_delta_mm"])

    def test_paired_comparison_rejects_nonidentical_sample_sets(self):
        with self.assertRaisesRegex(ValueError, "sample sets differ"):
            paired_comparison({"a": 1.0}, {"b": 1.0}, bootstrap_repetitions=20)

    def test_holm_adjust_is_monotonic_in_sorted_p_values(self):
        adjusted = holm_adjust([0.01, 0.04, 0.03, 0.20])
        order = np.argsort([0.01, 0.04, 0.03, 0.20])
        sorted_adjusted = np.asarray(adjusted)[order]

        self.assertTrue(np.all(np.diff(sorted_adjusted) >= 0))
        self.assertTrue(np.all((np.asarray(adjusted) >= 0) & (np.asarray(adjusted) <= 1)))
        self.assertAlmostEqual(adjusted[0], 0.04)

    def test_summarize_evaluation_recomputes_statistics_from_samples(self):
        rows = [
            {"sample": "a", "rms_tre_mm": 2.0, "success_20mm": 1},
            {"sample": "b", "rms_tre_mm": 6.0, "success_20mm": 1},
            {"sample": "c", "rms_tre_mm": 22.0, "success_20mm": 0},
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "eval.json"
            path.write_text(json.dumps({"method": {}, "samples": rows}))
            evaluation = load_evaluation(path, "candidate")

        summary = summarize_evaluation(evaluation, bootstrap_repetitions=200, seed=2)

        self.assertEqual(summary["n"], 3)
        self.assertEqual(summary["mean_rms_tre_mm"], 10.0)
        self.assertEqual(summary["median_rms_tre_mm"], 6.0)
        self.assertAlmostEqual(summary["success_rate_20mm"], 2.0 / 3.0)
        self.assertLessEqual(summary["mean_ci_low_mm"], 10.0)
        self.assertGreaterEqual(summary["mean_ci_high_mm"], 10.0)

    def test_factorial_interaction_uses_difference_of_differences(self):
        baseline = {"a": 10.0, "b": 8.0}
        rtor = {"a": 8.0, "b": 7.0}
        a3 = {"a": 7.0, "b": 7.0}
        combined = {"a": 4.0, "b": 6.0}

        result = factorial_interaction(
            baseline, rtor, a3, combined, bootstrap_repetitions=200, seed=3
        )

        self.assertEqual(result["n"], 2)
        self.assertAlmostEqual(result["interaction_mean_mm"], -0.5)
        self.assertEqual(result["synergistic_fraction"], 0.5)

    def test_build_tables_keeps_condition_and_comparator_semantics(self):
        def evaluation(name, values):
            samples = [
                {"sample": sample, "rms_tre_mm": value, "success_20mm": 1}
                for sample, value in values.items()
            ]
            return Evaluation(
                name=name,
                path=Path(f"{name}.json"),
                summary={},
                samples=__import__("pandas").DataFrame(samples),
            )

        evaluations = {
            "noise-free": {
                "GeoTransformer": evaluation("GeoTransformer", {"a": 5.0, "b": 7.0}),
                "RTOR": evaluation("RTOR", {"a": 4.0, "b": 6.0}),
                "A3": evaluation("A3", {"a": 3.0, "b": 6.0}),
                "RTOR+A3": evaluation("RTOR+A3", {"a": 2.0, "b": 5.0}),
            }
        }

        summaries, comparisons, interactions = build_tables(
            evaluations, bootstrap_repetitions=200, seed=4
        )

        self.assertEqual(len(summaries), 4)
        row = comparisons.loc[comparisons["candidate"] == "RTOR"].iloc[0]
        self.assertEqual(row["comparator"], "GeoTransformer")
        self.assertEqual(row["condition"], "noise-free")
        self.assertEqual(row["mean_delta_mm"], -1.0)
        self.assertEqual(len(interactions), 1)
        self.assertEqual(interactions.iloc[0]["interaction_mean_mm"], 0.0)

    def test_write_evidence_emits_reusable_tables_figures_and_manifest(self):
        methods = {}
        for offset, name in enumerate(
            ["GeoTransformer", "RTOR", "A3", "RTOR+A3"]
        ):
            methods[name] = Evaluation(
                name=name,
                path=Path(f"{name}.json"),
                summary={},
                samples=__import__("pandas").DataFrame(
                    [
                        {
                            "sample": f"case-{index}",
                            "rms_tre_mm": float(index + offset),
                            "success_20mm": 1,
                            "visibility": 0.25 + 0.1 * index,
                        }
                        for index in range(4)
                    ]
                ),
            )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            with warnings.catch_warnings():
                warnings.simplefilter("error", FutureWarning)
                write_evidence(
                    {"noise-free": methods},
                    output,
                    bootstrap_repetitions=50,
                    seed=5,
                )

            for relative in (
                "tables/summary.csv",
                "tables/paired_comparisons.csv",
                "tables/factorial_interactions.csv",
                "tables/cooperative_profile_comparisons.csv",
                "figures/performance_overview.png",
                "figures/paired_effects.png",
                "figures/visibility_curves.png",
                "manifest.json",
            ):
                artifact = output / relative
                self.assertTrue(artifact.is_file(), relative)
                self.assertGreater(artifact.stat().st_size, 0, relative)

    def test_cooperative_profile_comparison_uses_legacy_full_as_comparator(self):
        def make(name, values):
            return Evaluation(
                name=name,
                path=Path(f"{name}.json"),
                summary={},
                samples=__import__("pandas").DataFrame(
                    [
                        {"sample": sample, "rms_tre_mm": value}
                        for sample, value in values.items()
                    ]
                ),
            )

        evaluations = {
            "phantom": {
                "RTOR+A3": make("legacy", {"a": 5.0, "b": 7.0}),
                "RTOR+A3 (cooperative)": make("cooperative", {"a": 4.0, "b": 6.0}),
            }
        }
        rows = cooperative_profile_comparisons(
            evaluations, bootstrap_repetitions=100, seed=6
        )

        self.assertEqual(rows.iloc[0]["comparator"], "RTOR+A3")
        self.assertEqual(rows.iloc[0]["candidate"], "RTOR+A3 (cooperative)")
        self.assertEqual(rows.iloc[0]["mean_delta_mm"], -1.0)


if __name__ == "__main__":
    unittest.main()
