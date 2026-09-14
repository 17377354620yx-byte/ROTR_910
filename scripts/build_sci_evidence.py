"""Build auditable statistics and figures for the RTORv2 paper audit."""

import argparse
from dataclasses import dataclass
from pathlib import Path

import hashlib
import json
import matplotlib
import numpy as np
import pandas as pd
from scipy import stats

matplotlib.use("Agg")
import matplotlib.pyplot as plt


METHOD_ORDER = [
    "GeoTransformer",
    "RTOR",
    "A3",
    "RTOR+A3",
    "RTOR+A3 (cooperative)",
]
COLORS = {
    "GeoTransformer": "#0077BB",
    "RTOR": "#33BBEE",
    "A3": "#009988",
    "RTOR+A3": "#EE7733",
    "RTOR+A3 (cooperative)": "#CC3311",
}
MARKERS = {
    "GeoTransformer": "o",
    "RTOR": "s",
    "A3": "^",
    "RTOR+A3": "D",
    "RTOR+A3 (cooperative)": "P",
}


@dataclass(frozen=True)
class Evaluation:
    name: str
    path: Path
    summary: dict
    samples: pd.DataFrame


def load_evaluation(path, name):
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "method" in payload:
        summary = payload["method"]
    elif "GeoTransformer" in payload:
        summary = payload["GeoTransformer"]
    else:
        raise KeyError(f"{path} has neither 'method' nor 'GeoTransformer'")
    samples = pd.DataFrame(payload["samples"])
    if "sample" not in samples or "rms_tre_mm" not in samples:
        raise KeyError(f"{path} samples require sample and rms_tre_mm fields")
    if samples["sample"].duplicated().any():
        raise ValueError(f"{path} contains duplicate sample names")
    return Evaluation(name=name, path=path, summary=summary, samples=samples)


def holm_adjust(p_values):
    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values)
    ranked = values[order]
    adjusted_sorted = np.maximum.accumulate(
        np.minimum(1.0, ranked * (len(ranked) - np.arange(len(ranked))))
    )
    adjusted = np.empty_like(adjusted_sorted)
    adjusted[order] = adjusted_sorted
    return adjusted.tolist()


def paired_comparison(
    baseline,
    candidate,
    bootstrap_repetitions=10000,
    seed=7351,
):
    if set(baseline) != set(candidate):
        raise ValueError("sample sets differ between baseline and candidate")
    names = sorted(baseline)
    base = np.asarray([baseline[name] for name in names], dtype=float)
    cand = np.asarray([candidate[name] for name in names], dtype=float)
    delta = cand - base
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(delta), size=(bootstrap_repetitions, len(delta)))
    bootstrap_means = delta[indices].mean(axis=1)
    wilcoxon = stats.wilcoxon(delta, alternative="two-sided", zero_method="pratt")
    return {
        "n": int(len(delta)),
        "mean_delta_mm": float(delta.mean()),
        "median_delta_mm": float(np.median(delta)),
        "mean_ci_low_mm": float(np.quantile(bootstrap_means, 0.025)),
        "mean_ci_high_mm": float(np.quantile(bootstrap_means, 0.975)),
        "win_rate": float(np.mean(delta < 0)),
        "tie_rate": float(np.mean(delta == 0)),
        "wilcoxon_statistic": float(wilcoxon.statistic),
        "p_value": float(wilcoxon.pvalue),
        "standardized_mean_delta": float(delta.mean() / delta.std(ddof=1))
        if len(delta) > 1 and delta.std(ddof=1) > 0
        else 0.0,
    }


def summarize_evaluation(evaluation, bootstrap_repetitions=10000, seed=7351):
    values = evaluation.samples["rms_tre_mm"].to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(bootstrap_repetitions, len(values)))
    bootstrap_means = values[indices].mean(axis=1)
    if "success_20mm" in evaluation.samples:
        success = evaluation.samples["success_20mm"].to_numpy(dtype=float)
    else:
        success = (values < 20.0).astype(float)
    return {
        "n": int(len(values)),
        "mean_rms_tre_mm": float(values.mean()),
        "std_rms_tre_mm": float(values.std(ddof=0)),
        "median_rms_tre_mm": float(np.median(values)),
        "q95_rms_tre_mm": float(np.quantile(values, 0.95)),
        "mean_ci_low_mm": float(np.quantile(bootstrap_means, 0.025)),
        "mean_ci_high_mm": float(np.quantile(bootstrap_means, 0.975)),
        "success_rate_20mm": float(success.mean()),
    }


def factorial_interaction(
    baseline,
    rtor,
    a3,
    combined,
    bootstrap_repetitions=10000,
    seed=7351,
):
    sample_sets = [set(values) for values in (baseline, rtor, a3, combined)]
    if any(names != sample_sets[0] for names in sample_sets[1:]):
        raise ValueError("sample sets differ across factorial conditions")
    names = sorted(sample_sets[0])
    interaction = np.asarray(
        [
            combined[name] - rtor[name] - a3[name] + baseline[name]
            for name in names
        ],
        dtype=float,
    )
    rng = np.random.default_rng(seed)
    indices = rng.integers(
        0, len(interaction), size=(bootstrap_repetitions, len(interaction))
    )
    means = interaction[indices].mean(axis=1)
    return {
        "n": int(len(interaction)),
        "interaction_mean_mm": float(interaction.mean()),
        "interaction_median_mm": float(np.median(interaction)),
        "interaction_ci_low_mm": float(np.quantile(means, 0.025)),
        "interaction_ci_high_mm": float(np.quantile(means, 0.975)),
        "synergistic_fraction": float(np.mean(interaction < 0)),
    }


def _sample_map(evaluation):
    return dict(
        zip(
            evaluation.samples["sample"],
            evaluation.samples["rms_tre_mm"].astype(float),
        )
    )


def build_tables(evaluations, bootstrap_repetitions=10000, seed=7351):
    summary_rows = []
    comparison_rows = []
    interaction_rows = []
    for condition, methods in evaluations.items():
        for method, evaluation in methods.items():
            row = summarize_evaluation(
                evaluation,
                bootstrap_repetitions=bootstrap_repetitions,
                seed=seed,
            )
            row.update(
                condition=condition,
                method=method,
                source_file=str(evaluation.path),
            )
            summary_rows.append(row)

        baseline = _sample_map(methods["GeoTransformer"])
        for method, evaluation in methods.items():
            if method == "GeoTransformer":
                continue
            row = paired_comparison(
                baseline,
                _sample_map(evaluation),
                bootstrap_repetitions=bootstrap_repetitions,
                seed=seed,
            )
            row.update(
                condition=condition,
                comparator="GeoTransformer",
                candidate=method,
            )
            comparison_rows.append(row)

        factorial_methods = {"GeoTransformer", "RTOR", "A3", "RTOR+A3"}
        if factorial_methods.issubset(methods):
            row = factorial_interaction(
                *(
                    _sample_map(methods[name])
                    for name in ("GeoTransformer", "RTOR", "A3", "RTOR+A3")
                ),
                bootstrap_repetitions=bootstrap_repetitions,
                seed=seed,
            )
            row["condition"] = condition
            interaction_rows.append(row)

    summaries = pd.DataFrame(summary_rows)
    comparisons = pd.DataFrame(comparison_rows)
    if len(comparisons):
        comparisons["p_holm"] = holm_adjust(comparisons["p_value"].tolist())
    interactions = pd.DataFrame(interaction_rows)
    return summaries, comparisons, interactions


def cooperative_profile_comparisons(
    evaluations, bootstrap_repetitions=10000, seed=7351
):
    rows = []
    for condition, methods in evaluations.items():
        if not {"RTOR+A3", "RTOR+A3 (cooperative)"}.issubset(methods):
            continue
        row = paired_comparison(
            _sample_map(methods["RTOR+A3"]),
            _sample_map(methods["RTOR+A3 (cooperative)"]),
            bootstrap_repetitions=bootstrap_repetitions,
            seed=seed,
        )
        row.update(
            condition=condition,
            comparator="RTOR+A3",
            candidate="RTOR+A3 (cooperative)",
        )
        rows.append(row)
    frame = pd.DataFrame(rows)
    if len(frame):
        frame["p_holm"] = holm_adjust(frame["p_value"].tolist())
    return frame


def _configure_plotting():
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
            "font.size": 8,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 7,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _save_figure(fig, output_base):
    fig.savefig(output_base.with_suffix(".png"))
    fig.savefig(output_base.with_suffix(".pdf"))
    plt.close(fig)


def _plot_performance_overview(summaries, output_base):
    conditions = list(dict.fromkeys(summaries["condition"]))
    fig, axes = plt.subplots(
        1, len(conditions), figsize=(3.35 * len(conditions), 3.2), squeeze=False
    )
    for axis, condition in zip(axes[0], conditions):
        subset = summaries[summaries["condition"] == condition].copy()
        subset["rank"] = subset["method"].map({m: i for i, m in enumerate(METHOD_ORDER)})
        subset = subset.sort_values("rank", ascending=False)
        for y, row in enumerate(subset.itertuples()):
            lower = row.mean_rms_tre_mm - row.mean_ci_low_mm
            upper = row.mean_ci_high_mm - row.mean_rms_tre_mm
            axis.errorbar(
                row.mean_rms_tre_mm,
                y,
                xerr=[[lower], [upper]],
                fmt=MARKERS[row.method],
                color=COLORS[row.method],
                capsize=2,
                markersize=5,
            )
        axis.set_yticks(range(len(subset)))
        axis.set_yticklabels(subset["method"] if axis is axes[0, 0] else [])
        axis.set_title(condition)
        axis.set_xlabel("Mean RMS-TRE (mm; 95% bootstrap CI)")
        axis.grid(axis="x", color="#DDDDDD", linewidth=0.5)
    fig.tight_layout()
    _save_figure(fig, output_base)


def _plot_paired_effects(comparisons, output_base):
    display = comparisons.copy()
    display["label"] = display["condition"] + " | " + display["candidate"]
    display = display.iloc[::-1].reset_index(drop=True)
    fig, axis = plt.subplots(figsize=(6.9, max(3.4, 0.28 * len(display) + 1.2)))
    for y, row in enumerate(display.itertuples()):
        axis.errorbar(
            row.mean_delta_mm,
            y,
            xerr=[
                [row.mean_delta_mm - row.mean_ci_low_mm],
                [row.mean_ci_high_mm - row.mean_delta_mm],
            ],
            fmt=MARKERS[row.candidate],
            color=COLORS[row.candidate],
            capsize=2,
            markersize=5,
        )
    axis.axvline(0, color="black", linewidth=0.8, linestyle="--")
    axis.set_yticks(range(len(display)))
    axis.set_yticklabels(display["label"])
    axis.set_xlabel("Paired change in RMS-TRE (candidate − GeoTransformer, mm)")
    axis.grid(axis="x", color="#DDDDDD", linewidth=0.5)
    fig.tight_layout()
    _save_figure(fig, output_base)


def _visibility_rows(evaluations):
    rows = []
    bin_edges = np.arange(0.2, 1.01, 0.1)
    for condition, methods in evaluations.items():
        for method, evaluation in methods.items():
            if "visibility" not in evaluation.samples:
                continue
            frame = evaluation.samples
            for index, (lower, upper) in enumerate(zip(bin_edges[:-1], bin_edges[1:])):
                mask = (frame["visibility"] >= lower) & (
                    (frame["visibility"] <= upper)
                    if index == len(bin_edges) - 2
                    else (frame["visibility"] < upper)
                )
                values = frame.loc[mask, "rms_tre_mm"].astype(float)
                if len(values):
                    rows.append(
                        {
                            "condition": condition,
                            "method": method,
                            "visibility_midpoint": (lower + upper) / 2,
                            "visibility_range": f"[{lower:.1f}, {upper:.1f}"
                            + ("]" if index == len(bin_edges) - 2 else ")"),
                            "n": int(len(values)),
                            "mean_rms_tre_mm": float(values.mean()),
                            "std_rms_tre_mm": float(values.std(ddof=0)),
                        }
                    )
    return pd.DataFrame(rows)


def _plot_visibility_curves(visibility, output_base):
    conditions = list(dict.fromkeys(visibility["condition"]))
    fig, axes = plt.subplots(
        1, len(conditions), figsize=(3.35 * len(conditions), 3.2), squeeze=False
    )
    for axis, condition in zip(axes[0], conditions):
        subset = visibility[visibility["condition"] == condition]
        for method in METHOD_ORDER:
            rows = subset[subset["method"] == method]
            if rows.empty:
                continue
            axis.plot(
                rows["visibility_midpoint"],
                rows["mean_rms_tre_mm"],
                marker=MARKERS[method],
                color=COLORS[method],
                linewidth=1.2,
                markersize=3.5,
                label=method,
            )
        axis.set_title(condition)
        axis.set_xlabel("Visible surface fraction")
        axis.set_ylabel("Mean RMS-TRE (mm)" if axis is axes[0, 0] else "")
        axis.grid(color="#DDDDDD", linewidth=0.5)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(5, len(labels)), frameon=False)
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    _save_figure(fig, output_base)


def _plot_factorial_interactions(interactions, output_base):
    display = interactions.iloc[::-1].reset_index(drop=True)
    fig, axis = plt.subplots(figsize=(5.2, max(2.5, 0.55 * len(display) + 1.1)))
    axis.errorbar(
        display["interaction_mean_mm"].to_numpy(dtype=float),
        np.arange(len(display)),
        xerr=np.vstack(
            [
                (
                    display["interaction_mean_mm"]
                    - display["interaction_ci_low_mm"]
                ).to_numpy(dtype=float),
                (
                    display["interaction_ci_high_mm"]
                    - display["interaction_mean_mm"]
                ).to_numpy(dtype=float),
            ]
        ),
        fmt="D",
        color="#EE7733",
        capsize=3,
    )
    axis.axvline(0, color="black", linewidth=0.8, linestyle="--")
    axis.set_yticks(range(len(display)))
    axis.set_yticklabels(display["condition"])
    axis.set_xlabel("RTOR × A3 difference-of-differences in RMS-TRE (mm)")
    axis.set_title("Negative values favor synergistic error reduction")
    axis.grid(axis="x", color="#DDDDDD", linewidth=0.5)
    fig.tight_layout()
    _save_figure(fig, output_base)


def _plot_case_changes(evaluations, output_base):
    condition = "In-silico, no added noise"
    if condition not in evaluations:
        condition = next(iter(evaluations))
    methods = evaluations[condition]
    if "RTOR+A3 (cooperative)" not in methods:
        return
    baseline = methods["GeoTransformer"].samples.set_index("sample")
    candidate = methods["RTOR+A3 (cooperative)"].samples.set_index("sample")
    joined = baseline[["rms_tre_mm"]].join(
        candidate[["rms_tre_mm"]], lsuffix="_base", rsuffix="_candidate"
    )
    joined["delta"] = joined["rms_tre_mm_candidate"] - joined["rms_tre_mm_base"]
    selected = pd.concat([joined.nsmallest(8, "delta"), joined.nlargest(8, "delta")])
    selected = selected.sort_values("delta")
    colors = np.where(selected["delta"] < 0, "#0077BB", "#CC3311")
    fig, axis = plt.subplots(figsize=(6.9, 4.5))
    y = np.arange(len(selected))
    axis.barh(y, selected["delta"], color=colors)
    axis.axvline(0, color="black", linewidth=0.8)
    axis.set_yticks(y)
    axis.set_yticklabels(
        ["/".join(Path(name).with_suffix("").parts[-3:]) for name in selected.index]
    )
    axis.set_xlabel("Paired RMS-TRE change (cooperative − GeoTransformer, mm)")
    axis.set_ylabel("Eight largest improvements and regressions")
    axis.grid(axis="x", color="#DDDDDD", linewidth=0.5)
    fig.tight_layout()
    _save_figure(fig, output_base)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_evidence(
    evaluations,
    output_dir,
    bootstrap_repetitions=10000,
    seed=7351,
):
    output_dir = Path(output_dir)
    table_dir = output_dir / "tables"
    figure_dir = output_dir / "figures"
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    summaries, comparisons, interactions = build_tables(
        evaluations,
        bootstrap_repetitions=bootstrap_repetitions,
        seed=seed,
    )
    visibility = _visibility_rows(evaluations)
    profile_comparisons = cooperative_profile_comparisons(
        evaluations,
        bootstrap_repetitions=bootstrap_repetitions,
        seed=seed,
    )
    summaries.to_csv(table_dir / "summary.csv", index=False)
    comparisons.to_csv(table_dir / "paired_comparisons.csv", index=False)
    interactions.to_csv(table_dir / "factorial_interactions.csv", index=False)
    profile_comparisons.to_csv(
        table_dir / "cooperative_profile_comparisons.csv", index=False
    )
    visibility.to_csv(table_dir / "visibility_strata.csv", index=False)
    _configure_plotting()
    _plot_performance_overview(summaries, figure_dir / "performance_overview")
    _plot_paired_effects(comparisons, figure_dir / "paired_effects")
    if not visibility.empty:
        _plot_visibility_curves(visibility, figure_dir / "visibility_curves")
    if not interactions.empty:
        _plot_factorial_interactions(interactions, figure_dir / "factorial_interactions")
    _plot_case_changes(evaluations, figure_dir / "case_level_changes")
    input_rows = []
    for condition, methods in evaluations.items():
        for method, evaluation in methods.items():
            row = {
                "condition": condition,
                "method": method,
                "path": str(evaluation.path.resolve()),
            }
            if evaluation.path.is_file():
                row["sha256"] = _sha256(evaluation.path)
            input_rows.append(row)
    manifest = {
        "schema": "rtorv2-sci-evidence/1.0",
        "bootstrap_repetitions": int(bootstrap_repetitions),
        "random_seed": int(seed),
        "paired_difference_direction": "candidate_minus_geotransformer",
        "multiple_testing": "Holm adjustment across all baseline comparisons",
        "inputs": input_rows,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summaries, comparisons, interactions, visibility


def discover_evaluations(ablation_root, cooperative_root):
    ablation_root = Path(ablation_root)
    cooperative_root = Path(cooperative_root)
    directories = {
        "GeoTransformer": "geotransformer",
        "RTOR": "rtor_only",
        "A3": "a3_only",
        "RTOR+A3": "rtor_a3",
    }
    conditions = {
        "In-silico, no added noise": ("in_silico_noise_none.json", "in_silico_noise_none.json"),
        "In-silico, 2 mm noise": ("in_silico_noise_2.json", "in_silico_noise_2.json"),
        "In-silico, 4 mm noise": ("in_silico_noise_4.json", "in_silico_noise_4.json"),
        "In-vitro phantom": ("in_vitro_noise_none.json", "in_vitro.json"),
    }
    evaluations = {}
    for condition, (ablation_file, cooperative_file) in conditions.items():
        methods = {}
        for method, slug in directories.items():
            path = (
                ablation_root
                / f"geotransformer.p2p_liver.ablation.scratch_v2.{slug}.seed7351"
                / "evaluation_epoch150"
                / ablation_file
            )
            methods[method] = load_evaluation(path, method)
        methods["RTOR+A3 (cooperative)"] = load_evaluation(
            cooperative_root / cooperative_file, "RTOR+A3 (cooperative)"
        )
        evaluations[condition] = methods
    return evaluations


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ablation-root",
        default="/home/yangx/code/new_deform/RTORv6/output",
    )
    parser.add_argument(
        "--cooperative-root",
        default="output/geotransformer.p2p_liver.rtor_a3_cooperative_v1_epoch150",
    )
    parser.add_argument("--output", default="artifacts/rtorv2_model_audit")
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=7351)
    args = parser.parse_args()
    evaluations = discover_evaluations(args.ablation_root, args.cooperative_root)
    write_evidence(
        evaluations,
        args.output,
        bootstrap_repetitions=args.bootstrap,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
