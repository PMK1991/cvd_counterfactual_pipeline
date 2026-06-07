# conda-env: mtech-env
"""
Run the improvement-focused causal recourse analysis.

Where the main pipeline keeps only counterfactuals whose SCM intervention
flips the disease label (``target`` 1 -> 0), this looks at the CFs that did
NOT flip and asks a softer question: did the intervention still push the
patient's downstream symptoms (cp, restecg, thalach, exang, slope, oldpeak)
in a clinically-beneficial direction?

It re-scores the *already-generated* DiCE counterfactuals from a completed SCM
run (``fresh_cf_iterations/iteration_NNN/``) through the SAME offline-fitted
SCM, retaining every row instead of filtering to flips. Because the SCM step
is deterministic (per patient-CF seed + load-only artifact), re-scoring
reproduces exactly the propagated rows the original run discarded — no new
DiCE generation required.

Outputs go to ``<iterations_dir>/aggregated_results_recourse/`` so they sit
beside ``aggregated_results/`` (SCM flips) and ``aggregated_results_no_scm/``
(unfiltered ablation) without overwriting either.

Usage:
    python scripts/run_improvement_recourse.py
    python scripts/run_improvement_recourse.py --iterations_dir fresh_cf_iterations
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.pipeline.recourse_analyzer import RecourseAnalyzer
from src.pipeline.metrics_calculator import MetricsCalculator
from src.pipeline.ci_computer import CIComputer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("improvement_recourse")


# The deployed SCM config (pipeline_config.yaml -> scm). Critically,
# intervention_targets is 'chol_only' (do(chol) only) — NOT SCMAnalyzer's
# 'both' default. Used as the fallback when PyYAML / the YAML file is
# unavailable, so the re-score matches the run even without yaml installed.
_DEPLOYED_SCM_DEFAULTS = {
    'graph_structure': 'full',
    'intervention_targets': 'chol_only',
    'n_samples': 1000,
    'fit_seed': 42,
    'model_dir': 'model',
}


def _load_scm_config() -> dict:
    """Return the deployed ``scm`` config block.

    Reads pipeline_config.yaml when PyYAML and the file are available;
    otherwise falls back to ``_DEPLOYED_SCM_DEFAULTS`` (NOT SCMAnalyzer's
    built-in defaults, whose ``intervention_targets='both'`` would not match
    the deployed ``chol_only`` run and would corrupt the flip partition).
    """
    path = _PROJECT_ROOT / "pipeline_config.yaml"
    try:
        import yaml
    except ImportError:
        logger.warning(
            "PyYAML unavailable; falling back to deployed SCM defaults %s",
            _DEPLOYED_SCM_DEFAULTS,
        )
        return dict(_DEPLOYED_SCM_DEFAULTS)
    if not path.exists():
        logger.warning(
            "pipeline_config.yaml not found; falling back to deployed SCM defaults"
        )
        return dict(_DEPLOYED_SCM_DEFAULTS)
    scm_cfg = (yaml.safe_load(path.read_text()) or {}).get('scm', {})
    return scm_cfg or dict(_DEPLOYED_SCM_DEFAULTS)


def _carry_over_denominators(iteration_dir: Path, metrics: dict) -> dict:
    """Reuse the SCM run's per-iteration accounting for comparable rates."""
    src_metrics_file = iteration_dir / "metrics.json"
    total_generated = total_requested = total_patients = None
    if src_metrics_file.exists():
        try:
            with open(src_metrics_file) as f:
                src = json.load(f)
            total_generated = src.get("total_generated_cfs")
            total_requested = src.get("total_requested_cfs")
            total_patients = src.get("total_patients")
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Could not read {src_metrics_file}: {e}")

    metrics["total_patients"] = total_patients
    metrics["total_requested_cfs"] = total_requested
    metrics["total_generated_cfs"] = total_generated
    return metrics


def run_recourse(iterations_dir: Path, graph_structure: str,
                 max_iterations: int = 0) -> pd.DataFrame:
    if not iterations_dir.exists():
        raise FileNotFoundError(
            f"Iterations directory not found: {iterations_dir}. "
            f"Run the SCM pipeline first to generate counterfactuals."
        )

    # Start from SCMAnalyzer defaults, then apply the deployed scm config from
    # pipeline_config.yaml so the re-score uses the SAME intervention the run was
    # generated under (notably intervention_targets='chol_only', NOT the 'both'
    # default). Otherwise the re-scored flip partition would not match the run.
    analyzer = RecourseAnalyzer()
    analyzer.config.update(_load_scm_config())
    if graph_structure:  # explicit CLI override
        analyzer.config['graph_structure'] = graph_structure
    logger.info(
        "SCM re-score config: graph_structure=%s, intervention_targets=%s, n_samples=%s",
        analyzer.config.get('graph_structure'),
        analyzer.config.get('intervention_targets'),
        analyzer.config.get('n_samples'),
    )
    analyzer.initialize_analyzer()
    metrics_calc = MetricsCalculator()

    iteration_dirs = sorted(
        d for d in iterations_dir.iterdir()
        if d.is_dir() and d.name.startswith("iteration_")
    )
    if not iteration_dirs:
        raise FileNotFoundError(f"No iteration_* directories found in {iterations_dir}")

    if max_iterations and max_iterations > 0:
        iteration_dirs = iteration_dirs[:max_iterations]

    logger.info(f"Re-scoring {len(iteration_dirs)} iterations for improvement-focused recourse")

    all_metrics = []
    for iteration_dir in iteration_dirs:
        iteration_num = int(iteration_dir.name.replace("iteration_", ""))

        scored = analyzer.analyze_iteration_all(str(iteration_dir))
        n_scored = len(scored)
        if n_scored == 0:
            non_flip = scored
            n_flip = 0
        else:
            n_flip = int(scored['flipped'].sum())
            non_flip = scored[~scored['flipped']].reset_index(drop=True)
        n_non_flip = n_scored - n_flip

        # Persist the non-flip cohort for audit / spot-checks.
        out_iter = iteration_dir / "non_flip_recourse"
        out_iter.mkdir(parents=True, exist_ok=True)
        non_flip.to_csv(out_iter / "non_flip_counterfactuals.csv", index=False)

        # Per-symptom improvement (reuses the standard, outcome-agnostic
        # metrics) plus the recourse-specific headline metrics, both over the
        # non-flip cohort.
        metrics = metrics_calc.compute_all_metrics(non_flip)
        metrics.update(metrics_calc.compute_recourse_metrics(non_flip))

        metrics["iteration"] = iteration_num
        metrics["n_scored_cfs"] = n_scored
        metrics["n_flip_cfs"] = n_flip
        metrics["n_non_flip_cfs"] = n_non_flip
        metrics["non_flip_rate_pct"] = (n_non_flip / n_scored * 100) if n_scored else 0.0
        metrics = _carry_over_denominators(iteration_dir, metrics)

        all_metrics.append(metrics)
        logger.info(
            f"iteration_{iteration_num:03d}: {n_non_flip} non-flip CFs "
            f"(IRR strict {metrics['recourse_irr_strict_pct']:.1f}%, "
            f"any-improvement {metrics['recourse_any_improvement_pct']:.1f}%)"
        )

    return pd.DataFrame(all_metrics)


def _write_recourse_summary(ci_results: pd.DataFrame, aggregated: pd.DataFrame,
                            confidence_level: float, output_path: Path) -> None:
    """Write the recourse headline summary (IRR / net improvement) markdown."""
    def ci(metric, field='mean'):
        row = ci_results[ci_results['metric'] == metric]
        return None if row.empty else row.iloc[0][field]

    n_iters = int(aggregated['iteration'].nunique())
    pct = confidence_level * 100

    def line(label, metric, unit=""):
        m, lo, hi = ci(metric), ci(metric, 'ci_lower'), ci(metric, 'ci_upper')
        if m is None:
            return f"| {label} | — | — |"
        return f"| {label} | {m:.1f}{unit} | [{lo:.1f}, {hi:.1f}]{unit} |"

    mean_non_flip = aggregated['n_non_flip_cfs'].mean()
    mean_flip = aggregated['n_flip_cfs'].mean()

    lines = [
        "# Improvement-Focused Causal Recourse",
        "",
        f"**Iterations:** {n_iters}",
        f"**Confidence Level:** {pct:.0f}%",
        f"**Mean flipped CFs / iteration:** {mean_flip:.1f}",
        f"**Mean non-flip CFs / iteration:** {mean_non_flip:.1f}",
        "",
        "Among the counterfactuals whose SCM intervention did **not** flip the "
        "disease label (`target` stayed 1), how often did downstream symptoms "
        "still improve? Downstream symptoms scored: cp, restecg, thalach, "
        "exang, slope, oldpeak.",
        "",
        f"| Recourse metric | Mean | {pct:.0f}% CI |",
        "|-----------------|------|---------|",
        line("Any improvement (≥1 symptom better)", "recourse_any_improvement_pct", "%"),
        line("IRR — strict (≥1 better, none worse)", "recourse_irr_strict_pct", "%"),
        line("IRR — lenient (≥1 better, net ≥ 0)", "recourse_irr_lenient_pct", "%"),
        line("Mean # symptoms improved / CF", "recourse_mean_n_improved"),
        line("Mean # symptoms worsened / CF", "recourse_mean_n_worsened"),
        line("Mean net improvement / CF", "recourse_mean_net_improvement"),
        "",
        "Per-symptom improvement/worsening breakdown for this non-flip cohort "
        "is in `summary_report.md`; full numbers in `ci_results.csv`.",
        "",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"Wrote recourse summary to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Run the improvement-focused causal recourse analysis"
    )
    parser.add_argument(
        "--iterations_dir", default="fresh_cf_iterations",
        help="Directory holding the completed SCM run's iteration_NNN folders",
    )
    parser.add_argument(
        "--graph_structure", default=None,
        help="SCM graph variant to load. Default: value from pipeline_config.yaml "
             "(the variant the run was generated under).",
    )
    parser.add_argument("--confidence_level", type=float, default=0.95)
    parser.add_argument(
        "--max_iterations", type=int, default=0,
        help="Process only the first N iterations (0 = all). Useful for smoke runs.",
    )
    args = parser.parse_args()

    iterations_dir = (_PROJECT_ROOT / args.iterations_dir).resolve()

    aggregated = run_recourse(iterations_dir, args.graph_structure, args.max_iterations)

    suffix = f"_smoke{args.max_iterations}" if args.max_iterations else ""
    out_dir = iterations_dir / f"aggregated_results_recourse{suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_metrics_path = out_dir / "all_iteration_metrics.csv"
    aggregated.to_csv(all_metrics_path, index=False)
    logger.info(f"Saved per-iteration metrics to {all_metrics_path}")

    ci_computer = CIComputer(confidence_level=args.confidence_level)
    ci_results = ci_computer.compute_confidence_intervals(aggregated)
    ci_computer.save_results(ci_results, str(out_dir))
    _write_recourse_summary(
        ci_results, aggregated, args.confidence_level,
        out_dir / "recourse_summary.md",
    )

    n_iter = len(aggregated)
    mean_non_flip = aggregated["n_non_flip_cfs"].mean()
    mean_irr = aggregated["recourse_irr_strict_pct"].mean()
    mean_any = aggregated["recourse_any_improvement_pct"].mean()
    logger.info("=" * 60)
    logger.info(f"IMPROVEMENT-FOCUSED RECOURSE COMPLETE ({n_iter} iterations)")
    logger.info(f"  Mean non-flip CFs / iteration: {mean_non_flip:.1f}")
    logger.info(f"  Mean any-improvement: {mean_any:.1f}%")
    logger.info(f"  Mean IRR (strict): {mean_irr:.1f}%")
    logger.info(f"  Results: {out_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
