# conda-env: mtech-env
"""
Run the "no SCM" (unfiltered) ablation arm.

Reviewer 3, Comment 4b — SCM-filtered vs. unfiltered ablation.

This re-scores the *already-generated* DiCE counterfactuals from a completed
SCM pipeline run (``fresh_cf_iterations/iteration_NNN/``) directly with the
deployed prediction model, with no causal-validation layer. Reusing the same
on-disk counterfactuals keeps the comparison apples-to-apples: the DiCE
proposals are identical across both arms and only the acceptance criterion
differs (SCM interventional flip vs. model-predicted class flip).

Outputs go to ``<iterations_dir>/aggregated_results_no_scm/`` so the SCM and
no-SCM results sit side by side without overwriting each other.

Usage:
    python scripts/run_unfiltered_ablation.py
    python scripts/run_unfiltered_ablation.py --iterations_dir fresh_cf_iterations
"""

import argparse
import json
import logging
import pickle
import sys
from pathlib import Path

import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.pipeline.unfiltered_scorer import UnfilteredScorer
from src.pipeline.metrics_calculator import MetricsCalculator
from src.pipeline.ci_computer import CIComputer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("unfiltered_ablation")


def _carry_over_denominators(iteration_dir: Path, metrics: dict) -> dict:
    """Reuse the SCM run's per-iteration accounting so flip rates are comparable."""
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
    metrics["target_flip_rate_pct"] = (
        metrics["total_successful_cfs"] / total_generated * 100
        if total_generated else 0.0
    )
    return metrics


def run_ablation(iterations_dir: Path, model_path: Path, confidence_level: float) -> pd.DataFrame:
    if not iterations_dir.exists():
        raise FileNotFoundError(
            f"Iterations directory not found: {iterations_dir}. "
            f"Run the SCM pipeline first to generate counterfactuals."
        )
    if not model_path.exists():
        raise FileNotFoundError(f"Model artifact not found: {model_path}")

    with open(model_path, "rb") as f:
        model = pickle.load(f)
    logger.info(f"Loaded prediction model from {model_path}")

    scorer = UnfilteredScorer(model)
    metrics_calc = MetricsCalculator()

    iteration_dirs = sorted(
        d for d in iterations_dir.iterdir()
        if d.is_dir() and d.name.startswith("iteration_")
    )
    if not iteration_dirs:
        raise FileNotFoundError(f"No iteration_* directories found in {iterations_dir}")

    logger.info(f"Scoring {len(iteration_dirs)} iterations with the unfiltered (no-SCM) arm")

    all_metrics = []
    for iteration_dir in iteration_dirs:
        iteration_num = int(iteration_dir.name.replace("iteration_", ""))

        successful = scorer.analyze_iteration(
            iteration_dir=str(iteration_dir),
            output_dir=str(iteration_dir / "successful_unfiltered"),
        )

        metrics = metrics_calc.compute_all_metrics(successful)
        metrics["iteration"] = iteration_num
        metrics = _carry_over_denominators(iteration_dir, metrics)

        all_metrics.append(metrics)
        logger.info(
            f"iteration_{iteration_num:03d}: kept {metrics['total_successful_cfs']} "
            f"model-accepted CFs (flip rate {metrics['target_flip_rate_pct']:.1f}%)"
        )

    return pd.DataFrame(all_metrics)


def main():
    parser = argparse.ArgumentParser(description="Run the no-SCM (unfiltered) ablation arm")
    parser.add_argument(
        "--iterations_dir", default="fresh_cf_iterations",
        help="Directory holding the completed SCM run's iteration_NNN folders",
    )
    parser.add_argument(
        "--model_path", default="model/xgb_pipeline.pkl",
        help="Path to the deployed prediction model pickle",
    )
    parser.add_argument("--confidence_level", type=float, default=0.95)
    args = parser.parse_args()

    iterations_dir = (_PROJECT_ROOT / args.iterations_dir).resolve()
    model_path = (_PROJECT_ROOT / args.model_path).resolve()

    aggregated = run_ablation(iterations_dir, model_path, args.confidence_level)

    out_dir = iterations_dir / "aggregated_results_no_scm"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_metrics_path = out_dir / "all_iteration_metrics.csv"
    aggregated.to_csv(all_metrics_path, index=False)
    logger.info(f"Saved per-iteration metrics to {all_metrics_path}")

    ci_computer = CIComputer(confidence_level=args.confidence_level)
    ci_results = ci_computer.compute_confidence_intervals(aggregated)
    ci_computer.save_results(ci_results, str(out_dir))

    # Console comparison summary
    n_iter = len(aggregated)
    mean_success = aggregated["total_successful_cfs"].mean()
    mean_flip = aggregated["target_flip_rate_pct"].mean()
    logger.info("=" * 60)
    logger.info(f"NO-SCM ABLATION COMPLETE ({n_iter} iterations)")
    logger.info(f"  Mean model-accepted CFs / iteration: {mean_success:.1f}")
    logger.info(f"  Mean flip rate: {mean_flip:.1f}%")
    logger.info(f"  Results: {out_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
