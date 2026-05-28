# conda-env: mtech-env
"""
Patient-level bootstrap confidence intervals for persisted CF outputs.

This module reads `iteration_*/successful/successful_counterfactuals.csv`,
resamples patient clusters with replacement, recomputes diagnostic metrics,
and writes inferential percentile intervals.
"""

import sys
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from typing import List
import logging

import numpy as np
import pandas as pd

from src.pipeline.metrics_calculator import MetricsCalculator

logger = logging.getLogger(__name__)


class PatientBootstrap:
    """Compute patient-level bootstrap CIs from cached successful CF rows."""

    def __init__(
        self,
        iterations: int = 1000,
        confidence_level: float = 0.95,
        random_state: int = 42,
    ):
        self.iterations = iterations
        self.confidence_level = confidence_level
        self.random_state = random_state
        self.metrics_calculator = MetricsCalculator()

    def load_successful_cfs(self, output_dir: Path) -> pd.DataFrame:
        frames: List[pd.DataFrame] = []
        for csv_path in sorted(output_dir.glob("iteration_*/successful/successful_counterfactuals.csv")):
            try:
                df = pd.read_csv(csv_path)
            except pd.errors.EmptyDataError:
                continue
            if df.empty:
                continue
            if 'iteration' not in df.columns:
                iteration_name = csv_path.parents[1].name.replace('iteration_', '')
                df['iteration'] = int(iteration_name)
            frames.append(df)

        if not frames:
            return pd.DataFrame()

        result = pd.concat(frames, ignore_index=True)
        if 'patient_id' not in result.columns:
            raise ValueError("Successful CF files must include patient_id for patient bootstrap")
        result['patient_id'] = result['patient_id'].astype(str)
        return result

    def compute(self, output_dir: Path) -> pd.DataFrame:
        successful = self.load_successful_cfs(Path(output_dir))
        if successful.empty:
            return pd.DataFrame()

        patient_ids = successful['patient_id'].drop_duplicates().to_numpy()
        grouped = {
            patient_id: group
            for patient_id, group in successful.groupby('patient_id', sort=False)
        }

        rng = np.random.default_rng(self.random_state)
        rows = []
        for bootstrap_idx in range(self.iterations):
            sampled_ids = rng.choice(patient_ids, size=len(patient_ids), replace=True)
            sampled_frames = []
            for draw_idx, patient_id in enumerate(sampled_ids):
                sampled = grouped[patient_id].copy()
                sampled['bootstrap_draw'] = draw_idx
                sampled_frames.append(sampled)
            sample_df = pd.concat(sampled_frames, ignore_index=True)
            metrics = self.metrics_calculator.compute_all_metrics(sample_df)
            metrics['bootstrap'] = bootstrap_idx
            rows.append(metrics)

        bootstrap_metrics = pd.DataFrame(rows)
        return self.compute_intervals(bootstrap_metrics)

    def compute_intervals(self, bootstrap_metrics: pd.DataFrame) -> pd.DataFrame:
        lower = (1 - self.confidence_level) / 2 * 100
        upper = (1 + self.confidence_level) / 2 * 100
        rows = []
        for col in bootstrap_metrics.select_dtypes(include='number').columns:
            if col == 'bootstrap':
                continue
            values = bootstrap_metrics[col].dropna()
            if values.empty:
                continue
            ci_lower = np.percentile(values, lower)
            ci_upper = np.percentile(values, upper)
            rows.append({
                'metric': col,
                'mean': values.mean(),
                'median': values.median(),
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'ci_width': ci_upper - ci_lower,
                'n_bootstrap': len(values),
                'interval_type': 'patient_bootstrap_inferential',
            })
        return pd.DataFrame(rows)

    def save(self, results: pd.DataFrame, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(output_path, index=False)
        logger.info(f"Saved patient bootstrap CIs to {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compute patient-level bootstrap CIs")
    parser.add_argument("output_dir", help="Pipeline output directory containing iteration_* folders")
    parser.add_argument("--iterations", type=int, default=1000, help="Bootstrap replicates")
    parser.add_argument("--confidence_level", type=float, default=0.95, help="CI confidence level")
    parser.add_argument(
        "--output",
        default=None,
        help="Output CSV path (default: <output_dir>/aggregated_results/patient_bootstrap_ci.csv)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_path = (
        Path(args.output)
        if args.output
        else output_dir / "aggregated_results" / "patient_bootstrap_ci.csv"
    )
    bootstrap = PatientBootstrap(
        iterations=args.iterations,
        confidence_level=args.confidence_level,
    )
    bootstrap.save(bootstrap.compute(output_dir), output_path)
