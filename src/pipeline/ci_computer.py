# conda-env: mtech-env
"""
CI Computer Module

Aggregates iteration results and computes confidence intervals.

Author: PMK
Date: 2026-01-26
"""

import pandas as pd
import numpy as np
from typing import List, Dict
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CIComputer:
    """
    Confidence Interval Computer
    
    Aggregates metrics from multiple iterations and computes
    percentile-based confidence intervals.
    """
    
    def __init__(self, confidence_level: float = 0.95):
        """
        Initialize CI Computer
        
        Args:
            confidence_level: Confidence level (default 0.95 for 95% CI)
        """
        self.confidence_level = confidence_level
        logger.info(f"Initialized CIComputer with {confidence_level*100}% confidence level")
    
    def aggregate_iteration_results(self, results_list: List[Dict]) -> pd.DataFrame:
        """
        Aggregate metrics from all iterations
        
        Args:
            results_list: List of metric dictionaries from each iteration
            
        Returns:
            DataFrame with all iteration metrics
        """
        df = pd.DataFrame(results_list)
        logger.info(f"Aggregated {len(df)} iteration results")
        return df
    
    def compute_confidence_intervals(self, aggregated_results: pd.DataFrame) -> pd.DataFrame:
        """
        Compute confidence intervals for all metrics
        
        Args:
            aggregated_results: DataFrame with metrics from all iterations
            
        Returns:
            DataFrame with CI results
        """
        results = []
        
        lower_percentile = (1 - self.confidence_level) / 2 * 100
        upper_percentile = (1 + self.confidence_level) / 2 * 100
        
        for col in aggregated_results.select_dtypes(include='number').columns:
            if col == 'iteration':
                continue

            values = aggregated_results[col].dropna()

            if len(values) == 0:
                continue

            # Mode columns (categorical) — report the overall mode, not CI
            if col.endswith('_mode_before') or col.endswith('_mode_after'):
                results.append({
                    'metric': col,
                    'mean': values.mode().iloc[0],
                    'std': 0.0,
                    'median': values.mode().iloc[0],
                    'min': values.min(),
                    'max': values.max(),
                    'ci_lower': values.mode().iloc[0],
                    'ci_upper': values.mode().iloc[0],
                    'ci_width': 0.0,
                    'n_iterations': len(values)
                })
                continue

            results.append({
                'metric': col,
                'mean': values.mean(),
                'std': values.std(),
                'median': values.median(),
                'min': values.min(),
                'max': values.max(),
                'ci_lower': np.percentile(values, lower_percentile),
                'ci_upper': np.percentile(values, upper_percentile),
                'ci_width': np.percentile(values, upper_percentile) - np.percentile(values, lower_percentile),
                'n_iterations': len(values)
            })
        
        ci_df = pd.DataFrame(results)
        logger.info(f"Computed CIs for {len(ci_df)} metrics")
        
        return ci_df
    
    def _get_ci_val(self, ci_results: pd.DataFrame, metric: str, field: str = 'mean'):
        """Helper to extract a value from ci_results for a given metric."""
        row = ci_results[ci_results['metric'] == metric]
        if row.empty:
            return None
        return row.iloc[0][field]

    def generate_summary_report(self, ci_results: pd.DataFrame, output_path: str) -> None:
        """
        Generate markdown summary report

        Args:
            ci_results: DataFrame with CI results
            output_path: Path to save report
        """
        n_iters = ci_results['n_iterations'].iloc[0] if len(ci_results) > 0 else 0
        total_cfs = self._get_ci_val(ci_results, 'total_successful_cfs')
        total_cfs_ci_lo = self._get_ci_val(ci_results, 'total_successful_cfs', 'ci_lower')
        total_cfs_ci_hi = self._get_ci_val(ci_results, 'total_successful_cfs', 'ci_upper')

        report_lines = [
            "# Fresh Counterfactual Generation - CI Results",
            "",
            f"**Iterations:** {n_iters}",
            f"**Confidence Level:** {self.confidence_level * 100}%",
            f"**Successful CFs per iteration:** {total_cfs:.1f} (95% CI: [{total_cfs_ci_lo:.1f}, {total_cfs_ci_hi:.1f}])" if total_cfs is not None else "",
            "",
            "## Diagnostic Metrics Summary",
            "",
            "| Metric | ↓ Improve (%) | ↑ Worsen (%) | ↔ No Change (%) | Mode Before → After | Δ Mean | 95% CI (↓ Improve %) |",
            "|--------|--------------|-------------|-----------------|---------------------|--------|---------------------|",
        ]

        # Define feature rows: (label, key, unit, is_continuous)
        features = [
            ("Resting BP (trestbps)", "trestbps", "mmHg", True),
            ("Chest Pain (cp)", "cp", None, False),
            ("Exang (1→0 / 0→1)", "exang", None, False),
            ("ST Depression (oldpeak)", "oldpeak", "mm", True),
            ("Max Heart Rate (thalach)", "thalach", "bpm", True),
            ("ST Slope (slope)", "slope", None, False),
            ("Resting ECG (restecg)", "restecg", None, False),
        ]

        for label, key, unit, is_continuous in features:
            imp = self._get_ci_val(ci_results, f'{key}_improved_pct')
            wor = self._get_ci_val(ci_results, f'{key}_worsened_pct')
            noc = self._get_ci_val(ci_results, f'{key}_no_change_pct')
            ci_lo = self._get_ci_val(ci_results, f'{key}_improved_pct', 'ci_lower')
            ci_hi = self._get_ci_val(ci_results, f'{key}_improved_pct', 'ci_upper')

            imp_s = f"{imp:.1f}" if imp is not None else "—"
            wor_s = f"{wor:.1f}" if wor is not None else "—"
            noc_s = f"{noc:.1f}" if noc is not None else "—"
            ci_s = f"[{ci_lo:.1f}%, {ci_hi:.1f}%]" if ci_lo is not None else "—"

            if is_continuous:
                mean_diff = self._get_ci_val(ci_results, f'mean_diff_{key}')
                if mean_diff is not None:
                    sign = "+" if mean_diff > 0 else ""
                    delta_s = f"{sign}{mean_diff:.2f} {unit}"
                else:
                    delta_s = "—"
                mode_s = "—"
            else:
                mode_before = self._get_ci_val(ci_results, f'{key}_mode_before')
                mode_after = self._get_ci_val(ci_results, f'{key}_mode_after')
                if mode_before is not None and mode_after is not None:
                    mode_s = f"{int(mode_before)} → {int(mode_after)}"
                else:
                    mode_s = "—"
                delta_s = "—"

            report_lines.append(
                f"| {label} | {imp_s} | {wor_s} | {noc_s} | {mode_s} | {delta_s} | {ci_s} |"
            )

        report_lines.extend([
            "",
            "## Detailed Results",
            "",
            "See `ci_results.csv` for complete results.",
            ""
        ])
        
        # Write report
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        logger.info(f"Generated summary report: {output_path}")
    
    def save_results(self, ci_results: pd.DataFrame, output_dir: str) -> None:
        """
        Save CI results and generate report
        
        Args:
            ci_results: DataFrame with CI results
            output_dir: Output directory
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save CSV
        csv_path = output_path / "ci_results.csv"
        ci_results.to_csv(csv_path, index=False)
        logger.info(f"Saved CI results to {csv_path}")
        
        # Generate report
        report_path = output_path / "summary_report.md"
        self.generate_summary_report(ci_results, str(report_path))


if __name__ == "__main__":
    print("CI Computer Module")
    print("This module should be imported and used by the pipeline orchestrator")
