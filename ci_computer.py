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
        
        for col in aggregated_results.columns:
            if col == 'iteration':
                continue
            
            values = aggregated_results[col].dropna()
            
            if len(values) == 0:
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
    
    def generate_summary_report(self, ci_results: pd.DataFrame, output_path: str) -> None:
        """
        Generate markdown summary report
        
        Args:
            ci_results: DataFrame with CI results
            output_path: Path to save report
        """
        report_lines = [
            "# Fresh Counterfactual Generation - CI Results",
            "",
            f"**Iterations:** {ci_results['n_iterations'].iloc[0] if len(ci_results) > 0 else 0}",
            f"**Confidence Level:** {self.confidence_level * 100}%",
            "",
            "## Summary Table",
            "",
            "| Metric | Mean | 95% CI | Std Dev |",
            "|--------|------|--------|---------|"
        ]
        
        # Add key metrics
        key_metrics = [
            'total_successful_cfs',
            'trestbps_improved_pct', 'trestbps_worsened_pct',
            'cp_improved_pct', 'cp_worsened_pct',
            'exang_improved_pct', 'exang_worsened_pct',
            'oldpeak_improved_pct', 'oldpeak_worsened_pct',
            'thalach_improved_pct', 'thalach_worsened_pct',
            'slope_improved_pct', 'slope_worsened_pct',
            'restecg_improved_pct', 'restecg_worsened_pct'
        ]
        
        for metric in key_metrics:
            row = ci_results[ci_results['metric'] == metric]
            if not row.empty:
                r = row.iloc[0]
                report_lines.append(
                    f"| {metric} | {r['mean']:.2f} | [{r['ci_lower']:.2f}, {r['ci_upper']:.2f}] | {r['std']:.2f} |"
                )
        
        report_lines.extend([
            "",
            "## Detailed Results",
            "",
            "See `ci_results.csv` for complete results.",
            ""
        ])
        
        # Write report
        with open(output_path, 'w') as f:
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
