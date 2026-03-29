# conda-env: mtech-env
"""
Sensitivity Analyzer Module

Performs one-at-a-time (OAT) sensitivity analysis by varying key pipeline
parameters and comparing their impact on counterfactual generation metrics.

Author: PMK
Date: 2026-03-28
"""

import copy
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional
import logging
import time

from ci_computer import CIComputer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Key metrics to track across sensitivity variants
KEY_METRICS = [
    'total_successful_cfs',
    'trestbps_improved_pct',
    'mean_diff_trestbps',
    'cp_improved_pct',
    'exang_improved_pct',
    'oldpeak_improved_pct',
    'thalach_improved_pct',
    'slope_improved_pct',
]

# Parameter grid: each entry defines the parameter path, values to sweep, and baseline
PARAMETER_GRID = {
    'total_cfs': {
        'config_path': ['dice', 'total_cfs'],
        'values': [3, 5, 7, 10],
        'baseline': 5,
        'label': 'Number of CFs per patient',
    },
    'trestbps_range': {
        'config_path': ['dice', 'permitted_range', 'trestbps'],
        'values': [[90, 110], [100, 120], [110, 130], [120, 140]],
        'baseline': [100, 120],
        'label': 'Permitted BP range (mmHg)',
    },
    'chol_lower': {
        'config_path': ['dice', 'permitted_range', 'chol'],
        'values': [100, 150, 200],
        'baseline': 150,
        'label': 'Cholesterol lower bound (mg/dL)',
    },
    'confidence_level': {
        'config_path': ['ci', 'confidence_level'],
        'values': [0.90, 0.95, 0.99],
        'baseline': 0.95,
        'label': 'Confidence level',
    },
    # --- SCM / SEM parameters ---
    'graph_structure': {
        'config_path': ['scm', 'graph_structure'],
        'values': ['minimal', 'full', 'extended'],
        'baseline': 'full',
        'label': 'SCM graph structure',
    },
    'intervention_targets': {
        'config_path': ['scm', 'intervention_targets'],
        'values': ['chol_only', 'trestbps_only', 'both'],
        'baseline': 'both',
        'label': 'SCM intervention targets',
    },
    'n_samples': {
        'config_path': ['scm', 'n_samples'],
        'values': [1, 5, 10],
        'baseline': 1,
        'label': 'SCM interventional samples',
    },
}


class SensitivityAnalyzer:
    """
    Sensitivity Analyzer for the CVD Counterfactual Pipeline.

    Varies one parameter at a time, runs reduced pipeline executions,
    and compares metrics across parameter settings.
    """

    def __init__(
        self,
        baseline_config: Dict,
        output_dir: str = 'fresh_cf_iterations/sensitivity_results',
        n_iterations: int = 10,
        n_patients: int = 10,
        n_workers: int = 2,
        baseline_results_path: Optional[str] = None,
    ):
        self.baseline_config = baseline_config
        self.output_dir = Path(output_dir)
        self.n_iterations = n_iterations
        self.n_patients = n_patients
        self.n_workers = n_workers
        self.baseline_results_path = baseline_results_path or (
            'fresh_cf_iterations/aggregated_results/all_iteration_metrics.csv'
        )
        logger.info(
            f"Initialized SensitivityAnalyzer: {n_iterations} iterations, "
            f"{n_patients} patients per variant"
        )

    # ------------------------------------------------------------------
    # Config helpers
    # ------------------------------------------------------------------

    def _make_variant_config(
        self, param_name: str, config_path: List[str], value
    ) -> Dict:
        """Deep-copy baseline config and set one parameter to *value*."""
        config = copy.deepcopy(self.baseline_config)

        # Navigate to the parent of the leaf key
        node = config
        for key in config_path[:-1]:
            node = node[key]

        # Special handling for chol_lower: value is the lower bound only
        if param_name == 'chol_lower':
            node[config_path[-1]] = [value, None]
        else:
            node[config_path[-1]] = value

        # Override pipeline settings for reduced run
        config['pipeline']['n_iterations'] = self.n_iterations
        config['pipeline']['n_patients'] = self.n_patients
        config['pipeline']['n_workers'] = self.n_workers

        # Unique output directory per variant
        value_label = self._value_label(value)
        config['output']['base_dir'] = str(
            self.output_dir / param_name / f'variant_{value_label}'
        )

        return config

    @staticmethod
    def _value_label(value) -> str:
        """Human-readable label for a parameter value."""
        if isinstance(value, list):
            return '_'.join(str(v) for v in value)
        if isinstance(value, float):
            return str(value).replace('.', '')
        return str(value)

    # ------------------------------------------------------------------
    # Baseline loading
    # ------------------------------------------------------------------

    def _load_baseline_results(self) -> pd.DataFrame:
        """Load the existing 100-iteration baseline metrics CSV."""
        path = Path(self.baseline_results_path)
        if not path.exists():
            logger.warning(f"Baseline results not found at {path}")
            return pd.DataFrame()
        df = pd.read_csv(path)
        logger.info(f"Loaded baseline results: {len(df)} iterations from {path}")
        return df

    def _baseline_summary(self, baseline_df: pd.DataFrame) -> Dict:
        """Compute summary statistics for the baseline."""
        if baseline_df.empty:
            return {}
        ci = CIComputer(confidence_level=0.95)
        ci_df = ci.compute_confidence_intervals(baseline_df)
        summary = {'parameter_value': 'Baseline (100 iter)'}
        for metric in KEY_METRICS:
            row = ci_df[ci_df['metric'] == metric]
            if row.empty:
                continue
            r = row.iloc[0]
            summary[f'{metric}_mean'] = r['mean']
            summary[f'{metric}_ci_lower'] = r['ci_lower']
            summary[f'{metric}_ci_upper'] = r['ci_upper']
        return summary

    # ------------------------------------------------------------------
    # Pipeline execution
    # ------------------------------------------------------------------

    def _run_parameter_sweep(self, param_name: str, param_spec: Dict) -> pd.DataFrame:
        """Run pipeline for each value of a single parameter."""
        # Import here to avoid circular imports
        from fresh_cf_pipeline import FreshCFPipeline

        rows = []
        for value in param_spec['values']:
            value_label = self._value_label(value)
            logger.info(
                f"[{param_name}] Running variant: {value_label}"
            )
            start = time.time()

            config = self._make_variant_config(
                param_name, param_spec['config_path'], value
            )
            pipeline = FreshCFPipeline(config=config)
            agg_df = pipeline.run_concurrent_pipeline()

            # Compute CIs for this variant
            ci = CIComputer(confidence_level=config['ci']['confidence_level'])
            ci_df = ci.compute_confidence_intervals(agg_df)

            # Also save per-variant aggregated results
            pipeline.compute_and_save_results(agg_df)

            elapsed = time.time() - start

            # Extract key metrics
            row = {
                'parameter': param_name,
                'parameter_label': param_spec['label'],
                'parameter_value': str(value),
                'is_baseline_value': str(value) == str(param_spec['baseline']),
                'n_iterations': len(agg_df),
                'elapsed_seconds': round(elapsed, 1),
            }
            for metric in KEY_METRICS:
                ci_row = ci_df[ci_df['metric'] == metric]
                if ci_row.empty:
                    continue
                r = ci_row.iloc[0]
                row[f'{metric}_mean'] = r['mean']
                row[f'{metric}_ci_lower'] = r['ci_lower']
                row[f'{metric}_ci_upper'] = r['ci_upper']

            rows.append(row)
            logger.info(
                f"[{param_name}] variant {value_label} done in {elapsed:.1f}s"
            )

        return pd.DataFrame(rows)

    def _handle_confidence_level(self) -> pd.DataFrame:
        """Recompute CIs from baseline data at different confidence levels."""
        baseline_df = self._load_baseline_results()
        if baseline_df.empty:
            logger.warning("Cannot run confidence_level sensitivity without baseline data")
            return pd.DataFrame()

        spec = PARAMETER_GRID['confidence_level']
        rows = []
        for cl in spec['values']:
            ci = CIComputer(confidence_level=cl)
            ci_df = ci.compute_confidence_intervals(baseline_df)

            row = {
                'parameter': 'confidence_level',
                'parameter_label': spec['label'],
                'parameter_value': str(cl),
                'is_baseline_value': cl == spec['baseline'],
                'n_iterations': len(baseline_df),
                'elapsed_seconds': 0.0,
            }
            for metric in KEY_METRICS:
                ci_row = ci_df[ci_df['metric'] == metric]
                if ci_row.empty:
                    continue
                r = ci_row.iloc[0]
                row[f'{metric}_mean'] = r['mean']
                row[f'{metric}_ci_lower'] = r['ci_lower']
                row[f'{metric}_ci_upper'] = r['ci_upper']
            rows.append(row)

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run_sensitivity_analysis(
        self, parameters: Optional[List[str]] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Run the full sensitivity analysis.

        Args:
            parameters: Subset of parameter names to sweep (default: all).

        Returns:
            Dict mapping parameter name to comparison DataFrame.
        """
        params = parameters or list(PARAMETER_GRID.keys())
        results: Dict[str, pd.DataFrame] = {}

        logger.info("=" * 80)
        logger.info("SENSITIVITY ANALYSIS")
        logger.info(f"Parameters: {params}")
        logger.info("=" * 80)

        start_time = time.time()

        for param_name in params:
            if param_name not in PARAMETER_GRID:
                logger.warning(f"Unknown parameter: {param_name}, skipping")
                continue

            spec = PARAMETER_GRID[param_name]

            if param_name == 'confidence_level':
                df = self._handle_confidence_level()
            else:
                df = self._run_parameter_sweep(param_name, spec)

            if not df.empty:
                results[param_name] = df
                # Save per-parameter comparison
                param_dir = self.output_dir / param_name
                param_dir.mkdir(parents=True, exist_ok=True)
                df.to_csv(param_dir / 'comparison.csv', index=False)

        # Save combined results and report
        self.save_all_results(results)
        self.generate_sensitivity_report(results)

        elapsed = time.time() - start_time
        logger.info("=" * 80)
        logger.info(f"Sensitivity analysis completed in {elapsed / 60:.1f} minutes")
        logger.info(f"Results saved to: {self.output_dir}")
        logger.info("=" * 80)

        return results

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def save_all_results(self, results: Dict[str, pd.DataFrame]) -> None:
        """Save combined sensitivity results CSV."""
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if results:
            combined = pd.concat(results.values(), ignore_index=True)
            combined.to_csv(self.output_dir / 'all_sensitivity_results.csv', index=False)
            logger.info("Saved all_sensitivity_results.csv")

    def generate_sensitivity_report(self, results: Dict[str, pd.DataFrame]) -> None:
        """Generate a markdown sensitivity report."""
        baseline_df = self._load_baseline_results()
        baseline_summary = self._baseline_summary(baseline_df)

        lines = [
            "# Sensitivity Analysis Report",
            "",
            f"**Date:** {time.strftime('%Y-%m-%d %H:%M')}",
            f"**Iterations per variant:** {self.n_iterations}",
            f"**Patients per variant:** {self.n_patients}",
            f"**Baseline reference:** 100 iterations, 48 patients",
            "",
        ]

        for param_name, df in results.items():
            spec = PARAMETER_GRID[param_name]
            lines.append(f"## {spec['label']} (`{param_name}`)")
            lines.append("")

            # Build comparison table
            header_metrics = ['total_successful_cfs', 'trestbps_improved_pct',
                              'cp_improved_pct', 'thalach_improved_pct']
            col_labels = ['Value', 'Successful CFs', 'BP Improved %',
                          'CP Improved %', 'Thalach Improved %']

            lines.append('| ' + ' | '.join(col_labels) + ' |')
            lines.append('|' + '|'.join(['------'] * len(col_labels)) + '|')

            # Baseline row
            if baseline_summary:
                cells = [baseline_summary.get('parameter_value', 'Baseline')]
                for m in header_metrics:
                    mean = baseline_summary.get(f'{m}_mean')
                    ci_lo = baseline_summary.get(f'{m}_ci_lower')
                    ci_hi = baseline_summary.get(f'{m}_ci_upper')
                    if mean is not None:
                        cells.append(f"{mean:.1f} [{ci_lo:.1f}, {ci_hi:.1f}]")
                    else:
                        cells.append("—")
                lines.append('| ' + ' | '.join(cells) + ' |')

            # Variant rows
            for _, row in df.iterrows():
                marker = " *" if row.get('is_baseline_value') else ""
                cells = [f"{row['parameter_value']}{marker}"]
                for m in header_metrics:
                    mean = row.get(f'{m}_mean')
                    ci_lo = row.get(f'{m}_ci_lower')
                    ci_hi = row.get(f'{m}_ci_upper')
                    if pd.notna(mean):
                        cells.append(f"{mean:.1f} [{ci_lo:.1f}, {ci_hi:.1f}]")
                    else:
                        cells.append("—")
                lines.append('| ' + ' | '.join(cells) + ' |')

            lines.append("")
            lines.append(f"*Baseline value marked with \\**")
            lines.append("")

        # Key findings section
        lines.extend(self._generate_findings(results))

        lines.extend([
            "",
            "## Detailed Results",
            "",
            "See `all_sensitivity_results.csv` for complete numeric results.",
            "Per-parameter comparisons are in each parameter's `comparison.csv`.",
            "",
        ])

        report_path = self.output_dir / 'sensitivity_report.md'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        logger.info(f"Generated sensitivity report: {report_path}")

    def _generate_findings(self, results: Dict[str, pd.DataFrame]) -> List[str]:
        """Generate key findings bullet points from sensitivity results."""
        lines = ["## Key Findings", ""]

        for param_name, df in results.items():
            spec = PARAMETER_GRID[param_name]
            # Compute range of total_successful_cfs_mean across variants
            col = 'total_successful_cfs_mean'
            if col in df.columns:
                min_val = df[col].min()
                max_val = df[col].max()
                spread = max_val - min_val
                lines.append(
                    f"- **{spec['label']}**: Successful CFs ranged from "
                    f"{min_val:.1f} to {max_val:.1f} (spread: {spread:.1f}) "
                    f"across {len(df)} variants."
                )

            # Check trestbps_improved_pct spread
            col2 = 'trestbps_improved_pct_mean'
            if col2 in df.columns:
                min_val = df[col2].min()
                max_val = df[col2].max()
                spread = max_val - min_val
                if spread > 5:
                    lines.append(
                        f"  - BP improvement % varied by {spread:.1f} pp "
                        f"({min_val:.1f}% – {max_val:.1f}%)."
                    )

        if len(lines) == 2:
            lines.append("- No significant variation detected across parameters.")

        return lines


if __name__ == "__main__":
    print("Sensitivity Analyzer Module")
    print("Use: python fresh_cf_pipeline.py --sensitivity")
    print("Or import and use SensitivityAnalyzer directly")
