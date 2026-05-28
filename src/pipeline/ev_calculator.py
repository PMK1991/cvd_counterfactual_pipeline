# conda-env: mtech-env
"""
Target-flip robustness index for the CVD counterfactual pipeline.

The pipeline records the percentage of generated DiCE counterfactuals that
ultimately flip the classifier/SCM target. This module converts that single
flip probability `p` to odds `p / (1 - p)` (mirrored above 1 when p < 0.5),
then plugs that odds into the VanderWeele-Ding E-value formula
`RR + sqrt(RR * (RR - 1))`.

This is *not* the standard VanderWeele-Ding E-value, which is defined over a
risk ratio comparing two arms (exposed vs unexposed). Treating a single-arm
odds as the RR is a derivative robustness summary specific to this pipeline,
useful for indicating how concentrated the target-flip mass is. The output is
exposed under the name `target_flip_robustness_index` to avoid implying
identification with the published E-value quantity.
"""

import json
import math
from pathlib import Path
from typing import Dict, Optional

import pandas as pd


class EVCalculator:
    """Compute and persist E-values from aggregated iteration metrics."""

    def __init__(self, confidence_level: float = 0.95):
        self.confidence_level = confidence_level

    @staticmethod
    def _evalue(rr: float) -> Optional[float]:
        if rr < 1:
            return None
        return rr + math.sqrt(rr * (rr - 1))

    @staticmethod
    def _probability_to_rr(probability: float) -> Optional[float]:
        if probability <= 0 or probability >= 1:
            return None
        odds = probability / (1 - probability)
        return odds if odds >= 1 else 1 / odds

    def compute(
        self,
        aggregated_results: pd.DataFrame,
        ci_results: Optional[pd.DataFrame] = None,
    ) -> Dict:
        if 'target_flip_rate_pct' not in aggregated_results.columns:
            return {
                'computed': False,
                'reason': 'target_flip_rate_pct column is unavailable',
            }

        values = aggregated_results['target_flip_rate_pct'].dropna() / 100.0
        if values.empty:
            return {
                'computed': False,
                'reason': 'target_flip_rate_pct contains no numeric values',
            }

        probability = float(values.mean())
        rr = self._probability_to_rr(probability)
        if rr is None:
            return {
                'computed': False,
                'reason': 'target flip probability must be strictly between 0 and 1',
                'target_flip_probability': probability,
            }

        lower_probability = None
        upper_probability = None
        if ci_results is not None:
            row = ci_results[ci_results['metric'] == 'target_flip_rate_pct']
            if not row.empty:
                lower_probability = float(row.iloc[0]['ci_lower']) / 100.0
                upper_probability = float(row.iloc[0]['ci_upper']) / 100.0

        lower_rr_raw = (
            self._probability_to_rr(lower_probability)
            if lower_probability is not None else None
        )
        upper_rr_raw = (
            self._probability_to_rr(upper_probability)
            if upper_probability is not None else None
        )
        rr_bounds = [
            value for value in (lower_rr_raw, upper_rr_raw)
            if value is not None
        ]
        lower_rr = min(rr_bounds) if rr_bounds else None
        upper_rr = max(rr_bounds) if rr_bounds else None

        e_value_bounds = [
            self._evalue(value) for value in rr_bounds
            if self._evalue(value) is not None
        ]
        e_value_lower = min(e_value_bounds) if e_value_bounds else None
        e_value_upper = max(e_value_bounds) if e_value_bounds else None

        return {
            'computed': True,
            'quantity': 'target_flip_rate',
            'formula': (
                'odds(p) = p / (1 - p); rr_like = max(odds, 1/odds); '
                'index = rr_like + sqrt(rr_like * (rr_like - 1))'
            ),
            'scale_note': (
                'Derivative robustness summary, not the published '
                'VanderWeele-Ding E-value. Single-arm odds substituted for '
                'the risk ratio.'
            ),
            'confidence_level': self.confidence_level,
            'target_flip_probability': probability,
            'risk_ratio_like_estimate': rr,
            'risk_ratio_like_lower': lower_rr,
            'risk_ratio_like_upper': upper_rr,
            'target_flip_robustness_index': self._evalue(rr),
            'target_flip_robustness_index_lower': e_value_lower,
            'target_flip_robustness_index_upper': e_value_upper,
            'n_iterations': int(len(values)),
        }

    def to_ci_row(self, payload: Dict) -> Optional[Dict]:
        if not payload.get('computed'):
            return None
        index_value = payload.get('target_flip_robustness_index')
        if index_value is None:
            return None
        lower = payload.get('target_flip_robustness_index_lower')
        upper = payload.get('target_flip_robustness_index_upper')
        return {
            'metric': 'target_flip_robustness_index',
            'mean': index_value,
            'std': None,
            'median': index_value,
            'min': lower,
            'max': upper,
            'ci_lower': lower,
            'ci_upper': upper,
            'ci_width': (
                upper - lower
                if lower is not None and upper is not None else None
            ),
            'n_iterations': payload.get('n_iterations'),
            'interval_type': 'target_flip_robustness_index',
        }

    def save(self, payload: Dict, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(payload, f, indent=2)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compute target-flip E-value")
    parser.add_argument("metrics_csv", help="Path to all_iteration_metrics.csv")
    parser.add_argument("--output", default="evalue.json", help="Output JSON path")
    args = parser.parse_args()

    df = pd.read_csv(args.metrics_csv)
    calculator = EVCalculator()
    result = calculator.compute(df)
    calculator.save(result, Path(args.output))
