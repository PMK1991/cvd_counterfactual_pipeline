# conda-env: mtech-env
"""
Metrics Calculator Module

Computes all diagnostic metrics for successful counterfactuals.
Reuses logic from existing diagnostic_metrics_ci.py.

Author: PMK
Date: 2026-01-26
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _safe_mode_int(series: pd.Series, default: int = 0) -> int:
    """Return the integer mode of a series, or `default` if it is empty/all-NaN."""
    mode = series.mode()
    return int(mode.iloc[0]) if len(mode) else default


class MetricsCalculator:
    """
    Diagnostic Metrics Calculator
    
    Computes all diagnostic metrics for successful counterfactuals.
    Thread-safe for concurrent execution.
    """
    
    def __init__(self):
        """Initialize Metrics Calculator"""
        logger.info("Initialized MetricsCalculator")
    
    def compute_all_metrics(self, df: pd.DataFrame) -> Dict:
        """
        Compute all diagnostic metrics
        
        Args:
            df: DataFrame with successful counterfactuals
            
        Returns:
            Dictionary with all metrics
        """
        if df is None or len(df) == 0:
            return self._empty_metrics()
        
        n_total = len(df)
        
        metrics = {
            'total_successful_cfs': n_total,
            
            # Resting Blood Pressure
            **self.compute_trestbps_metrics(df, n_total),
            
            # Chest Pain
            **self.compute_chest_pain_metrics(df, n_total),
            
            # Exercise-Induced Angina
            **self.compute_exang_metrics(df, n_total),
            
            # ST Depression
            **self.compute_oldpeak_metrics(df, n_total),
            
            # Max Heart Rate
            **self.compute_thalach_metrics(df, n_total),
            
            # ST Slope
            **self.compute_slope_metrics(df, n_total),
            
            # Resting ECG
            **self.compute_restecg_metrics(df, n_total)
        }
        
        return metrics
    
    def _empty_metrics(self) -> Dict:
        """Return empty metrics dictionary"""
        return {'total_successful_cfs': 0}
    
    def compute_trestbps_metrics(self, df: pd.DataFrame, n_total: int) -> Dict:
        """Compute resting blood pressure metrics"""
        return {
            'trestbps_improved_pct': (df['cf_trestbps'] < df['orig_trestbps']).sum() / n_total * 100,
            'trestbps_worsened_pct': (df['cf_trestbps'] > df['orig_trestbps']).sum() / n_total * 100,
            'trestbps_no_change_pct': (df['cf_trestbps'] == df['orig_trestbps']).sum() / n_total * 100,
            'mean_diff_trestbps': (df['cf_trestbps'] - df['orig_trestbps']).mean()
        }
    
    def compute_chest_pain_metrics(self, df: pd.DataFrame, n_total: int) -> Dict:
        """Compute chest pain metrics"""
        return {
            'cp_improved_pct': (df['cf_cp'] < df['orig_cp']).sum() / n_total * 100,
            'cp_worsened_pct': (df['cf_cp'] > df['orig_cp']).sum() / n_total * 100,
            'cp_no_change_pct': (df['cf_cp'] == df['orig_cp']).sum() / n_total * 100,
            'cp_changed_pct': (df['orig_cp'] != df['cf_cp']).sum() / n_total * 100,
            'cp_mode_before': _safe_mode_int(df['orig_cp']),
            'cp_mode_after': _safe_mode_int(df['cf_cp']),
        }
    
    def compute_exang_metrics(self, df: pd.DataFrame, n_total: int) -> Dict:
        """Compute exercise-induced angina metrics"""
        return {
            'exang_improved_pct': ((df['orig_exang'] == 1) & (df['cf_exang'] == 0)).sum() / n_total * 100,
            'exang_worsened_pct': ((df['orig_exang'] == 0) & (df['cf_exang'] == 1)).sum() / n_total * 100,
            'exang_no_change_pct': (df['orig_exang'] == df['cf_exang']).sum() / n_total * 100,
            'exang_mode_before': _safe_mode_int(df['orig_exang']),
            'exang_mode_after': _safe_mode_int(df['cf_exang']),
        }
    
    def compute_oldpeak_metrics(self, df: pd.DataFrame, n_total: int) -> Dict:
        """Compute ST depression metrics"""
        return {
            'oldpeak_improved_pct': (df['cf_oldpeak'] < df['orig_oldpeak']).sum() / n_total * 100,
            'oldpeak_worsened_pct': (df['cf_oldpeak'] > df['orig_oldpeak']).sum() / n_total * 100,
            'oldpeak_no_change_pct': (df['cf_oldpeak'] == df['orig_oldpeak']).sum() / n_total * 100,
            'mean_diff_oldpeak': (df['cf_oldpeak'] - df['orig_oldpeak']).mean()
        }
    
    def compute_thalach_metrics(self, df: pd.DataFrame, n_total: int) -> Dict:
        """Compute maximum heart rate metrics"""
        return {
            'thalach_improved_pct': (df['cf_thalach'] > df['orig_thalach']).sum() / n_total * 100,
            'thalach_worsened_pct': (df['cf_thalach'] < df['orig_thalach']).sum() / n_total * 100,
            'thalach_no_change_pct': (df['cf_thalach'] == df['orig_thalach']).sum() / n_total * 100,
            'mean_diff_thalach': (df['cf_thalach'] - df['orig_thalach']).mean()
        }
    
    def compute_slope_metrics(self, df: pd.DataFrame, n_total: int) -> Dict:
        """Compute ST slope metrics"""
        return {
            'slope_improved_pct': (df['cf_slope'] < df['orig_slope']).sum() / n_total * 100,
            'slope_worsened_pct': (df['cf_slope'] > df['orig_slope']).sum() / n_total * 100,
            'slope_no_change_pct': (df['cf_slope'] == df['orig_slope']).sum() / n_total * 100,
            'slope_mode_before': _safe_mode_int(df['orig_slope']),
            'slope_mode_after': _safe_mode_int(df['cf_slope']),
        }
    
    def compute_restecg_metrics(self, df: pd.DataFrame, n_total: int) -> Dict:
        """Compute resting ECG metrics"""
        return {
            'restecg_improved_pct': (df['cf_restecg'] < df['orig_restecg']).sum() / n_total * 100,
            'restecg_worsened_pct': (df['cf_restecg'] > df['orig_restecg']).sum() / n_total * 100,
            'restecg_no_change_pct': (df['orig_restecg'] == df['cf_restecg']).sum() / n_total * 100,
            'restecg_mode_before': _safe_mode_int(df['orig_restecg']),
            'restecg_mode_after': _safe_mode_int(df['cf_restecg']),
        }

    # ------------------------------------------------------------------
    # Improvement-focused causal recourse
    # ------------------------------------------------------------------

    # Downstream symptom variables (children of `target` in the SCM) and the
    # clinically-beneficial direction of change. These are the variables that
    # may still improve when an intervention does NOT flip the disease label.
    # +1 => improvement is an increase; -1 => improvement is a decrease.
    _SYMPTOM_DIRECTIONS = {
        'cp': -1,        # less severe chest pain
        'restecg': -1,   # toward normal resting ECG
        'thalach': +1,   # higher max heart rate
        'exang': -1,     # exercise-induced angina 1 -> 0
        'slope': -1,     # flatter/normal ST slope
        'oldpeak': -1,   # less ST depression
    }

    def _symptom_change_counts(self, df: pd.DataFrame):
        """Per-row counts of improved / worsened downstream symptoms.

        Returns two integer Series (n_improved, n_worsened) aligned with `df`,
        where each downstream symptom is scored in its clinically-beneficial
        direction. Symptoms absent from `df` are skipped.
        """
        n_improved = pd.Series(0, index=df.index, dtype=int)
        n_worsened = pd.Series(0, index=df.index, dtype=int)

        for sym, direction in self._SYMPTOM_DIRECTIONS.items():
            orig_col, cf_col = f'orig_{sym}', f'cf_{sym}'
            if orig_col not in df.columns or cf_col not in df.columns:
                continue
            delta = df[cf_col] - df[orig_col]
            beneficial = delta * direction  # > 0 => improved, < 0 => worsened
            n_improved += (beneficial > 0).astype(int)
            n_worsened += (beneficial < 0).astype(int)

        return n_improved, n_worsened

    def compute_recourse_metrics(self, df: pd.DataFrame) -> Dict:
        """Improvement-focused recourse metrics over the supplied CF rows.

        Intended for the NON-flip subset (CFs where the SCM left ``target`` at
        1): quantifies whether downstream symptoms still moved in a beneficial
        direction. Outcome-agnostic — the caller decides which rows to pass.

        All returned values are scalar numerics so :class:`CIComputer` picks
        them up for percentile CIs across iterations.
        """
        if df is None or len(df) == 0:
            return {
                'recourse_n_cfs': 0,
                'recourse_any_improvement_pct': 0.0,
                'recourse_irr_strict_pct': 0.0,
                'recourse_irr_lenient_pct': 0.0,
                'recourse_mean_n_improved': 0.0,
                'recourse_mean_n_worsened': 0.0,
                'recourse_mean_net_improvement': 0.0,
            }

        n_total = len(df)
        n_improved, n_worsened = self._symptom_change_counts(df)
        net = n_improved - n_worsened

        any_improvement = n_improved >= 1
        # Strict: at least one symptom improved and none worsened.
        irr_strict = any_improvement & (n_worsened == 0)
        # Lenient: at least one improved and net effect non-negative.
        irr_lenient = any_improvement & (net >= 0)

        return {
            'recourse_n_cfs': n_total,
            'recourse_any_improvement_pct': any_improvement.sum() / n_total * 100,
            'recourse_irr_strict_pct': irr_strict.sum() / n_total * 100,
            'recourse_irr_lenient_pct': irr_lenient.sum() / n_total * 100,
            'recourse_mean_n_improved': n_improved.mean(),
            'recourse_mean_n_worsened': n_worsened.mean(),
            'recourse_mean_net_improvement': net.mean(),
        }


if __name__ == "__main__":
    print("Metrics Calculator Module")
    print("This module should be imported and used by the pipeline orchestrator")
