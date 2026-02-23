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
            'cp_changed_pct': (df['orig_cp'] != df['cf_cp']).sum() / n_total * 100
        }
    
    def compute_exang_metrics(self, df: pd.DataFrame, n_total: int) -> Dict:
        """Compute exercise-induced angina metrics"""
        return {
            'exang_improved_pct': ((df['orig_exang'] == 1) & (df['cf_exang'] == 0)).sum() / n_total * 100,
            'exang_worsened_pct': ((df['orig_exang'] == 0) & (df['cf_exang'] == 1)).sum() / n_total * 100,
            'exang_no_change_pct': (df['orig_exang'] == df['cf_exang']).sum() / n_total * 100
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
            'slope_no_change_pct': (df['cf_slope'] == df['orig_slope']).sum() / n_total * 100
        }
    
    def compute_restecg_metrics(self, df: pd.DataFrame, n_total: int) -> Dict:
        """Compute resting ECG metrics"""
        return {
            'restecg_improved_pct': (df['cf_restecg'] < df['orig_restecg']).sum() / n_total * 100,
            'restecg_worsened_pct': (df['cf_restecg'] > df['orig_restecg']).sum() / n_total * 100,
            'restecg_no_change_pct': (df['orig_restecg'] == df['cf_restecg']).sum() / n_total * 100
        }


if __name__ == "__main__":
    print("Metrics Calculator Module")
    print("This module should be imported and used by the pipeline orchestrator")
