# conda-env: mtech-env
"""
SCM Analyzer Module

This module handles SCM-based counterfactual validation using DoWhy.
It wraps the existing CounterfactualAnalyzer for use in the pipeline.

Author: PMK
Date: 2026-01-26
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
import logging
import sys

# Import existing CounterfactualAnalyzer
from counterfactualAnalyzer import CounterfactualAnalyzer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SCMAnalyzer:
    """
    SCM-based Counterfactual Analyzer
    
    Validates counterfactuals using structural causal models (DoWhy).
    Thread-safe for concurrent execution.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize SCM Analyzer
        
        Args:
            config: Configuration dictionary for SCM parameters
        """
        self.config = config or self._default_config()
        self.analyzer = None
        
        logger.info("Initialized SCMAnalyzer")
    
    def _default_config(self) -> Dict:
        """Default SCM configuration"""
        return {
            'n_samples': 1000,
            'causal_model': 'default'
        }
    
    def initialize_analyzer(self) -> None:
        """Initialize the CounterfactualAnalyzer"""
        try:
            self.analyzer = CounterfactualAnalyzer()
            logger.info("CounterfactualAnalyzer initialized")
        except Exception as e:
            logger.error(f"Error initializing analyzer: {e}")
            raise
    
    def load_counterfactuals_for_iteration(
        self,
        iteration_dir: str
    ) -> List[Dict]:
        """
        Load all counterfactuals for a specific iteration
        
        Args:
            iteration_dir: Path to iteration directory
            
        Returns:
            List of dictionaries with original and CF data
        """
        iter_path = Path(iteration_dir)
        orig_dir = iter_path / "original"
        cf_dir = iter_path / "counterfactuals"
        
        if not orig_dir.exists() or not cf_dir.exists():
            logger.warning(f"Missing directories in {iteration_dir}")
            return []
        
        cf_pairs = []
        
        # Iterate through original files
        for orig_file in orig_dir.glob("patient_*.csv"):
            patient_id = orig_file.stem.replace("patient_", "")
            
            # Load original
            orig_df = pd.read_csv(orig_file)
            
            # Find corresponding CFs
            cf_files = list(cf_dir.glob(f"patient_{patient_id}_cf_*.csv"))
            
            for cf_file in cf_files:
                cf_df = pd.read_csv(cf_file)
                
                cf_pairs.append({
                    'patient_id': patient_id,
                    'original': orig_df,
                    'cf_suggestion': cf_df,
                    'cf_file': str(cf_file)
                })
        
        logger.info(f"Loaded {len(cf_pairs)} CF pairs from {iteration_dir}")
        return cf_pairs
    
    def apply_scm_intervention(
        self,
        original: pd.DataFrame,
        cf_suggestion: pd.DataFrame
    ) -> Optional[pd.DataFrame]:
        """
        Apply SCM intervention to validate counterfactual
        
        Args:
            original: Original patient data
            cf_suggestion: Suggested counterfactual
            
        Returns:
            SCM-validated counterfactual or None if failed
        """
        if self.analyzer is None:
            raise ValueError("Must call initialize_analyzer() first")
        
        try:
            # Extract intervention values
            chol_value = cf_suggestion['chol'].values[0]
            trestbps_value = cf_suggestion['trestbps'].values[0]
            
            # Apply SCM intervention using existing analyzer
            cf_result = self.analyzer.generate_counterfactual(
                original,
                chol_value,
                trestbps_value,
                n_samples=self.config['n_samples']
            )
            
            return cf_result
            
        except Exception as e:
            logger.debug(f"SCM intervention failed: {e}")
            return None
    
    def validate_counterfactual(
        self,
        cf_result: pd.DataFrame,
        original_target: int
    ) -> bool:
        """
        Validate if counterfactual successfully flips target
        
        Args:
            cf_result: SCM-generated counterfactual
            original_target: Original target value
            
        Returns:
            True if target flipped from 1 to 0
        """
        if cf_result is None or len(cf_result) == 0:
            return False
        
        # Check if target flipped from 1 to 0
        cf_target = cf_result['target'].values[0]
        
        return original_target == 1 and cf_target == 0
    
    def analyze_iteration(
        self,
        iteration_dir: str,
        output_dir: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Complete SCM analysis for one iteration
        
        Args:
            iteration_dir: Path to iteration directory
            output_dir: Optional custom output directory
            
        Returns:
            DataFrame with successful counterfactuals
        """
        if self.analyzer is None:
            self.initialize_analyzer()
        
        # Load CFs
        cf_pairs = self.load_counterfactuals_for_iteration(iteration_dir)
        
        successful_cfs = []
        
        for pair in cf_pairs:
            # Apply SCM
            cf_result = self.apply_scm_intervention(
                pair['original'],
                pair['cf_suggestion']
            )
            
            # Validate
            original_target = pair['original']['target'].values[0]
            
            if self.validate_counterfactual(cf_result, original_target):
                # Add metadata
                cf_result['patient_id'] = pair['patient_id']
                cf_result['orig_chol'] = pair['original']['chol'].values[0]
                cf_result['orig_trestbps'] = pair['original']['trestbps'].values[0]
                cf_result['cf_chol'] = pair['cf_suggestion']['chol'].values[0]
                cf_result['cf_trestbps'] = pair['cf_suggestion']['trestbps'].values[0]
                
                # Copy other original features
                for col in pair['original'].columns:
                    if col not in cf_result.columns and col != 'target':
                        cf_result[f'orig_{col}'] = pair['original'][col].values[0]
                
                # Copy other CF features
                for col in pair['cf_suggestion'].columns:
                    if col not in cf_result.columns and col != 'target':
                        cf_result[f'cf_{col}'] = pair['cf_suggestion'][col].values[0]
                
                successful_cfs.append(cf_result)
        
        # Combine into DataFrame
        if successful_cfs:
            result_df = pd.concat(successful_cfs, ignore_index=True)
            logger.info(f"Found {len(result_df)} successful CFs in iteration")
        else:
            result_df = pd.DataFrame()
            logger.warning("No successful CFs found in iteration")
        
        # Save if output_dir specified
        if output_dir:
            self.save_successful_cfs(result_df, iteration_dir, output_dir)
        
        return result_df
    
    def save_successful_cfs(
        self,
        successful_cfs: pd.DataFrame,
        iteration_dir: str,
        output_dir: Optional[str] = None
    ) -> str:
        """
        Save successful counterfactuals
        
        Args:
            successful_cfs: DataFrame with successful CFs
            iteration_dir: Iteration directory path
            output_dir: Optional custom output directory
            
        Returns:
            Path to saved file
        """
        if output_dir is None:
            output_dir = Path(iteration_dir) / "successful"
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = output_dir / "successful_counterfactuals.csv"
        successful_cfs.to_csv(output_file, index=False)
        
        logger.info(f"Saved {len(successful_cfs)} successful CFs to {output_file}")
        
        return str(output_file)


if __name__ == "__main__":
    print("SCM Analyzer Module")
    print("This module should be imported and used by the pipeline orchestrator")
