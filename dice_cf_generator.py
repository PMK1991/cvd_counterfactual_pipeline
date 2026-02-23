# conda-env: mtech-env
"""
DiCE Counterfactual Generator Module

This module encapsulates all DiCE-ML counterfactual generation logic.
It provides a clean interface for generating counterfactuals for CVD patients.

Author: PMK
Date: 2026-01-26
"""

import pandas as pd
import numpy as np
import dice_ml
from dice_ml import Data, Model, Dice
import threading
import queue
import os
import pickle
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DiceCFGenerator:
    """
    DiCE-based Counterfactual Generator
    
    Generates counterfactuals using DiCE-ML genetic algorithm.
    Thread-safe and supports concurrent execution.
    """
    
    def __init__(self, model_path: str, data_path: str, config: Optional[Dict] = None):
        """
        Initialize DiCE CF Generator
        
        Args:
            model_path: Path to trained model pickle file
            data_path: Path to training data CSV
            config: Configuration dictionary with DiCE parameters
        """
        self.model_path = model_path
        self.data_path = data_path
        self.config = config or self._default_config()
        
        self.model = None
        self.dice_data = None
        self.dice_model = None
        self.dice_exp = None
        
        logger.info(f"Initialized DiceCFGenerator with model: {model_path}")
    
    def _default_config(self) -> Dict:
        """Default DiCE configuration"""
        return {
            'method': 'genetic',
            'total_cfs': 5,
            'permitted_range': {
                'trestbps': [100, 120],
                'chol': [150, None]  # None means use 90% of original
            },
            'timeout': 30,
            'features_to_vary': None  # None means all actionable features
        }
    
    def load_model_and_data(self) -> None:
        """Load trained model and prepare data for DiCE"""
        try:
            # Load model
            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)
            logger.info(f"Loaded model from {self.model_path}")
            
            # Load data
            df = pd.read_csv(self.data_path)
            logger.info(f"Loaded data from {self.data_path}: {len(df)} rows")
            
            # Prepare DiCE data object
            self.dice_data = Data(
                dataframe=df,
                continuous_features=['age', 'trestbps', 'chol', 'thalach', 'oldpeak'],
                outcome_name='target'
            )
            
            # Prepare DiCE model object
            self.dice_model = Model(model=self.model, backend='sklearn')
            
            logger.info("Successfully prepared DiCE data and model objects")
            
        except Exception as e:
            logger.error(f"Error loading model and data: {e}")
            raise
    
    def setup_dice_explainer(self) -> None:
        """Setup DiCE explainer"""
        try:
            if self.dice_data is None or self.dice_model is None:
                raise ValueError("Must call load_model_and_data() first")
            
            self.dice_exp = Dice(
                self.dice_data,
                self.dice_model,
                method=self.config['method']
            )
            
            logger.info(f"DiCE explainer setup with method: {self.config['method']}")
            
        except Exception as e:
            logger.error(f"Error setting up DiCE explainer: {e}")
            raise
    
    def generate_counterfactuals(
        self,
        patient_data: pd.DataFrame,
        timeout: Optional[int] = None
    ) -> Optional[dice_ml.counterfactual_explanations.CounterfactualExplanations]:
        """
        Generate counterfactuals for a single patient
        
        Args:
            patient_data: DataFrame with single patient row
            timeout: Timeout in seconds (uses config default if None)
            
        Returns:
            DiCE CounterfactualExplanations object or None if failed
        """
        if self.dice_exp is None:
            raise ValueError("Must call setup_dice_explainer() first")
        
        timeout = timeout or self.config['timeout']
        
        # Prepare permitted range (handle dynamic chol limit)
        permitted_range = self.config['permitted_range'].copy()
        if permitted_range['chol'][1] is None:
            original_chol = patient_data['chol'].values[0]
            permitted_range['chol'][1] = original_chol - 0.1 * original_chol
        
        # Use threading with timeout for robustness
        result_queue = queue.Queue()
        
        def target():
            try:
                cf_result = self.dice_exp.generate_counterfactuals(
                    patient_data,
                    total_CFs=self.config['total_cfs'],
                    desired_class='opposite',
                    permitted_range=permitted_range,
                    features_to_vary=self.config['features_to_vary']
                )
                result_queue.put(cf_result)
            except Exception as e:
                logger.warning(f"DiCE generation failed: {e}")
                result_queue.put(None)
        
        thread = threading.Thread(target=target)
        thread.start()
        thread.join(timeout)
        
        if thread.is_alive():
            logger.warning(f"DiCE generation timed out after {timeout}s")
            return None
        
        return result_queue.get() if not result_queue.empty() else None
    
    def save_counterfactuals(
        self,
        cf_result: dice_ml.counterfactual_explanations.CounterfactualExplanations,
        iteration_num: int,
        patient_id: int,
        output_dir: str
    ) -> Tuple[str, List[str]]:
        """
        Save counterfactuals to disk
        
        Args:
            cf_result: DiCE counterfactual result
            iteration_num: Iteration number
            patient_id: Patient ID
            output_dir: Base output directory
            
        Returns:
            Tuple of (original_path, list of cf_paths)
        """
        # Create directory structure
        iter_dir = Path(output_dir) / f"iteration_{iteration_num:03d}"
        orig_dir = iter_dir / "original"
        cf_dir = iter_dir / "counterfactuals"
        
        orig_dir.mkdir(parents=True, exist_ok=True)
        cf_dir.mkdir(parents=True, exist_ok=True)
        
        # Save original patient data
        orig_path = orig_dir / f"patient_{patient_id}.csv"
        cf_result.test_instance_df.to_csv(orig_path, index=False)
        
        # Save counterfactuals
        cf_paths = []
        if cf_result.final_cfs_df is not None and len(cf_result.final_cfs_df) > 0:
            for i, cf_row in cf_result.final_cfs_df.iterrows():
                cf_path = cf_dir / f"patient_{patient_id}_cf_{i}.csv"
                cf_row.to_frame().T.to_csv(cf_path, index=False)
                cf_paths.append(str(cf_path))
        
        logger.debug(f"Saved {len(cf_paths)} CFs for patient {patient_id}, iteration {iteration_num}")
        
        return str(orig_path), cf_paths
    
    def generate_and_save_for_patient(
        self,
        patient_data: pd.DataFrame,
        patient_id: int,
        iteration_num: int,
        output_dir: str
    ) -> Dict:
        """
        Complete workflow: generate and save CFs for one patient
        
        Args:
            patient_data: Patient DataFrame
            patient_id: Patient ID
            iteration_num: Iteration number
            output_dir: Output directory
            
        Returns:
            Dictionary with results summary
        """
        result = {
            'patient_id': patient_id,
            'iteration': iteration_num,
            'success': False,
            'n_cfs_generated': 0,
            'original_path': None,
            'cf_paths': []
        }
        
        try:
            # Generate CFs
            cf_result = self.generate_counterfactuals(patient_data)
            
            if cf_result is not None and cf_result.final_cfs_df is not None:
                # Save CFs
                orig_path, cf_paths = self.save_counterfactuals(
                    cf_result, iteration_num, patient_id, output_dir
                )
                
                result['success'] = True
                result['n_cfs_generated'] = len(cf_paths)
                result['original_path'] = orig_path
                result['cf_paths'] = cf_paths
            else:
                logger.warning(f"No CFs generated for patient {patient_id}")
                
        except Exception as e:
            logger.error(f"Error in generate_and_save for patient {patient_id}: {e}")
        
        return result


if __name__ == "__main__":
    # Example usage
    print("DiCE CF Generator Module")
    print("This module should be imported and used by the pipeline orchestrator")
