# conda-env: mtech-env
"""
Fresh Counterfactual Generation Pipeline - Main Orchestrator

Scalable, concurrent pipeline for generating fresh counterfactuals 100 times
and computing confidence intervals.

Author: PMK
Date: 2026-01-26
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path so package imports work
# regardless of how the script is invoked.
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import logging
import json
import time
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import argparse

# Import pipeline modules
from src.pipeline.dice_cf_generator import DiceCFGenerator
from src.pipeline.scm_analyzer import SCMAnalyzer
from src.pipeline.metrics_calculator import MetricsCalculator
from src.pipeline.ci_computer import CIComputer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('fresh_cf_pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class FreshCFPipeline:
    """
    Fresh Counterfactual Generation Pipeline
    
    Scalable pipeline with concurrent execution support.
    Generates fresh CFs multiple times and computes confidence intervals.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize Pipeline
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or self._default_config()
        self.output_dir = Path(self.config['output']['base_dir'])
        
        # Initialize modules (will be created per-process for concurrency)
        self.dice_generator = None
        self.scm_analyzer = None
        self.metrics_calculator = None
        self.ci_computer = None
        self._cached_patient_data = None
        
        logger.info("Initialized FreshCFPipeline")
    
    def _default_config(self) -> Dict:
        """Default pipeline configuration"""
        return {
            'pipeline': {
                'n_iterations': 100,
                'n_patients': 48,
                'n_workers': 4,  # Number of concurrent workers
                'test_mode': False
            },
            'dice': {
                'model_path': 'model/xgb_pipeline.pkl',
                'data_path': 'data/heart_statlog_cleveland_hungary_final.csv',
                'method': 'genetic',
                'total_cfs': 5,
                'permitted_range': {
                    'trestbps': [100, 120],
                    'chol': [150, None]
                },
                'timeout': 30
            },
            'scm': {
                'n_samples': 1,
                'graph_structure': 'full',
                'intervention_targets': 'both',
            },
            'output': {
                'base_dir': 'fresh_cf_iterations',
                'keep_all_iterations': False
            },
            'ci': {
                'confidence_level': 0.95
            }
        }
    
    def initialize_modules(self):
        """Initialize pipeline modules"""
        # DiCE Generator
        self.dice_generator = DiceCFGenerator(
            model_path=self.config['dice']['model_path'],
            data_path=self.config['dice']['data_path'],
            config=self.config['dice']
        )
        self.dice_generator.load_model_and_data()
        self.dice_generator.setup_dice_explainer()
        
        # SCM Analyzer
        self.scm_analyzer = SCMAnalyzer(config=self.config['scm'])
        self.scm_analyzer.initialize_analyzer()
        
        # Metrics Calculator
        self.metrics_calculator = MetricsCalculator()
        
        # CI Computer
        self.ci_computer = CIComputer(
            confidence_level=self.config['ci']['confidence_level']
        )
        
        logger.info("All modules initialized")
    
    def load_patient_data(self) -> pd.DataFrame:
        """Load patient data for CF generation (cached after first call).

        Applies the same cleaning as train_model.py (DataLoader) so that
        patient feature values stay within the range DiCE was trained on.
        """
        if self._cached_patient_data is not None:
            logger.info(f"Using cached patient data: {len(self._cached_patient_data)} patients")
            return self._cached_patient_data

        from src.utils.dataLoader import DataLoader
        loader = DataLoader(self.config['dice']['data_path'])
        df = loader.load_data()
        if df is not None:
            df = loader.remove_outliers_iqr(df)

        # Filter high-risk patients (target=1)
        high_risk = df[df['target'] == 1].copy()

        n_patients = self.config['pipeline']['n_patients']
        if len(high_risk) > n_patients:
            high_risk = high_risk.head(n_patients)

        self._cached_patient_data = high_risk
        logger.info(f"Loaded {len(high_risk)} high-risk patients")
        return high_risk
    
    def run_single_iteration(self, iteration_num: int, patients_df: pd.DataFrame) -> Dict:
        """
        Run a single iteration: Generate CFs → SCM Analysis → Compute Metrics
        
        This function is designed to be run in a separate process for concurrency.
        
        Args:
            iteration_num: Iteration number
            patients_df: DataFrame with patient data
            
        Returns:
            Dictionary with iteration metrics
        """
        try:
            logger.info(f"Starting iteration {iteration_num}")
            
            # Initialize modules in this process
            dice_gen = DiceCFGenerator(
                model_path=self.config['dice']['model_path'],
                data_path=self.config['dice']['data_path'],
                config=self.config['dice']
            )
            dice_gen.load_model_and_data()
            dice_gen.setup_dice_explainer()
            
            # Step 1: Generate CFs for all patients
            iteration_dir = self.output_dir / f"iteration_{iteration_num:03d}"
            
            cf_results = []
            for idx, (_, patient_row) in enumerate(patients_df.iterrows()):
                patient_data = patient_row.to_frame().T
                # Restore original dtypes (to_frame().T casts int→float due to mixed Series)
                for col in patient_data.columns:
                    if col in patients_df.columns:
                        patient_data[col] = patient_data[col].astype(patients_df[col].dtype)
                # Drop target column — DiCE expects features only
                patient_data = patient_data.drop(columns=['target'], errors='ignore')
                result = dice_gen.generate_and_save_for_patient(
                    patient_data,
                    patient_id=idx,
                    iteration_num=iteration_num,
                    output_dir=str(self.output_dir)
                )
                cf_results.append(result)
            
            # Step 2: Run SCM Analysis
            scm_analyzer = SCMAnalyzer(config=self.config['scm'])
            scm_analyzer.initialize_analyzer()
            
            successful_cfs = scm_analyzer.analyze_iteration(
                iteration_dir=str(iteration_dir)
            )
            
            # Step 3: Compute Metrics
            metrics_calc = MetricsCalculator()
            metrics = metrics_calc.compute_all_metrics(successful_cfs)
            metrics['iteration'] = iteration_num
            
            # Save iteration metrics
            iteration_dir.mkdir(parents=True, exist_ok=True)
            metrics_file = iteration_dir / "metrics.json"
            with open(metrics_file, 'w') as f:
                json.dump(metrics, f, indent=2)
            
            logger.info(f"Completed iteration {iteration_num}: {metrics['total_successful_cfs']} successful CFs")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error in iteration {iteration_num}: {e}")
            return {'iteration': iteration_num, 'total_successful_cfs': 0, 'error': str(e)}
    
    def run_concurrent_pipeline(
        self,
        n_iterations: Optional[int] = None,
        n_workers: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Run full pipeline with concurrent execution
        
        Args:
            n_iterations: Number of iterations (uses config if None)
            n_workers: Number of concurrent workers (uses config if None)
            
        Returns:
            DataFrame with aggregated results
        """
        n_iterations = n_iterations or self.config['pipeline']['n_iterations']
        n_workers = n_workers or self.config['pipeline']['n_workers']
        
        logger.info(f"Starting concurrent pipeline: {n_iterations} iterations, {n_workers} workers")
        
        # Load patient data
        patients_df = self.load_patient_data()
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Run iterations concurrently
        all_metrics = []
        
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            # Submit all iterations
            futures = {
                executor.submit(self.run_single_iteration, i, patients_df): i
                for i in range(n_iterations)
            }
            
            # Collect results with progress bar
            with tqdm(total=n_iterations, desc="Running iterations") as pbar:
                for future in as_completed(futures):
                    iteration_num = futures[future]
                    try:
                        metrics = future.result()
                        all_metrics.append(metrics)
                        pbar.update(1)
                        pbar.set_postfix({'successful_cfs': metrics.get('total_successful_cfs', 0)})
                    except Exception as e:
                        logger.error(f"Iteration {iteration_num} failed: {e}")
                        pbar.update(1)
        
        logger.info(f"Completed all {n_iterations} iterations")
        
        return pd.DataFrame(all_metrics)
    
    def compute_and_save_results(self, aggregated_results: pd.DataFrame) -> None:
        """
        Compute CIs and save final results
        
        Args:
            aggregated_results: DataFrame with all iteration metrics
        """
        if self.ci_computer is None:
            self.ci_computer = CIComputer(
                confidence_level=self.config['ci']['confidence_level']
            )
        
        # Compute CIs
        ci_results = self.ci_computer.compute_confidence_intervals(aggregated_results)
        
        # Save results
        results_dir = self.output_dir / "aggregated_results"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Save all metrics
        all_metrics_path = results_dir / "all_iteration_metrics.csv"
        aggregated_results.to_csv(all_metrics_path, index=False)
        logger.info(f"Saved all iteration metrics to {all_metrics_path}")
        
        # Save CI results and report
        self.ci_computer.save_results(ci_results, str(results_dir))
        
        logger.info("Results saved successfully")
    
    def run_full_pipeline(
        self,
        n_iterations: Optional[int] = None,
        n_workers: Optional[int] = None
    ) -> None:
        """
        Run complete pipeline: Generate CFs → Analyze → Compute CIs
        
        Args:
            n_iterations: Number of iterations
            n_workers: Number of concurrent workers
        """
        start_time = time.time()
        
        logger.info("="*80)
        logger.info("FRESH COUNTERFACTUAL GENERATION PIPELINE")
        logger.info("="*80)
        
        # Run concurrent iterations
        aggregated_results = self.run_concurrent_pipeline(n_iterations, n_workers)
        
        # Compute and save results
        self.compute_and_save_results(aggregated_results)
        
        elapsed_time = time.time() - start_time
        logger.info("="*80)
        logger.info(f"Pipeline completed in {elapsed_time/60:.2f} minutes")
        logger.info(f"Results saved to: {self.output_dir / 'aggregated_results'}")
        logger.info("="*80)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Fresh CF Generation Pipeline')
    parser.add_argument('--n_iterations', type=int, default=100, help='Number of iterations')
    parser.add_argument('--n_patients', type=int, default=48, help='Number of patients')
    parser.add_argument('--n_workers', type=int, default=4, help='Number of concurrent workers')
    parser.add_argument('--test_mode', action='store_true', help='Run in test mode (5 patients, 5 iterations)')
    parser.add_argument('--sensitivity', action='store_true', help='Run sensitivity analysis')
    parser.add_argument('--sensitivity_iterations', type=int, default=10, help='Iterations per sensitivity variant')
    parser.add_argument('--sensitivity_patients', type=int, default=10, help='Patients per sensitivity variant')
    parser.add_argument('--sensitivity_params', nargs='*', default=None,
                        help='Parameters to vary (default: all). Options: total_cfs, trestbps_range, chol_lower, confidence_level, graph_structure, intervention_targets, n_samples')

    args = parser.parse_args()
    
    # Create config
    config = {
        'pipeline': {
            'n_iterations': 5 if args.test_mode else args.n_iterations,
            'n_patients': 5 if args.test_mode else args.n_patients,
            'n_workers': 2 if args.test_mode else args.n_workers
        },
        'dice': {
            'model_path': 'model/xgb_pipeline.pkl',
            'data_path': 'data/heart_statlog_cleveland_hungary_final.csv',
            'method': 'genetic',
            'total_cfs': 5,
            'permitted_range': {
                'trestbps': [100, 120],
                'chol': [150, None]
            },
            'timeout': 30
        },
        'scm': {
            'n_samples': 1,
            'graph_structure': 'full',
            'intervention_targets': 'both',
        },
        'output': {
            'base_dir': 'fresh_cf_iterations_test' if args.test_mode else 'fresh_cf_iterations',
            'keep_all_iterations': False
        },
        'ci': {
            'confidence_level': 0.95
        }
    }
    
    if args.sensitivity:
        from src.pipeline.sensitivity_analyzer import SensitivityAnalyzer
        analyzer = SensitivityAnalyzer(
            baseline_config=config,
            n_iterations=args.sensitivity_iterations,
            n_patients=args.sensitivity_patients,
            n_workers=config['pipeline']['n_workers'],
        )
        analyzer.run_sensitivity_analysis(parameters=args.sensitivity_params)
    else:
        pipeline = FreshCFPipeline(config=config)
        pipeline.run_full_pipeline()


if __name__ == "__main__":
    main()
