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
from typing import Dict, Optional
import logging
import json
import pickle
import shutil
import time
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
        self._cohort_counts = []

        logger.info("Initialized FreshCFPipeline")

    @staticmethod
    def _default_config() -> Dict:
        """Default pipeline configuration"""
        return {
            'pipeline': {
                'n_iterations': 100,
                'n_patients': 48,
                'n_workers': 4,  # Number of concurrent workers
                'test_mode': False,
                'run_patient_bootstrap': False,
                'bootstrap_iterations': 1000,
            },
            'dice': {
                'model_path': 'model/xgb_pipeline.pkl',
                'data_path': 'data/heart_statlog_cleveland_hungary_final.csv',
                'method': 'genetic',
                'total_cfs': 5,
                'permitted_range': {
                    'trestbps': [100, 120],
                    'chol': [150, 200]
                },
                'timeout': 45,
                'features_to_vary': None,
                'search_params': {
                    'maxiterations': 500,
                    'thresh': 0.01,
                    'proximity_weight': 0.5,
                    'sparsity_weight': 1.0,
                    'diversity_weight': 5.0,
                    'stopping_threshold': 0.5,
                    'posthoc_sparsity_algorithm': 'binary',
                    'posthoc_sparsity_param': 0.1,
                },
            },
            'scm': {
                'n_samples': 1000,
                'graph_structure': 'full',
                'intervention_targets': 'chol_only',
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

        from src.utils.dataLoader import (
            DataLoader,
            MODEL_RANDOM_STATE,
            MODEL_TEST_SIZE,
        )
        if MODEL_TEST_SIZE != 0.2 or MODEL_RANDOM_STATE != 42:
            raise RuntimeError(
                f"Pipeline cohort depends on train_model.py's split "
                f"(test_size=0.2, random_state=42); got {MODEL_TEST_SIZE=}, {MODEL_RANDOM_STATE=}"
            )

        loader = DataLoader(self.config['dice']['data_path'])
        high_risk = loader.test_set_high_risk(
            test_size=MODEL_TEST_SIZE,
            random_state=MODEL_RANDOM_STATE,
        )
        if high_risk is None:
            raise ValueError("Unable to load test-set high-risk cohort")

        rows_in = len(high_risk)
        with open(self.config['dice']['model_path'], 'rb') as f:
            model = pickle.load(f)
        predictions = model.predict(high_risk.drop(columns=['target'], errors='ignore'))
        high_risk = high_risk[predictions == 1].copy()
        loader.record_step('test_set_true_positives', rows_in, len(high_risk))

        n_patients = self.config['pipeline']['n_patients']
        if self.config['pipeline'].get('test_mode', False) and len(high_risk) > n_patients:
            rows_in = len(high_risk)
            high_risk = high_risk.head(n_patients)
            loader.record_step('test_mode_debug_cap', rows_in, len(high_risk))

        self._cohort_counts = loader.get_step_counts()

        self._cached_patient_data = high_risk
        logger.info(f"Loaded {len(high_risk)} test-set true-positive high-risk patients")
        return high_risk

    def _save_successful_cfs(self, successful_cfs: pd.DataFrame, iteration_dir: Path) -> str:
        """Persist successful CFs for downstream bootstrap and audit steps."""
        out_dir = iteration_dir / "successful"
        out_dir.mkdir(parents=True, exist_ok=True)
        output_file = out_dir / "successful_counterfactuals.csv"
        successful_cfs.to_csv(output_file, index=False)
        return str(output_file)

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
            if iteration_dir.exists():
                shutil.rmtree(iteration_dir)

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

            total_generated_cfs = sum(r['n_cfs_generated'] for r in cf_results)
            total_requested_cfs = len(patients_df) * self.config['dice']['total_cfs']

            # Step 2: SCM validation
            scm_analyzer = SCMAnalyzer(config=self.config['scm'])
            scm_analyzer.initialize_analyzer()

            successful_cfs = scm_analyzer.analyze_iteration(
                iteration_dir=str(iteration_dir)
            )

            self._save_successful_cfs(successful_cfs, iteration_dir)

            # Step 3: Compute Metrics
            metrics_calc = MetricsCalculator()
            metrics = metrics_calc.compute_all_metrics(successful_cfs)
            metrics['iteration'] = iteration_num
            metrics['total_patients'] = len(patients_df)
            metrics['total_requested_cfs'] = total_requested_cfs
            metrics['total_generated_cfs'] = total_generated_cfs
            metrics['target_flip_rate_pct'] = (
                metrics['total_successful_cfs'] / total_generated_cfs * 100
                if total_generated_cfs else 0.0
            )

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

        # Compute algorithmic-stability intervals across independent iterations.
        ci_results = self.ci_computer.compute_confidence_intervals(aggregated_results)
        ci_results['interval_type'] = 'algorithmic_stability'

        from src.pipeline.ev_calculator import EVCalculator
        ev_calculator = EVCalculator(confidence_level=self.config['ci']['confidence_level'])
        ev_payload = ev_calculator.compute(aggregated_results, ci_results)
        ev_row = ev_calculator.to_ci_row(ev_payload)
        if ev_row:
            ci_results = pd.concat([ci_results, pd.DataFrame([ev_row])], ignore_index=True)

        # Save results
        results_dir = self.output_dir / "aggregated_results"
        results_dir.mkdir(parents=True, exist_ok=True)

        # Save all metrics
        all_metrics_path = results_dir / "all_iteration_metrics.csv"
        aggregated_results.to_csv(all_metrics_path, index=False)
        logger.info(f"Saved all iteration metrics to {all_metrics_path}")

        # Save cohort accounting
        if self._cohort_counts:
            cohort_counts_path = results_dir / "cohort_counts.json"
            with open(cohort_counts_path, 'w') as f:
                json.dump(self._cohort_counts, f, indent=2)
            logger.info(f"Saved cohort counts to {cohort_counts_path}")

        # Save CI results and report
        self.ci_computer.save_results(ci_results, str(results_dir))

        ev_calculator.save(ev_payload, results_dir / "evalue.json")

        if self.config['pipeline'].get('run_patient_bootstrap', False):
            from src.pipeline.patient_bootstrap import PatientBootstrap
            bootstrap = PatientBootstrap(
                iterations=self.config['pipeline'].get('bootstrap_iterations', 1000),
                confidence_level=self.config['ci']['confidence_level'],
            )
            bootstrap_results = bootstrap.compute(self.output_dir)
            bootstrap.save(
                bootstrap_results,
                results_dir / "patient_bootstrap_ci.csv",
            )

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


def _deep_update(base: Dict, updates: Dict) -> Dict:
    """Recursively update a config dictionary."""
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_yaml_config(config_path: str = "pipeline_config.yaml") -> Dict:
    """Load pipeline YAML config when available."""
    path = Path(config_path)
    if not path.exists():
        return {}
    try:
        import yaml
    except ImportError:
        logger.warning("PyYAML is not installed; using built-in defaults")
        return {}
    with open(path, 'r') as f:
        return yaml.safe_load(f) or {}


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Fresh CF Generation Pipeline')
    parser.add_argument('--n_iterations', type=int, default=None, help='Number of iterations (overrides YAML)')
    parser.add_argument('--n_patients', type=int, default=None, help='Patient cap (used only in --test_mode)')
    parser.add_argument('--n_workers', type=int, default=None, help='Number of concurrent workers (overrides YAML)')
    parser.add_argument('--test_mode', action='store_true', help='Run in test mode (5 patients, 5 iterations)')
    parser.add_argument('--run_patient_bootstrap', action='store_true', help='Run patient-level bootstrap after the pipeline')
    parser.add_argument('--bootstrap_only', action='store_true', help='Skip the pipeline; only bootstrap from cached successful CFs in output.base_dir')
    parser.add_argument('--bootstrap_iterations', type=int, default=None, help='Patient bootstrap replicates')
    parser.add_argument('--sensitivity', action='store_true', help='Run sensitivity analysis')
    parser.add_argument('--sensitivity_iterations', type=int, default=10, help='Iterations per sensitivity variant')
    parser.add_argument('--sensitivity_patients', type=int, default=10, help='Patients per sensitivity variant')
    parser.add_argument('--sensitivity_params', nargs='*', default=None,
                        help='Parameters to vary (default: all). Options: total_cfs, trestbps_range, chol_lower, confidence_level, graph_structure, intervention_targets, n_samples')

    args = parser.parse_args()

    config = FreshCFPipeline._default_config()
    _deep_update(config, load_yaml_config())
    if args.test_mode:
        config['pipeline']['n_iterations'] = 5
        config['pipeline']['n_patients'] = 5
        config['pipeline']['n_workers'] = 2
    else:
        if args.n_iterations is not None:
            config['pipeline']['n_iterations'] = args.n_iterations
        if args.n_patients is not None:
            config['pipeline']['n_patients'] = args.n_patients
        if args.n_workers is not None:
            config['pipeline']['n_workers'] = args.n_workers
    config['pipeline']['test_mode'] = args.test_mode
    if args.run_patient_bootstrap or args.bootstrap_only:
        config['pipeline']['run_patient_bootstrap'] = True
    if args.bootstrap_iterations is not None:
        config['pipeline']['bootstrap_iterations'] = args.bootstrap_iterations

    if args.test_mode:
        config['output']['base_dir'] = 'fresh_cf_iterations_test'

    if args.sensitivity:
        from src.pipeline.sensitivity_analyzer import SensitivityAnalyzer
        analyzer = SensitivityAnalyzer(
            baseline_config=config,
            n_iterations=args.sensitivity_iterations,
            n_patients=args.sensitivity_patients,
            n_workers=config['pipeline']['n_workers'],
        )
        analyzer.run_sensitivity_analysis(parameters=args.sensitivity_params)
    elif args.bootstrap_only:
        from src.pipeline.patient_bootstrap import PatientBootstrap
        output_dir = Path(config['output']['base_dir'])
        bootstrap = PatientBootstrap(
            iterations=config['pipeline'].get('bootstrap_iterations', 1000),
            confidence_level=config['ci']['confidence_level'],
        )
        results = bootstrap.compute(output_dir)
        if results.empty:
            logger.warning(
                f"No cached successful CFs found under {output_dir}; "
                "run the pipeline first to populate iteration_*/successful/."
            )
        else:
            bootstrap.save(
                results,
                output_dir / "aggregated_results" / "patient_bootstrap_ci.csv",
            )
    else:
        pipeline = FreshCFPipeline(config=config)
        pipeline.run_full_pipeline()


if __name__ == "__main__":
    main()
