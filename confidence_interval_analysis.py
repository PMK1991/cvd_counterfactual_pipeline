# conda-env: mtech-env
"""
Confidence Interval Analysis for Counterfactual Metrics
Runs the counterfactual analysis 100 times and computes confidence intervals
"""

import pandas as pd
import numpy as np
from scipy import stats
import tempfile
import warnings
from tqdm import tqdm
import time
from dowhy import gcm
from dowhy.gcm.util.general import set_random_seed

# Import your existing analyzer
from counterfactualAnalyzer import CounterfactualAnalyzer


class ConfidenceIntervalAnalyzer:
    """
    Runs counterfactual analysis multiple times and computes confidence intervals
    """
    
    def __init__(self, causal_model, n_runs=100, confidence_level=0.95):
        """
        Initialize the CI analyzer
        
        Args:
            causal_model: Fitted causal model
            n_runs: Number of runs for bootstrap (default: 100)
            confidence_level: Confidence level (default: 0.95)
        """
        self.causal_model = causal_model
        self.n_runs = n_runs
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level
        
        # Storage for metrics across runs
        self.metrics_per_run = []
        
    def run_single_iteration(self, run_id, seed):
        """
        Run a single iteration of counterfactual analysis
        
        Args:
            run_id: Iteration number
            seed: Random seed for reproducibility
            
        Returns:
            dict: Metrics from this run
        """
        # Set random seed for this run
        set_random_seed(seed)
        np.random.seed(seed)
        
        # Create analyzer in a temporary directory that auto-cleans on exit
        with tempfile.TemporaryDirectory(prefix=f"temp_run_{run_id}_") as output_dir:
            analyzer = CounterfactualAnalyzer(
                self.causal_model,
                original_dir="original",
                cf_dir="counterfactuals",
                output_dir=output_dir
            )

            # Process all instances (suppress output)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                analyzer.process_all_instances(instance_range=range(48), show_progress=False)

            # Extract metrics before temp directory is cleaned up
            metrics = self._extract_metrics(analyzer)

        return metrics
    
    def _extract_metrics(self, analyzer):
        """
        Extract key metrics from analyzer results
        
        Returns:
            dict: Dictionary of metrics
        """
        if not analyzer.all_counterfactuals:
            return None
            
        df = pd.DataFrame(analyzer.all_counterfactuals)
        worked_df = df[df['target_changed'] == 1]
        
        metrics = {
            # Overall statistics
            'total_cf': analyzer.stats['total_cf'],
            'working_cf': analyzer.stats['working_cf'],
            'success_rate': analyzer.stats['working_cf'] / analyzer.stats['total_cf'] if analyzer.stats['total_cf'] > 0 else 0,
            
            # Distance metrics - all counterfactuals
            'euclidean_mean_all': df['euclidean_distance'].mean(),
            'euclidean_std_all': df['euclidean_distance'].std(),
            'euclidean_median_all': df['euclidean_distance'].median(),
            'manhattan_mean_all': df['manhattan_distance'].mean(),
            'manhattan_std_all': df['manhattan_distance'].std(),
            'manhattan_median_all': df['manhattan_distance'].median(),
        }
        
        # Distance metrics - working counterfactuals only
        if not worked_df.empty:
            metrics.update({
                'euclidean_mean_working': worked_df['euclidean_distance'].mean(),
                'euclidean_std_working': worked_df['euclidean_distance'].std(),
                'euclidean_median_working': worked_df['euclidean_distance'].median(),
                'manhattan_mean_working': worked_df['manhattan_distance'].mean(),
                'manhattan_std_working': worked_df['manhattan_distance'].std(),
                'manhattan_median_working': worked_df['manhattan_distance'].median(),
            })
            
            # Feature changes for working counterfactuals
            change_cols = [col for col in worked_df.columns if col.startswith('change_')]
            for col in change_cols:
                feature_name = col.replace('change_', '')
                metrics[f'change_{feature_name}_mean'] = worked_df[col].mean()
                metrics[f'change_{feature_name}_std'] = worked_df[col].std()
        
        return metrics
    
    def run_bootstrap_analysis(self):
        """
        Run the analysis n_runs times with different random seeds
        """
        print(f"Running {self.n_runs} iterations of counterfactual analysis...")
        print(f"Confidence level: {self.confidence_level * 100}%\n")
        
        start_time = time.time()
        
        for run_id in tqdm(range(self.n_runs), desc="Bootstrap iterations"):
            # Use different seed for each run
            seed = 42 + run_id
            
            try:
                metrics = self.run_single_iteration(run_id, seed)
                if metrics:
                    metrics['run_id'] = run_id
                    self.metrics_per_run.append(metrics)
            except Exception as e:
                print(f"\nWarning: Run {run_id} failed with error: {str(e)}")
                continue
        
        elapsed_time = time.time() - start_time
        print(f"\nCompleted {len(self.metrics_per_run)} successful runs in {elapsed_time:.2f} seconds")
        
    def calculate_confidence_intervals(self):
        """
        Calculate confidence intervals for all metrics
        
        Returns:
            pd.DataFrame: DataFrame with mean, std, CI lower, CI upper for each metric
        """
        if not self.metrics_per_run:
            raise ValueError("No metrics available. Run bootstrap_analysis first.")
        
        # Convert to DataFrame
        metrics_df = pd.DataFrame(self.metrics_per_run)
        
        # Calculate statistics for each metric
        results = []
        
        for col in metrics_df.columns:
            if col == 'run_id':
                continue
                
            values = metrics_df[col].dropna()
            
            if len(values) == 0:
                continue
            
            # Calculate mean and std
            mean_val = values.mean()
            std_val = values.std()
            
            # Calculate confidence interval using t-distribution
            ci = stats.t.interval(
                self.confidence_level,
                len(values) - 1,
                loc=mean_val,
                scale=stats.sem(values)
            )
            
            # Also calculate percentile-based CI (bootstrap CI)
            lower_percentile = (1 - self.confidence_level) / 2 * 100
            upper_percentile = (1 + self.confidence_level) / 2 * 100
            
            ci_percentile = (
                np.percentile(values, lower_percentile),
                np.percentile(values, upper_percentile)
            )
            
            results.append({
                'metric': col,
                'mean': mean_val,
                'std': std_val,
                'median': values.median(),
                'min': values.min(),
                'max': values.max(),
                'ci_lower_t': ci[0],
                'ci_upper_t': ci[1],
                'ci_width_t': ci[1] - ci[0],
                'ci_lower_percentile': ci_percentile[0],
                'ci_upper_percentile': ci_percentile[1],
                'ci_width_percentile': ci_percentile[1] - ci_percentile[0],
                'n_runs': len(values)
            })
        
        results_df = pd.DataFrame(results)
        return results_df
    
    def save_results(self, output_file='reports/confidence_intervals.csv'):
        """
        Save confidence interval results to CSV
        
        Args:
            output_file: Output filename
        """
        ci_results = self.calculate_confidence_intervals()
        ci_results.to_csv(output_file, index=False)
        print(f"\nConfidence intervals saved to: {output_file}")
        return ci_results
    
    def print_summary(self):
        """
        Print a summary of key metrics with confidence intervals
        """
        ci_results = self.calculate_confidence_intervals()
        
        print("\n" + "="*80)
        print("CONFIDENCE INTERVAL ANALYSIS SUMMARY")
        print("="*80)
        print(f"Number of runs: {self.n_runs}")
        print(f"Confidence level: {self.confidence_level * 100}%")
        print("="*80)
        
        # Key metrics to display
        key_metrics = [
            'total_cf',
            'working_cf', 
            'success_rate',
            'euclidean_mean_working',
            'manhattan_mean_working',
            'euclidean_median_working',
            'manhattan_median_working'
        ]
        
        print("\nKEY METRICS WITH CONFIDENCE INTERVALS:")
        print("-" * 80)
        
        for metric in key_metrics:
            row = ci_results[ci_results['metric'] == metric]
            if not row.empty:
                row = row.iloc[0]
                print(f"\n{metric}:")
                print(f"  Mean: {row['mean']:.4f}")
                print(f"  Std:  {row['std']:.4f}")
                print(f"  95% CI (t-dist):      [{row['ci_lower_t']:.4f}, {row['ci_upper_t']:.4f}]")
                print(f"  95% CI (percentile):  [{row['ci_lower_percentile']:.4f}, {row['ci_upper_percentile']:.4f}]")
        
        print("\n" + "="*80)
        
        # Feature changes
        print("\nFEATURE CHANGES (Working Counterfactuals):")
        print("-" * 80)
        
        change_metrics = ci_results[ci_results['metric'].str.startswith('change_') & 
                                   ci_results['metric'].str.endswith('_mean')]
        
        for _, row in change_metrics.iterrows():
            feature = row['metric'].replace('change_', '').replace('_mean', '')
            print(f"\n{feature}:")
            print(f"  Mean change: {row['mean']:.4f}")
            print(f"  95% CI: [{row['ci_lower_percentile']:.4f}, {row['ci_upper_percentile']:.4f}]")
        
        print("\n" + "="*80)


def main():
    """
    Main function to run confidence interval analysis
    """
    # This is a placeholder - you need to provide your fitted causal model
    # Example usage:
    
    print("Confidence Interval Analysis for CVD Counterfactuals")
    print("=" * 80)
    print("\nNOTE: This script requires a fitted causal model.")
    print("Please integrate this with your existing pipeline.\n")
    print("Example usage:")
    print("""
    # After fitting your causal model:
    from confidence_interval_analysis import ConfidenceIntervalAnalyzer
    
    ci_analyzer = ConfidenceIntervalAnalyzer(
        causal_model=your_fitted_model,
        n_runs=100,
        confidence_level=0.95
    )
    
    # Run bootstrap analysis
    ci_analyzer.run_bootstrap_analysis()
    
    # Save and display results
    ci_results = ci_analyzer.save_results('reports/confidence_intervals.csv')
    ci_analyzer.print_summary()
    """)


if __name__ == "__main__":
    main()
