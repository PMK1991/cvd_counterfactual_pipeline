# conda-env: mtech-env
"""
Confidence Interval Analysis for Downstream Diagnostic Metrics
Computes 95% CIs for diagnostic changes in successful counterfactuals
"""

import pandas as pd
import numpy as np
from scipy import stats
from tqdm import tqdm
import warnings


class DiagnosticMetricsCI:
    """
    Calculate confidence intervals for downstream diagnostic metrics
    in successful counterfactuals using bootstrap resampling
    """
    
    def __init__(self, n_bootstrap=100, confidence_level=0.95, random_seed=42):
        """
        Initialize the diagnostic metrics CI analyzer
        
        Args:
            n_bootstrap: Number of bootstrap samples (default: 100)
            confidence_level: Confidence level (default: 0.95)
            random_seed: Random seed for reproducibility
        """
        self.n_bootstrap = n_bootstrap
        self.confidence_level = confidence_level
        self.random_seed = random_seed
        self.bootstrap_results = []
        
    def load_successful_counterfactuals(self, filepath='worked/working_counterfactuals_with_distances.csv'):
        """
        Load successful counterfactuals data
        
        Args:
            filepath: Path to working counterfactuals CSV
            
        Returns:
            pd.DataFrame: Successful counterfactuals
        """
        df = pd.read_csv(filepath)
        print(f"Loaded {len(df)} successful counterfactuals")
        return df
    
    def calculate_diagnostic_metrics(self, df):
        """
        Calculate all diagnostic metrics for a given sample
        
        Args:
            df: DataFrame of successful counterfactuals
            
        Returns:
            dict: Dictionary of diagnostic metrics
        """
        n_total = len(df)
        
        metrics = {
            # Total successful counterfactuals
            'total_successful_cfs': n_total,
            
            # Resting Blood Pressure (trestbps)
            'trestbps_improved_count': (df['cf_trestbps'] < df['orig_trestbps']).sum(),
            'trestbps_improved_pct': (df['cf_trestbps'] < df['orig_trestbps']).sum() / n_total * 100,
            'trestbps_worsened_count': (df['cf_trestbps'] > df['orig_trestbps']).sum(),
            'trestbps_worsened_pct': (df['cf_trestbps'] > df['orig_trestbps']).sum() / n_total * 100,
            'trestbps_no_change_count': (df['cf_trestbps'] == df['orig_trestbps']).sum(),
            'trestbps_no_change_pct': (df['cf_trestbps'] == df['orig_trestbps']).sum() / n_total * 100,
            'mean_orig_trestbps': df['orig_trestbps'].mean(),
            'mean_cf_trestbps': df['cf_trestbps'].mean(),
            'mean_diff_trestbps': (df['cf_trestbps'] - df['orig_trestbps']).mean(),
            
            # Chest Pain (cp) changes
            'cp_improved_count': (df['cf_cp'] < df['orig_cp']).sum(),
            'cp_improved_pct': (df['cf_cp'] < df['orig_cp']).sum() / n_total * 100,
            'cp_worsened_count': (df['cf_cp'] > df['orig_cp']).sum(),
            'cp_worsened_pct': (df['cf_cp'] > df['orig_cp']).sum() / n_total * 100,
            'cp_no_change_count': (df['cf_cp'] == df['orig_cp']).sum(),
            'cp_no_change_pct': (df['cf_cp'] == df['orig_cp']).sum() / n_total * 100,
            'cp_changed_count': (df['orig_cp'] != df['cf_cp']).sum(),
            'cp_changed_pct': (df['orig_cp'] != df['cf_cp']).sum() / n_total * 100,
            
            # Exercise-Induced Angina (exang) changes
            'exang_improved_count': ((df['orig_exang'] == 1) & (df['cf_exang'] == 0)).sum(),
            'exang_improved_pct': ((df['orig_exang'] == 1) & (df['cf_exang'] == 0)).sum() / n_total * 100,
            'exang_worsened_count': ((df['orig_exang'] == 0) & (df['cf_exang'] == 1)).sum(),
            'exang_worsened_pct': ((df['orig_exang'] == 0) & (df['cf_exang'] == 1)).sum() / n_total * 100,
            'exang_no_change_count': (df['orig_exang'] == df['cf_exang']).sum(),
            'exang_no_change_pct': (df['orig_exang'] == df['cf_exang']).sum() / n_total * 100,
            # Legacy names for backward compatibility
            'exang_1_to_0_count': ((df['orig_exang'] == 1) & (df['cf_exang'] == 0)).sum(),
            'exang_1_to_0_pct': ((df['orig_exang'] == 1) & (df['cf_exang'] == 0)).sum() / n_total * 100,
            'exang_0_to_1_count': ((df['orig_exang'] == 0) & (df['cf_exang'] == 1)).sum(),
            'exang_0_to_1_pct': ((df['orig_exang'] == 0) & (df['cf_exang'] == 1)).sum() / n_total * 100,
            
            # ST Depression (oldpeak)
            'oldpeak_improved_count': (df['cf_oldpeak'] < df['orig_oldpeak']).sum(),
            'oldpeak_improved_pct': (df['cf_oldpeak'] < df['orig_oldpeak']).sum() / n_total * 100,
            'oldpeak_worsened_count': (df['cf_oldpeak'] > df['orig_oldpeak']).sum(),
            'oldpeak_worsened_pct': (df['cf_oldpeak'] > df['orig_oldpeak']).sum() / n_total * 100,
            'oldpeak_no_change_count': (df['cf_oldpeak'] == df['orig_oldpeak']).sum(),
            'oldpeak_no_change_pct': (df['cf_oldpeak'] == df['orig_oldpeak']).sum() / n_total * 100,
            # Legacy names
            'oldpeak_reduced_count': (df['cf_oldpeak'] < df['orig_oldpeak']).sum(),
            'oldpeak_reduced_pct': (df['cf_oldpeak'] < df['orig_oldpeak']).sum() / n_total * 100,
            'mean_orig_oldpeak': df['orig_oldpeak'].mean(),
            'mean_cf_oldpeak': df['cf_oldpeak'].mean(),
            'mean_diff_oldpeak': (df['cf_oldpeak'] - df['orig_oldpeak']).mean(),
            'median_orig_oldpeak': df['orig_oldpeak'].median(),
            'median_cf_oldpeak': df['cf_oldpeak'].median(),
            
            # Maximum Heart Rate (thalach)
            'thalach_improved_count': (df['cf_thalach'] > df['orig_thalach']).sum(),
            'thalach_improved_pct': (df['cf_thalach'] > df['orig_thalach']).sum() / n_total * 100,
            'thalach_worsened_count': (df['cf_thalach'] < df['orig_thalach']).sum(),
            'thalach_worsened_pct': (df['cf_thalach'] < df['orig_thalach']).sum() / n_total * 100,
            'thalach_no_change_count': (df['cf_thalach'] == df['orig_thalach']).sum(),
            'thalach_no_change_pct': (df['cf_thalach'] == df['orig_thalach']).sum() / n_total * 100,
            # Legacy names
            'thalach_increased_count': (df['cf_thalach'] > df['orig_thalach']).sum(),
            'thalach_increased_pct': (df['cf_thalach'] > df['orig_thalach']).sum() / n_total * 100,
            'mean_orig_thalach': df['orig_thalach'].mean(),
            'mean_cf_thalach': df['cf_thalach'].mean(),
            'mean_diff_thalach': (df['cf_thalach'] - df['orig_thalach']).mean(),
            'median_orig_thalach': df['orig_thalach'].median(),
            'median_cf_thalach': df['cf_thalach'].median(),
            
            # ST Slope (slope)
            'slope_improved_count': (df['cf_slope'] < df['orig_slope']).sum(),
            'slope_improved_pct': (df['cf_slope'] < df['orig_slope']).sum() / n_total * 100,
            'slope_worsened_count': (df['cf_slope'] > df['orig_slope']).sum(),
            'slope_worsened_pct': (df['cf_slope'] > df['orig_slope']).sum() / n_total * 100,
            'slope_no_change_count': (df['cf_slope'] == df['orig_slope']).sum(),
            'slope_no_change_pct': (df['cf_slope'] == df['orig_slope']).sum() / n_total * 100,
            # Legacy names
            'slope_decreased_count': (df['cf_slope'] < df['orig_slope']).sum(),
            'slope_decreased_pct': (df['cf_slope'] < df['orig_slope']).sum() / n_total * 100,
            
            # Resting ECG (restecg)
            'restecg_no_change_count': (df['orig_restecg'] == df['cf_restecg']).sum(),
            'restecg_no_change_pct': (df['orig_restecg'] == df['cf_restecg']).sum() / n_total * 100,
            'restecg_improved_count': (df['cf_restecg'] < df['orig_restecg']).sum(),
            'restecg_improved_pct': (df['cf_restecg'] < df['orig_restecg']).sum() / n_total * 100,
            'restecg_worsened_count': (df['cf_restecg'] > df['orig_restecg']).sum(),
            'restecg_worsened_pct': (df['cf_restecg'] > df['orig_restecg']).sum() / n_total * 100,
        }
        
        return metrics
    
    def bootstrap_resample(self, df):
        """
        Perform bootstrap resampling
        
        Args:
            df: Original DataFrame
            
        Returns:
            pd.DataFrame: Bootstrap sample
        """
        n = len(df)
        indices = np.random.choice(n, size=n, replace=True)
        return df.iloc[indices].reset_index(drop=True)
    
    def run_bootstrap_analysis(self, df):
        """
        Run bootstrap analysis to get confidence intervals
        
        Args:
            df: DataFrame of successful counterfactuals
        """
        print(f"\nRunning bootstrap analysis with {self.n_bootstrap} iterations...")
        print(f"Confidence level: {self.confidence_level * 100}%\n")
        
        np.random.seed(self.random_seed)
        
        # Calculate metrics for original sample
        original_metrics = self.calculate_diagnostic_metrics(df)
        
        # Bootstrap resampling
        self.bootstrap_results = []
        
        for i in tqdm(range(self.n_bootstrap), desc="Bootstrap iterations"):
            # Resample with replacement
            bootstrap_sample = self.bootstrap_resample(df)
            
            # Calculate metrics for this bootstrap sample
            metrics = self.calculate_diagnostic_metrics(bootstrap_sample)
            metrics['iteration'] = i
            self.bootstrap_results.append(metrics)
        
        print(f"\nCompleted {len(self.bootstrap_results)} bootstrap iterations")
        
        return original_metrics
    
    def calculate_confidence_intervals(self):
        """
        Calculate confidence intervals from bootstrap results
        
        Returns:
            pd.DataFrame: Confidence intervals for all metrics
        """
        if not self.bootstrap_results:
            raise ValueError("No bootstrap results available. Run bootstrap_analysis first.")
        
        # Convert to DataFrame
        bootstrap_df = pd.DataFrame(self.bootstrap_results)
        
        # Calculate CIs for each metric
        results = []
        
        for col in bootstrap_df.columns:
            if col == 'iteration':
                continue
            
            values = bootstrap_df[col].dropna()
            
            if len(values) == 0:
                continue
            
            # Calculate statistics
            mean_val = values.mean()
            std_val = values.std()
            median_val = values.median()
            
            # Percentile-based CI (recommended for bootstrap)
            lower_percentile = (1 - self.confidence_level) / 2 * 100
            upper_percentile = (1 + self.confidence_level) / 2 * 100
            
            ci_lower = np.percentile(values, lower_percentile)
            ci_upper = np.percentile(values, upper_percentile)
            
            results.append({
                'metric': col,
                'mean': mean_val,
                'std': std_val,
                'median': median_val,
                'min': values.min(),
                'max': values.max(),
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'ci_width': ci_upper - ci_lower,
                'n_bootstrap': len(values)
            })
        
        return pd.DataFrame(results)
    
    def print_summary(self, original_metrics, ci_results):
        """
        Print formatted summary of results
        
        Args:
            original_metrics: Metrics from original sample
            ci_results: DataFrame with confidence intervals
        """
        print("\n" + "="*80)
        print("DIAGNOSTIC METRICS CONFIDENCE INTERVAL ANALYSIS")
        print("="*80)
        print(f"Bootstrap iterations: {self.n_bootstrap}")
        print(f"Confidence level: {self.confidence_level * 100}%")
        print("="*80)
        
        # Helper function to get CI for a metric
        def get_ci(metric_name):
            row = ci_results[ci_results['metric'] == metric_name]
            if not row.empty:
                return row.iloc[0]
            return None
        
        # Print table header
        print("\n{:<20} {:<15} {:<15} {:<15} {:<20}".format(
            "Metric", "Improve (%)", "Worsen (%)", "No Change (%)", "Δ Mean"
        ))
        print("-" * 85)
        
        # Resting Blood Pressure
        imp = get_ci('trestbps_improved_pct')
        wor = get_ci('trestbps_worsened_pct')
        noc = get_ci('trestbps_no_change_pct')
        diff = get_ci('mean_diff_trestbps')
        if imp is not None and wor is not None and noc is not None and diff is not None:
            print("{:<20} {:<15} {:<15} {:<15} {:<20}".format(
                "Resting BP",
                f"{imp['mean']:.1f}",
                f"{wor['mean']:.1f}",
                f"{noc['mean']:.1f}",
                f"{diff['mean']:.2f} mmHg"
            ))
            print("{:<20} {:<15} {:<15} {:<15}".format(
                "  95% CI",
                f"[{imp['ci_lower']:.1f}, {imp['ci_upper']:.1f}]",
                f"[{wor['ci_lower']:.1f}, {wor['ci_upper']:.1f}]",
                f"[{noc['ci_lower']:.1f}, {noc['ci_upper']:.1f}]"
            ))
        
        # Chest Pain
        imp = get_ci('cp_improved_pct')
        wor = get_ci('cp_worsened_pct')
        noc = get_ci('cp_no_change_pct')
        if imp is not None and wor is not None and noc is not None:
            print("\n{:<20} {:<15} {:<15} {:<15} {:<20}".format(
                "Chest Pain (cp)",
                f"{imp['mean']:.1f}",
                f"{wor['mean']:.1f}",
                f"{noc['mean']:.1f}",
                "—"
            ))
            print("{:<20} {:<15} {:<15} {:<15}".format(
                "  95% CI",
                f"[{imp['ci_lower']:.1f}, {imp['ci_upper']:.1f}]",
                f"[{wor['ci_lower']:.1f}, {wor['ci_upper']:.1f}]",
                f"[{noc['ci_lower']:.1f}, {noc['ci_upper']:.1f}]"
            ))
        
        # Exercise-Induced Angina
        imp = get_ci('exang_improved_pct')
        wor = get_ci('exang_worsened_pct')
        noc = get_ci('exang_no_change_pct')
        if imp is not None and wor is not None and noc is not None:
            print("\n{:<20} {:<15} {:<15} {:<15} {:<20}".format(
                "Exang (1→0/0→1)",
                f"{imp['mean']:.1f}",
                f"{wor['mean']:.1f}",
                f"{noc['mean']:.1f}",
                "—"
            ))
            print("{:<20} {:<15} {:<15} {:<15}".format(
                "  95% CI",
                f"[{imp['ci_lower']:.1f}, {imp['ci_upper']:.1f}]",
                f"[{wor['ci_lower']:.1f}, {wor['ci_upper']:.1f}]",
                f"[{noc['ci_lower']:.1f}, {noc['ci_upper']:.1f}]"
            ))
        
        # ST Depression
        imp = get_ci('oldpeak_improved_pct')
        wor = get_ci('oldpeak_worsened_pct')
        noc = get_ci('oldpeak_no_change_pct')
        diff = get_ci('mean_diff_oldpeak')
        if imp is not None and wor is not None and noc is not None and diff is not None:
            print("\n{:<20} {:<15} {:<15} {:<15} {:<20}".format(
                "ST Depression",
                f"{imp['mean']:.1f}",
                f"{wor['mean']:.1f}",
                f"{noc['mean']:.1f}",
                f"{diff['mean']:.2f} mm"
            ))
            print("{:<20} {:<15} {:<15} {:<15}".format(
                "  95% CI",
                f"[{imp['ci_lower']:.1f}, {imp['ci_upper']:.1f}]",
                f"[{wor['ci_lower']:.1f}, {wor['ci_upper']:.1f}]",
                f"[{noc['ci_lower']:.1f}, {noc['ci_upper']:.1f}]"
            ))
        
        # Max Heart Rate
        imp = get_ci('thalach_improved_pct')
        wor = get_ci('thalach_worsened_pct')
        noc = get_ci('thalach_no_change_pct')
        diff = get_ci('mean_diff_thalach')
        if imp is not None and wor is not None and noc is not None and diff is not None:
            print("\n{:<20} {:<15} {:<15} {:<15} {:<20}".format(
                "Max Heart Rate",
                f"{imp['mean']:.1f}",
                f"{wor['mean']:.1f}",
                f"{noc['mean']:.1f}",
                f"{diff['mean']:.1f} bpm"
            ))
            print("{:<20} {:<15} {:<15} {:<15}".format(
                "  95% CI",
                f"[{imp['ci_lower']:.1f}, {imp['ci_upper']:.1f}]",
                f"[{wor['ci_lower']:.1f}, {wor['ci_upper']:.1f}]",
                f"[{noc['ci_lower']:.1f}, {noc['ci_upper']:.1f}]"
            ))
        
        # ST Slope
        imp = get_ci('slope_improved_pct')
        wor = get_ci('slope_worsened_pct')
        noc = get_ci('slope_no_change_pct')
        if imp is not None and wor is not None and noc is not None:
            print("\n{:<20} {:<15} {:<15} {:<15} {:<20}".format(
                "ST Slope",
                f"{imp['mean']:.1f}",
                f"{wor['mean']:.1f}",
                f"{noc['mean']:.1f}",
                "—"
            ))
            print("{:<20} {:<15} {:<15} {:<15}".format(
                "  95% CI",
                f"[{imp['ci_lower']:.1f}, {imp['ci_upper']:.1f}]",
                f"[{wor['ci_lower']:.1f}, {wor['ci_upper']:.1f}]",
                f"[{noc['ci_lower']:.1f}, {noc['ci_upper']:.1f}]"
            ))
        
        # Resting ECG
        imp = get_ci('restecg_improved_pct')
        wor = get_ci('restecg_worsened_pct')
        noc = get_ci('restecg_no_change_pct')
        if imp is not None and wor is not None and noc is not None:
            print("\n{:<20} {:<15} {:<15} {:<15} {:<20}".format(
                "Resting ECG",
                f"{imp['mean']:.1f}",
                f"{wor['mean']:.1f}",
                f"{noc['mean']:.1f}",
                "—"
            ))
            print("{:<20} {:<15} {:<15} {:<15}".format(
                "  95% CI",
                f"[{imp['ci_lower']:.1f}, {imp['ci_upper']:.1f}]",
                f"[{wor['ci_lower']:.1f}, {wor['ci_upper']:.1f}]",
                f"[{noc['ci_lower']:.1f}, {noc['ci_upper']:.1f}]"
            ))
        
        print("\n" + "="*80)
    
    def save_results(self, ci_results, output_file='diagnostic_metrics_ci.csv'):
        """
        Save confidence interval results to CSV
        
        Args:
            ci_results: DataFrame with confidence intervals
            output_file: Output filename
        """
        ci_results.to_csv(output_file, index=False)
        print(f"\nResults saved to: {output_file}")


def main():
    """
    Main function to run diagnostic metrics CI analysis
    """
    # Initialize analyzer
    analyzer = DiagnosticMetricsCI(
        n_bootstrap=100,
        confidence_level=0.95,
        random_seed=42
    )
    
    # Load successful counterfactuals
    df = analyzer.load_successful_counterfactuals()
    
    # Run bootstrap analysis
    original_metrics = analyzer.run_bootstrap_analysis(df)
    
    # Calculate confidence intervals
    ci_results = analyzer.calculate_confidence_intervals()
    
    # Print summary
    analyzer.print_summary(original_metrics, ci_results)
    
    # Save results
    analyzer.save_results(ci_results)
    
    return analyzer, ci_results


if __name__ == "__main__":
    analyzer, ci_results = main()
