"""
Quick script to print the confidence interval results in a formatted way
"""
import pandas as pd

# Load results
ci_results = pd.read_csv('diagnostic_metrics_ci.csv')

print("\n" + "="*80)
print("CONFIDENCE INTERVALS FOR DOWNSTREAM DIAGNOSTIC METRICS")
print("="*80)
print("Bootstrap iterations: 100 | Confidence level: 95% | Sample size: 94")
print("="*80)

# Format and print key metrics
metrics_display = {
    'cp_changed_pct': 'Chest Pain Changed (%)',
    'exang_1_to_0_pct': 'Angina Resolved (1→0) (%)',
    'exang_0_to_1_pct': 'Angina Developed (0→1) (%)',
    'oldpeak_reduced_pct': 'ST Depression Reduced (%)',
    'mean_orig_oldpeak': 'Original ST Depression (mm)',
    'mean_cf_oldpeak': 'CF ST Depression (mm)',
    'mean_diff_oldpeak': 'ST Depression Change (mm)',
    'thalach_increased_pct': 'Max Heart Rate Increased (%)',
    'mean_orig_thalach': 'Original Max HR (bpm)',
    'mean_cf_thalach': 'CF Max HR (bpm)',
    'mean_diff_thalach': 'Max HR Change (bpm)',
    'slope_decreased_pct': 'ST Slope Improved (%)',
    'restecg_no_change_pct': 'Resting ECG No Change (%)',
    'restecg_improved_pct': 'Resting ECG Improved (%)',
    'restecg_worsened_pct': 'Resting ECG Worsened (%)',
}

for metric_key, metric_name in metrics_display.items():
    row = ci_results[ci_results['metric'] == metric_key]
    if not row.empty:
        row = row.iloc[0]
        print(f"\n{metric_name}:")
        print(f"  Mean: {row['mean']:.2f}")
        print(f"  95% CI: [{row['ci_lower']:.2f}, {row['ci_upper']:.2f}]")
        print(f"  CI Width: {row['ci_width']:.2f}")

print("\n" + "="*80)
print("\nFull results saved to: diagnostic_metrics_ci.csv")
print("Detailed report saved to: diagnostic_metrics_ci_report.md")
print("="*80 + "\n")
