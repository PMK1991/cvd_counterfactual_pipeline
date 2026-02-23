# Fresh CF Pipeline - Quick Start Guide

## Overview

This pipeline generates fresh counterfactuals **100 times** using DiCE-ML and computes confidence intervals for all diagnostic metrics. It features:

- **Modular Architecture:** 5 independent modules
- **Concurrent Execution:** Runs iterations in parallel
- **Scalable:** Adjust workers based on CPU cores
- **Configurable:** All parameters in `pipeline_config.yaml`

## Quick Start

### 1. Test Mode (Recommended First)

Run with 5 patients and 5 iterations to verify everything works:

```bash
cd c:\\Users\\pmkul\\Dropbox\\Counterfactual_Analysis\\cvd_counterfactual_pipeline
python fresh_cf_pipeline.py --test_mode
```

**Expected Output:**
- Creates `fresh_cf_iterations_test/` folder
- Runs 5 iterations concurrently
- Generates CI results in ~5-10 minutes

### 2. Full Pipeline

Run with all 48 patients and 100 iterations:

```bash
python fresh_cf_pipeline.py --n_iterations 100 --n_patients 48 --n_workers 4
```

**Expected Runtime:** 2-4 hours (depending on hardware)

### 3. Custom Configuration

Adjust parameters:

```bash
python fresh_cf_pipeline.py --n_iterations 50 --n_patients 30 --n_workers 8
```

## Command-Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--n_iterations` | 100 | Number of times to generate fresh CFs |
| `--n_patients` | 48 | Number of high-risk patients |
| `--n_workers` | 4 | Number of concurrent workers |
| `--test_mode` | False | Quick test (5 patients, 5 iterations) |

## Output Structure

```
fresh_cf_iterations/
├── iteration_000/
│   ├── original/          # Original patient data
│   ├── counterfactuals/   # Generated CFs
│   ├── successful/        # SCM-validated CFs
│   └── metrics.json       # Iteration metrics
├── iteration_001/
│   └── ...
├── aggregated_results/
│   ├── all_iteration_metrics.csv  # All iteration results
│   ├── ci_results.csv             # Confidence intervals
│   └── summary_report.md          # Summary report
└── fresh_cf_pipeline.log          # Detailed logs
```

## Modules

1. **dice_cf_generator.py** - DiCE counterfactual generation
2. **scm_analyzer.py** - SCM validation using DoWhy
3. **metrics_calculator.py** - Diagnostic metrics computation
4. **ci_computer.py** - Confidence interval calculation
5. **fresh_cf_pipeline.py** - Main orchestrator

## Performance Tips

- **More Workers:** Increase `--n_workers` if you have more CPU cores
- **Fewer Iterations:** Start with 20-50 iterations for faster results
- **Test Mode:** Always test first to catch issues early

## Troubleshooting

**Issue:** Pipeline runs slowly
- **Solution:** Increase `--n_workers` (try 6-8 on modern CPUs)

**Issue:** Out of memory errors
- **Solution:** Reduce `--n_workers` to 2-3

**Issue:** DiCE timeouts
- **Solution:** Increase `timeout` in `pipeline_config.yaml`

## Example Workflow

```bash
# Step 1: Test with small dataset
python fresh_cf_pipeline.py --test_mode

# Step 2: Check results
cat fresh_cf_iterations_test/aggregated_results/summary_report.md

# Step 3: Run full pipeline
python fresh_cf_pipeline.py --n_iterations 100 --n_workers 6

# Step 4: View final results
cat fresh_cf_iterations/aggregated_results/summary_report.md
```

## Next Steps

After running the pipeline:
1. Review `aggregated_results/summary_report.md`
2. Compare with existing results in `diagnostic_ci_summary.md`
3. Analyze differences in confidence intervals
4. Adjust DiCE parameters if needed
