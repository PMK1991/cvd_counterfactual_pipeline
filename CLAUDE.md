# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **Cardiovascular Disease (CVD) Counterfactual Analysis Pipeline** for generating and validating counterfactual explanations. The pipeline generates fresh counterfactuals multiple times (default 100 iterations) using DiCE-ML, validates them with Structural Causal Models (SCM) via DoWhy, and computes confidence intervals for diagnostic metrics.

## Common Commands

### Running the Pipeline

**Test mode** (5 patients, 5 iterations - recommended first):
```bash
python fresh_cf_pipeline.py --test_mode
```

**Full pipeline** (48 patients, 100 iterations):
```bash
python fresh_cf_pipeline.py --n_iterations 100 --n_patients 48 --n_workers 4
```

**Custom configuration**:
```bash
python fresh_cf_pipeline.py --n_iterations 50 --n_patients 30 --n_workers 8
```

### Environment Setup

The project uses conda environment defined in `mtech-env.yml`:
```bash
conda env create -f mtech-env.yml
conda activate base
```

Key dependencies: `dice-ml==0.11`, `dowhy`, `numpy==1.26.4`, `pandas==1.5.3`

Alternative minimal setup from `requirements.txt`:
```bash
pip install -r requirements.txt
```

### Configuration

Edit `pipeline_config.yaml` to adjust:
- Number of iterations and patients
- Number of concurrent workers
- DiCE parameters (method, total_cfs, permitted_range, timeout)
- SCM sampling parameters
- Confidence interval level

## Architecture

### Modular 5-Component Pipeline

The pipeline follows a **modular, concurrent architecture** with these independent modules:

1. **`fresh_cf_pipeline.py`** - Main orchestrator
   - Manages concurrent execution using `ProcessPoolExecutor`
   - Coordinates workflow: Load data → Generate CFs → SCM validation → Compute metrics → Calculate CIs
   - Supports parallel execution with configurable worker count

2. **`dice_cf_generator.py`** - DiCE counterfactual generation
   - Uses DiCE-ML genetic algorithm
   - Thread-safe with timeout support
   - Generates counterfactuals for high-risk patients (target=1)
   - Permits intervention on `trestbps` and `chol` within specified ranges

3. **`scm_analyzer.py`** - SCM validation wrapper
   - Validates DiCE-generated counterfactuals using causal models
   - Wraps `counterfactualAnalyzer.py` (legacy DoWhy implementation)
   - Applies interventions on `chol` and `trestbps`
   - Only accepts counterfactuals that flip target from 1→0

4. **`metrics_calculator.py`** - Diagnostic metrics
   - Computes improvement/worsening/no-change percentages for each feature
   - Tracks metrics for: trestbps, chol, cp (chest pain), exang (exercise angina), oldpeak (ST depression), thalach (max heart rate), slope (ST slope), restecg (resting ECG)

5. **`ci_computer.py`** - Confidence interval calculation
   - Aggregates results across iterations
   - Computes percentile-based 95% confidence intervals
   - Generates summary reports in markdown format

### Key Data Flow

```
Load high-risk patients (target=1)
↓
[Parallel Execution: n_workers processes]
↓
For each iteration:
  - DiceCFGenerator: Generate 5 CFs per patient
  - SCMAnalyzer: Validate CFs using DoWhy interventions
  - MetricsCalculator: Compute diagnostic metrics
  - Save: iteration_XXX/{original,counterfactuals,successful,metrics.json}
↓
CIComputer: Aggregate all iterations
↓
Output: aggregated_results/{all_iteration_metrics.csv, ci_results.csv, summary_report.md}
```

### Legacy Components

- **`counterfactualAnalyzer.py`** - Original SCM analyzer using DoWhy's `gcm.interventional_samples()`
  - Used internally by `scm_analyzer.py`
  - Generates counterfactual samples via interventions
  - Applies physiological constraints (e.g., oldpeak ≥ 0)

- **Jupyter Notebooks** (in root directory):
  - `nb_cvd_pipeline.ipynb` - Original pipeline development
  - `nb_cvd_eda.ipynb` - Exploratory data analysis
  - `nb_confidence_intervals.ipynb` - CI analysis
  - `nb_heart_disease_scm.ipynb` - SCM experimentation

- **Standalone Analysis Scripts**:
  - `confidence_interval_analysis.py` - Standalone CI computation
  - `diagnostic_metrics_ci.py` - Metrics with CI computation
  - `display_ci_results.py` - Results visualization

## Data

**Primary dataset**: `heart_statlog_cleveland_hungary_final.csv`

**Features**:
- **Continuous**: age, trestbps (resting BP), chol (cholesterol), thalach (max heart rate), oldpeak (ST depression)
- **Categorical**: cp (chest pain type: 1-4), restecg (resting ECG: 0-2), slope (ST slope: 1-3)
- **Binary**: sex, fbs (fasting blood sugar), exang (exercise-induced angina)
- **Target**: target (1=disease, 0=no disease)

**Actionable features for intervention**: trestbps, chol (cholesterol lowering is the primary intervention)

## Output Structure

```
fresh_cf_iterations/              # or fresh_cf_iterations_test/ in test mode
├── iteration_000/
│   ├── original/                 # Original high-risk patient data
│   │   └── patient_*.csv
│   ├── counterfactuals/          # DiCE-generated CFs
│   │   └── patient_*_cf_*.csv
│   ├── successful/               # SCM-validated CFs (target flipped 1→0)
│   │   └── successful_counterfactuals.csv
│   └── metrics.json              # Iteration diagnostic metrics
├── iteration_001/
│   └── ...
├── aggregated_results/
│   ├── all_iteration_metrics.csv # All iteration results
│   ├── ci_results.csv            # Confidence intervals for all metrics
│   └── summary_report.md         # Human-readable summary
└── fresh_cf_pipeline.log         # Detailed execution logs
```

## Important Notes

### Concurrent Execution
- The pipeline uses `ProcessPoolExecutor` for parallel iteration execution
- Each worker process initializes its own DiCE and SCM modules to avoid shared state
- Adjust `--n_workers` based on CPU cores (increase for faster results, decrease if memory-constrained)

### DiCE Configuration
- **Method**: genetic (more robust than gradient-based for this dataset)
- **Timeout**: 30 seconds per patient (configurable in pipeline_config.yaml)
- **Permitted ranges**:
  - `trestbps`: [100, 120] (target healthy BP range)
  - `chol`: [150, 90% of original] (cholesterol reduction)

### SCM Validation
- Only counterfactuals that flip target from 1→0 are considered "successful"
- Uses DoWhy's interventional sampling with n_samples=1000
- Applies physiological constraints to ensure realistic outputs

### Performance Considerations
- Test mode completes in ~5-10 minutes
- Full pipeline (100 iterations, 48 patients) takes 2-4 hours on typical hardware
- Increase `--n_workers` to 6-8 on modern CPUs for faster execution
- Reduce `--n_workers` to 2-3 if encountering out-of-memory errors

## Working with This Codebase

### Adding New Metrics
Edit `metrics_calculator.py`:
- Add a new `compute_*_metrics()` method
- Include it in `compute_all_metrics()` return dictionary
- Metrics will automatically be included in CI computation

### Modifying Intervention Strategy
Edit `dice_cf_generator.py`:
- Update `permitted_range` in `_default_config()` or `pipeline_config.yaml`
- Modify `features_to_vary` to include/exclude features

### Changing Causal Model
Edit `scm_analyzer.py` and `counterfactualAnalyzer.py`:
- Modify the causal graph structure in `counterfactualAnalyzer.__init__()`
- Update intervention logic in `apply_scm_intervention()`

### Debugging
- Check `fresh_cf_pipeline.log` for detailed execution logs
- Each iteration's `metrics.json` shows success counts
- Failed iterations return `{'error': 'message'}` in results
