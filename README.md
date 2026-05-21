# CVD Counterfactual Analysis Pipeline

A reproducible pipeline for generating and validating counterfactual explanations for cardiovascular disease (CVD) risk prediction. The pipeline combines **DiCE-ML** counterfactual generation with **Structural Causal Model (SCM)** validation using **DoWhy**, and computes **95% algorithmic-stability intervals** across 100 independent iterations.

## Research Context

Counterfactual explanations answer: *"What minimal changes to a patient's clinical profile would flip their CVD risk prediction from high-risk to low-risk?"*

This pipeline addresses a key challenge: counterfactual generators like DiCE produce statistically valid but not necessarily **causally plausible** explanations. The final analysis lets DiCE search broadly, projects each generated counterfactual to a cholesterol-only change, and validates that intervention through an SCM that encodes cardiovascular domain knowledge.

## Pipeline Architecture

```
                        ┌─────────────────────┐
                        │   Raw CVD Dataset    │
                        │      1190 rows       │
                        └──────────┬──────────┘
                                   │
                                   ▼
                        ┌─────────────────────┐
                        │    Data Cleaning     │
                        │  dropna, chol > 0,   │
                        │  trestbps > 0,       │
                        │  dedupe, IQR filter  │
                        │                      │
                        │  1190 ──> 707 rows   │
                        └──────────┬──────────┘
                                   │
                                   ▼
                        ┌─────────────────────┐
                        │  Held-out Test Set   │
                        │  test_size = 0.2     │
                        │  random_state = 42   │
                        │                      │
                        │  707 ──> 142 rows    │
                        └──────────┬──────────┘
                                   │
                                   ▼
                        ┌─────────────────────┐
                        │  True-positive       │
                        │  High-risk Cohort    │
                        │                      │
                        │  target = 1          │
                        │  model prediction=1  │
                        │                      │
                        │  52 ──> 48 patients  │
                        └──────────┬──────────┘
                                   │
                                   ▼
              ┌─────────────────────────────────────────────┐
              │     100 Independent Pipeline Iterations      │
              │                                             │
              │  ┌───────────────────────────────────────┐  │
              │  │ DiCE Generator                         │  │
              │  │ Genetic algorithm, broad search        │  │
              │  │ 5 CFs requested per patient            │  │
              │  │ chol constrained to [150, 200]         │  │
              │  └──────────────────┬────────────────────┘  │
              │                     │                       │
              │                     ▼                       │
              │  ┌───────────────────────────────────────┐  │
              │  │ Cholesterol-only Projection            │  │
              │  │ Keep cf_chol from DiCE                 │  │
              │  │ Reset all other features to original   │  │
              │  └──────────────────┬────────────────────┘  │
              │                     │                       │
              │                     ▼                       │
              │  ┌───────────────────────────────────────┐  │
              │  │ SCM Validation (DoWhy) using full DAG  │  │
              │  │ InvertibleStructuralCausalModel        │  │
              │  │                                       │  │
              │  │ Intervention: do(chol = cf_chol)       │  │
              │  │                                       │  │
              │  │ DAG: risk factors -> target ->         │  │
              │  │ downstream variables, plus full-DAG    │  │
              │  │ cross-links                            │  │
              │  └──────────────────┬────────────────────┘  │
              │                     │                       │
              │                     ▼                       │
              │  ┌───────────────────────────────────────┐  │
              │  │ Target flipped?                        │  │
              │  │ original target = 1                    │  │
              │  │ SCM counterfactual target = 0          │  │
              │  │ YES -> valid CF; NO -> rejected CF     │  │
              │  └──────────────────┬────────────────────┘  │
              │                     │                       │
              │                     ▼                       │
              │  ┌───────────────────────────────────────┐  │
              │  │ Per-iteration Metrics                  │  │
              │  │ successful CF count, target flip rate, │  │
              │  │ downstream deltas and change ranges    │  │
              │  └───────────────────────────────────────┘  │
              │                                             │
              └──────────────────────┬──────────────────────┘
                                     │
                                     ▼
                          ┌─────────────────────┐
                          │ Aggregate Results    │
                          │ Across 100 Runs      │
                          └──────────┬──────────┘
                                     │
                                     ▼
                          ┌─────────────────────┐
                          │ 95% Algorithmic-     │
                          │ Stability Intervals  │
                          │ + Target-flip        │
                          │ Robustness Index     │
                          └─────────────────────┘
```

**Key design choices:**
- **True-positive cohort** restricts full-mode analysis to held-out test-set patients with `target=1` and model prediction `1`
- **Broad DiCE search with cholesterol-only projection** preserves the high-yield historical workflow while analyzing only `do(chol)`
- **Fixed random seed per patient-CF pair** ensures deterministic SCM results for a given projected CF; variation comes from DiCE's stochastic CF generation across iterations
- **Parallel execution** via `ProcessPoolExecutor` with configurable worker count
- **Physiological constraints** enforce clinically valid ranges (e.g., oldpeak >= 0, cp in [1,4])

## Repository Structure

```
cvd_counterfactual_pipeline/
│
├── src/                                # Source code
│   ├── pipeline/                       # Core pipeline modules
│   │   ├── fresh_cf_pipeline.py        #   Main pipeline orchestrator
│   │   ├── dice_cf_generator.py        #   DiCE counterfactual generation
│   │   ├── scm_analyzer.py            #   SCM validation (DoWhy)
│   │   ├── metrics_calculator.py       #   Diagnostic metrics computation
│   │   ├── ci_computer.py             #   Confidence interval computation
│   │   ├── ev_calculator.py           #   Target-flip robustness index (E-value-like)
│   │   ├── patient_bootstrap.py       #   Patient-level bootstrap CIs
│   │   ├── cohort_flowchart.py        #   Cohort flowchart rendering
│   │   └── sensitivity_analyzer.py     #   Sensitivity analysis
│   ├── training/                       # Model training
│   │   └── train_model.py             #   XGBoost model training
│   ├── utils/                          # Utility classes
│   │   ├── dataLoader.py              #   Data loading + IQR outlier removal
│   │   ├── plotter.py                 #   Visualization utilities
│   │   └── hyperParameterTuning.py    #   GridSearchCV wrapper
│   └── legacy/                         # Standalone analysis scripts
│       ├── counterfactualAnalyzer.py   #   Legacy SCM analyzer
│       ├── confidence_interval_analysis.py
│       ├── diagnostic_metrics_ci.py
│       └── display_ci_results.py
│
├── data/                              # Dataset
│   ├── heart_statlog_cleveland_hungary_final.csv
│   └── heart_statlog_cleveland_hungary_final.xlsx
│
├── model/                             # Trained model
│   └── xgb_pipeline.pkl              # XGBoost + StandardScaler + OneHotEncoder
│
├── notebooks/                         # Jupyter notebooks (not tracked in git)
│   ├── eda/                           #   Exploratory data analysis
│   ├── counterfactual/                #   CF generation experiments
│   ├── causal/                        #   SCM/causal model experiments
│   └── analysis/                      #   CI/results analysis
│
├── fresh_cf_iterations/               # Pipeline output (gitignored)
│   ├── iteration_000/ ... iteration_099/
│   │   ├── original/                  # Original patient data
│   │   ├── counterfactuals/           # DiCE-generated CFs
│   │   ├── successful/               # SCM-validated CFs
│   │   └── metrics.json              # Per-iteration metrics
│   ├── aggregated_results/
│   │   ├── all_iteration_metrics.csv
│   │   ├── ci_results.csv
│   │   └── summary_report.md
│   └── sensitivity_results/           # Sensitivity analysis output
│
├── docs/                              # Paper drafts, reviewer comments
├── reports/                           # Standalone analysis reports
├── plots/                             # Visualizations
│
├── pipeline_config.yaml               # Pipeline configuration
├── mtech-env.yml                      # Conda environment (full)
├── requirements.txt                   # Pip requirements (minimal)
└── README.md
```

## Reproducing Results

### 1. Environment Setup

**Option A: Conda (recommended)**
```bash
conda env create -f mtech-env.yml
conda activate base
```

**Option B: Pip**
```bash
pip install -r requirements.txt
```

**Core dependencies:**
| Package | Version | Purpose |
|---------|---------|---------|
| dice-ml | 0.12 | Counterfactual generation (genetic algorithm) |
| dowhy | 0.14 | Structural causal models, interventional sampling |
| xgboost | 3.0.5 | CVD risk classifier |
| scikit-learn | 1.6.1 | Pipeline, preprocessing, evaluation |
| numpy | 1.26.4 | Numerical computation |
| pandas | 1.5.3+ | Data manipulation |
| networkx | 3.4.2 | Causal graph construction |
| matplotlib | 3.9.3 | Plotting |
| openpyxl | 3.1.5 | Excel I/O |

### 2. Train the Model

```bash
python src/training/train_model.py
```

Trains an XGBoost classifier (`max_depth=3, learning_rate=0.01, n_estimators=300`) on the CVD dataset with IQR outlier removal, StandardScaler for continuous features, and OneHotEncoder for categorical features. Saves to `model/xgb_pipeline.pkl` and reports both train and held-out test sensitivity/specificity to two decimal places.

To rerun the randomized balanced-accuracy hyperparameter search and optional TabPFN comparison:

```bash
python src/training/tune_model.py
```

Full model performance from `reports/model_tuning_results.json` (baseline = original hyperparameters; tuned = best from randomised search over an expanded grid):

| Split | Accuracy | Precision | Recall | F1 | ROC-AUC | Sensitivity | Specificity |
|-------|--------:|----------:|-------:|---:|--------:|------------:|------------:|
| Baseline — Train | 0.8973 | 0.8840 | 0.9152 | 0.8993 | 0.9611 | 0.9152 | 0.8794 |
| Baseline — Test  | 0.9155 | 0.8571 | 0.9231 | 0.8889 | 0.9746 | 0.9231 | 0.9111 |
| Tuned — Train    | 0.9097 | 0.8919 | 0.9329 | 0.9119 | 0.9652 | 0.9329 | 0.8865 |
| Tuned — Test     | 0.9155 | 0.8571 | 0.9231 | 0.8889 | 0.9675 | 0.9231 | 0.9111 |

Tuned best params: `max_depth=4, learning_rate=0.05, n_estimators=200, subsample=0.6, colsample_bytree=0.6, min_child_weight=5, gamma=0.5, reg_alpha=0.01, reg_lambda=2.0, scale_pos_weight=1.0`. Despite the broader search, test-set sensitivity and specificity are identical across both models, confirming that the baseline hyperparameters were already near-optimal for this dataset.

The corresponding confusion matrix on the cleaned held-out test set (both models) is `[[82, 8], [4, 48]]` with rows as true labels and columns as predictions. Sensitivity is `TP / (TP + FN) = 48 / 52 = 0.9231`, specificity is `TN / (TN + FP) = 82 / 90 = 0.9111`.

**Clinical implications of sensitivity and specificity.** In a CVD screening context the two metrics have asymmetric clinical consequences. A sensitivity of **92.3%** means the model correctly flags approximately 9 in every 10 genuinely high-risk patients, keeping the false-negative rate low (4 missed cases out of 52). This is the more safety-critical metric: a missed high-risk patient receives no recourse recommendations and may not take the lifestyle steps that could defer a cardiovascular event. A specificity of **91.1%** means 9 in 10 truly low-risk patients are correctly left out of the counterfactual pipeline, avoiding unnecessary intervention recommendations and wasted computation (8 false positives out of 90). Both values exceed the 80–85% benchmarks reported in comparable CVD prediction studies on this dataset, and together they establish that the true-positive high-risk cohort entering the downstream SCM validation is both **comprehensive** (few genuine high-risk patients missed) and **precise** (few low-risk patients misrouted into recourse computation). The model's operating point therefore favours the clinically preferred direction — higher sensitivity at modest specificity cost — consistent with standard practice for preventive-screening classifiers where the cost of a false negative substantially exceeds the cost of a false positive.

### 3. Run the Pipeline

**Quick test** (5 patients, 5 iterations, ~6 min):
```bash
python src/pipeline/fresh_cf_pipeline.py --test_mode
```

**Full run** (test-set true-positive high-risk cohort, 100 iterations):
```bash
python src/pipeline/fresh_cf_pipeline.py --n_iterations 100 --n_workers 4
```

Full mode uses the same cleaned 80/20 split as model training (`test_size=0.2`, `random_state=42`) and selects true-positive high-risk patients from the test set (`target=1` and model prediction `1`). `--n_patients` is a debug cap only in `--test_mode`.

**Patient-level bootstrap** (inferential intervals from cached successful CFs):
```bash
python src/pipeline/fresh_cf_pipeline.py --run_patient_bootstrap --bootstrap_iterations 1000
```

Adjust `--n_workers` based on CPU cores (increase for faster execution, decrease if memory-constrained).

### 4. Sensitivity Analysis

```bash
python src/pipeline/fresh_cf_pipeline.py --sensitivity
```

### 5. View Results

Results are saved to `fresh_cf_iterations/aggregated_results/`:
- `summary_report.md` — human-readable table with algorithmic-stability intervals
- `ci_results.csv` — full algorithmic-stability interval data for all metrics
- `all_iteration_metrics.csv` — raw metrics from each iteration
- `evalue.json` — target-flip robustness index (single-arm odds plugged into the E-value formula; not a published E-value)
- `cohort_counts.json` — row counts for cohort construction
- `patient_bootstrap_ci.csv` — optional patient-level inferential intervals
- `structural_equations.md` / `structural_equations.json` — per-node SCM mechanisms in open form (`X = f(parents) + ε`) and closed form where the auto-assigned estimator allows it; regenerate with `python scripts/dump_structural_equations.py --output_dir fresh_cf_iterations/aggregated_results`
- `structural_equations_open_closed.md` — longer-form write-up pairing open and closed forms with empirical noise summaries

## Dataset

**Source:** Combined heart disease dataset from Statlog, Cleveland, and Hungary databases.

| Property | Value |
|----------|-------|
| Instances | 1190 |
| Features | 11 (5 continuous, 6 categorical/binary) |
| Target | Binary (1 = CVD, 0 = healthy) |
| Class balance | ~55% positive |

**Features:**

| Feature | Type | Description | Range |
|---------|------|-------------|-------|
| age | Continuous | Age in years | 28-77 |
| sex | Binary | Sex (0=F, 1=M) | 0, 1 |
| cp | Categorical | Chest pain type | 1-4 |
| trestbps | Continuous | Resting blood pressure (mmHg) | 0-200 |
| chol | Continuous | Serum cholesterol (mg/dL) | 0-603 |
| fbs | Binary | Fasting blood sugar > 120 mg/dL | 0, 1 |
| restecg | Categorical | Resting ECG results | 0-2 |
| thalach | Continuous | Maximum heart rate achieved (bpm) | 60-202 |
| exang | Binary | Exercise-induced angina | 0, 1 |
| oldpeak | Continuous | ST depression (mm) | -2.6 to 6.2 |
| slope | Categorical | Slope of peak exercise ST segment | 1-3 |
| target | Binary | CVD diagnosis | 0, 1 |

**Direct intervention target in the final analysis:** `chol`. Blood pressure (`trestbps`) and symptom features are downstream SCM-propagated variables, not direct interventions in the final run.

## Cohort Definition and Interval Semantics

The analysis cohort is the true-positive high-risk subset of the held-out test split used by `src/training/train_model.py` (`test_size=0.2`, `random_state=42`): `target=1` and model prediction `1`. Test mode applies an additional debug cap for faster runs.

`ci_results.csv` reports percentile intervals across independent DiCE-generation iterations; these are **algorithmic-stability intervals**. `patient_bootstrap_ci.csv`, when enabled, reports **inferential intervals** from patient-level cluster bootstrap resampling.

## Configuration

All parameters are configurable via `pipeline_config.yaml`:

```yaml
pipeline:
  n_iterations: 100        # Number of fresh CF generation rounds
  n_patients: 48            # Test-mode debug cap only
  n_workers: 4              # Concurrent worker processes
  run_patient_bootstrap: false
  bootstrap_iterations: 1000

dice:
  method: "genetic"         # DiCE algorithm
  total_cfs: 5              # CFs generated per patient
  features_to_vary: null    # Broad DiCE search
  permitted_range:
    trestbps: [100, 120]
    chol: [150, 200]        # Cholesterol intervention range
  timeout: 45               # Seconds per patient

scm:
  n_samples: 1000           # SCM Monte Carlo samples per intervention
  graph_structure: "full"   # minimal, full, full_with_symptom_links, or extended
  intervention_targets: "chol_only"

ci:
  confidence_level: 0.95    # 95% algorithmic-stability intervals
```

## Causal Graph

The SCM uses DoWhy's `InvertibleStructuralCausalModel` with a 3-layer directed acyclic graph (from `notebooks/causal/nb_cvd_scm.ipynb`):

![DAG for Statlog Heart Disease Dataset](plots/scm_dag.png)

The graph encodes three layers:

- **Layer 1 — Risk Factors** (root nodes): `age`, `sex`, `chol`, `fbs`, `trestbps`
- **Layer 2 — Disease**: `target`
- **Layer 3 — Symptoms**: `cp`, `restecg`, `thalach`, `exang`, `slope`, `oldpeak`

With edges: risk factors → target → symptoms, plus direct risk-factor linkages (`age → chol`, `age → trestbps`, `sex → trestbps`, `sex → chol`, `chol → trestbps`). Symptom-to-symptom cross-links (`thalach → exang`, `exang → cp`) are *excluded* from the default `full` variant to preserve the conditional-independence assumption that symptoms depend on each other only through `target`; the legacy graph that included them is reachable via `graph_structure: full_with_symptom_links`.

**Intervention mechanism:** the final run uses `do(chol=X)` only. DiCE may search broadly, but each candidate is projected back onto the original patient row with only `chol` changed before SCM validation. DoWhy's `gcm.interventional_samples()` then propagates the cholesterol intervention through the DAG; `trestbps`, `cp`, `exang`, `oldpeak`, `thalach`, `slope`, and `restecg` are interpreted as downstream SCM effects. A fixed random seed derived from patient features ensures deterministic results per patient-CF pair.

**Categorical columns** (`target`, `exang`, `fbs`, `cp`, `restecg`, `slope`) are cast to `category` dtype before model fitting, matching the notebook setup.

## Results (100 iterations, 48 patients, 95% algorithmic-stability intervals)

**Successful SCM-validated CFs per iteration:** 83.3 (95% algorithmic-stability interval: [72.0, 98.0])

**Target flip rate:** 35.1% (95% algorithmic-stability interval: [31.0%, 41.1%])

**Target-flip robustness index:** 3.09 (interval: [2.22, 3.89]). Computed by plugging the single-arm flip-rate odds into the VanderWeele-Ding E-value formula; reported as a derivative robustness summary specific to this pipeline, *not* the published two-arm E-value and not proof of causal identification or clinical effectiveness.

| Metric | Improve (%) | Worsen (%) | No Change (%) | Mode Before/After | Mean Change | 95% Algorithmic-Stability Interval (Improve %) |
|--------|-------------|------------|---------------|-------------------|-------------|---------------------|
| Resting BP (trestbps) | 52.5 | 45.7 | 1.8 | -- | -2.86 mmHg | [42.6%, 65.8%] |
| Chest Pain (cp) | 68.2 | 4.6 | 27.2 | 4 to 3 | -- | [58.9%, 79.1%] |
| Exercise Angina (exang) | 81.7 | 0.0 | 18.3 | 1 to 0 | -- | [76.2%, 86.8%] |
| ST Depression (oldpeak) | 78.5 | 21.5 | 0.0 | -- | -1.15 mm | [73.4%, 83.1%] |
| Max Heart Rate (thalach) | 79.3 | 20.4 | 0.2 | -- | +22.74 bpm | [72.3%, 84.8%] |
| ST Slope (slope) | 92.9 | 0.0 | 7.1 | 2 to 1 | -- | [89.2%, 95.0%] |
| Resting ECG (restecg) | 54.9 | 0.0 | 45.1 | 0 to 0 | -- | [47.4%, 60.6%] |

Intervals in this table are percentile intervals across 100 independent DiCE-generation runs. They quantify algorithmic stability under repeated stochastic counterfactual generation and should not be interpreted as patient-level inferential confidence intervals.

Observed change ranges across all 8,326 successful SCM-validated rows:

| Variable | Signed Change Range | Absolute Change Range | Mean Signed Change |
|----------|---------------------|-----------------------|--------------------|
| `chol` | -191 to +35 mg/dL | 0 to 191 mg/dL | -57.41 mg/dL |
| `trestbps` | -37 to +32 mmHg | 0 to 37 mmHg | -2.93 mmHg |
| `cp` | -1 to +3 category levels | 0 to 3 levels | -0.58 |
| `exang` | -1 to 0 | 0 to 1 | -0.82 |
| `oldpeak` | -5.60 to +0.87 mm | 0 to 5.60 mm | -1.15 mm |
| `thalach` | -32 to +70 bpm | 0 to 70 bpm | +22.66 bpm |
| `slope` | -2 to 0 category levels | 0 to 2 levels | -1.04 |
| `restecg` | -2 to 0 category levels | 0 to 2 levels | -0.80 |

No `exang`, `slope`, or `restecg` worsening occurred in the corrected run; the zero-worsening rows reflect the observed true-positive cohort and fitted SCM response under cholesterol-only intervention.

## Sensitivity Analysis

One-at-a-time (OAT) parameter sweeps are supported for `total_cfs`, `trestbps_range`, `chol_lower`, `confidence_level`, `graph_structure`, `intervention_targets`, and `n_samples`. Sensitivity outputs should be regenerated after changing the primary intervention strategy, because the final analysis now uses broad DiCE search followed by cholesterol-only projection and SCM `do(chol)`.

See `fresh_cf_iterations/sensitivity_results/sensitivity_report.md` for full comparison tables.

## License

For academic and research use.
