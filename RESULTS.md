# CVD Counterfactual Pipeline — Results

**Run date:** 2026-05-30
**Configuration:** 100 iterations · 4 workers · test-set true-positive high-risk cohort · cholesterol-only SCM intervention `do(chol)`
**SCM:** leakage-free pre-fitted validator (`model/scm_full.pkl`, fitted on the 565-row cleaned training split). All 4 workers loaded the pre-fitted model — 0 in-process fits, 0 errors. Wall time 106.91 min.

> Intervals reported below are **95% algorithmic-stability intervals** — percentile intervals across the 100 independent DiCE generation runs. They quantify stability under repeated stochastic counterfactual generation and are **not** patient-level inferential confidence intervals. Source: `fresh_cf_iterations/aggregated_results/ci_results.csv`.

---

## 1. Headline metrics

| Metric | Value | 95% stability interval |
|--------|-------|------------------------|
| Successful SCM-validated CFs per iteration | **82.9** | [76.0, 88.0] |
| Target flip rate | **34.8%** | [32.7%, 36.7%] |
| Target-flip robustness index | **3.16** | [2.85, 3.53] |
| Total successful SCM-validated rows | 8,286 | — |
| Patients in cohort | 48 | — |

The **target-flip robustness index** plugs the single-arm flip-rate odds into the VanderWeele–Ding E-value formula. It is a derivative robustness summary specific to this pipeline — *not* the published two-arm E-value, and not proof of causal identification or clinical effectiveness.

---

## 2. Effect on variables

The only intervention is `do(chol)`. Every other change is a **propagated SCM effect** through the DAG (chol → target → symptoms), not a directly-set value. Direction below is the clinically healthy direction.

| Variable | Direction | Mean Δ | Δ 95% CI | Improve % | Improve 95% CI | Worsen % | Worsen 95% CI |
|----------|-----------|--------|----------|-----------|----------------|----------|---------------|
| **chol** (intervened) | ↓ lowered | -56.70 mg/dL | — | — | — | — | — |
| Resting BP (trestbps) | ↓ better | -3.48 mmHg | [-4.47, -2.90] | 60.3 | [57.0, 64.6] | 39.5 | [35.4, 42.7] |
| Chest Pain (cp) | ↓ better | — | — | 87.5 | [85.6, 88.6] | 12.1 | [11.4, 13.2] |
| Exercise Angina (exang) | ↓ better | — | — | 62.5 | [58.7, 65.3] | **0.0** | [0.0, 0.0] |
| ST Depression (oldpeak) | ↓ better | -1.51 mm | [-1.62, -1.42] | 72.8 | [69.6, 77.3] | 27.1 | [22.7, 30.4] |
| Max Heart Rate (thalach) | ↑ better | +16.15 bpm | [+14.98, +17.05] | 75.0 | [72.1, 77.3] | 24.1 | [22.5, 26.3] |
| ST Slope (slope) | ↓ better | — | — | 93.7 | [92.4, 94.3] | **0.0** | [0.0, 0.0] |
| Resting ECG (restecg) | ↓ better | — | — | 38.6 | [35.9, 42.3] | **0.0** | [0.0, 0.0] |

**Observations**

- **Three variables never worsen** — `exang`, `slope`, `restecg` move only toward health (worsen-% CI = [0.0, 0.0]).
- **Strongest, most reliable responses:** `slope` (93.7% improve) and `cp` (87.5% improve).
- **Weakest response:** `restecg` (38.6% improve, 61.4% no-change) — largely insensitive to the cholesterol intervention.
- **Bidirectional variables** (`trestbps`, `oldpeak`, `thalach`) have some worsening but their means all point in the improving direction; `thalach ↑` is the healthy direction.

### Observed change ranges (all 8,286 successful rows)

| Variable | Signed Change Range | Absolute Change Range | Mean Signed Change |
|----------|---------------------|-----------------------|--------------------|
| `chol` | -132 to +36 mg/dL | 0 to 132 mg/dL | -56.70 mg/dL |
| `trestbps` | -31 to +19 mmHg | 0 to 31 mmHg | -3.47 mmHg |
| `cp` | -1 to +2 levels | 0 to 2 levels | -0.63 |
| `exang` | -1 to 0 | 0 to 1 | -0.63 |
| `oldpeak` | -6.13 to +1.07 mm | 0 to 6.13 mm | -1.50 mm |
| `thalach` | -34 to +67 bpm | 0 to 67 bpm | +16.16 bpm |
| `slope` | -2 to 0 levels | 0 to 2 levels | -1.02 |
| `restecg` | -2 to 0 levels | 0 to 2 levels | -0.57 |

---

## 3. Cohort funnel

| Step | Rows in | Rows out | Dropped |
|------|---------|----------|---------|
| Raw dataset | 1190 | 1190 | 0 |
| Drop NA | 1190 | 1190 | 0 |
| chol > 0 | 1190 | 1018 | 172 |
| trestbps > 0 | 1018 | 1018 | 0 |
| Drop duplicates | 1018 | 746 | 272 |
| Remove IQR outliers | 746 | 707 | 39 |
| Train/test split (test set) | 707 | 142 | 565 |
| Test-set high-risk (target=1) | 142 | 52 | 90 |
| Test-set true positives (pred=1) | 52 | **48** | 4 |

The 565 rows removed at the split step are exactly the cleaned training split used to fit the leakage-free SCM, so no test-set patient leaks into the SCM fit. Source: `fresh_cf_iterations/aggregated_results/cohort_counts.json`.

---

## 4. Robustness index detail

| Field | Value |
|-------|-------|
| Quantity | target_flip_rate |
| Target-flip probability | 0.3477 |
| Risk-ratio-like estimate | 1.876 [1.727, 2.054] |
| Target-flip robustness index | **3.159** [2.848, 3.525] |
| Formula | `odds(p)=p/(1-p); rr_like=max(odds,1/odds); index=rr_like+sqrt(rr_like·(rr_like-1))` |

Source: `fresh_cf_iterations/aggregated_results/evalue.json`.

---

## 5. Notes & caveats

- **Statistical shift (expected):** the leakage-free 565-row SCM fit yields a higher success rate than the earlier raw-CSV fallback fit. This is the anticipated X3 effect of fitting on the cleaned training split — not a regression.
- **Interval semantics:** all intervals here are algorithmic-stability percentile intervals. For patient-cluster inferential CIs, run the pipeline with `--run_patient_bootstrap` (not run for these results).
- **Artifacts:** full machine-readable outputs are in `fresh_cf_iterations/aggregated_results/` — `summary_report.md`, `ci_results.csv`, `all_iteration_metrics.csv`, `evalue.json`, `cohort_counts.json`.
