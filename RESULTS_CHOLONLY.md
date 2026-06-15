# Results — Cholesterol-Only Recourse (do(chol)), Unified Run

All numbers below come from a **single, internally consistent 100-iteration run**
(`fresh_cf_iterations_cholonly/`), generated and validated with one method:

- **DiCE** performs an unconstrained search across all features
  (`features_to_vary = null`); only **`chol`** is range-constrained to the
  clinical-target corridor **[150, 200] mg/dL** (`trestbps` is no longer
  constrained).
- The **SCM projects cholesterol only**: it forward-simulates from the original
  patient row with the cholesterol node clamped to the recourse value —
  **`do(chol)`** — and propagates every other variable (incl. `trestbps`, via the
  `chol → trestbps` edge) through the structural equations. No variable other
  than `chol` is clamped.
- Environment: `mtech-env` (dowhy 0.12 / scikit-learn 1.6.1), matching the
  fitted SCM artifact exactly (no version skew). 100 iterations, 48 test-set
  true-positive high-risk patients, `n_samples = 1000`, graph variant `full`.

**Cohort accounting (per iteration):** 237.9 counterfactuals generated →
**82.8 flipped** + **155.1 non-flip**. Target flip rate **34.8%**
(95% algorithmic-stability interval [33.1%, 36.7%]). Both the flip and non-flip
analyses below are derived from this same run, so the flip count is consistent
across them.

---

## 1. Flip cohort — SCM-validated successful counterfactuals

Successful CFs per iteration: **82.8** (95% interval [79.0, 87.5]).
Target-flip robustness index (E-value-like, single-arm): **3.15**
([2.84, 3.47]); flip probability 0.348. *Derivative robustness summary, not the
published VanderWeele–Ding E-value.*

| Metric | ↓ Improve (%) | ↑ Worsen (%) | ↔ No Change (%) | Mode Before → After | Δ Mean | 95% Interval (Improve %) |
|--------|--------------|-------------|-----------------|---------------------|--------|--------------------------|
| Resting BP (trestbps) | 59.5 | 40.3 | 0.1 | — | −3.31 mmHg | [56.4%, 62.3%] |
| Chest Pain (cp) | 87.7 | 11.8 | 0.5 | 4 → 3 | — | [85.3%, 89.2%] |
| Exang (1→0 / 0→1) | 63.3 | 0.0 | 36.7 | 1 → 0 | — | [60.2%, 66.5%] |
| ST Depression (oldpeak) | 71.8 | 28.1 | 0.1 | — | −1.49 mm | [69.2%, 75.2%] |
| Max Heart Rate (thalach) | 75.5 | 23.9 | 0.6 | — | +16.29 bpm | [73.4%, 77.3%] |
| ST Slope (slope) | 93.8 | 0.0 | 6.2 | 2 → 1 | — | [92.7%, 94.3%] |
| Resting ECG (restecg) | 38.6 | 0.0 | 61.4 | 0 → 0 | — | [35.9%, 41.8%] |

### Observed change ranges across all 8,282 successful rows

| Variable | Signed Change Range | Absolute Change Range | Mean Signed Change |
|----------|---------------------|-----------------------|--------------------|
| `chol` | −132 to +36 mg/dL | 0 to 132 mg/dL | −56.17 mg/dL |
| `trestbps` | −31 to +19.5 mmHg | 0 to 31 mmHg | −3.31 mmHg |
| `cp` | −1 to +2 levels | 0 to 2 levels | −0.64 |
| `exang` | −1 to 0 | 0 to 1 | −0.63 |
| `oldpeak` | −6.10 to +1.07 mm | 0 to 6.10 mm | −1.49 mm |
| `thalach` | −34 to +68 bpm | 0 to 68 bpm | +16.30 bpm |
| `slope` | −2 to 0 levels | 0 to 2 levels | −1.02 |
| `restecg` | −2 to 0 levels | 0 to 2 levels | −0.58 |

`exang`, `slope`, and `restecg` show zero worsening across all successful rows.

---

## 2. Non-flip cohort — improvement-focused causal recourse

Among the counterfactuals whose SCM intervention did **not** flip the label
(`target` stayed 1), how often do downstream symptoms still move in a
clinically-beneficial direction? Non-flip CFs per iteration: **155.1**
([150.0, 160.0]). Improvement is scored over the six downstream symptoms
(cp, restecg, thalach, exang, slope, oldpeak); `chol`/`trestbps` are
intervention-linked inputs shown for context.

| Recourse metric | Mean | 95% Interval |
|-----------------|------|--------------|
| Any improvement (≥1 symptom better) | 100.0% | [100.0, 100.0] |
| **IRR — strict** (≥1 better, none worse) | **36.2%** | [34.2, 37.7] |
| **IRR — lenient** (≥1 better, net ≥ 0) | **90.5%** | [90.0, 91.1] |
| Mean # symptoms improved / CF | 2.4 | [2.3, 2.4] |
| Mean # symptoms worsened / CF | 0.9 | [0.9, 1.0] |
| Mean net improvement / CF | +1.4 | [1.4, 1.5] |

### Per-symptom breakdown (non-flip cohort)

| Metric | ↓ Improve (%) | ↑ Worsen (%) | ↔ No Change (%) | Mode Before → After | Δ Mean |
|--------|--------------|-------------|-----------------|---------------------|--------|
| Resting BP (trestbps) | 67.0 | 32.1 | 0.9 | — | −1.26 mmHg |
| Chest Pain (cp) | 0.4 | 19.2 | 80.5 | 4 → 4 | — |
| Exang | 1.5 | 24.5 | 74.0 | 1 → 1 | — |
| ST Depression (oldpeak) | 76.7 | 23.3 | 0.0 | — | −0.51 mm |
| Max Heart Rate (thalach) | 80.3 | 19.3 | 0.4 | — | +10.48 bpm |
| ST Slope (slope) | 11.8 | 6.4 | 81.8 | 2 → 2 | — |
| Resting ECG (restecg) | 65.8 | 0.0 | 34.2 | 2 → 0 | — |

**Reading:** the continuous downstream symptoms (oldpeak, thalach, restecg, and
the indirectly-moved trestbps) improve in most non-flip cases, while the
categorical disease-severity markers (cp, exang) register ~0% improvement and
stay at their most severe modal categories — they are children of the disease
node and shift only when the predicted label itself changes, which the non-flip
subset excludes. Because `gcm.interventional_samples` redraws symptom noise
rather than abducting the patient's own, these non-flip shifts are
conditional-distribution shifts (regression toward the `target=1` mean), not
abduction-based individual counterfactuals — the same propagation used for the
flip-cohort metrics.

---

## 3. Manuscript correction map (do(both) → do(chol))

The manuscript's improvement-focused recourse table (**Table 6** / paras 223–226)
was generated with the SCM intervening on **both** chol and trestbps, which is
inconsistent with the rest of the paper (and that section's own prose) using
`do(chol)`. The tell was an implied 107.5 flips/iter (vs the main results' 82.9)
and a trestbps drop of −23.93 mmHg / 81.2% (a direct-intervention signature).
The corrected, consistent values:

| Quantity | Table 6 as printed (do both) | Corrected (do chol) |
|----------|------------------------------|---------------------|
| Non-flip CFs / iteration | 130.8 [125.0, 136.5] | **155.1 [150.0, 160.0]** |
| Implied flips / iteration | 107.5 | **82.8** (matches main results) |
| trestbps improve % / Δ | 81.2% / −23.93 mmHg | **67.0% / −1.26 mmHg** |
| oldpeak improve % / Δ | 78.1% / −0.51 mm | 76.7% / −0.51 mm |
| thalach improve % / Δ | 79.2% / +9.70 bpm | 80.3% / +10.48 bpm |
| restecg improve % | 66.4% | 65.8% |
| slope improve % | 9.5% | 11.8% |
| cp improve / worsen % | 0.0 / 19.8 | 0.4 / 19.2 |
| exang improve / worsen % | 0.0 / 23.9 | 1.5 / 24.5 |
| IRR strict / lenient | 35.9% / 90.8% | 36.2% / 90.5% |
| Net symptoms / CF | +1.4 | +1.4 |

Also update para 223's "130.8 non-flip per iteration" → **155.1**, and drop the
sentence attributing the extra flips to decision-boundary / library-version
effects — with `do(chol)` the re-score reproduces the main run's flip partition
(82.8 ≈ 82.9). The headline recourse conclusion (any-improvement 100%, IRR ~36%
strict / ~90% lenient, +1.4 net symptoms, continuous symptoms ease while
cp/exang do not) is unchanged.

---

*Source artifacts (gitignored, local):*
`fresh_cf_iterations_cholonly/aggregated_results/` (flip) and
`fresh_cf_iterations_cholonly/aggregated_results_recourse/` (non-flip).
Regenerate: `python src/pipeline/fresh_cf_pipeline.py --n_iterations 100 --output_base fresh_cf_iterations_cholonly`
then `python scripts/run_improvement_recourse.py --iterations_dir fresh_cf_iterations_cholonly`.
