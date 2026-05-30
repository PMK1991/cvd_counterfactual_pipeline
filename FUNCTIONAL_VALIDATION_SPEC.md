# Functional & Results Validation Spec — CVD Counterfactual Pipeline

**Audience:** a coding agent tasked with validating that the pipeline *runs* and
*produces correct results*. This is **not** a unit-test assignment — do not write a
`pytest` suite. Instead, execute the pipeline end-to-end, inspect the artifacts it
writes, and report pass/fail against the checks below. Use throwaway scripts or shell
one-liners to inspect outputs; do not modify production code to make a check pass. If a
check fails, capture the error and the relevant log lines and report — do not paper over it.

---

## 1. Environment

- Recreate the pinned environment (`mtech-env.yml`, or `requirements.txt`):
  `dice-ml==0.11`, `dowhy`, `numpy==1.26.4`, `pandas==1.5.3`, `xgboost`, `scikit-learn`.
  Version skew is a real failure mode — if the trained `model/xgb_pipeline.pkl` was
  pickled under a different sklearn/xgboost, loading may warn or break. Treat that as a
  finding, not something to silence.
- Required artifacts before running:
  - `data/heart_statlog_cleveland_hungary_final.csv` (bundled).
  - `model/xgb_pipeline.pkl` — if absent, run `python src/training/train_model.py` first.
- Run everything from the repo root so relative paths (`data/`, `model/`) resolve.

---

## 2. Known-good reference values (result oracles)

Validate results against these. They were derived from the cleaned data and the trained
model; a mismatch is a real regression, not a tolerance issue.

| Quantity | Expected |
|---|---|
| Cleaned dataset rows (`load_clean_data`) | 707 |
| Train / test split (`test_size=0.2`, `random_state=42`) | 565 / 142 |
| Inference cohort = TP high-risk (`y=1 & pred=1`) | 48 |
| DiCE counterfactuals per patient | 5 |
| DiCE intervention ranges | `trestbps ∈ [100,120]`, `chol ∈ [150,200]` |
| Features DiCE may change | `trestbps`, `chol` only |
| "Successful" counterfactual | SCM flips `target` 1 → 0 |
| Physiological constraints on SCM output | `oldpeak ≥ 0`, `cp ∈ [1,4]`, `slope ∈ [1,3]`, `restecg ∈ [0,2]`, `exang ∈ {0,1}` |
| SCM fit seed | 42 |
| Graph variants | `minimal`, `full`, `full_with_symptom_links`, `extended` |

---

## 3. Functional checks — does it run?

Run each command and record exit status, wall-clock time, and whether the expected
artifacts appear. All of these must complete without an unhandled exception.

**F1 — Test mode (primary smoke run).**
```bash
python src/pipeline/fresh_cf_pipeline.py --test_mode
```
Expect: completes in ~5–10 min; creates `fresh_cf_iterations_test/` with per-iteration
folders and an `aggregated_results/` folder. No iteration ends in `{'error': ...}`.

**F2 — Single-worker determinism run.**
```bash
python src/pipeline/fresh_cf_pipeline.py --test_mode --n_workers 1
```
Expect: same cohort and the same per-patient *successful-CF counts* as F1 (the SCM stage
is seeded/deterministic; DiCE generation is not, so do not expect identical CF values).

**F3 — Train the SCM artifact (refactor path, if `train_scm.py` is present).**
```bash
python src/training/train_scm.py --all
```
Expect: one `model/scm_<variant>.pkl` per graph variant; each unpickles to a dict with
`graph_structure`, `fit_seed=42`, `fit_data`, `n_rows`, `versions`.

**F4 — Sensitivity analysis (quick variant).**
```bash
python src/pipeline/fresh_cf_pipeline.py --sensitivity \
  --sensitivity_iterations 3 --sensitivity_patients 5 --sensitivity_params total_cfs
```
Expect: `sensitivity_results/total_cfs/comparison.csv` with one row per variant and a
`sensitivity_report.md`.

**F5 — Optional patient bootstrap.**
```bash
python src/pipeline/fresh_cf_pipeline.py --test_mode --run_patient_bootstrap --bootstrap_iterations 50
```
Expect: `aggregated_results/patient_bootstrap_ci.csv` with `lower ≤ upper` on every row.

---

## 4. Output-tree checks — are the artifacts there?

After F1, confirm the documented structure exists and is non-empty:

```
fresh_cf_iterations_test/
  iteration_000/
    original/patient_*.csv
    counterfactuals/patient_*_cf_*.csv
    successful/successful_counterfactuals.csv
    metrics.json
  ...
  aggregated_results/
    all_iteration_metrics.csv
    ci_results.csv
    summary_report.md
    evalue.json
    cohort_counts.json
  fresh_cf_pipeline.log
```

Checks:
- Every `iteration_XXX/` has all four members; `metrics.json` parses and has success counts.
- `summary_report.md` is human-readable and has a line per tracked metric.
- `fresh_cf_pipeline.log` contains no unhandled tracebacks.

---

## 5. Results-correctness checks — are the numbers right?

Inspect the written files (e.g. with `pandas`/`jq`) and verify:

**R1 — Cohort.** In a full run (or from the cohort accounting), the selected cohort is
**48** patients and `cohort_counts.json` is internally consistent (each funnel stage ≤ the
previous; final stage == cohort size). In `--test_mode` the cohort is the debug cap
(`--n_patients`), so assert ≤ cap and every selected patient has `target==1`.

**R2 — Counterfactual generation.** For sampled `counterfactuals/patient_*_cf_*.csv`:
every CF changes only `trestbps`/`chol` vs. its `original/` row, and both lie within the
permitted ranges.

**R3 — SCM validation contract.** Every row in `successful_counterfactuals.csv`
corresponds to a `target` that flipped 1 → 0. No "successful" CF should have `cf_target==1`.

**R4 — Physiological sanity.** Across SCM outputs, the constraints in §2 hold (no negative
`oldpeak`, categorical features within their valid integer ranges).

**R5 — Metrics coherence.** In `all_iteration_metrics.csv`, for each tracked feature the
improvement/worsening/no-change percentages sum to ~100% and lie in [0,100].

**R6 — Confidence intervals.** In `ci_results.csv`, every metric has `lower ≤ point ≤ upper`
and bounds within [0,100] (or the metric's natural range). CIs should narrow as the number
of iterations grows (spot-check test mode vs. a longer run).

**R7 — E-value.** `evalue.json` parses; the index is finite and ≥ 1 where defined; note in
the report that this is the single-arm flip-odds variant (per project docs), not the
published two-arm E-value.

---

## 6. Refactor validation — functional (only if `train_scm.py` + load path are wired)

These confirm the serialization refactor is behaviour-preserving, by running, not by
unit-testing:

**X1 — Load == fit equivalence.** Run F1 once with `model/scm_*.pkl` present (load path)
and once with those files removed/renamed (fit-per-worker fallback), holding all seeds and
config equal and using the **same DiCE counterfactuals** as input to the SCM stage (reuse
the `counterfactuals/` from the first run so DiCE randomness is excluded). The set of
SCM-validated successful CFs must be **identical** between the two runs.

**X2 — Backward compatibility.** With no artifacts present, the pipeline must run exactly as
before (fallback fit). Confirm via the log that it fitted the SCM, and that outputs are
well-formed.

**X3 — Leakage-free fit (scientific check, expected to *differ*).** Confirm `train_scm.py`
with `--fit-data train` fits the SCM on **565** rows that exclude all **142** test patients.
Because the legacy default fit on the full 707 rows, the validity numbers **will change** —
do not flag this as a regression; flag it only if the fit row count or test-disjointness is
wrong.

---

## 7. Deliverable — the validation report

Produce `VALIDATION_REPORT.md` containing:
- Environment used (package versions; note any skew from the pinned versions).
- A table of F1–F5 and X1–X3 with pass/fail, runtime, and the command run.
- A table of R1–R7 with the observed value vs. the expected oracle.
- For any failure: the error, the offending log/file excerpt, and the smallest repro command.
- A one-line overall verdict: does the pipeline run, and are the results correct?

---

## 8. Acceptance criteria (definition of done)

- [ ] F1 (test mode) completes with no errored iterations and the full output tree.
- [ ] R1–R7 all match the oracles in §2 (cohort, CF ranges, 1→0 flips, constraints, metric
      coherence, valid CIs, finite E-value).
- [ ] If the refactor is present: X1 shows load == fit, X2 shows the fallback still works,
      X3 confirms the 565-row test-disjoint fit.
- [ ] `VALIDATION_REPORT.md` is written and gives a clear overall verdict.

> Note: this supersedes the `pytest`-based `tests/TEST_SPEC.md` for the purpose of
> "test the pipeline for functionality and results." Keep that file only if a unit suite is
> wanted later.
