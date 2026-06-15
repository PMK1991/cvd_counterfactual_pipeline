# Improvement-Focused Causal Recourse — Implementation Plan

Branch: `improvement-focussed-causal-recourse`

## 1. The idea

The main pipeline keeps a counterfactual (CF) only when the SCM intervention
**flips the disease label** (`target` 1 → 0). Everything else is discarded.
But an intervention on `chol` / `trestbps` can be clinically meaningful even when
it does **not** reverse the binary diagnosis: it may still push downstream
symptom variables (`oldpeak`, `exang`, `cp`, `thalach`, `slope`, `restecg`,
`trestbps`) in a beneficial direction.

**Improvement-focused causal recourse** = among the CFs whose `target` stayed 1,
quantify how often, and how much, the SCM-propagated downstream symptoms still
*improve*. This reframes recourse from a binary "did the label flip" to a graded
"did the patient's physiological state get better."

## 2. Can we use existing run results? — Yes

A completed run (`fresh_cf_iterations/`, 100 iterations) already persists
everything needed:

- `iteration_NNN/original/patient_*.csv` — the 48 true-positive high-risk patients.
- `iteration_NNN/counterfactuals/patient_*_cf_*.csv` — **all** DiCE proposals
  (5 per patient), flipped *and* non-flipped. This is the raw material.

What is **not** persisted: the SCM-propagated full symptom row for the
**non-flip** CFs. The original run computes it inside
`SCMAnalyzer.apply_scm_intervention()` but `analyze_iteration()` only keeps the
row when `validate_counterfactual()` returns True, then drops the rest.

The SCM step is **deterministic** (per patient–CF seed in `apply_scm_intervention`,
+ offline-loaded `model/scm_full.pkl`), so re-running *only* the SCM propagation
on the persisted DiCE CFs reproduces exactly the rows the original run discarded.

**Conclusion:** No new DiCE generation (the expensive part) is required. We
re-score the existing CFs through the SCM and this time **retain all rows**,
partitioning into flip / non-flip. New runs are only needed if we later want the
main pipeline to persist non-flip rows natively, or want a larger cohort.

This mirrors the existing no-SCM ablation pattern
(`scripts/run_unfiltered_ablation.py` + `unfiltered_scorer.py`), which already
re-scores persisted DiCE CFs without re-running DiCE.

## 3. Metric definitions

Per symptom, "improved" uses the clinically-beneficial direction already encoded
in `MetricsCalculator`:

| Feature   | Improvement = |
|-----------|---------------|
| trestbps  | decrease |
| chol      | decrease |
| oldpeak   | decrease |
| cp        | decrease (less severe chest pain) |
| exang     | 1 → 0 |
| slope     | decrease |
| restecg   | decrease |
| thalach   | increase (higher max heart rate) |

Over the **non-flip subset** (`cf_target == 1`):

1. **Per-symptom improvement / worsening / no-change %** — reuse
   `MetricsCalculator.compute_all_metrics()` unchanged (it is outcome-agnostic;
   it just needs the non-flip rows fed in).
2. **Improvement Recourse Rate (IRR)** — fraction of non-flip CFs with
   ≥1 symptom improved and 0 worsened (strict), plus a lenient variant
   (≥1 improved, net improvement ≥ 0).
3. **Net symptom-burden delta** — mean of (#improved − #worsened) per CF, and a
   standardized continuous version (sum of z-scored beneficial deltas) so
   magnitude, not just direction, is captured.
4. **Patient coverage** — fraction of the 48 patients for whom *some* non-flip CF
   achieves improvement, i.e. partial recourse is available even when no full
   flip exists.

All four reported with the existing CI machinery (`CIComputer`,
percentile 95% across the 100 iterations) for algorithmic-stability intervals.

## 4. Implementation steps

1. **`src/pipeline/recourse_analyzer.py`** (new). Subclass/wrap `SCMAnalyzer`;
   override iteration analysis to retain **every** scored row with a
   `flipped` boolean (`cf_target == 0`) instead of filtering. Output schema =
   existing `orig_*/cf_*/target/patient_id` + `flipped`, so `MetricsCalculator`
   works unchanged.
2. **`scripts/run_improvement_recourse.py`** (new, modeled on
   `run_unfiltered_ablation.py`). Walk `iteration_NNN/`, re-score CFs, split
   flip vs. non-flip, compute the §3 metrics on the non-flip subset, aggregate
   across iterations. Write to `fresh_cf_iterations/aggregated_results_recourse/`
   (sits beside `aggregated_results/` and `aggregated_results_no_scm/`).
3. **Metric additions** — add IRR / net-burden-delta helpers. Prefer a small
   `compute_recourse_metrics()` next to the existing per-symptom methods rather
   than touching the per-symptom ones, to keep the main pipeline untouched.
4. **Outputs** — `recourse_metrics.csv` (per-iteration), `recourse_ci.csv`
   (aggregated + CIs), `recourse_summary.md`, and a stacked
   improved/worsened/no-change bar chart per symptom for the non-flip cohort.
5. **(Optional) Persist-going-forward** — add a flag to the main pipeline so
   future runs also dump non-flip rows (e.g. `iteration_NNN/all_scored/`),
   removing the re-score step for later runs. Not required for the first result.

## 5. Validation / sanity checks

- Re-score the **flip** subset and confirm its metrics match the existing
  `aggregated_results/ci_results.csv` byte-for-byte → proves the re-score
  reproduces the original SCM propagation (determinism check).
- flip count + non-flip count == total DiCE CFs scored per iteration (accounting).
- Spot-check a handful of non-flip CFs by hand against the SCM samples.

## 6. Open questions for confirmation

- **Cohort:** non-flip CFs *of the current true-positive high-risk patients*
  (recommended — direct partial-recourse story), or broaden to all generated CFs?
- **"Improvement" threshold:** strict (no worsening allowed) vs. net-positive.
- **Reporting:** standalone section, or fold into the existing ablation/ results
  narrative for the rebuttal?

---

## 7. Implementation status (shipped on this branch)

Decisions taken: current true-positive high-risk cohort; report **both** strict
and lenient IRR; standalone output dir.

Delivered:

- `MetricsCalculator.compute_recourse_metrics()` — `recourse_n_cfs`,
  `recourse_any_improvement_pct`, `recourse_irr_strict_pct`,
  `recourse_irr_lenient_pct`, `recourse_mean_n_improved`,
  `recourse_mean_n_worsened`, `recourse_mean_net_improvement`, over the 6
  downstream symptoms (cp, restecg, thalach, exang, slope, oldpeak).
  `trestbps`/`chol` are intervention inputs and are **excluded** from the
  recourse score (shown only as context in the per-symptom table).
- `src/pipeline/recourse_analyzer.py` — `RecourseAnalyzer(SCMAnalyzer)` retains
  every scored CF with a `flipped` flag (`analyze_iteration_all`).
- `scripts/run_improvement_recourse.py` — re-scores a completed run, splits
  flip/non-flip, writes `aggregated_results_recourse/`
  (`all_iteration_metrics.csv`, `ci_results.csv`, `summary_report.md`,
  `recourse_summary.md`) and per-iteration `non_flip_recourse/`. Loads the `scm`
  block from `pipeline_config.yaml` (falls back to the deployed defaults —
  `chol_only` — if PyYAML is unavailable). Flags: `--iterations_dir`,
  `--graph_structure`, `--confidence_level`, `--max_iterations` (smoke runs
  write a `_smoke<N>` suffixed dir).

**Results (100 iterations, `do(chol)`, ~155 non-flip CFs/iter):** any-improvement
100%, IRR-strict **36.4%** [34.5, 38.2], IRR-lenient **90.3%** [89.9, 90.6], mean
net **+1.4** symptoms/CF. Continuous symptoms ease (oldpeak −0.52 mm, thalach
+10.5 bpm, restecg→normal 66%) while categorical disease markers (cp, exang) do
not improve. Mean flipped CFs/iter = **82.9**, matching the deployed run's
published flip count exactly.

## 8. Reproduction note & limitations

**Determinism is exact — confirmed.** Re-scoring must use the SAME
`intervention_targets` the run was generated under. The deployed runs use
`do(chol)` only (`pipeline_config.yaml` → `scm.intervention_targets: chol_only`).
With that config, re-scoring reproduces the original run's flip partition
exactly: `iteration_000` gives **87 flips** and the 100-iteration mean is
**82.9 flips/iter**, matching the deployed pipeline's published number — despite
the `InconsistentVersionWarning` from the `sklearn 1.6.1 → 1.8.0` pickle skew, so
the version drift has **no measurable effect** here, and the per-patient seeding
(unsalted float hashing) is fully deterministic. An earlier draft wrongly
attributed an 87 → 107 difference to version skew; that gap was entirely a config
bug — the analyzer defaulted to `intervention_targets='both'` (also intervening
on `trestbps`), compounded by PyYAML being absent in the run environment so the
YAML config silently failed to load. The script now carries the deployed
`chol_only` default independent of PyYAML and logs the resolved config at
startup.

**Limitation — non-flip "improvement" mixes effect with resampling.** Symptoms are
   children of `target`; `gcm.interventional_samples` *redraws* symptom noise
   rather than abducting the patient's own, so for non-flip CFs the continuous
   symptoms (oldpeak, thalach, restecg) partly regress toward the target=1
   conditional mean. This is the **same propagation the main pipeline uses**
   for flipped-CF symptom metrics (so the arm is methodologically consistent),
   but the graded "improvement" should be read as conditional-distribution
   shift, not abduction-based individual counterfactual change. Note the
   categorical disease-severity symptoms (cp, exang) do **not** improve in the
   non-flip cohort — they stay severe or worsen — which is the expected and
   honest signal.
