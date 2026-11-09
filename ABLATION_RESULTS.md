# Ablation Study: SCM-Filtered vs. Unfiltered Counterfactuals

_Refreshed 2026-08-02: re-quoted from the canonical `fresh_cf_iterations/` run (deterministically
seeded). The previous `fresh_cf_iterations_cholonly/` run is archived as
`fresh_cf_iterations_archive_published_20260607/`; the two agree to within noise on every quantity
below (see `PAIRED_ABLATION_BOOTSTRAP.md`)._

**Reviewer 3, Comment 4b** — does the Structural Causal Model (SCM) validation
layer materially change which counterfactuals (CFs) are accepted, and the
resulting recourse recommendations?

This ablation removes the SCM and replaces it with **direct model scoring**: a
DiCE-proposed CF is accepted iff the deployed prediction model itself predicts
class 0 (low risk) for it. Everything upstream is held fixed.

## Design

- **Same cohort, same candidate CFs.** Both arms re-use the *identical* DiCE
  counterfactuals from the leakage-free 100-iteration cholesterol-only run
  (48 test-set true-positive high-risk patients, 237.9 CFs generated per
  iteration on average). Only the **acceptance criterion** differs.
- **SCM arm (filtered):** accept a CF if DoWhy `gcm` interventional propagation
  of the cholesterol intervention — **`do(chol)`** — flips `target` 1 → 0
  **under the fitted SCM**. Every other variable (incl. `trestbps`, via the
  `chol → trestbps` edge) propagates through the structural equations; nothing
  but `chol` is clamped. The reported `cf_*` symptom values are the SCM's
  **causally-propagated** estimates (and are physiologically clipped, e.g.
  `slope` ∈ [1, 3], `oldpeak` ≥ 0).
- **No-SCM arm (unfiltered):** accept a CF if the XGBoost model predicts
  `target = 0`. The reported `cf_*` values are **DiCE's raw proposals** (DiCE
  varies all features freely — only `chol` is range-constrained to [150, 200] —
  with no physiological clipping).
- Both arms feed the same `MetricsCalculator` / `CIComputer`; 95% percentile
  CIs are over the 100 iterations.

> **This is a candidate-filter ablation, not a regenerated no-SCM pipeline, and
> the no-SCM arm is intentionally circular.** DiCE generated these candidates by
> optimising against this *same* classifier, so direct model scoring is not an
> independent validity check — it is the **model-only acceptance upper bound**
> (the share of DiCE candidates that still score class 0 when reloaded and
> re-predicted). The two arms therefore also report **different estimands**: SCM
> `cf_*` are causal consequences of the `do(chol)` intervention, while no-SCM `cf_*`
> are arbitrary model-facing edits across all features, measured over different
> accepted subsets. The per-feature deltas below describe *what the accepted rows
> contain*, not a like-for-like causal-recourse comparison.

Reproduce with:

```bash
python scripts/run_unfiltered_ablation.py --iterations_dir fresh_cf_iterations
# writes fresh_cf_iterations/aggregated_results_no_scm/
```

## Headline Result

| Quantity | SCM-filtered | Unfiltered (no SCM) |
|---|---|---|
| Accepted CFs / iteration | **82.5** [78.0, 87.0] | **159.5** [148.0, 167.5] |
| Target-flip / retention rate | **34.7%** [32.7, 36.8] | **67.0%** [62.0, 70.4] |
| Total accepted CFs (100 iter) | 8,255 | 15,946 |
| Accepted CFs / patient | 1.72 [1.62, 1.81] | 3.32 [3.08, 3.49] |
| Mean CFs generated / iteration | 237.9 | 237.9 |

**The SCM filter is ~2× more conservative.** Removing it nearly doubles the
acceptance rate (34.7% → 67.0%), because direct model scoring accepts *any*
DiCE proposal the classifier happens to like — including causally implausible
ones — whereas the SCM only keeps CFs whose propagation flips the outcome
**under the fitted SCM**. (The 67.0% is a model-only upper bound, not an
independent validation rate — see the design caveat above.)

## Effect on Recommended Feature Changes

Mean **improvement %** per diagnostic feature (95% CI), with the mean Δ for
continuous features:

| Feature | SCM-filtered improve % | No-SCM improve % | Δ mean (SCM) | Δ mean (no-SCM) |
|---|---|---|---|---|
| Resting BP (trestbps) | 59.6 [56.7, 63.0] | 61.6 [56.4, 67.1] | −3.31 mmHg | **−11.63 mmHg** |
| Max heart rate (thalach) | 75.4 [73.1, 77.6] | 95.4 [93.2, 97.4] | +16.41 bpm | **+43.02 bpm** |
| ST depression (oldpeak) | 71.6 [69.1, 74.0] | 77.3 [73.7, 80.9] | −1.49 | −0.96 |
| Chest pain (cp) | **87.8** [85.9, 89.6] | 51.9 [46.8, 57.2] | mode 4→3 | mode 4→**4** |
| ST slope (slope) | **93.8** [92.5, 94.3] | 42.2 [38.0, 48.2] | mode 2→1 | mode 2→1 |
| Exercise angina (exang) | 63.1 [60.0, 66.3] | 32.4 [27.4, 37.8] | mode 1→0 | mode 1→0 |
| Resting ECG (restecg) | 38.2 [35.5, 40.7] | 26.5 [22.1, 31.6] | mode 0→0 | mode 0→0 |

### Interpretation

The accepted rows of the two arms **contain different feature changes** — the
no-SCM rows are raw DiCE perturbations, the SCM rows are causally-propagated
consequences of the `chol`/`trestbps` intervention (see the estimand caveat in
**Design**). With that framing:

1. **The unfiltered arm makes large raw edits to continuous features — including
   downstream/diagnostic ones.** DiCE freely drops `trestbps` by ~11.6 mmHg and
   raises `thalach` by ~43 bpm to satisfy the classifier — roughly 3.5× and 2.6×
   the changes the SCM propagates (−3.3 mmHg, +16.4 bpm). Note `thalach` is a
   *downstream symptom* in the causal graph, yet the no-SCM arm moves it the
   most, because it is a cheap, high-leverage feature for the classifier — not
   necessarily a clinically coherent change.

2. **For the categorical disease-severity marker `cp` the pattern reverses: the
   unfiltered arm moves it far less than the SCM-propagated arm.** cp improves
   87.8% (SCM) vs 51.9% (no-SCM); its mode shifts under SCM (4→3) but stays put
   without it (4→4). Under the SCM these symptoms move *because the fitted model
   propagates the upstream intervention onto them*; the unfiltered model has no
   reason to touch `cp` when a continuous tweak already flips the score. `slope`
   improves more under SCM too (93.8% vs 42.2%), though its mode shifts (2→1) in
   both arms.

3. **Worsening is higher and more scattered without the SCM on features the SCM
   never worsens** — restecg 7.6% and slope 3.9% (both 0.0% under SCM) — i.e.
   unconstrained edits that move symptoms the wrong way. (The reverse-looking
   trestbps figure — worsened 40.3% SCM vs 21.8% no-SCM — is an artifact of the
   SCM's smaller, bidirectional propagated changes vs. DiCE's one-directional
   larger drop.)

## Conclusion

The SCM is **not cosmetic**. It (i) roughly halves the acceptance rate by
rejecting CFs that flip the classifier without flipping under the fitted causal
model, and (ii) yields accepted rows whose feature changes are causally
propagated across the symptom set rather than concentrated in whichever
continuous inputs are cheapest for the classifier. The unfiltered arm's higher
retention (a model-only upper bound) and headline "improvements" (e.g. 95%
thalach) overstate benefit because they reflect causally ungrounded, unclipped
perturbations — which is the motivation for the SCM-filtered design. A stronger,
strictly paired recourse claim would require comparing raw-DiCE vs. SCM-propagated
values on the *same* candidate CFs; this ablation establishes the acceptance-rate
and accepted-row-composition differences.

## Files

- Code: `src/pipeline/unfiltered_scorer.py`, `scripts/run_unfiltered_ablation.py`
  (or the faster, checkpointed `scripts/run_unfiltered_ablation_fast.py`)
- SCM-filtered results: `fresh_cf_iterations/aggregated_results/`
- Unfiltered results: `fresh_cf_iterations/aggregated_results_no_scm/`
  (`all_iteration_metrics.csv`, `ci_results.csv`, `summary_report.md`, and
  `successful_counterfactuals_unfiltered_all.csv`)
