# Ablation Study: SCM-Filtered vs. Unfiltered Counterfactuals

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
python scripts/run_unfiltered_ablation.py --iterations_dir fresh_cf_iterations_cholonly
# writes fresh_cf_iterations_cholonly/aggregated_results_no_scm/
```

## Headline Result

| Quantity | SCM-filtered | Unfiltered (no SCM) |
|---|---|---|
| Accepted CFs / iteration | **82.8** [79.0, 87.5] | **160.1** [148.5, 170.5] |
| Target-flip / retention rate | **34.8%** [33.1, 36.7] | **67.3%** [62.5, 71.4] |
| Total accepted CFs (100 iter) | 8,282 | 16,012 |
| Accepted CFs / patient | 1.73 [1.65, 1.82] | 3.34 [3.09, 3.55] |
| Mean CFs generated / iteration | 237.9 | 237.9 |

**The SCM filter is ~2× more conservative.** Removing it nearly doubles the
acceptance rate (34.8% → 67.3%), because direct model scoring accepts *any*
DiCE proposal the classifier happens to like — including causally implausible
ones — whereas the SCM only keeps CFs whose propagation flips the outcome
**under the fitted SCM**. (The 67.3% is a model-only upper bound, not an
independent validation rate — see the design caveat above.)

## Effect on Recommended Feature Changes

Mean **improvement %** per diagnostic feature (95% CI), with the mean Δ for
continuous features:

| Feature | SCM-filtered improve % | No-SCM improve % | Δ mean (SCM) | Δ mean (no-SCM) |
|---|---|---|---|---|
| Resting BP (trestbps) | 59.5 [56.4, 62.3] | 61.4 [56.9, 65.9] | −3.31 mmHg | **−11.65 mmHg** |
| Max heart rate (thalach) | 75.5 [73.4, 77.3] | 95.4 [93.3, 97.5] | +16.29 bpm | **+43.07 bpm** |
| ST depression (oldpeak) | 71.8 [69.2, 75.2] | 77.7 [74.5, 81.2] | −1.49 | −0.96 |
| Chest pain (cp) | **87.7** [85.3, 89.2] | 51.5 [46.3, 56.0] | mode 4→3 | mode 4→**4** |
| ST slope (slope) | **93.8** [92.7, 94.3] | 42.2 [36.9, 48.4] | mode 2→1 | mode 2→1 |
| Exercise angina (exang) | 63.3 [60.2, 66.5] | 32.5 [26.9, 37.7] | mode 1→0 | mode 1→0 |
| Resting ECG (restecg) | 38.6 [35.9, 41.8] | 26.3 [21.5, 31.0] | mode 0→0 | mode 0→0 |

### Interpretation

The accepted rows of the two arms **contain different feature changes** — the
no-SCM rows are raw DiCE perturbations, the SCM rows are causally-propagated
consequences of the `chol`/`trestbps` intervention (see the estimand caveat in
**Design**). With that framing:

1. **The unfiltered arm makes large raw edits to continuous features — including
   downstream/diagnostic ones.** DiCE freely drops `trestbps` by ~11.7 mmHg and
   raises `thalach` by ~43 bpm to satisfy the classifier — roughly 3.5× and 2.6×
   the changes the SCM propagates (−3.3 mmHg, +16.3 bpm). Note `thalach` is a
   *downstream symptom* in the causal graph, yet the no-SCM arm moves it the
   most, because it is a cheap, high-leverage feature for the classifier — not
   necessarily a clinically coherent change.

2. **For the categorical disease-severity marker `cp` the pattern reverses: the
   unfiltered arm moves it far less than the SCM-propagated arm.** cp improves
   87.7% (SCM) vs 51.5% (no-SCM); its mode shifts under SCM (4→3) but stays put
   without it (4→4). Under the SCM these symptoms move *because the fitted model
   propagates the upstream intervention onto them*; the unfiltered model has no
   reason to touch `cp` when a continuous tweak already flips the score. `slope`
   improves more under SCM too (93.8% vs 42.2%), though its mode shifts (2→1) in
   both arms.

3. **Worsening is higher and more scattered without the SCM on features the SCM
   never worsens** — restecg 7.1% and slope 3.6% (both 0.0% under SCM) — i.e.
   unconstrained edits that move symptoms the wrong way. (The reverse-looking
   trestbps figure — worsened 40.3% SCM vs 22.2% no-SCM — is an artifact of the
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
- SCM-filtered results: `fresh_cf_iterations_cholonly/aggregated_results/`
- Unfiltered results: `fresh_cf_iterations_cholonly/aggregated_results_no_scm/`
  (`all_iteration_metrics.csv`, `ci_results.csv`, `summary_report.md`, and
  per-iteration `successful_unfiltered/` CSVs)
