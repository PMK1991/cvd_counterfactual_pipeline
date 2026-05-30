# Ablation Study: SCM-Filtered vs. Unfiltered Counterfactuals

**Reviewer 3, Comment 4b** — does the Structural Causal Model (SCM) validation
layer materially change which counterfactuals (CFs) are accepted, and the
resulting recourse recommendations?

This ablation removes the SCM and replaces it with **direct model scoring**: a
DiCE-proposed CF is accepted iff the deployed prediction model itself predicts
class 0 (low risk) for it. Everything upstream is held fixed.

## Design

- **Same cohort, same candidate CFs.** Both arms re-use the *identical* DiCE
  counterfactuals from the leakage-free 100-iteration run (48 test-set
  true-positive high-risk patients, 238.3 CFs generated per iteration on
  average). Only the **acceptance criterion** differs.
- **SCM arm (filtered):** accept a CF if DoWhy `gcm` interventional propagation
  of the actionable edits (`chol`, `trestbps`) flips `target` 1 → 0 **under the
  fitted SCM**. The reported `cf_*` symptom values are the SCM's
  **causally-propagated** estimates (and are physiologically clipped, e.g.
  `slope` ∈ [1, 3], `oldpeak` ≥ 0).
- **No-SCM arm (unfiltered):** accept a CF if the XGBoost model predicts
  `target = 0`. The reported `cf_*` values are **DiCE's raw proposals** (DiCE
  varies all features freely, with no physiological clipping).
- Both arms feed the same `MetricsCalculator` / `CIComputer`; 95% percentile
  CIs are over the 100 iterations.

> **This is a candidate-filter ablation, not a regenerated no-SCM pipeline, and
> the no-SCM arm is intentionally circular.** DiCE generated these candidates by
> optimising against this *same* classifier, so direct model scoring is not an
> independent validity check — it is the **model-only acceptance upper bound**
> (the share of DiCE candidates that still score class 0 when reloaded and
> re-predicted). The two arms therefore also report **different estimands**: SCM
> `cf_*` are causal consequences of a 2-feature intervention, while no-SCM `cf_*`
> are arbitrary model-facing edits across all features, measured over different
> accepted subsets. The per-feature deltas below describe *what the accepted rows
> contain*, not a like-for-like causal-recourse comparison.

Reproduce with:

```bash
python scripts/run_unfiltered_ablation.py
# writes fresh_cf_iterations/aggregated_results_no_scm/
```

## Headline Result

| Quantity | SCM-filtered | Unfiltered (no SCM) |
|---|---|---|
| Accepted CFs / iteration | **82.9** [76.0, 88.0] | **157.5** [148.5, 166.5] |
| Target-flip / retention rate | **34.8%** [32.7, 36.7] | **66.1%** [62.3, 69.7] |
| Total accepted CFs (100 iter) | 8,286 | 15,750 |
| Mean CFs generated / iteration | 238.3 | 238.3 |

**The SCM filter is ~2× more conservative.** Removing it nearly doubles the
acceptance rate (34.8% → 66.1%), because direct model scoring accepts *any*
DiCE proposal the classifier happens to like — including causally implausible
ones — whereas the SCM only keeps CFs whose propagation flips the outcome
**under the fitted SCM**. (The 66.1% is a model-only upper bound, not an
independent validation rate — see the design caveat above.)

## Effect on Recommended Feature Changes

Mean **improvement %** per diagnostic feature (95% CI), with the mean Δ for
continuous features:

| Feature | SCM-filtered improve % | No-SCM improve % | Δ mean (SCM) | Δ mean (no-SCM) |
|---|---|---|---|---|
| Resting BP (trestbps) | 60.3 [57.0, 64.6] | 79.7 [77.1, 82.4] | −3.48 mmHg | **−23.35 mmHg** |
| Max heart rate (thalach) | 75.0 [72.1, 77.3] | 95.5 [93.2, 97.5] | +16.15 bpm | **+42.79 bpm** |
| ST depression (oldpeak) | 72.8 [69.6, 77.3] | 78.3 [74.4, 81.4] | −1.51 | −0.97 |
| Chest pain (cp) | **87.5** [85.6, 88.6] | 50.5 [45.2, 56.1] | mode 4→3 | mode 4→**4** |
| ST slope (slope) | **93.7** [92.4, 94.3] | 41.8 [36.2, 47.5] | mode 2→1 | mode 2→**2** |
| Exercise angina (exang) | 62.5 [58.7, 65.3] | 31.9 [26.3, 35.9] | mode 1→0 | mode 1→0 |
| Resting ECG (restecg) | 38.6 [35.9, 42.3] | 26.1 [22.1, 30.5] | mode 0→0 | mode 0→0 |

### Interpretation

The accepted rows of the two arms **contain different feature changes** — the
no-SCM rows are raw DiCE perturbations, the SCM rows are causally-propagated
consequences of the `chol`/`trestbps` intervention (see the estimand caveat in
**Design**). With that framing:

1. **The unfiltered arm makes large raw edits to continuous features — including
   downstream/diagnostic ones.** DiCE freely drops `trestbps` by ~23 mmHg and
   raises `thalach` by ~43 bpm to satisfy the classifier — roughly 6× and 2.5×
   the changes the SCM propagates (−3.5 mmHg, +16 bpm). Note `thalach` is a
   *downstream symptom* in the causal graph, yet the no-SCM arm moves it the
   most, because it is a cheap, high-leverage feature for the classifier — not
   necessarily a clinically coherent change.

2. **For categorical symptoms the pattern reverses: the unfiltered arm moves
   `cp`/`slope` less than the SCM-propagated arm.** cp improves 87.5% (SCM) vs
   50.5% (no-SCM) and slope 93.7% vs 41.8%; their modes shift under SCM (4→3,
   2→1) but stay put without it (4→4, 2→2). Under the SCM these symptoms move
   *because the fitted model propagates the upstream intervention onto them*; the
   unfiltered model has no reason to touch them when a continuous tweak already
   flips the score. (Caveat: no-SCM `slope` is unclipped — DiCE emits `slope = 0`
   rows that the SCM arm would clip to [1, 3], so some no-SCM slope "improvements"
   reflect an encoding/scale difference, not a clinically comparable change.)

3. **Worsening is higher and more scattered without the SCM on features the SCM
   never worsens** — restecg 7.2% and slope 3.2% (both 0.0% under SCM) — i.e.
   unconstrained edits that move symptoms the wrong way. (The reverse-looking
   trestbps figure — worsened 39.5% SCM vs 2.4% no-SCM — is an artifact of the
   SCM's smaller, bidirectional propagated changes vs. DiCE's one-directional
   large drop.)

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
- SCM-filtered results: `fresh_cf_iterations/aggregated_results/`
- Unfiltered results: `fresh_cf_iterations/aggregated_results_no_scm/`
  (`all_iteration_metrics.csv`, `ci_results.csv`, `summary_report.md`, and
  per-iteration `successful_unfiltered/` CSVs)
