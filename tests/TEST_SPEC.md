# Test Specification — CVD Counterfactual Pipeline

**Audience:** a coding agent implementing the test suite.
**Goal:** validate the whole pipeline *and* lock in the SCM serialization refactor
(`train_scm.py` + load-at-inference). Write tests with `pytest`. Do not change
production behaviour to make a test pass — if a test reveals a bug, stop and report it.

---

## 0. Ground rules

- **Framework:** `pytest`. Put everything under `tests/`. Mirror the source layout
  (`tests/pipeline/`, `tests/training/`, `tests/utils/`).
- **No network, no GPU.** Everything runs on CPU against the bundled CSV.
- **Determinism over flakiness.** The DiCE genetic algorithm is stochastic and the
  SCM `gcm.fit` draws on the global NumPy RNG. Tests must either (a) assert structural
  properties (counts, ranges, dtypes) rather than exact stochastic values, or
  (b) pin seeds and assert exact equality only where the code already guarantees it
  (SCM with `fit_seed=42`; SCM intervention with its per-patient derived seed).
- **Markers** (register in `pyproject.toml`/`pytest.ini`):
  - `unit` — fast, no model/SCM fitting (< 1 s each).
  - `integration` — fits a model/SCM or runs a mini end-to-end (seconds).
  - `slow` — anything over ~30 s; excluded from the default run.
  Default run: `pytest -m "not slow"`.
- **Tolerances:** exact equality for deterministic paths; for aggregate metrics
  compare with `pytest.approx(..., abs=1e-9)` after seeding, or document why a
  looser tolerance is used.

---

## 1. Known invariants (verified — use as oracles)

These were confirmed against the current code and the trained model. Treat them as
expected values; if a test disagrees, the regression is real.

| Quantity | Value | Source |
|---|---|---|
| Cleaned dataset rows | **707** | `DataLoader.load_clean_data()` = dropna → drop_duplicates → IQR filter on `chol`,`trestbps` per target class |
| Split | `test_size=0.2`, `random_state=42`, no stratify | `train_model.load_train_test_data()` |
| Train split rows | **565** | 707 × 0.8 |
| Test split rows | **142** | 707 × 0.2 |
| Inference cohort (TP high-risk, `y=1 & pred=1`) | **48** | scoring `xgb_pipeline.pkl` on the 142-row test set; matches `n_patients: 48` |
| SCM fit seed | **42** | `scm_analyzer` / `train_scm` |
| DiCE counterfactuals per patient | **5** (`total_cfs`) | `pipeline_config.yaml` |
| DiCE permitted ranges | `trestbps ∈ [100,120]`, `chol ∈ [150,200]` | config |
| Features varied by DiCE | `trestbps`, `chol` only | config |
| "Successful" CF criterion | SCM flips `target` 1 → 0 only | `SCMAnalyzer.validate_counterfactual` |
| Physiological constraints post-intervention | `oldpeak ≥ 0`, `cp ∈ [1,4]`, `slope ∈ [1,3]`, `restecg ∈ [0,2]`, `exang ∈ {0,1}` | `apply_scm_intervention` |
| Graph variants | `minimal`, `full`, `full_with_symptom_links`, `extended` | `SCMAnalyzer.GRAPH_VARIANTS` |
| Fitted SCM pickle round-trip | bit-identical interventional output | proven by spike |

---

## 2. Shared fixtures (`tests/conftest.py`)

- `clean_df` — `DataLoader(DATA).load_clean_data()`, session-scoped.
- `split` — `load_train_test_data()` results, session-scoped.
- `xgb_model` — `pickle.load("model/xgb_pipeline.pkl")`, session-scoped; `skip` with a
  clear message if the file is absent.
- `fitted_scm_full` — a `SCMAnalyzer(config={'graph_structure':'full','fit_seed':42})`
  with `initialize_analyzer()` called; session-scoped (fitting is the slow part — fit once).
- `golden_cf_pairs` — a small, **checked-in** set of (original patient row, DiCE CF
  suggestion) pairs saved as CSV under `tests/fixtures/`. These decouple SCM/metrics/CI
  tests from DiCE's randomness: SCM validation, metrics, and CI must be deterministic
  given fixed CF inputs.
- `tmp_iteration_dir` — builds the `original/` + `counterfactuals/` directory layout that
  `SCMAnalyzer.analyze_iteration` and `CIComputer` consume, from `golden_cf_pairs`.

Generate `golden_cf_pairs` once with a helper script (not at test time) and commit it.

---

## 3. Module-by-module requirements

### 3.1 `utils/dataLoader.py` — `unit`
- `load_data()` removes NaNs and duplicates; row count is monotonically non-increasing
  per step and `step_summary()` records each step.
- `load_clean_data()` returns **707** rows; column set unchanged; `target ∈ {0,1}`.
- `remove_outliers_iqr` only filters on `chol`/`trestbps` and never drops a whole class.
- `test_set_high_risk(...)` raises/warns if `test_size`/`random_state` differ from the
  pinned `MODEL_TEST_SIZE`/`MODEL_RANDOM_STATE` (guards split drift).

### 3.2 `training/train_model.py` — `integration`
- `load_train_test_data()` returns **565/142** train/test rows; the split is
  reproducible across two calls (same indices).
- Trained pipeline exposes `predict`; on the test set, TP high-risk count == **48**.
- Smoke: `train_and_export()` writes a non-empty `xgb_pipeline.pkl` to a tmp path.

### 3.3 `training/train_scm.py` (NEW) — `integration`
This is the heart of the refactor. Required tests:
- **Train-split fit:** with default `--fit-data train`, the data passed to `gcm.fit`
  has **565** rows and contains **none** of the 142 test indices (assert disjoint).
  Use the same `load_train_test_data()` source to get the held-out indices.
- **Artifact contents:** running for variant `full` writes `model/scm_full.pkl` whose
  unpickled artifact is a dict with keys `causal_model`, `graph_structure=='full'`,
  `fit_seed==42`, `fit_data=='train'`, `n_rows==565`, `data_sha256_16`, `versions`.
- **All variants:** `--all` writes one `scm_<variant>.pkl` per key in `GRAPH_VARIANTS`.
- **`--fit-data full`** fits on **707** rows (legacy path still available).
- **Determinism:** two runs with the same seed/variant produce artifacts whose
  interventional output is identical (compare via §3.4 intervention, not file bytes —
  pickle bytes may differ).

### 3.4 `pipeline/scm_analyzer.py` — `integration`
- **Graph construction:** each variant builds the expected edge set and node set; `full`
  excludes symptom-to-symptom links, `full_with_symptom_links` includes them.
- **Serialization equivalence (CRITICAL):** the SCM is fitted offline by
  `src/training/train_scm.py` (`fit_one()`); pickle+reload its artifact; run
  `apply_scm_intervention` on a fixed patient with a fixed seed on both the in-memory
  fitted model and the reloaded artifact; assert the returned DataFrames are **identical**
  (proves the serialize→load round-trip is lossless).
- **Load-only inference (CRITICAL):** after the refactor, `initialize_analyzer()` runs in a
  single mode — it loads `model/scm_<variant>.pkl` and never fits in-process:
  - loads the artifact when present (assert `gcm.fit` is patched and **not** called),
  - raises `FileNotFoundError` when the artifact is absent (no fallback fit),
  - raises `ValueError` for a mismatched artifact (wrong `graph_structure`, `fit_seed`, or
    graph edges) or a non-dict/`causal_model`-less artifact,
  - raises `RuntimeError` when the pickle cannot be unpickled,
  - `FreshCFPipeline.run_concurrent_pipeline()` preflights the load in the parent process so
    a missing/stale artifact fails fast instead of yielding zero-CF "error" iterations.
- **Intervention correctness:** `apply_scm_intervention` returns `orig_*`/`cf_*` columns
  for every graph node plus a scalar `target`; physiological constraints hold (see §1).
- **Validation:** `validate_counterfactual` is `True` only when `original_target==1` and
  the SCM `target` rounds to `0`; `False` for `None`/empty input.
- **`analyze_iteration`:** given `tmp_iteration_dir`, returns only the flipped CFs and
  writes `successful_counterfactuals.csv`.

### 3.5 `pipeline/dice_cf_generator.py` — `integration`
DiCE is stochastic — assert **structure**, not values:
- Generates exactly `total_cfs` (5) suggestions per patient (or documents fewer on
  timeout, and the timeout path returns gracefully).
- Every CF varies **only** `trestbps`/`chol`; all other features equal the original row.
- `trestbps ∈ [100,120]` and `chol ∈ [150,200]` for every CF.
- The model predicts the desired (low-risk) class on generated CFs (DiCE's own contract).

### 3.6 `pipeline/metrics_calculator.py` — `unit`
- On `golden_cf_pairs`, improvement/worsening/no-change percentages for each tracked
  feature (trestbps, chol, cp, exang, oldpeak, thalach, slope, restecg) sum to ~100%.
- Hand-computed expected values for 2–3 crafted rows match exactly (deterministic).
- Empty input yields a well-formed zero/empty result, not an exception.

### 3.7 `pipeline/ci_computer.py` — `unit`
- Percentile 95% CI on a known array equals the hand-computed `np.percentile` values.
- `lower ≤ point ≤ upper` for every metric; CI degenerates correctly with one iteration.
- `summary_report.md` is generated and contains a row per tracked metric.

### 3.8 `pipeline/ev_calculator.py` — `unit`
- The flip-odds → E-value-like index matches the VanderWeele–Ding formula on a hand
  example; document that this is the single-arm variant (per CLAUDE.md), and guard the
  edge cases (flip rate 0 and 1).

### 3.9 `pipeline/patient_bootstrap.py` — `integration`
- With a fixed seed and small `bootstrap_iterations`, CI bounds are reproducible and
  `lower ≤ upper`; resampling is at the patient cluster level (not row level).

### 3.10 `pipeline/sensitivity_analyzer.py` — `integration`/`slow`
- A single-parameter OAT sweep (e.g. `total_cfs` with 2 tiny values, 1 iter, 2 patients)
  produces a `comparison.csv` with one row per variant.
- `confidence_level` sweep recomputes CIs from baseline **without** re-running the
  pipeline (assert no new iteration dirs are created).
- `graph_structure` sweep picks up serialized variants when present (ties to §3.4).

### 3.11 `pipeline/cohort_flowchart.py` — `unit`
- Given a `cohort_counts.json`, renders a non-empty PNG and the numbers are consistent
  (each stage ≤ the previous; final stage == cohort size).

### 3.12 `pipeline/fresh_cf_pipeline.py` — `integration`/`slow`
- **Cohort selection:** the selected high-risk cohort has exactly **48** patients and
  every member satisfies `y==1 & pred==1`.
- **Config plumbing:** `scm_dir` default is `model`; per-worker init uses the load path.
- **Output contract:** a `--test_mode` run with tiny `n_iterations`/`n_patients` and
  `n_workers=1` produces the documented tree (`iteration_XXX/{original,counterfactuals,
  successful,metrics.json}` and `aggregated_results/{all_iteration_metrics.csv,
  ci_results.csv,summary_report.md,evalue.json,cohort_counts.json}`).
- **Worker isolation:** runs with `n_workers=1` and `n_workers=2` both complete and yield
  the same cohort and the same set of successful-CF counts per patient (SCM stage is
  deterministic; assert on the SCM-validated counts, not DiCE-internal randomness).

---

## 4. Refactor equivalence & regression (the key acceptance gate)

1. **SCM load == SCM fit.** (§3.4) The loaded artifact must reproduce the built model's
   interventional output exactly. This guarantees the refactor changes *only* where the
   SCM comes from, not what it does.
2. **Backward compatibility.** With no `.pkl` artifacts present, the pipeline behaves
   exactly as before (fits per worker). Test by deleting/ignoring artifacts.
3. **Scientific-change flag.** `--fit-data train` (565 rows) is the corrected,
   leakage-free fit and **will** change validity numbers vs. the legacy full-data SCM.
   Add an explicit test asserting the train-fit SCM is fit on 565 rows disjoint from the
   142 test patients. Do **not** write a test asserting train-fit results equal
   full-fit results — they are expected to differ.
4. **Golden regression (decoupled from DiCE).** Using `golden_cf_pairs` as fixed inputs,
   snapshot the SCM-validation → metrics → CI outputs to `tests/fixtures/golden_metrics.json`
   and assert future runs match exactly. This catches unintended changes without DiCE flakiness.

---

## 5. How to run

```bash
# fast suite (default for CI and local dev)
pytest -m "not slow" -q

# include integration (fits a model/SCM, mini end-to-end)
pytest -m "not slow or integration" -q

# everything, including the slow end-to-end and sensitivity sweeps
pytest -q

# a single module
pytest tests/pipeline/test_scm_analyzer.py -q
```

Prerequisites: `model/xgb_pipeline.pkl` must exist (run `python src/training/train_model.py`);
SCM-load tests need `python src/training/train_scm.py --all` to have produced
`model/scm_*.pkl`. Tests that need these must `pytest.skip` with an actionable message
when the artifacts are missing, never silently pass.

---

## 6. CI & coverage

- Add a GitHub Actions (or equivalent) job: set up the env from `requirements.txt`,
  cache the trained model, run `pytest -m "not slow"` on push, and the full suite nightly.
- Coverage target: **≥ 85%** line coverage on `src/pipeline/` and `src/training/`
  (`pytest --cov=src --cov-report=term-missing`). Legacy code under `src/legacy/` is
  out of scope.
- Treat warnings from version mismatches (pickle/sklearn/dowhy) as a signal: pin the
  versions in the test env to match `mtech-env.yml`; a load failure due to version skew
  is a real risk this suite must surface, not hide.

---

## 7. Acceptance criteria (definition of done)

- [ ] `pytest -m "not slow"` is green locally and in CI.
- [ ] §3.4 serialization-equivalence and load-then-fallback tests pass.
- [ ] Cohort-size test asserts exactly 48; split tests assert 565/142.
- [ ] `train_scm.py` train-fit test proves the SCM never sees the 142 test patients.
- [ ] Golden regression snapshot committed and enforced.
- [ ] Coverage ≥ 85% on `src/pipeline/` and `src/training/`.
- [ ] No test mutates files under `model/` or the repo data; all writes go to `tmp_path`.
