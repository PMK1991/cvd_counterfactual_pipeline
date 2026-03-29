# TODO - Tomorrow's Implementation Checklist

**Date:** 2026-02-11
**Estimated Time:** 6-10 hours (depending on phases completed)
**Status:** Not Started

---

## Pre-Implementation (15 min)

- [ ] Read `OPTIMIZATION_PLAN.md` sections 1-4 (critical fixes)
- [ ] Verify environment: `conda activate base` or appropriate env
- [ ] Check current directory: `pwd` should be `.../cvd_counterfactual_pipeline`
- [ ] Create backups:
  ```bash
  cp fresh_cf_pipeline.py fresh_cf_pipeline.py.bak
  cp scm_analyzer.py scm_analyzer.py.bak
  cp dice_cf_generator.py dice_cf_generator.py.bak
  cp confidence_interval_analysis.py confidence_interval_analysis.py.bak
  ```
- [ ] Clean existing temp directories: `rm -rf temp_run_*`

---

## Phase 1: Critical Fixes (4-6 hours)

### 1.1 Fix Missing Causal Model (3-4 hours) 🔴 CRITICAL
**File:** `scm_analyzer.py`
**Reference:** `OPTIMIZATION_PLAN.md` - Section 3

- [ ] Review current `scm_analyzer.py:56` - confirm bug exists
- [ ] Add `import networkx as nx` and `from dowhy import gcm`
- [ ] Add `self.causal_model = None` to `__init__()`
- [ ] Implement `_build_causal_model()` method (copy from plan)
- [ ] Update `_default_config()` to add `train_data_path` and set `n_samples: 1`
- [ ] Modify `initialize_analyzer()` to call `_build_causal_model()`
- [ ] Add `apply_scm_intervention()` method (replaces reliance on counterfactualAnalyzer)
- [ ] Test initialization:
  ```python
  from scm_analyzer import SCMAnalyzer
  analyzer = SCMAnalyzer()
  analyzer.initialize_analyzer()
  print("✓ Causal model initialized successfully")
  ```
- [ ] **Decision Point:** Review causal graph edges - do they match CVD domain knowledge?

### 1.2 Reduce SCM Samples (15 min) ⚡ QUICK WIN
**File:** `scm_analyzer.py`
**Reference:** `OPTIMIZATION_PLAN.md` - Section 4

- [ ] Already done in 1.1 if following plan (`n_samples: 1` in `_default_config()`)
- [ ] Verify `apply_scm_intervention()` uses `num_samples_to_draw=1`
- [ ] Optional: Add `estimate_cf_uncertainty()` method for future use

### 1.3 Temp Directory Cleanup (30 min)
**File:** `confidence_interval_analysis.py`
**Reference:** `OPTIMIZATION_PLAN.md` - Section 1

- [ ] Add `import tempfile` at top
- [ ] Find `run_single_iteration()` method
- [ ] Replace `output_dir = f"temp_run_{run_id}"` with:
  ```python
  with tempfile.TemporaryDirectory(prefix=f"temp_run_{run_id}_") as output_dir:
  ```
- [ ] Indent all iteration code inside context manager
- [ ] Test: No temp dirs should remain after run

### 1.4 End-to-End Test (30 min)
- [ ] Run test mode:
  ```bash
  python fresh_cf_pipeline.py --test_mode 2>&1 | tee test_phase1.log
  ```
- [ ] **Success Criteria:**
  - No errors in log
  - Pipeline completes successfully
  - Creates `fresh_cf_iterations_test/` directory
  - Contains `aggregated_results/` with CSV and report
  - **No temp_run_* directories remain**
- [ ] Check memory usage during run (Task Manager / htop)
- [ ] **If tests fail:** Review errors, fix, re-test before proceeding

---

## Phase 2: Performance Boost (3-4 hours)

### 2.1 Model Loading Optimization (2 hours)
**File:** `fresh_cf_pipeline.py`
**Reference:** `OPTIMIZATION_PLAN.md` - Section 2

- [ ] Add globals: `_worker_dice_gen = None`, `_worker_scm_analyzer = None`
- [ ] Implement `init_worker(model_path, data_path, dice_config, scm_config)`
- [ ] Implement `worker_process_iteration(args)` to replace `run_single_iteration()`
- [ ] Update `run_concurrent_pipeline()`:
  - Add `initializer=init_worker` to `ProcessPoolExecutor`
  - Add `initargs=(...)` with config paths
  - Convert patients to `to_dict('records')`
  - Use `worker_process_iteration` instead of `self.run_single_iteration`
- [ ] Test with test mode: `python fresh_cf_pipeline.py --test_mode`
- [ ] **Verification:** Should see "Worker XXXX initialized" messages in log

### 2.2 In-Memory Pipeline (2-3 hours)
**File:** `fresh_cf_pipeline.py`
**Reference:** `OPTIMIZATION_PLAN.md` - Section 5

- [ ] Implement `worker_process_iteration_inmemory(args)` (copy from plan)
- [ ] Add `in_memory` parameter to `run_concurrent_pipeline()` (default True)
- [ ] Update worker selection:
  ```python
  worker_fn = (worker_process_iteration_inmemory if in_memory
              else worker_process_iteration)
  ```
- [ ] Test in-memory mode:
  ```bash
  python fresh_cf_pipeline.py --test_mode 2>&1 | tee test_inmemory.log
  ```
- [ ] **Comparison Test:** Run both modes, verify results match:
  ```bash
  # Disk mode
  python fresh_cf_pipeline.py --n_iterations 5 --n_patients 5 --n_workers 2
  # In-memory mode (should be default)
  # Compare aggregated_results/all_iteration_metrics.csv
  ```

### 2.3 Benchmark Test (1 hour)
- [ ] Run 20 iterations with optimizations:
  ```bash
  time python fresh_cf_pipeline.py --n_iterations 20 --n_workers 4 2>&1 | tee bench_optimized.log
  ```
- [ ] Record:
  - Runtime: _______ minutes
  - Peak memory: _______ MB
  - Successful CFs: _______
- [ ] Compare with estimates (should be ~2× faster)

---

## Phase 3: Polish & Documentation (1-2 hours)

### 3.1 Remove Redundant Copies (1 hour)
**Files:** `dice_cf_generator.py`, `fresh_cf_pipeline.py`
**Reference:** `OPTIMIZATION_PLAN.md` - Section 7

- [ ] In `dice_cf_generator.py:137`:
  - Only copy `permitted_range` if modifying
- [ ] In `fresh_cf_pipeline.py:169`:
  - Use `pd.DataFrame([patient_row.to_dict()])` instead of `.to_frame().T`
- [ ] Search for other `.copy()` calls: `grep -n "\.copy()" *.py`
- [ ] Review and optimize each

### 3.2 Update Documentation (30 min)
- [ ] Update `CLAUDE.md`:
  - Add "Optimizations" section
  - Note in-memory mode is default
  - Update performance expectations
- [ ] Update `FRESH_CF_README.md`:
  - Add note about optimizations
  - Update expected runtime (was 2-4 hours, now 1-2 hours)
- [ ] Add comments to code explaining optimizations

### 3.3 Final Validation (30 min)
- [ ] Run full 100-iteration test:
  ```bash
  python fresh_cf_pipeline.py --n_iterations 100 --n_workers 6 2>&1 | tee production_test.log
  ```
- [ ] Monitor:
  - Memory stays < 300 MB
  - No temp directories created
  - Completes in < 2 hours
- [ ] Verify results:
  - Check `aggregated_results/summary_report.md`
  - Compare CI results with previous runs (if available)

---

## Rollback Plan (If Needed)

If any phase fails critically:

```bash
# Restore backups
cp fresh_cf_pipeline.py.bak fresh_cf_pipeline.py
cp scm_analyzer.py.bak scm_analyzer.py
cp dice_cf_generator.py.bak dice_cf_generator.py
cp confidence_interval_analysis.py.bak confidence_interval_analysis.py

# Or restore specific file
cp <filename>.bak <filename>
```

---

## Success Criteria

### Minimum (Phase 1 Complete)
- ✅ Pipeline runs without errors
- ✅ Causal model initializes correctly
- ✅ No temp directories remain after run
- ✅ Results are scientifically valid

### Target (Phase 1 + 2 Complete)
- ✅ Memory usage < 300 MB (down from 500 MB)
- ✅ Runtime < 2 hours for 100 iterations (down from 3-4 hours)
- ✅ No intermediate CSV files unless configured
- ✅ All tests pass

### Stretch (All 3 Phases Complete)
- ✅ All redundant operations removed
- ✅ Documentation updated
- ✅ Production-ready code
- ✅ Full 100-iteration test completes successfully

---

## Notes & Decisions

### Decision Log
**Date** | **Decision** | **Rationale**
---------|--------------|---------------
2026-02-11 | (TBD) | (TBD)

### Issues Encountered
- (Document any issues and solutions here)

### Performance Metrics
**Metric** | **Before** | **After** | **Improvement**
-----------|------------|-----------|----------------
Memory | 500 MB | ___ MB | ___%
Runtime (100 iter) | 3-4 hours | ___ hours | ___×
Temp files | 28 MB | ___ MB | ___

---

## Quick Commands Reference

```bash
# Navigate to project
cd C:\Users\pmkul\Dropbox\Counterfactual_Analysis\cvd_counterfactual_pipeline

# Test mode (5 patients, 5 iterations)
python fresh_cf_pipeline.py --test_mode

# Small benchmark (20 iterations)
python fresh_cf_pipeline.py --n_iterations 20 --n_workers 4

# Full production run
python fresh_cf_pipeline.py --n_iterations 100 --n_workers 6

# Monitor logs
tail -f fresh_cf_pipeline.log

# Check memory (Windows Task Manager or Git Bash)
ps aux | grep python

# Clean temp directories
rm -rf temp_run_*
```

---

**Start Time:** ____________
**End Time:** ____________
**Total Duration:** ____________
**Status:** ☐ Not Started | ☐ In Progress | ☐ Completed | ☐ Blocked
