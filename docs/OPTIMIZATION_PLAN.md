# CVD Counterfactual Pipeline - Optimization Plan

**Date Created:** 2026-02-10
**Session Status:** Analysis Complete, Implementation Pending
**Estimated Impact:** 60% memory reduction, 2× speedup, clean temp files

---

## Executive Summary

Comprehensive analysis of the CVD counterfactual pipeline identified **8 major optimization opportunities** across memory usage, performance, and code quality. Implementing all optimizations will reduce memory from 500 MB to 200 MB, cut runtime from 3-4 hours to 1-2 hours, and eliminate 28 MB of abandoned temp files.

---

## 1. CRITICAL ISSUE: Temp Directory Cleanup (28 MB Wasted)

**Priority:** 🔴 CRITICAL
**Impact:** Disk space waste, file system clutter
**Effort:** Low (30 minutes)

### Problem
100 temporary directories (`temp_run_0` through `temp_run_99`) are created by `confidence_interval_analysis.py` but never cleaned up:
- Total waste: 28 MB per 100-iteration run
- File count: 100 directories × 50 files = 5,000 files
- Potential I/O slowdown from directory scanning

### Location
- `confidence_interval_analysis.py:62` - Creates `temp_run_{run_id}` directories
- No cleanup code exists in any file

### Solution Option 1: Add Cleanup Method
```python
# In confidence_interval_analysis.py
import shutil
from pathlib import Path

class ConfidenceIntervalAnalyzer:
    def __init__(self, causal_model, n_runs=100, confidence_level=0.95):
        # ... existing code ...
        self.temp_dirs = []  # Track temp directories

    def run_single_iteration(self, run_id, seed):
        output_dir = f"temp_run_{run_id}"
        self.temp_dirs.append(output_dir)

        # ... existing code ...

        return metrics

    def cleanup_temp_directories(self):
        """Remove all temporary directories created during analysis"""
        for temp_dir in self.temp_dirs:
            if Path(temp_dir).exists():
                shutil.rmtree(temp_dir, ignore_errors=True)
        logger.info(f"Cleaned up {len(self.temp_dirs)} temporary directories")

    def run_bootstrap_analysis(self):
        try:
            # ... existing code ...
        finally:
            # Always cleanup, even if analysis fails
            if not self.config.get('keep_temp_dirs', False):
                self.cleanup_temp_directories()
```

### Solution Option 2: Use TemporaryDirectory (Preferred)
```python
# In confidence_interval_analysis.py
import tempfile

def run_single_iteration(self, run_id, seed):
    with tempfile.TemporaryDirectory(prefix=f"temp_run_{run_id}_") as output_dir:
        analyzer = CounterfactualAnalyzer(
            self.causal_model,
            output_dir=output_dir
        )
        analyzer.process_all_instances(show_progress=False)
        metrics = self._extract_metrics(analyzer)
        # Directory auto-deleted when context exits
        return metrics
```

### Implementation Steps
1. Backup current `confidence_interval_analysis.py`
2. Implement Option 2 (cleaner, automatic)
3. Test with `--test_mode` first
4. Manually clean existing temp directories: `rm -rf temp_run_*`

---

## 2. CRITICAL ISSUE: Redundant Model Loading (Memory × Workers)

**Priority:** 🔴 CRITICAL
**Impact:** 75% memory reduction (4 copies → 1)
**Effort:** Medium (2 hours)

### Problem
Each worker process loads the entire ML model independently via `pickle.load()`, multiplying memory usage:
- Model size: 5-50 MB (typical XGBoost)
- With 4 workers: 20-200 MB total
- Redundant disk I/O on model file

### Locations
- `fresh_cf_pipeline.py:156-162` - Creates new `DiceCFGenerator` in each process
- `dice_cf_generator.py:74-76` - Loads model from pickle file

### Current Inefficient Code
```python
# In fresh_cf_pipeline.py:run_single_iteration()
def run_single_iteration(self, iteration_num: int, patients_df: pd.DataFrame):
    # ❌ Each process loads model independently
    dice_gen = DiceCFGenerator(
        model_path=self.config['dice']['model_path'],  # Re-loads model!
        data_path=self.config['dice']['data_path'],
        config=self.config['dice']
    )
    dice_gen.load_model_and_data()  # Redundant pickle.load()
```

### Solution: Load Once Per Worker Process
```python
# In fresh_cf_pipeline.py

# Global worker resources (initialized once per process)
_worker_dice_gen = None
_worker_scm_analyzer = None

def init_worker(model_path, data_path, dice_config, scm_config):
    """Initialize worker process with shared resources (called once per worker)"""
    global _worker_dice_gen, _worker_scm_analyzer

    # Load model once per worker
    _worker_dice_gen = DiceCFGenerator(model_path, data_path, dice_config)
    _worker_dice_gen.load_model_and_data()
    _worker_dice_gen.setup_dice_explainer()

    # Initialize SCM analyzer once per worker
    _worker_scm_analyzer = SCMAnalyzer(config=scm_config)
    _worker_scm_analyzer.initialize_analyzer()

    logger.info(f"Worker {os.getpid()} initialized with shared resources")

def worker_process_iteration(args):
    """Worker function that reuses loaded resources"""
    iteration_num, patients_df, config = args
    global _worker_dice_gen, _worker_scm_analyzer

    try:
        logger.info(f"Worker {os.getpid()} processing iteration {iteration_num}")

        # Create output directory
        output_dir = Path(config['output']['base_dir'])
        iteration_dir = output_dir / f"iteration_{iteration_num:03d}"

        # Step 1: Generate CFs for all patients using pre-loaded model
        cf_results = []
        for idx, patient_row in enumerate(patients_df):
            patient_data = pd.DataFrame([patient_row])
            result = _worker_dice_gen.generate_and_save_for_patient(
                patient_data,
                patient_id=idx,
                iteration_num=iteration_num,
                output_dir=str(output_dir)
            )
            cf_results.append(result)

        # Step 2: Run SCM Analysis using pre-loaded analyzer
        successful_cfs = _worker_scm_analyzer.analyze_iteration(
            iteration_dir=str(iteration_dir)
        )

        # Step 3: Compute Metrics
        metrics_calc = MetricsCalculator()
        metrics = metrics_calc.compute_all_metrics(successful_cfs)
        metrics['iteration'] = iteration_num

        # Save iteration metrics
        metrics_file = iteration_dir / "metrics.json"
        metrics_file.parent.mkdir(parents=True, exist_ok=True)
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)

        logger.info(f"Completed iteration {iteration_num}: {metrics['total_successful_cfs']} successful CFs")

        return metrics

    except Exception as e:
        logger.error(f"Error in iteration {iteration_num}: {e}")
        return {'iteration': iteration_num, 'total_successful_cfs': 0, 'error': str(e)}

class FreshCFPipeline:
    def run_concurrent_pipeline(self, n_iterations=None, n_workers=None):
        """Run full pipeline with concurrent execution"""
        n_iterations = n_iterations or self.config['pipeline']['n_iterations']
        n_workers = n_workers or self.config['pipeline']['n_workers']

        logger.info(f"Starting concurrent pipeline: {n_iterations} iterations, {n_workers} workers")

        # Load patient data once
        patients_df = self.load_patient_data()

        # Convert to list of dicts for efficient serialization
        patient_records = patients_df.to_dict('records')

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Run iterations concurrently with worker initialization
        all_metrics = []

        with ProcessPoolExecutor(
            max_workers=n_workers,
            initializer=init_worker,
            initargs=(
                self.config['dice']['model_path'],
                self.config['dice']['data_path'],
                self.config['dice'],
                self.config['scm']
            )
        ) as executor:
            # Submit all iterations
            futures = {
                executor.submit(
                    worker_process_iteration,
                    (i, patient_records, self.config)
                ): i
                for i in range(n_iterations)
            }

            # Collect results with progress bar
            with tqdm(total=n_iterations, desc="Running iterations") as pbar:
                for future in as_completed(futures):
                    iteration_num = futures[future]
                    try:
                        metrics = future.result()
                        all_metrics.append(metrics)
                        pbar.update(1)
                        pbar.set_postfix({'successful_cfs': metrics.get('total_successful_cfs', 0)})
                    except Exception as e:
                        logger.error(f"Iteration {iteration_num} failed: {e}")
                        pbar.update(1)

        logger.info(f"Completed all {n_iterations} iterations")

        return pd.DataFrame(all_metrics)
```

### Implementation Steps
1. Backup `fresh_cf_pipeline.py`
2. Add `init_worker()` and `worker_process_iteration()` functions
3. Modify `run_concurrent_pipeline()` to use `ProcessPoolExecutor` initializer
4. Remove `run_single_iteration()` method (replaced by `worker_process_iteration()`)
5. Add `import os` at top of file
6. Test with `--test_mode` (should see "Worker XXXX initialized" messages)

---

## 3. CRITICAL BUG: Missing Causal Model in SCMAnalyzer

**Priority:** 🔴 CRITICAL (Pipeline Won't Run)
**Impact:** Fixes crash on initialization
**Effort:** High (3-4 hours - requires domain knowledge)

### Problem
`scm_analyzer.py:56` calls `CounterfactualAnalyzer()` without the required `causal_model` parameter:
```python
self.analyzer = CounterfactualAnalyzer()  # ❌ Missing causal_model!
```

This will cause: `TypeError: __init__() missing 1 required positional argument: 'causal_model'`

### Locations
- `scm_analyzer.py:56` - Initializes without causal model
- `counterfactualAnalyzer.py:16` - Requires `causal_model` as first positional argument

### Solution: Build and Pass Causal Model
```python
# In scm_analyzer.py
import networkx as nx
from dowhy import gcm
import logging

logger = logging.getLogger(__name__)

class SCMAnalyzer:
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._default_config()
        self.analyzer = None
        self.causal_model = None  # Add field for causal model

    def _build_causal_model(self):
        """
        Build and fit the causal model for CVD counterfactual analysis.

        Based on cardiovascular disease domain knowledge:
        - Cholesterol (chol) affects blood pressure and chest pain
        - Blood pressure (trestbps) influences heart rate and ST depression
        - Exercise angina and ST slope are intermediate indicators
        - Target (CVD risk) is influenced by multiple pathways
        """
        logger.info("Building causal graph for CVD...")

        # Define causal graph based on medical knowledge
        causal_graph = nx.DiGraph([
            # Primary interventions
            ('chol', 'trestbps'),
            ('chol', 'cp'),

            # Blood pressure effects
            ('trestbps', 'thalach'),
            ('trestbps', 'oldpeak'),

            # Chest pain pathway
            ('cp', 'exang'),
            ('cp', 'target'),

            # Heart rate and exercise
            ('thalach', 'exang'),
            ('thalach', 'target'),

            # ST depression pathway
            ('oldpeak', 'slope'),
            ('slope', 'target'),

            # Exercise-induced angina
            ('exang', 'target'),

            # ECG effects
            ('restecg', 'target'),

            # Note: 'age', 'sex', 'fbs' are considered exogenous (no incoming edges)
            # They influence outcomes but aren't intervened upon
        ])

        # Create probabilistic causal model
        causal_model = gcm.ProbabilisticCausalModel(causal_graph)

        # Load training data for fitting the causal mechanisms
        logger.info("Loading training data for causal model fitting...")
        train_data_path = self.config.get('train_data_path', 'heart_statlog_cleveland_hungary_final.csv')
        train_data = pd.read_csv(train_data_path)

        # Auto-assign causal mechanisms (ANM, functional models, etc.)
        logger.info("Auto-assigning causal mechanisms...")
        gcm.auto.assign_causal_mechanisms(causal_model, train_data)

        # Fit the causal model to data
        logger.info("Fitting causal model to training data...")
        gcm.fit(causal_model, train_data)

        logger.info("Causal model built and fitted successfully")
        return causal_model

    def _default_config(self) -> Dict:
        """Default SCM configuration"""
        return {
            'n_samples': 1,  # Only need 1 sample for validation (changed from 1000)
            'train_data_path': 'heart_statlog_cleveland_hungary_final.csv',
            'causal_model': 'default'
        }

    def initialize_analyzer(self) -> None:
        """Initialize the CounterfactualAnalyzer with fitted causal model"""
        try:
            # Build causal model if not exists
            if self.causal_model is None:
                logger.info("Building causal model...")
                self.causal_model = self._build_causal_model()

            # Initialize analyzer with model
            self.analyzer = CounterfactualAnalyzer(
                causal_model=self.causal_model,
                original_dir="temp_original",
                cf_dir="temp_cf",
                output_dir="temp_output"
            )
            logger.info("CounterfactualAnalyzer initialized with causal model")
        except Exception as e:
            logger.error(f"Error initializing analyzer: {e}")
            raise

    def apply_scm_intervention(self, original, cf_suggestion):
        """
        Apply SCM intervention to validate counterfactual

        Uses DoWhy's interventional sampling to generate counterfactual
        based on causal model.
        """
        if self.causal_model is None:
            raise ValueError("Causal model not initialized")

        try:
            # Extract intervention values
            chol_value = cf_suggestion['chol'].values[0]
            trestbps_value = cf_suggestion['trestbps'].values[0]

            # Define intervention
            intervention_dict = {
                'chol': lambda chol: chol_value,
                'trestbps': lambda trestbps: trestbps_value
            }

            # Generate counterfactual samples
            cf_samples = gcm.interventional_samples(
                self.causal_model,
                intervention_dict,
                observed_data=original
            )

            # Apply physiological constraints
            cf_samples['oldpeak'] = cf_samples['oldpeak'].clip(lower=0)
            cf_samples['exang'] = cf_samples['exang'].clip(lower=0, upper=1).round()
            cf_samples['cp'] = cf_samples['cp'].clip(lower=1, upper=4).round()
            cf_samples['slope'] = cf_samples['slope'].clip(lower=1, upper=3).round()
            cf_samples['restecg'] = cf_samples['restecg'].clip(lower=0, upper=2).round()

            # Return first sample (n_samples=1 in config)
            return cf_samples.iloc[0:1]

        except Exception as e:
            logger.debug(f"SCM intervention failed: {e}")
            return None
```

### Implementation Steps
1. **Backup** `scm_analyzer.py`
2. **Review causal graph** - Verify edges match domain knowledge (consult CVD literature if needed)
3. **Implement** `_build_causal_model()` method
4. **Update** `initialize_analyzer()` to call `_build_causal_model()`
5. **Add** `apply_scm_intervention()` method for direct intervention (replaces reliance on counterfactualAnalyzer.generate_counterfactual)
6. **Update** `_default_config()` to include `train_data_path` and reduce `n_samples` to 1
7. **Test** with small dataset first:
   ```python
   # Test script
   analyzer = SCMAnalyzer()
   analyzer.initialize_analyzer()
   print("✓ Causal model initialized successfully")
   ```

### Important Notes
- Causal graph structure is critical - edges represent direct causal relationships
- `gcm.auto.assign_causal_mechanisms()` automatically selects appropriate models (linear, ANM, etc.)
- First run will be slower due to model fitting (~30-60 seconds)
- Consider caching fitted causal model to disk for subsequent runs

---

## 4. CRITICAL ISSUE: Excessive SCM Sampling (99.9% Memory Waste)

**Priority:** 🔴 CRITICAL
**Impact:** 99.9% memory reduction for SCM operations
**Effort:** Low (15 minutes)

### Problem
Generates 1,000 samples per counterfactual validation but only uses 1:
- Memory: 1,000 rows × features × data type size per CF
- Computation: Generates 999 unused samples
- Total waste: 48 patients × 5 CFs × 1,000 samples × 100 iterations = **24 million unnecessary samples**

### Locations
- `scm_analyzer.py:136` - `n_samples=self.config['n_samples']` (default 1,000)
- `counterfactualAnalyzer.py:83-85` - Generates samples then only uses first

### Current Wasteful Code
```python
# scm_analyzer.py - default config
def _default_config(self):
    return {
        'n_samples': 1000,  # ❌ Way too many!
        'causal_model': 'default'
    }

# counterfactualAnalyzer.py
cf_samples = gcm.interventional_samples(
    self.causal_model,
    intervention_dict,
    observed_data=orig_data
)  # Returns 1,000 samples

# Only first sample used for validation
for _, row in cf_samples.iterrows():
    cf_result = self.create_comparison_row(...)
    results.append(cf_result)
    break  # Implicit - only processes first
```

### Solution: Reduce to 1 Sample
```python
# In scm_analyzer.py
def _default_config(self) -> Dict:
    """Default SCM configuration"""
    return {
        'n_samples': 1,  # ✅ Only need 1 sample for target flip validation
        'train_data_path': 'heart_statlog_cleveland_hungary_final.csv',
        'causal_model': 'default'
    }

# In apply_scm_intervention (from optimization #3)
def apply_scm_intervention(self, original, cf_suggestion):
    # ... intervention setup ...

    # Generate only 1 sample
    cf_samples = gcm.interventional_samples(
        self.causal_model,
        intervention_dict,
        observed_data=original,
        num_samples_to_draw=1  # ✅ Explicit single sample
    )

    # Apply constraints
    cf_samples['oldpeak'] = cf_samples['oldpeak'].clip(lower=0)
    # ... other constraints ...

    return cf_samples.iloc[0:1]  # Return single row DataFrame
```

### Optional: Add Uncertainty Estimation Method
```python
# In scm_analyzer.py - for future uncertainty analysis
def estimate_cf_uncertainty(
    self,
    original: pd.DataFrame,
    cf_suggestion: pd.DataFrame,
    n_samples: int = 100
) -> pd.DataFrame:
    """
    Generate multiple samples for uncertainty quantification.

    Use this method when you need to understand variability
    in counterfactual outcomes, NOT for basic validation.
    """
    samples = []
    for i in range(n_samples):
        cf = self.apply_scm_intervention(original, cf_suggestion)
        if cf is not None:
            cf['sample_id'] = i
            samples.append(cf)

    return pd.concat(samples, ignore_index=True) if samples else pd.DataFrame()
```

### Implementation Steps
1. In `scm_analyzer.py`, change `_default_config()` to set `'n_samples': 1`
2. In `apply_scm_intervention()`, add explicit `num_samples_to_draw=1` parameter
3. Test with `--test_mode` to verify no errors
4. Monitor memory usage (should drop significantly)

---

## 5. PERFORMANCE: Eliminate 28,800 CSV Operations

**Priority:** 🟡 HIGH IMPACT
**Impact:** 2-3× speedup, eliminates I/O bottleneck
**Effort:** High (4-5 hours)

### Problem
Extensive CSV writing/reading in tight loops:
- **Write**: 48 patients × 6 files/patient × 100 iterations = 28,800 CSV writes
- **Read**: All files re-read during SCM validation
- Each CSV operation includes: file creation, serialization, disk sync, deserialization

### Locations
- `dice_cf_generator.py:198-206` - Writes 1 original + 5 CFs per patient
- `scm_analyzer.py:90-96` - Re-reads all CSVs for validation

### Solution: In-Memory Pipeline
```python
# In fresh_cf_pipeline.py

def worker_process_iteration_inmemory(args):
    """Worker function with in-memory processing (no intermediate CSV files)"""
    iteration_num, patients_df, config = args
    global _worker_dice_gen, _worker_scm_analyzer

    try:
        logger.info(f"Processing iteration {iteration_num} (in-memory mode)")

        # Step 1: Generate CFs in memory
        cf_results = []
        for idx, patient_row in enumerate(patients_df):
            patient_data = pd.DataFrame([patient_row])

            # Generate counterfactuals (in memory, no file write)
            cf_result = _worker_dice_gen.generate_counterfactuals(patient_data)

            if cf_result and cf_result.final_cfs_df is not None:
                # Store in memory
                for cf_idx, (_, cf_row) in enumerate(cf_result.final_cfs_df.iterrows()):
                    cf_results.append({
                        'patient_id': idx,
                        'cf_id': cf_idx,
                        'original': patient_data.iloc[0].to_dict(),
                        'counterfactual': cf_row.to_dict(),
                        'original_df': patient_data,
                        'cf_df': cf_row.to_frame().T
                    })

        # Step 2: SCM validation in memory
        successful_cfs = []
        for result in cf_results:
            cf_scm = _worker_scm_analyzer.apply_scm_intervention(
                result['original_df'],
                result['cf_df']
            )

            if cf_scm is not None:
                original_target = result['original_df']['target'].values[0]
                if _worker_scm_analyzer.validate_counterfactual(cf_scm, original_target):
                    # Enrich with metadata
                    cf_scm['patient_id'] = result['patient_id']
                    cf_scm['cf_id'] = result['cf_id']
                    successful_cfs.append(cf_scm)

        # Step 3: Compute metrics
        if successful_cfs:
            successful_df = pd.concat(successful_cfs, ignore_index=True)
        else:
            successful_df = pd.DataFrame()

        metrics_calc = MetricsCalculator()
        metrics = metrics_calc.compute_all_metrics(successful_df)
        metrics['iteration'] = iteration_num

        # Step 4: Save only final results (if configured)
        if config['output'].get('keep_all_iterations', False):
            output_dir = Path(config['output']['base_dir'])
            iteration_dir = output_dir / f"iteration_{iteration_num:03d}"
            iteration_dir.mkdir(parents=True, exist_ok=True)

            # Save only successful CFs
            if not successful_df.empty:
                successful_df.to_csv(
                    iteration_dir / "successful_counterfactuals.csv",
                    index=False
                )

            # Save metrics
            with open(iteration_dir / "metrics.json", 'w') as f:
                json.dump(metrics, f, indent=2)

        logger.info(f"Completed iteration {iteration_num}: {len(successful_cfs)} successful CFs")

        return metrics

    except Exception as e:
        logger.error(f"Error in iteration {iteration_num}: {e}")
        return {'iteration': iteration_num, 'total_successful_cfs': 0, 'error': str(e)}


class FreshCFPipeline:
    def run_concurrent_pipeline(self, n_iterations=None, n_workers=None, in_memory=True):
        """Run pipeline with optional in-memory mode"""
        # ... existing setup ...

        # Choose worker function
        worker_fn = (worker_process_iteration_inmemory if in_memory
                    else worker_process_iteration)

        with ProcessPoolExecutor(...) as executor:
            futures = {
                executor.submit(worker_fn, (i, patient_records, self.config)): i
                for i in range(n_iterations)
            }
            # ... collect results ...
```

### Implementation Steps
1. Add `worker_process_iteration_inmemory()` function
2. Add `in_memory` parameter to `run_concurrent_pipeline()` (default True)
3. Test with `--test_mode` first, verify results match original
4. For large-scale runs, consider hybrid: in-memory processing but save every 10th iteration

---

## 6-8. Additional Optimizations (Lower Priority)

### 6. Data Loading Caching
```python
# In fresh_cf_pipeline.py
class FreshCFPipeline:
    def __init__(self, config):
        self._cached_patient_data = None

    def load_patient_data(self, use_cache=True):
        if use_cache and self._cached_patient_data is not None:
            return self._cached_patient_data
        # ... load data ...
        if use_cache:
            self._cached_patient_data = high_risk
        return high_risk
```

### 7. Remove Redundant DataFrame Copies
```python
# In dice_cf_generator.py:137
# BEFORE: permitted_range = self.config['permitted_range'].copy()
# AFTER: Only copy if modifying
permitted_range = self.config['permitted_range']
if permitted_range['chol'][1] is None:
    permitted_range = permitted_range.copy()  # Now copy
    permitted_range['chol'][1] = original_chol * 0.9
```

### 8. Threading Optimization
```python
# In dice_cf_generator.py - Remove threading for operations < 60s
def generate_counterfactuals(self, patient_data, timeout=None):
    timeout = timeout or self.config.get('timeout', 30)

    # Run directly without threading overhead
    try:
        return self.dice_exp.generate_counterfactuals(...)
    except Exception as e:
        logger.warning(f"DiCE generation failed: {e}")
        return None
```

---

## Implementation Roadmap

### Phase 1: Critical Fixes (Day 1 - 4-6 hours)
**Must complete before production use**

1. ✅ **Fix Missing Causal Model** (3-4 hours)
   - Build causal graph
   - Implement `_build_causal_model()`
   - Test initialization
   - **Blocker**: Pipeline won't run without this

2. ✅ **Reduce SCM Samples** (15 min)
   - Change `n_samples: 1000 → 1`
   - Add explicit parameter
   - Quick win: 99.9% memory reduction

3. ✅ **Temp Directory Cleanup** (30 min)
   - Implement `TemporaryDirectory` approach
   - Manually clean existing: `rm -rf temp_run_*`
   - Prevents disk bloat

4. ✅ **Test End-to-End** (30 min)
   ```bash
   python fresh_cf_pipeline.py --test_mode
   # Should complete without errors
   ```

### Phase 2: Performance Boost (Day 2 - 3-4 hours)
**Significant speed and memory improvements**

5. ✅ **Model Loading Optimization** (2 hours)
   - Implement `init_worker()` and worker pattern
   - Test with multiple workers
   - Verify memory usage: `htop` or Task Manager

6. ✅ **In-Memory Pipeline** (2-3 hours)
   - Implement `worker_process_iteration_inmemory()`
   - Add `in_memory` flag
   - Benchmark: compare runtime before/after

7. ✅ **Full Pipeline Test** (1 hour)
   ```bash
   python fresh_cf_pipeline.py --n_iterations 20 --n_workers 4
   # Verify 2-3× speedup
   ```

### Phase 3: Code Quality (Day 3 - 1-2 hours)
**Polish and minor optimizations**

8. ✅ **Remove Redundant Copies** (1 hour)
   - Search for `.copy()` calls
   - Optimize DataFrame operations
   - Run benchmarks

9. ✅ **Update Documentation** (30 min)
   - Update CLAUDE.md with optimizations
   - Add performance notes to README
   - Document in-memory mode usage

10. ✅ **Final Validation** (30 min)
    ```bash
    python fresh_cf_pipeline.py --n_iterations 100 --n_workers 6
    # Full production run
    ```

---

## Testing Strategy

### Unit Tests
```python
# test_optimizations.py
def test_temp_cleanup():
    """Verify no temp directories remain after run"""
    # Run analysis
    # Assert: glob.glob("temp_run_*") == []

def test_model_loading_once_per_worker():
    """Verify model loaded once per worker, not per iteration"""
    # Mock pickle.load
    # Run with 4 workers, 10 iterations
    # Assert: pickle.load called 4 times (not 40)

def test_in_memory_matches_disk():
    """Verify in-memory results match disk-based results"""
    # Run same config with in_memory=True and False
    # Assert: results are identical

def test_scm_single_sample():
    """Verify SCM generates only 1 sample"""
    # Mock gcm.interventional_samples
    # Assert: num_samples_to_draw == 1
```

### Integration Tests
```bash
# Test mode validation
python fresh_cf_pipeline.py --test_mode
# Expected: 5 iterations, 5 patients, <10 minutes

# Memory profiling
python -m memory_profiler fresh_cf_pipeline.py --test_mode
# Expected: Peak < 300 MB

# Performance benchmark
time python fresh_cf_pipeline.py --n_iterations 20 --n_workers 4
# Expected: <30 minutes (vs. 60 minutes before)
```

---

## Rollback Plan

### Before Starting
```bash
# Create backup branch
cd /c/Users/pmkul/Dropbox/Counterfactual_Analysis/cvd_counterfactual_pipeline
git checkout -b optimization-backup
git add .
git commit -m "Pre-optimization backup"
git checkout main  # or master
```

### If Issues Arise
```bash
# Restore from backup
git checkout optimization-backup -- <filename>

# Or revert all changes
git checkout optimization-backup
```

### Keep Copies
```bash
# Backup critical files
cp fresh_cf_pipeline.py fresh_cf_pipeline.py.bak
cp scm_analyzer.py scm_analyzer.py.bak
cp dice_cf_generator.py dice_cf_generator.py.bak
```

---

## Expected Performance Metrics

### Before Optimization
| Metric | Value |
|--------|-------|
| Memory (peak) | ~500 MB |
| Runtime (100 iter) | 3-4 hours |
| Disk (temp files) | 28 MB |
| CSV operations | 28,800 |
| Model loads | 400 (4 workers × 100 iter) |

### After Optimization
| Metric | Value | Improvement |
|--------|-------|-------------|
| Memory (peak) | ~200 MB | **60% reduction** |
| Runtime (100 iter) | 1-2 hours | **2× faster** |
| Disk (temp files) | 0 MB | **Clean** |
| CSV operations | 200 | **99% reduction** |
| Model loads | 4 (once per worker) | **99% reduction** |

---

## Monitoring Commands

### During Execution
```bash
# Memory usage (Linux/Mac)
watch -n 5 'ps aux | grep python | grep fresh_cf'

# Memory usage (Windows)
# Task Manager → Details → python.exe → Memory

# Disk I/O
iotop -p $(pgrep -f fresh_cf_pipeline)

# Progress
tail -f fresh_cf_pipeline.log

# Temp directory count
ls -d temp_run_* 2>/dev/null | wc -l
```

### After Completion
```bash
# Check output
ls -lh fresh_cf_iterations/aggregated_results/

# Verify no temp dirs
ls -d temp_run_* 2>/dev/null

# Check log for errors
grep -i error fresh_cf_pipeline.log
```

---

## Risk Assessment

### Low Risk (Safe to implement)
- ✅ Temp directory cleanup
- ✅ Reduce SCM samples to 1
- ✅ Data loading caching
- ✅ Remove redundant copies

### Medium Risk (Test thoroughly)
- ⚠️ Model loading optimization
  - Risk: ProcessPoolExecutor initialization failures
  - Mitigation: Test with `--test_mode` first

- ⚠️ In-memory pipeline
  - Risk: Different results than disk-based
  - Mitigation: Run comparison test, validate metrics match

### High Risk (Requires domain knowledge)
- 🔴 Causal model building
  - Risk: Incorrect causal graph → invalid counterfactuals
  - Mitigation: Literature review, expert consultation, extensive testing

---

## Session Context for Tomorrow

### What We've Done
1. ✅ Created CLAUDE.md with comprehensive codebase documentation
2. ✅ Analyzed entire pipeline for optimization opportunities
3. ✅ Identified 8 major optimization areas
4. ✅ Created detailed implementation plan with code examples

### What's Next
- Decide on implementation approach (all at once vs. incremental)
- Start with Phase 1 (Critical Fixes) - approximately 4-6 hours
- Test each optimization before moving to next
- Monitor performance improvements

### Key Files to Modify
1. `confidence_interval_analysis.py` - Temp cleanup
2. `scm_analyzer.py` - Causal model + reduce samples
3. `fresh_cf_pipeline.py` - Model loading + in-memory pipeline
4. `dice_cf_generator.py` - Remove redundant operations

### Questions to Answer Tomorrow
1. **Causal graph structure**: Do you have domain knowledge of CVD causality, or should we research?
2. **Risk tolerance**: Implement all optimizations, or start with safe ones?
3. **Testing strategy**: Full test suite or manual validation?
4. **Production timeline**: When do you need this running in production?

---

## Resources

### Relevant Documentation
- DiCE-ML: https://github.com/interpretml/DiCE
- DoWhy: https://github.com/py-why/dowhy
- Python ProcessPoolExecutor: https://docs.python.org/3/library/concurrent.futures.html

### CVD Causal Modeling Papers
- Add references if implementing causal model from literature
- Consider consulting domain expert for graph structure

### Code References
- `counterfactualAnalyzer.py:16-36` - Example causal model structure
- `nb_heart_disease_scm.ipynb` - Exploratory causal analysis

---

**End of Optimization Plan**
**Ready to implement tomorrow - let's optimize this pipeline! 🚀**
