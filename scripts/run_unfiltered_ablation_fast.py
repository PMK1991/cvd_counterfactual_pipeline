# conda-env: mtech-env
"""
Fast, resumable, checkpointed reimplementation of run_unfiltered_ablation.py.

The original script scores every raw DiCE counterfactual one row at a time
through the full sklearn Pipeline, which is prohibitively slow on a synced
(OneDrive/Dropbox-style) filesystem where per-file I/O latency dominates
(hundreds of ms x tens of thousands of CF files). This version:

  1. Batches all predictions for an iteration into a single vectorized
     model.predict() call instead of one call per CF row.
  2. Parallelizes file reads within an iteration with a thread pool, which
     hides per-file I/O latency on network/synced filesystems.
  3. Checkpoints each completed iteration to disk immediately, so a run that
     gets interrupted (timeout, machine sleep, etc.) can be resumed by
     re-invoking the script — already-checkpointed iterations are skipped.

Two-step usage (checkpointing step, then finalize step):

    python scripts/run_unfiltered_ablation_fast.py fresh_cf_iterations \
        --model_path model/xgb_pipeline.pkl --max_iters 15 --time_budget 38
    # re-run the same command until it reports "ALL DONE"

    python scripts/run_unfiltered_ablation_fast.py fresh_cf_iterations --finalize

Output matches run_unfiltered_ablation.py's format exactly:
    <iterations_dir>/aggregated_results_no_scm/{all_iteration_metrics.csv,ci_results.csv,summary_report.md}
plus a combined successful_counterfactuals_unfiltered_all.csv (with patient_id
and iteration columns) for downstream per-patient / paired-bootstrap analyses.

Checkpoints live in <iterations_dir>/aggregated_results_no_scm/_partial/ and
can be deleted to force a clean rerun.
"""
import argparse
import glob
import json
import os
import pickle
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.pipeline.metrics_calculator import MetricsCalculator
from src.pipeline.ci_computer import CIComputer

GRAPH_NODES = ['age', 'sex', 'chol', 'fbs', 'trestbps', 'target',
               'cp', 'restecg', 'thalach', 'exang', 'slope', 'oldpeak']
FEATURE_COLS = [c for c in GRAPH_NODES if c != 'target']


def _read(f):
    try:
        return f, pd.read_csv(f)
    except Exception:
        return f, None


def process_iteration(it_dir, model, calc, executor):
    itn = int(re.search(r'iteration_(\d+)', it_dir).group(1))
    orig_files = glob.glob(os.path.join(it_dir, 'original', 'patient_*.csv'))
    orig_cache = {}
    for f in orig_files:
        pid = re.search(r'patient_(\d+)\.csv', os.path.basename(f)).group(1)
        orig_cache[pid] = pd.read_csv(f).iloc[0].to_dict()

    cf_files = glob.glob(os.path.join(it_dir, 'counterfactuals', 'patient_*_cf_*.csv'))
    cf_by_pid = {}
    for f, df in executor.map(_read, cf_files):
        if df is None or df.empty:
            continue
        pid = re.search(r'patient_(\d+)_cf_', os.path.basename(f)).group(1)
        cf_by_pid.setdefault(pid, []).append(df.iloc[0].to_dict())

    rows = []
    for pid, cfs in cf_by_pid.items():
        orig = orig_cache.get(pid)
        if orig is None:
            continue
        for cfrow in cfs:
            rows.append((pid, orig, cfrow))

    if not rows:
        metrics = calc.compute_all_metrics(pd.DataFrame())
        metrics['_successful_rows'] = []
    else:
        feat_df = pd.DataFrame([{c: r[2].get(c, np.nan) for c in FEATURE_COLS} for r in rows])
        preds = model.predict(feat_df)
        succ = []
        for (pid, orig, cfrow), pred in zip(rows, preds):
            if orig.get('target', 1) != 1 or int(pred) != 0:
                continue
            rec = {f'orig_{c}': orig.get(c) for c in FEATURE_COLS}
            rec.update({f'cf_{c}': cfrow.get(c) for c in FEATURE_COLS})
            rec['target'] = int(pred)
            rec['patient_id'] = pid
            succ.append(rec)
        succ_df = pd.DataFrame(succ) if succ else pd.DataFrame()
        metrics = calc.compute_all_metrics(succ_df)
        metrics['_successful_rows'] = succ

    metrics['iteration'] = itn

    mfile = os.path.join(it_dir, 'metrics.json')
    total_generated = total_requested = total_patients = None
    if os.path.exists(mfile):
        with open(mfile) as fh:
            src = json.load(fh)
        total_generated = src.get('total_generated_cfs')
        total_requested = src.get('total_requested_cfs')
        total_patients = src.get('total_patients')
    metrics['total_patients'] = total_patients
    metrics['total_requested_cfs'] = total_requested
    metrics['total_generated_cfs'] = total_generated
    metrics['target_flip_rate_pct'] = (
        metrics['total_successful_cfs'] / total_generated * 100 if total_generated else 0.0
    )
    return metrics


def run_batch(iterations_dir, model_path, max_iters, time_budget_s, n_workers=128):
    partial_dir = os.path.join(iterations_dir, 'aggregated_results_no_scm', '_partial')
    os.makedirs(partial_dir, exist_ok=True)

    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    calc = MetricsCalculator()

    all_iter_dirs = sorted(glob.glob(os.path.join(iterations_dir, 'iteration_*')))
    done = {os.path.splitext(os.path.basename(p))[0] for p in glob.glob(os.path.join(partial_dir, '*.json'))}
    todo = [d for d in all_iter_dirs if os.path.basename(d) not in done]

    print(f'{len(all_iter_dirs)} total iterations, {len(done)} already done, {len(todo)} remaining', flush=True)

    t_start = time.time()
    n_processed = 0
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        for it_dir in todo:
            if n_processed >= max_iters or (time.time() - t_start) > time_budget_s:
                break
            t0 = time.time()
            metrics = process_iteration(it_dir, model, calc, executor)
            iname = os.path.basename(it_dir)
            with open(os.path.join(partial_dir, f'{iname}.json'), 'w') as fh:
                json.dump(metrics, fh)
            n_processed += 1
            print(f'  {iname}: {metrics["total_successful_cfs"]} accepted, '
                  f'flip {metrics["target_flip_rate_pct"]:.1f}%  ({time.time() - t0:.1f}s)', flush=True)

    remaining = len(all_iter_dirs) - (len(done) + n_processed)
    print(f'Processed {n_processed} this call in {time.time() - t_start:.1f}s. {remaining} remaining.', flush=True)
    if remaining <= 0:
        print('ALL DONE', flush=True)


def finalize(iterations_dir):
    partial_dir = os.path.join(iterations_dir, 'aggregated_results_no_scm', '_partial')
    files = sorted(glob.glob(os.path.join(partial_dir, '*.json')))
    print(f'{len(files)} partial files found')

    all_metrics = []
    all_succ_rows = []
    for f in files:
        with open(f) as fh:
            m = json.load(fh)
        succ_rows = m.pop('_successful_rows', [])
        for r in succ_rows:
            r2 = dict(r)
            r2['iteration'] = m['iteration']
            all_succ_rows.append(r2)
        all_metrics.append(m)

    agg = pd.DataFrame(all_metrics).sort_values('iteration').reset_index(drop=True)
    out_dir = os.path.join(iterations_dir, 'aggregated_results_no_scm')
    os.makedirs(out_dir, exist_ok=True)
    agg.to_csv(os.path.join(out_dir, 'all_iteration_metrics.csv'), index=False)

    ci = CIComputer(confidence_level=0.95)
    ci_results = ci.compute_confidence_intervals(agg)
    ci.save_results(ci_results, out_dir)

    succ_df = pd.DataFrame(all_succ_rows)
    succ_df.to_csv(os.path.join(out_dir, 'successful_counterfactuals_unfiltered_all.csv'), index=False)

    print(f'n_iterations: {len(agg)}')
    print(f'Mean accepted/iter: {agg["total_successful_cfs"].mean():.2f}')
    print(f'Mean flip rate: {agg["target_flip_rate_pct"].mean():.2f}%')
    print(f'Total successful rows (all iters): {len(succ_df)}')
    print(f'Written to {out_dir}')


def main():
    parser = argparse.ArgumentParser(description='Fast, resumable no-SCM ablation scorer')
    parser.add_argument('iterations_dir')
    parser.add_argument('--model_path', default='model/xgb_pipeline.pkl')
    parser.add_argument('--max_iters', type=int, default=15,
                         help='Max iterations to process in this invocation (for time-boxed environments)')
    parser.add_argument('--time_budget', type=float, default=38.0,
                         help='Soft wall-clock budget in seconds for this invocation')
    parser.add_argument('--n_workers', type=int, default=128)
    parser.add_argument('--finalize', action='store_true',
                         help='Skip batch processing; just aggregate existing checkpoints into final CSVs')
    args = parser.parse_args()

    if args.finalize:
        finalize(args.iterations_dir)
    else:
        run_batch(args.iterations_dir, args.model_path, args.max_iters, args.time_budget, args.n_workers)


if __name__ == '__main__':
    main()
