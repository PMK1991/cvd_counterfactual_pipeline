"""
Chunked, resumable patient-cluster bootstrap for Table 7 (single-arm SCM
patient-level inferential intervals), matching src/pipeline/patient_bootstrap.py's
methodology exactly but able to accumulate 5000 replicates across multiple
time-boxed invocations.

Usage (re-run until it reports ALL DONE):
    python patient_bootstrap_chunked.py fresh_cf_iterations --target 5000 --chunk 1500

Then:
    python patient_bootstrap_chunked.py fresh_cf_iterations --finalize
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, '/sessions/practical-gallant-noether/mnt/cvd_counterfactual_pipeline')
from src.pipeline.metrics_calculator import MetricsCalculator


def load_successful(output_dir):
    frames = []
    for f in sorted(glob.glob(os.path.join(output_dir, 'iteration_*', 'successful', 'successful_counterfactuals.csv'))):
        try:
            df = pd.read_csv(f)
        except Exception:
            continue
        if df.empty:
            continue
        frames.append(df)
    result = pd.concat(frames, ignore_index=True)
    result['patient_id'] = result['patient_id'].astype(str)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('output_dir')
    ap.add_argument('--target', type=int, default=5000)
    ap.add_argument('--chunk', type=int, default=1500)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--finalize', action='store_true')
    args = ap.parse_args()

    ckpt_dir = os.path.join(args.output_dir, 'aggregated_results', '_patient_bootstrap_chunks')
    os.makedirs(ckpt_dir, exist_ok=True)

    if args.finalize:
        files = sorted(glob.glob(os.path.join(ckpt_dir, 'chunk_*.csv')))
        print(f'{len(files)} chunk files found')
        dfs = [pd.read_csv(f) for f in files]
        all_rows = pd.concat(dfs, ignore_index=True)
        print(f'Total replicates: {len(all_rows)}')
        lower, upper = 2.5, 97.5
        out_rows = []
        for col in all_rows.select_dtypes(include='number').columns:
            if col == 'bootstrap':
                continue
            vals = all_rows[col].dropna()
            if vals.empty:
                continue
            ci_lo = np.percentile(vals, lower)
            ci_hi = np.percentile(vals, upper)
            out_rows.append({
                'metric': col, 'mean': vals.mean(), 'median': vals.median(),
                'ci_lower': ci_lo, 'ci_upper': ci_hi, 'ci_width': ci_hi - ci_lo,
                'n_bootstrap': len(vals), 'interval_type': 'patient_bootstrap_inferential',
            })
        out_df = pd.DataFrame(out_rows)
        out_path = os.path.join(args.output_dir, 'aggregated_results', 'patient_bootstrap_ci.csv')
        out_df.to_csv(out_path, index=False)
        print(f'Written to {out_path}')
        for _, r in out_df.iterrows():
            print(f"{r['metric']:<30} mean={r['mean']:.2f}  [{r['ci_lower']:.2f}, {r['ci_upper']:.2f}]")
        return

    done_files = glob.glob(os.path.join(ckpt_dir, 'chunk_*.csv'))
    done_reps = 0
    for f in done_files:
        done_reps += len(pd.read_csv(f))
    remaining = args.target - done_reps
    print(f'{done_reps} replicates done, {remaining} remaining', flush=True)
    if remaining <= 0:
        print('ALL DONE')
        return

    n_this_chunk = min(args.chunk, remaining)
    chunk_idx = len(done_files)

    successful = load_successful(args.output_dir)
    patient_ids = successful['patient_id'].drop_duplicates().to_numpy()
    grouped = {pid: g for pid, g in successful.groupby('patient_id', sort=False)}
    calc = MetricsCalculator()

    # offset the RNG stream by however many reps are already done, for a single
    # continuous deterministic stream across chunks
    rng = np.random.default_rng(args.seed)
    for _ in range(done_reps):
        rng.choice(patient_ids, size=len(patient_ids), replace=True)

    t0 = time.time()
    rows = []
    for i in range(n_this_chunk):
        sampled_ids = rng.choice(patient_ids, size=len(patient_ids), replace=True)
        sampled_frames = [grouped[pid] for pid in sampled_ids]
        sample_df = pd.concat(sampled_frames, ignore_index=True)
        metrics = calc.compute_all_metrics(sample_df)
        rows.append(metrics)

    chunk_df = pd.DataFrame(rows)
    chunk_path = os.path.join(ckpt_dir, f'chunk_{chunk_idx:03d}.csv')
    chunk_df.to_csv(chunk_path, index=False)
    print(f'Wrote {n_this_chunk} replicates to {chunk_path} in {time.time()-t0:.1f}s', flush=True)
    print(f'{done_reps + n_this_chunk}/{args.target} total', flush=True)


if __name__ == '__main__':
    main()
