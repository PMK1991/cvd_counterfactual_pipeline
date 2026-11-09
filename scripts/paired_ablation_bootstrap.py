# conda-env: mtech-env
"""
Paired patient-cluster bootstrap on the SCM-vs-unfiltered ablation contrast
(Reviewer 3, Comment 4b, applied to the ablation rather than just the flip rate).

Recreated 2026-08-02: the original script of this name was never committed to
this repo (only referenced in R3_REVISION_HANDOFF.md / PAIRED_ABLATION_BOOTSTRAP.md
from a prior session on a different machine) and could not be recovered. This is
a from-scratch reimplementation of the same methodology, validated to reproduce
the original 50-iteration findings on an independent 100-iteration seeded run
(see PAIRED_ABLATION_BOOTSTRAP.md for the comparison).

Requires both arms already scored:
    fresh_cf_iterations/iteration_*/successful/successful_counterfactuals.csv   (SCM arm)
    fresh_cf_iterations/aggregated_results_no_scm/successful_counterfactuals_unfiltered_all.csv
        (no-SCM arm; produced by run_unfiltered_ablation.py or run_unfiltered_ablation_fast.py --finalize)

Each of N_BOOT resamples draws 48 patients (with replacement) once; BOTH arms
are scored on that same resampled patient set, so the contrast is paired.

Sign convention: contrast = SCM − no-SCM. A positive value means the SCM arm
scores higher; negative means the unfiltered arm scores higher.

Usage:
    python scripts/paired_ablation_bootstrap.py --iterations_dir fresh_cf_iterations --n_boot 5000
"""
import argparse
import glob
import os
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.pipeline.metrics_calculator import MetricsCalculator

FEATURES = ['trestbps', 'cp', 'exang', 'oldpeak', 'thalach', 'slope', 'restecg']
METRIC_KEYS = {f: f'{f}_improved_pct' for f in FEATURES}


def load_attempts(iterations_dir):
    att = Counter()
    for f in glob.glob(os.path.join(iterations_dir, 'iteration_*', 'counterfactuals', 'patient_*_cf_*.csv')):
        pid = re.search(r'patient_(\d+)_cf_', os.path.basename(f)).group(1)
        att[pid] += 1
    return att


def load_scm_successful(iterations_dir):
    frames = []
    for f in glob.glob(os.path.join(iterations_dir, 'iteration_*', 'successful', 'successful_counterfactuals.csv')):
        try:
            d = pd.read_csv(f)
        except Exception:
            continue
        if len(d):
            frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    df['patient_id'] = df['patient_id'].astype(int).astype(str)
    return df


def main():
    parser = argparse.ArgumentParser(description='Paired patient-cluster bootstrap: SCM vs no-SCM ablation contrast')
    parser.add_argument('--iterations_dir', default='fresh_cf_iterations')
    parser.add_argument('--n_boot', type=int, default=5000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', default=None)
    args = parser.parse_args()

    iterations_dir = args.iterations_dir
    noscm_path = os.path.join(iterations_dir, 'aggregated_results_no_scm',
                               'successful_counterfactuals_unfiltered_all.csv')
    if not os.path.exists(noscm_path):
        raise FileNotFoundError(
            f'{noscm_path} not found. Run run_unfiltered_ablation.py (or '
            f'run_unfiltered_ablation_fast.py --finalize) on {iterations_dir} first.'
        )

    print('Loading attempts and successful CFs...')
    att = load_attempts(iterations_dir)
    scm_df = load_scm_successful(iterations_dir)
    noscm_df = pd.read_csv(noscm_path)
    noscm_df['patient_id'] = noscm_df['patient_id'].astype(str)

    patients = np.array(sorted(att.keys(), key=int))
    A = np.array([att[p] for p in patients], float)
    scm_by_pid = {p: g for p, g in scm_df.groupby('patient_id')}
    noscm_by_pid = {p: g for p, g in noscm_df.groupby('patient_id')}
    S_scm = np.array([len(scm_by_pid[p]) if p in scm_by_pid else 0 for p in patients], float)
    S_nscm = np.array([len(noscm_by_pid[p]) if p in noscm_by_pid else 0 for p in patients], float)

    print(f'{len(patients)} patients, {int(A.sum())} attempts, '
          f'{int(S_scm.sum())} SCM successes, {int(S_nscm.sum())} no-SCM successes')

    calc = MetricsCalculator()
    n = len(patients)
    idx = np.arange(n)
    rng = np.random.default_rng(args.seed)

    results = {k: [] for k in ['acceptance'] + list(METRIC_KEYS.values())}

    for b in range(args.n_boot):
        samp = rng.choice(idx, size=n, replace=True)
        samp_patients = patients[samp]

        a_sum = A[samp].sum()
        scm_rate = 100 * S_scm[samp].sum() / a_sum
        nscm_rate = 100 * S_nscm[samp].sum() / a_sum
        results['acceptance'].append(scm_rate - nscm_rate)  # SCM - no-SCM

        scm_frames = [scm_by_pid[p] for p in samp_patients if p in scm_by_pid]
        nscm_frames = [noscm_by_pid[p] for p in samp_patients if p in noscm_by_pid]
        scm_sample = pd.concat(scm_frames, ignore_index=True) if scm_frames else pd.DataFrame()
        nscm_sample = pd.concat(nscm_frames, ignore_index=True) if nscm_frames else pd.DataFrame()

        scm_metrics = calc.compute_all_metrics(scm_sample)
        nscm_metrics = calc.compute_all_metrics(nscm_sample)

        for f, key in METRIC_KEYS.items():
            results[key].append(scm_metrics.get(key, 0.0) - nscm_metrics.get(key, 0.0))  # SCM - no-SCM

        if (b + 1) % 1000 == 0:
            print(f'{b + 1}/{args.n_boot} done', flush=True)

    print()
    print(f"{'Contrast (SCM - no-SCM)':<28}{'Mean':>8}{'95% CI':>22}{'Survives?':>12}")
    labels = {'acceptance': 'Acceptance rate', **{v: f'{k} improved' for k, v in METRIC_KEYS.items()}}
    rows_out = []
    for key, vals in results.items():
        vals = np.array(vals)
        mean = vals.mean()
        lo, hi = np.percentile(vals, [2.5, 97.5])
        survives = 'yes' if (lo > 0 or hi < 0) else 'no'
        print(f"{labels[key]:<28}{mean:>8.1f}{f'[{lo:.1f}, {hi:.1f}]':>22}{survives:>12}")
        rows_out.append({'contrast': labels[key], 'mean': mean, 'ci_lower': lo, 'ci_upper': hi, 'survives': survives})

    out_path = args.output or os.path.join(iterations_dir, 'aggregated_results', 'paired_ablation_bootstrap.csv')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    pd.DataFrame(rows_out).to_csv(out_path, index=False)
    print(f'\nSaved to {out_path}')


if __name__ == '__main__':
    main()
