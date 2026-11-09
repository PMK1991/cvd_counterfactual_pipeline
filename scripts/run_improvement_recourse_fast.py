"""
Checkpointed, process-parallel wrapper around RecourseAnalyzer.analyze_iteration_all,
for time-boxed environments where a full 100-iteration serial re-score (each
iteration costs ~15-25s of CPU-bound SCM Monte Carlo sampling) can't complete
in one invocation.

Each worker process loads the SCM once (via a module-level cache) and then
processes assigned iterations; results are checkpointed to disk immediately
so a killed/timed-out invocation loses at most the in-flight iterations.

Usage (re-run repeatedly until it reports "ALL DONE"):
    python recourse_batch.py <iterations_dir> --max_iters 4 --time_budget 40

Then finalize:
    python recourse_batch.py <iterations_dir> --finalize
"""
import argparse
import glob
import json
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, '/sessions/practical-gallant-noether/mnt/cvd_counterfactual_pipeline')

_ANALYZER = None


def _get_analyzer():
    global _ANALYZER
    if _ANALYZER is None:
        from src.pipeline.recourse_analyzer import RecourseAnalyzer
        import yaml
        cfg_path = '/sessions/practical-gallant-noether/mnt/cvd_counterfactual_pipeline/pipeline_config.yaml'
        scm_cfg = {'graph_structure': 'full', 'n_samples': 1000, 'fit_seed': 42, 'model_dir': 'model'}
        try:
            with open(cfg_path) as f:
                y = yaml.safe_load(f)
            scm_cfg.update(y.get('scm', {}))
        except Exception:
            pass
        analyzer = RecourseAnalyzer()
        analyzer.config.update(scm_cfg)
        analyzer.initialize_analyzer()
        _ANALYZER = analyzer
    return _ANALYZER


def process_iteration(it_dir):
    import json as _json
    sys.path.insert(0, '/sessions/practical-gallant-noether/mnt/cvd_counterfactual_pipeline')
    from src.pipeline.metrics_calculator import MetricsCalculator
    analyzer = _get_analyzer()
    calc = MetricsCalculator()

    iteration_num = int(re.search(r'iteration_(\d+)', os.path.basename(it_dir)).group(1))
    scored = analyzer.analyze_iteration_all(it_dir)
    n_scored = len(scored)
    if n_scored == 0:
        non_flip = scored
        n_flip = 0
    else:
        n_flip = int(scored['flipped'].sum())
        non_flip = scored[~scored['flipped']].reset_index(drop=True)
    n_non_flip = n_scored - n_flip

    metrics = calc.compute_all_metrics(non_flip)
    metrics.update(calc.compute_recourse_metrics(non_flip))
    metrics['iteration'] = iteration_num
    metrics['n_scored_cfs'] = n_scored
    metrics['n_flip_cfs'] = n_flip
    metrics['n_non_flip_cfs'] = n_non_flip
    metrics['non_flip_rate_pct'] = (n_non_flip / n_scored * 100) if n_scored else 0.0

    mfile = os.path.join(it_dir, 'metrics.json')
    total_generated = total_requested = total_patients = None
    if os.path.exists(mfile):
        with open(mfile) as fh:
            src = _json.load(fh)
        total_generated = src.get('total_generated_cfs')
        total_requested = src.get('total_requested_cfs')
        total_patients = src.get('total_patients')
    metrics['total_patients'] = total_patients
    metrics['total_requested_cfs'] = total_requested
    metrics['total_generated_cfs'] = total_generated

    # sanitize for JSON (numpy types etc.)
    clean = {}
    for k, v in metrics.items():
        try:
            _json.dumps(v)
            clean[k] = v
        except TypeError:
            clean[k] = float(v)
    return iteration_num, clean


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('iterations_dir')
    parser.add_argument('--max_iters', type=int, default=4)
    parser.add_argument('--time_budget', type=float, default=40.0)
    parser.add_argument('--n_workers', type=int, default=2)
    parser.add_argument('--finalize', action='store_true')
    args = parser.parse_args()

    partial_dir = os.path.join(args.iterations_dir, 'aggregated_results_recourse', '_partial')
    os.makedirs(partial_dir, exist_ok=True)

    if args.finalize:
        import pandas as pd
        sys.path.insert(0, '/sessions/practical-gallant-noether/mnt/cvd_counterfactual_pipeline')
        from src.pipeline.ci_computer import CIComputer
        files = sorted(glob.glob(os.path.join(partial_dir, '*.json')))
        print(f'{len(files)} partial files found')
        all_metrics = []
        for f in files:
            with open(f) as fh:
                all_metrics.append(json.load(fh))
        agg = pd.DataFrame(all_metrics).sort_values('iteration').reset_index(drop=True)
        out_dir = os.path.join(args.iterations_dir, 'aggregated_results_recourse')
        agg.to_csv(os.path.join(out_dir, 'all_iteration_metrics.csv'), index=False)
        ci = CIComputer(confidence_level=0.95)
        ci_results = ci.compute_confidence_intervals(agg)
        ci.save_results(ci_results, out_dir)
        print(f'n_iterations: {len(agg)}')
        print(f'Mean non-flip CFs/iter: {agg["n_non_flip_cfs"].mean():.2f}')
        print(f'Mean IRR strict: {agg["recourse_irr_strict_pct"].mean():.2f}%')
        print(f'Mean any-improvement: {agg["recourse_any_improvement_pct"].mean():.2f}%')
        print(f'Written to {out_dir}')
        return

    all_iter_dirs = sorted(glob.glob(os.path.join(args.iterations_dir, 'iteration_*')))
    done = {os.path.splitext(os.path.basename(p))[0] for p in glob.glob(os.path.join(partial_dir, '*.json'))}
    todo = [d for d in all_iter_dirs if os.path.basename(d) not in done]
    print(f'{len(all_iter_dirs)} total, {len(done)} done, {len(todo)} remaining', flush=True)

    t0 = time.time()
    n_done = 0
    batch = todo[:max(args.max_iters, args.n_workers)]
    with ProcessPoolExecutor(max_workers=args.n_workers) as ex:
        futures = {ex.submit(process_iteration, d): d for d in batch}
        for fut in as_completed(futures):
            if time.time() - t0 > args.time_budget and n_done >= args.n_workers:
                break
            try:
                itn, metrics = fut.result()
            except Exception as e:
                print(f'FAILED {futures[fut]}: {e}', flush=True)
                continue
            with open(os.path.join(partial_dir, f'iteration_{itn:03d}.json'), 'w') as fh:
                json.dump(metrics, fh)
            n_done += 1
            print(f'  iteration_{itn:03d}: {metrics["n_non_flip_cfs"]} non-flip, '
                  f'IRR-strict {metrics["recourse_irr_strict_pct"]:.1f}%  ({time.time()-t0:.1f}s elapsed)', flush=True)
            if n_done >= args.max_iters:
                break

    remaining = len(all_iter_dirs) - (len(done) + n_done)
    print(f'Processed {n_done} this call in {time.time()-t0:.1f}s. {remaining} remaining.', flush=True)
    if remaining <= 0:
        print('ALL DONE', flush=True)


if __name__ == '__main__':
    main()
