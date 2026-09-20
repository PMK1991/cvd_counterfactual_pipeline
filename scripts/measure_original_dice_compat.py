"""Frozen-original-model, paired legacy/shared-adapter measurement (no retraining)."""
import argparse
import contextlib
import copy
import hashlib
import importlib.metadata
import io
import json
import logging
import os
from pathlib import Path
import pickle
import random
import subprocess
import sys
import time
import types
from concurrent.futures import ProcessPoolExecutor, as_completed

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[key] = "1"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits
import yaml
from src.pipeline.dice_cf_generator import DiceCFGenerator, derive_seed
from src.pipeline.dice_compat import NativeNumericClassifier, CATEGORICAL_SCHEMA
from src.pipeline.scm_analyzer import SCMAnalyzer
from src.utils.dataLoader import DataLoader

logging.disable(logging.WARNING)
ENDPOINTS = ("raw_flip", "scm_accept", "propagated_flip", "joint")
PROTECTED = ("data", "model", "fresh_cf_iterations",
             "fresh_cf_iterations_archive_published_20260607", "backups",
             "fresh_cf_iterations_run_unseeded_20260727_153650",
             "kfold_runs/primary_seed42")
SOURCE_PATHS = ("src/pipeline/dice_cf_generator.py", "src/pipeline/dice_compat.py",
                "src/pipeline/scm_analyzer.py", "src/utils/dataLoader.py",
                "src/training/train_model.py", "pipeline_config.yaml",
                "scripts/measure_original_dice_compat.py")


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_suffix(path.suffix + ".partial")
    part.write_text(json.dumps(value, indent=2, default=str, allow_nan=False))
    part.replace(path)


def legacy_class():
    source = subprocess.check_output(
        ["git", "show", "a3ce882a23d80a8be502032fea7262bc5aba0d16:src/pipeline/dice_cf_generator.py"],
        text=True)
    module = types.ModuleType("legacy_dice_measurement")
    module.__file__ = str(ROOT / "src/pipeline/dice_cf_generator.py")
    exec(compile(source, module.__file__, "exec"), module.__dict__)
    return module.DiceCFGenerator, hashlib.sha256(source.encode()).hexdigest()


def load_inputs():
    cfg = yaml.safe_load((ROOT / "pipeline_config.yaml").read_text())
    loader = DataLoader(cfg["dice"]["data_path"])
    with contextlib.redirect_stdout(io.StringIO()):
        data = loader.load_clean_data()
    with open(cfg["dice"]["model_path"], "rb") as f:
        model = pickle.load(f)
    train, test = train_test_split(data, test_size=0.2, random_state=42)
    cohort = test[(test.target == 1) & (model.predict(test.drop(columns="target")) == 1)]
    return cfg, data, train, test, cohort, model


def generators(cfg):
    old, _ = legacy_class()
    result = {}
    for arm, cls in (("legacy", old), ("corrected", DiceCFGenerator)):
        gen = cls(cfg["dice"]["model_path"], cfg["dice"]["data_path"], copy.deepcopy(cfg["dice"]))
        with contextlib.redirect_stdout(io.StringIO()):
            gen.load_model_and_data()
            gen.setup_dice_explainer()
        result[arm] = gen
    return result


def protected_inventory():
    result = {}
    for directory in PROTECTED:
        for path in sorted((ROOT / directory).rglob("*")):
            if path.is_file():
                result[str(path.relative_to(ROOT))] = {
                    "bytes": path.stat().st_size, "sha256": sha(path)}
    # Preserve dirty manuscripts and existing tracked files, too.
    tracked = subprocess.check_output(["git", "ls-files"], text=True).splitlines()
    for name in tracked:
        path = ROOT / name
        if path.is_file():
            result[str(path.relative_to(ROOT))] = {
                "bytes": path.stat().st_size, "sha256": sha(path)}
    return result


def prediction_diff(native, other):
    d = np.abs(native - other)
    return {"n": len(native), "probability_changed_gt_1e-7": int((d > 1e-7).sum()),
            "max_abs_probability_difference": float(d.max()),
            "mean_abs_probability_difference": float(d.mean()),
            "class_changed": int(((native >= .5) != (other >= .5)).sum())}


def audit(output):
    if (output / "manifest.json").exists():
        raise RuntimeError("Audit output already exists; use a new directory.")
    cfg, data, train, test, cohort, model = load_inputs()
    gens = generators(cfg)
    versions = {p: importlib.metadata.version(p) for p in
                ("numpy", "pandas", "scikit-learn", "xgboost", "dowhy", "dice-ml")}
    with open(ROOT / "model/scm_full.pkl", "rb") as f:
        scm_art = pickle.load(f)
    scm_meta = {k: v for k, v in scm_art.items() if k != "causal_model"}
    archive = ROOT / "fresh_cf_iterations_archive_published_20260607"
    historical = pd.read_csv(archive / "aggregated_results/all_iteration_metrics.csv")
    archive_checks = []
    for _, row in historical.iterrows():
        d = archive / f"iteration_{int(row.iteration):03d}"
        m = json.loads((d / "metrics.json").read_text())
        archive_checks.append(all(m[k] == row[k] for k in
                                  ("total_generated_cfs", "total_successful_cfs", "total_patients")))
    audit_result = {"cohort": {"eligible": len(data), "train": len(train), "test": len(test),
                             "test_positive": int(test.target.sum()), "true_positive": len(cohort)},
                    "slope_zero": {n: int((d.slope == 0).sum()) for n, d in
                                   (("eligible", data), ("train", train), ("test", test), ("TP", cohort))},
                    "historical": {
                        "archive": str(archive), "iterations": len(historical),
                        "attempts": int(historical.total_patients.sum()),
                        "requested": int(historical.total_requested_cfs.sum()),
                        "returned": int(historical.total_generated_cfs.sum()),
                        "scm_accepted": int(historical.total_successful_cfs.sum()),
                        "pooled_scm_acceptance_pct": float(100 * historical.total_successful_cfs.sum() /
                                                         historical.total_generated_cfs.sum()),
                        "mean_repeat_acceptance_pct": float(historical.target_flip_rate_pct.mean()),
                        "mean_repeat_accepted": float(historical.total_successful_cfs.mean()),
                        "iteration_metrics_match_aggregate": all(archive_checks)},
                    "paths": {}, "rows": []}
    numeric_wrapper = NativeNumericClassifier(model, list(cohort.drop(columns="target")))
    for name, frame in (("eligible", data), ("train", train), ("test", test), ("TP", cohort)):
        x = frame.drop(columns="target")
        native = model.predict_proba(x)[:, 1]
        paths = {}
        for arm, gen in gens.items():
            q = gen.dice_data.prepare_query_instance(x)
            encoded = gen.dice_exp.label_encode(q.copy()).to_numpy()
            paths[f"{arm}_genetic"] = gen.dice_exp.predict_fn_scores(encoded)[:, 1]
            # Identical call and prepared frame used by build_KD_tree.
            paths[f"{arm}_kdtree"] = gen.dice_model.get_output(q, model_score=True)[:, 1]
            paths[f"{arm}_posthoc_callable"] = gen.dice_exp.predict_fn_for_sparsity(q)[:, 1]
            if arm == "legacy":
                paths["numeric_cast_only"] = numeric_wrapper.predict_proba(gen.dice_exp.label_decode(encoded))[:, 1]
        audit_result["paths"][name] = {k: prediction_diff(native, v) for k, v in paths.items()}
        if name == "TP":
            for i, (source, row) in enumerate(frame.iterrows()):
                saved = pd.read_csv(archive / f"iteration_000/original/patient_{i}.csv")
                differences = np.abs(saved[x.columns].to_numpy(dtype=float) -
                                     x.iloc[[i]].to_numpy(dtype=float))
                audit_result["rows"].append({
                    "record_id": i, "source_row_id": int(source), "slope": int(row.slope),
                    "native_probability": float(native[i]),
                    **{k: float(v[i]) for k, v in paths.items()},
                    "legacy_opposite_target": int(paths["legacy_genetic"][i] < .5),
                    "corrected_opposite_target": int(paths["corrected_genetic"][i] < .5),
                    "archive_factual_target": int(saved.target.iloc[0]),
                    "archive_factual_max_feature_difference": float(differences.max())})
    output.mkdir(parents=True, exist_ok=True)
    cohort.assign(record_id=range(len(cohort)), source_row_id=cohort.index).to_csv(
        output / "cohort.csv", index=False)
    _, legacy_hash = legacy_class()
    manifest = {
        "created_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "branch": subprocess.check_output(["git", "branch", "--show-current"], text=True).strip(),
        "python": sys.executable, "python_version": sys.version, "versions": versions,
        "source_hashes": {p: sha(ROOT / p) for p in SOURCE_PATHS},
        "legacy_source_sha256": legacy_hash, "scm_metadata": scm_meta, "config": cfg,
        "records": [{"record_id": i, "source_row_id": int(v)} for i, v in enumerate(cohort.index)],
        "comparison": "Matched reproducible reconstruction, not exact historical unseeded reproduction",
        "arms": ["legacy HEAD generator", "current shared-adapter generator"],
        "desired_class": "opposite (unchanged)",
        "seeds": "derive_seed(iteration, record_id, 42), both numpy and Python random",
        "scm_seeding": "Unchanged SCMAnalyzer feature/proposal hash seed",
        "factual_handoff": "DiCE test_instance_df CSV round-trip, exactly as original save/load pipeline",
        "joint": "same proposal: historical SCM acceptance AND propagated native classifier prediction 0",
        "raw_flip": "numeric/native classifier prediction of returned proposal == 0",
        "scm_accept": "SCMAnalyzer.validate_counterfactual: saved factual target 1 and SCM target 0",
        "posthoc_note": "DiCE 0.11 genetic creates sparse copy; find_counterfactuals does not call sparsity search",
        "package_drift": "Original mtech-env remains available; historical per-run package manifest unavailable",
    }
    write_json(output / "manifest.json", manifest)
    write_json(output / "audit.json", audit_result)
    print(json.dumps({k: v for k, v in audit_result.items() if k != "rows"}, indent=2))
    inventory = protected_inventory()
    write_json(output / "protected_before.json", inventory)
    print(f"Protected hashes: {len(inventory)}")


_STATE = None


def init_worker():
    global _STATE
    threadpool_limits(1)
    cfg, data, train, test, cohort, model = load_inputs()
    gens = generators(cfg)
    scm = SCMAnalyzer(copy.deepcopy(cfg["scm"]))
    scm.initialize_analyzer()
    _STATE = cfg, cohort, model, gens, scm


def csv_roundtrip(frame):
    return pd.read_csv(io.StringIO(frame.to_csv(index=False)))


def measure_pair(task):
    output, iteration, record_id = task
    cfg, cohort, model, gens, scm = _STATE
    x = cohort.iloc[[record_id]].drop(columns="target")
    feature_names = list(x)
    results = {}
    for arm in ("legacy", "corrected"):
        path = Path(output) / "attempts" / arm / f"repeat_{iteration:03d}_record_{record_id:02d}.json"
        if path.exists():
            results[arm] = json.loads(path.read_text())
            continue
        start = time.monotonic()
        seed = derive_seed(iteration, record_id, 42)
        record = {"arm": arm, "iteration": iteration, "record_id": record_id,
                  "source_row_id": int(cohort.index[record_id]), "seed": seed,
                  "requested": cfg["dice"]["total_cfs"], "status": "ok",
                  "error": None, "candidates": []}
        try:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                result = gens[arm].generate_counterfactuals(x.copy(), seed=seed, strict=True)
            if result is None or not result.cf_examples_list:
                record["status"] = "no_cf"
            else:
                example = result.cf_examples_list[0]
                original = csv_roundtrip(example.test_instance_df)
                record["saved_factual"] = original.iloc[0].to_dict()
                record["desired_class"] = int(example.desired_class)
                proposals = example.final_cfs_df
                if proposals is None or not len(proposals):
                    record["status"] = "no_cf"
                else:
                    for i in range(len(proposals)):
                        proposal = csv_roundtrip(proposals.iloc[[i]])
                        raw_prob = float(model.predict_proba(proposal[feature_names])[0, 1])
                        candidate = {"proposal_id": i, "raw": proposal.iloc[0].to_dict(),
                                     "raw_probability": raw_prob,
                                     "raw_prediction": int(model.predict(proposal[feature_names])[0]),
                                     "scm_error": None}
                        try:
                            propagated = scm.apply_scm_intervention(original, proposal, strict=True)
                            covariates = pd.DataFrame([{c: propagated[f"cf_{c}"].iloc[0]
                                                       for c in feature_names}])[feature_names]
                            pred = int(model.predict(covariates)[0])
                            accepted = scm.validate_counterfactual(propagated, int(original.target.iloc[0]))
                            candidate.update(
                                propagated=covariates.iloc[0].to_dict(),
                                scm_target=int(propagated.target.iloc[0]),
                                propagated_prediction=pred,
                                propagated_probability=float(model.predict_proba(covariates)[0, 1]),
                                flags={"raw_flip": candidate["raw_prediction"] == 0,
                                       "scm_accept": bool(accepted),
                                       "propagated_flip": pred == 0,
                                       "joint": bool(accepted and pred == 0)})
                        except Exception as exc:
                            candidate["scm_error"] = f"{type(exc).__name__}: {exc}"
                            candidate["flags"] = {"raw_flip": candidate["raw_prediction"] == 0,
                                                  "scm_accept": False, "propagated_flip": False,
                                                  "joint": False}
                        record["candidates"].append(candidate)
        except Exception as exc:
            record["status"] = "generation_error"
            record["error"] = f"{type(exc).__name__}: {exc}"
        record["seconds"] = time.monotonic() - start
        record["returned"] = len(record["candidates"])
        record["counts"] = {e: sum(c["flags"][e] for c in record["candidates"]) for e in ENDPOINTS}
        record["any"] = {e: record["counts"][e] > 0 for e in ENDPOINTS}
        write_json(path, record)
        results[arm] = record
    return {arm: {k: r[k] for k in ("status", "returned", "counts", "seconds")}
            for arm, r in results.items()}


def check_sources(output):
    manifest = json.loads((output / "manifest.json").read_text())
    changed = [p for p, h in manifest["source_hashes"].items() if sha(ROOT / p) != h]
    if changed:
        raise RuntimeError(f"Source changed after audit: {changed}")


def run(output, repeats, records, workers):
    check_sources(output)
    n_records = len(json.loads((output / "manifest.json").read_text())["records"])
    ids = range(n_records) if records == "all" else [int(i) for i in records.split(",")]
    tasks = [(str(output), rep, i) for rep in range(repeats) for i in ids
             if not all((output / "attempts" / a / f"repeat_{rep:03d}_record_{i:02d}.json").exists()
                        for a in ("legacy", "corrected"))]
    write_json(output / "run_plan.json", {"repeats": repeats, "record_ids": list(ids),
                                          "workers": workers, "pending_pairs": len(tasks)})
    print(f"Running {len(tasks)} pending pairs; max workers={workers}", flush=True)
    start = time.monotonic()
    with ProcessPoolExecutor(max_workers=workers, initializer=init_worker) as pool:
        futures = {pool.submit(measure_pair, t): t for t in tasks}
        for n, future in enumerate(as_completed(futures), 1):
            value = future.result()
            if n <= 5 or n % 24 == 0 or n == len(tasks):
                print(json.dumps({"pairs_completed": n, "elapsed": time.monotonic() - start,
                                  "task": futures[future][1:], "result": value}), flush=True)
    summarize(output)


def summarize(output):
    rows = []
    for arm in ("legacy", "corrected"):
        for path in sorted((output / "attempts" / arm).glob("*.json")):
            r = json.loads(path.read_text())
            rows.append({k: r[k] for k in ("arm", "iteration", "record_id", "status", "returned")} |
                        {e: r["counts"][e] for e in ENDPOINTS} |
                        {"any_" + e: int(r["any"][e]) for e in ENDPOINTS} |
                        {"scm_errors": sum(c["scm_error"] is not None for c in r["candidates"])})
    if not rows:
        return
    df = pd.DataFrame(rows)
    df.to_csv(output / "attempt_summary.csv", index=False)
    summaries = {}
    for arm, group in df.groupby("arm"):
        sums = group[["returned", *ENDPOINTS, *("any_" + e for e in ENDPOINTS)]].sum()
        summaries[arm] = {
            "completed_attempts": len(group), "distinct_records": int(group.record_id.nunique()),
            "statuses": group.status.value_counts().to_dict(), "returned": int(sums.returned),
            "scm_errors": int(group.scm_errors.sum()),
            "candidate_counts": {e: int(sums[e]) for e in ENDPOINTS},
            "candidate_rates": {e: float(sums[e] / sums.returned) if sums.returned else None for e in ENDPOINTS},
            "attempt_counts": {e: int(sums["any_" + e]) for e in ENDPOINTS},
            "attempt_rates": {e: float(sums["any_" + e] / len(group)) for e in ENDPOINTS}}
    keys = df.groupby(["iteration", "record_id"]).arm.nunique()
    paired_keys = set(keys[keys == 2].index)
    paired = df[[tuple(v) in paired_keys for v in df[["iteration", "record_id"]].to_numpy()]]
    per_record = paired.groupby(["arm", "record_id"])[
        ["returned", *ENDPOINTS, *("any_" + e for e in ENDPOINTS)]].sum()
    record_ids = sorted(paired.record_id.unique())
    rng = np.random.default_rng(20260912)
    boot = rng.integers(0, len(record_ids), size=(10000, len(record_ids)))
    contrasts = {}
    for e in ENDPOINTS:
        arm_rates, arm_boot = {}, {}
        for arm in ("legacy", "corrected"):
            values = per_record.loc[arm].reindex(record_ids)
            num, den = values[e].to_numpy(), values.returned.to_numpy()
            arm_rates[arm] = float(num.sum() / den.sum())
            arm_boot[arm] = num[boot].sum(axis=1) / den[boot].sum(axis=1)
        dif = arm_boot["corrected"] - arm_boot["legacy"]
        contrasts[e] = {"delta_pp": 100 * (arm_rates["corrected"] - arm_rates["legacy"]),
                        "paired_record_bootstrap_95pct_pp": (100 * np.quantile(dif, [.025, .975])).tolist()}
    plan = json.loads((output / "run_plan.json").read_text()) if (output / "run_plan.json").exists() else {}
    result = {"arms": summaries, "paired_pairs": len(paired_keys),
              "planned_attempts_per_arm": plan.get("repeats", 0) * len(plan.get("record_ids", [])),
              "contrasts": contrasts,
              "bootstrap": "10,000 paired record-cluster resamples; fixed models, no retraining; records not verified patients"}
    write_json(output / "summary.json", result)
    print(json.dumps(result, indent=2), flush=True)


def verify(output):
    before = json.loads((output / "protected_before.json").read_text())
    differences = []
    for name, expected in before.items():
        path = ROOT / name
        if not path.exists() or path.stat().st_size != expected["bytes"] or sha(path) != expected["sha256"]:
            differences.append(name)
    result = {"files_verified": len(before), "changed_or_missing": differences}
    write_json(output / "protected_verification.json", result)
    print(json.dumps(result))
    if differences:
        raise RuntimeError("Protected files changed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("audit", "run", "summary", "verify"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--records", default="all")
    parser.add_argument("--workers", type=int, choices=range(1, 5), default=4)
    args = parser.parse_args()
    with threadpool_limits(1):
        if args.mode == "audit":
            audit(args.output)
        elif args.mode == "run":
            run(args.output, args.repeats, args.records, args.workers)
        elif args.mode == "summary":
            summarize(args.output)
        else:
            verify(args.output)
