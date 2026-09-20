"""Leakage-isolated, prespecified five-fold CVD recourse experiment."""
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import pickle
import platform
import random
import sys
import time
import traceback

# Set before importing numerical libraries, also in Windows spawned workers.
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_key] = "1"

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits
import yaml

SCHEMA = 1
ENDPOINTS = ("raw_flip", "scm_accept", "propagated_flip", "both")
SOURCE_FILES = (
    "scripts/run_kfold_recourse.py", "src/training/train_model.py",
    "src/training/train_scm.py", "src/pipeline/dice_cf_generator.py",
    "src/pipeline/scm_analyzer.py", "src/utils/dataLoader.py",
    "src/pipeline/kfold_dice.py", "src/pipeline/dice_compat.py",
    "requirements-kfold.txt",
)


def json_bytes(value):
    return json.dumps(value, sort_keys=True, allow_nan=False,
                      separators=(",", ":")).encode("utf-8")


def digest(value):
    return hashlib.sha256(json_bytes(value)).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_bytes(path, content):
    """Crash leftovers are recognizable .partial files, never committed records."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    with open(partial, "wb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(partial, path)


def write_record(path, payload):
    atomic_bytes(path, json_bytes({"sha256": digest(payload), "payload": payload}))


def read_record(path):
    with open(path, encoding="utf-8") as stream:
        envelope = json.load(stream)
    if set(envelope) != {"sha256", "payload"}:
        raise ValueError(f"Invalid record envelope: {path}")
    if digest(envelope["payload"]) != envelope["sha256"]:
        raise ValueError(f"Checksum mismatch: {path}")
    return envelope["payload"]


@contextmanager
def run_lock(output):
    output.mkdir(parents=True, exist_ok=True)
    lock = output / "RUN.lock"
    fd = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump({"pid": os.getpid(), "host": platform.node()}, stream)
        yield
    finally:
        lock.unlink()


def eligible_data(path):
    # Preserve zero-based input CSV row indices; no synthetic ID enters features.
    raw = pd.read_csv(path)
    eligible = raw.dropna()
    eligible = eligible[(eligible.chol > 0) & (eligible.trestbps > 0)]
    eligible = eligible.drop_duplicates()
    if not eligible.index.is_unique or set(eligible.target.unique()) != {0, 1}:
        raise ValueError("Expected unique source row indices and binary target")
    if not np.isfinite(eligible.to_numpy(dtype=float)).all():
        raise ValueError("Non-finite eligible data")
    return eligible


def filter_training(training):
    """Fit/apply target-specific sequential IQR fences to training rows only."""
    retained, fences = [], []
    for target in (0, 1):
        group = training[training.target == target].copy()
        for column in ("chol", "trestbps"):
            q1, q3 = group[column].quantile([0.25, 0.75])
            lo, hi = float(q1 - 1.5 * (q3 - q1)), float(q3 + 1.5 * (q3 - q1))
            before = len(group)
            group = group[group[column].between(lo, hi)]
            fences.append({"target": target, "column": column, "lower": lo,
                           "upper": hi, "before": before, "after": len(group)})
        if group.empty:
            raise ValueError(f"Empty training class {target} after IQR filtering")
        retained.append(group)
    return pd.concat(retained), fences


def fold_plan(data, seed=42):
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    plans = []
    for fold, (train_pos, validation_pos) in enumerate(splitter.split(data, data.target)):
        train, fences = filter_training(data.iloc[train_pos])
        plans.append({
            "fold": fold,
            "outer_train_ids": data.index[train_pos].tolist(),
            "training_ids": train.index.tolist(),
            "validation_ids": data.index[validation_pos].tolist(),
            "iqr_fences": fences,
        })
    validate_plan(data, plans)
    return plans


def validate_plan(data, plans):
    seen = []
    all_ids = set(data.index)
    for plan in plans:
        outer = set(plan["outer_train_ids"])
        valid = set(plan["validation_ids"])
        train = set(plan["training_ids"])
        if outer & valid or outer | valid != all_ids or not train <= outer:
            raise ValueError("Fold leakage or missing training provenance")
        seen.extend(plan["validation_ids"])
    if len(seen) != len(set(seen)) or set(seen) != all_ids:
        raise ValueError("OOF coverage is incomplete or duplicated")


def make_manifest(data_path, config_path):
    from src.training.train_model import BASELINE_XGB_PARAMS
    from src.pipeline.kfold_dice import CATEGORICAL_SCHEMA
    with open(config_path, encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not config["dice"].get("deterministic_seeding", False):
        raise ValueError("This experiment requires deterministic DiCE seeds")
    if config["pipeline"]["n_iterations"] < 1:
        raise ValueError("At least one repeat required")
    versions = {dist.metadata["Name"].lower(): dist.version
                for dist in importlib.metadata.distributions() if dist.metadata["Name"]}
    data = eligible_data(data_path)
    manifest = {
        "schema": SCHEMA, "design": "prespecified stratified outer five-fold; no tuning",
        "seed": 42, "data_path": str(data_path.resolve()),
        "classifier_params": dict(BASELINE_XGB_PARAMS, random_state=42, n_jobs=1),
        "categorical_schema": {c: list(v) for c, v in CATEGORICAL_SCHEMA.items()},
        "dice_desired_class": 0,
        "data_sha256": file_hash(data_path), "config": config,
        "source_sha256": {p: file_hash(ROOT / p) for p in SOURCE_FILES},
        "python": platform.python_version(), "platform": platform.platform(),
        "packages": versions,
        "eligible_ids": data.index.tolist(), "columns": data.columns.tolist(),
        "folds": fold_plan(data), "expected_repeats": config["pipeline"]["n_iterations"],
    }
    return manifest, data


def assert_manifest(saved, current):
    if saved != current:
        raise ValueError("Manifest mismatch: data/config/code/environment changed; use a NEW output directory")


def frame_rows(frame):
    return json.loads(frame.to_json(orient="records"))


def check_fold(output, manifest, plan):
    directory = output / f"fold_{plan['fold']}"
    ready = read_record(directory / "ready.json")
    if ready["manifest_id"] != digest(manifest) or ready["plan"] != plan:
        raise ValueError("Fold provenance mismatch")
    for name, checksum in ready["files"].items():
        if file_hash(directory / name) != checksum:
            raise ValueError(f"Fold artifact corrupt: {directory / name}")
    return ready


def prepare(output, manifest, data):
    from dowhy import gcm
    from src.training.train_model import build_xgb_pipeline, compute_metrics
    from src.training.train_scm import fit_one
    gcm.config.set_default_n_jobs(1)
    for plan in manifest["folds"]:
        fold = plan["fold"]
        directory = output / f"fold_{fold}"
        if (directory / "ready.json").exists():
            check_fold(output, manifest, plan)
            print(f"Fold {fold}: verified existing artifacts", flush=True)
            continue
        # An incomplete fold has no consumable ready marker and is retrained.
        training = data.loc[plan["training_ids"]].copy()
        validation = data.loc[plan["validation_ids"]].copy()
        X = training.drop(columns="target")
        print(f"Fold {fold}: fitting classifier + SCM on {len(training)} rows; "
              f"validation={len(validation)}", flush=True)
        np.random.seed(42)
        random.seed(42)
        with threadpool_limits(limits=1):
            classifier = build_xgb_pipeline(X, manifest["classifier_params"])
            classifier.fit(X, training.target)
            causal_model = fit_one(manifest["config"]["scm"]["graph_structure"],
                                   training, manifest["config"]["scm"].get("fit_seed", 42))
            predictions = classifier.predict(validation.drop(columns="target"))
            scores = classifier.predict_proba(validation.drop(columns="target"))[:, 1]
        oof = [{"source_row_id": int(row_id), "fold": fold,
                "target": int(validation.loc[row_id, "target"]),
                "prediction": int(pred), "score": float(score)}
               for row_id, pred, score in zip(validation.index, predictions, scores)]
        scm_config = manifest["config"]["scm"]
        scm_artifact = {
            "causal_model": causal_model, "graph_structure": scm_config["graph_structure"],
            "fit_seed": scm_config.get("fit_seed", 42), "fit_data": f"outer_fold_{fold}_training",
            "n_rows": len(training), "training_ids": plan["training_ids"],
            "manifest_id": digest(manifest), "versions": {},
        }
        scm_name = f"scm_{scm_config['graph_structure']}.pkl"
        files = {
            "classifier.pkl": pickle.dumps(classifier),
            scm_name: pickle.dumps(scm_artifact),
            "training.pkl": pickle.dumps(training),
            "validation.pkl": pickle.dumps(validation),
            "training.csv": training.to_csv(index_label="source_row_id").encode(),
            "validation.csv": validation.to_csv(index_label="source_row_id").encode(),
        }
        for name, content in files.items():
            atomic_bytes(directory / name, content)
        ready = {"manifest_id": digest(manifest), "plan": plan,
                 "files": {name: file_hash(directory / name) for name in files},
                 "oof": oof,
                 "metrics": compute_metrics(validation.target, predictions, scores),
                 "tp_ids": [r["source_row_id"] for r in oof
                            if r["target"] == 1 and r["prediction"] == 1]}
        write_record(directory / "ready.json", ready)


def preflight(output, manifest, data, prepared=False):
    from src.pipeline.kfold_dice import FoldDiceGenerator, preflight_generator
    from src.training.train_model import build_xgb_pipeline
    reports = []
    for plan in manifest["folds"]:
        training = data.loc[plan["training_ids"]]
        validation = data.loc[plan["validation_ids"]]
        with threadpool_limits(limits=1):
            if prepared:
                with open(output / f"fold_{plan['fold']}" / "classifier.pkl", "rb") as stream:
                    classifier = pickle.load(stream)
            else:
                classifier = build_xgb_pipeline(training.drop(columns="target"),
                                                manifest["classifier_params"])
                classifier.fit(training.drop(columns="target"), training.target)
            generator = FoldDiceGenerator("", "", manifest["config"]["dice"])
            generator.initialize_for_training(classifier, training)
            report = preflight_generator(generator, validation)
            report["fold"] = plan["fold"]
            reports.append(report)
            print(f"Preflight fold={plan['fold']}: all {report['checked_tp_count']} TPs "
                  f"encoded; max probability difference={report['max_probability_difference']}", flush=True)
    write_record(output / ("preflight_prepared.json" if prepared else "preflight.json"),
                 {"manifest_id": digest(manifest), "folds": reports,
                  "checked_tp_count": sum(r["checked_tp_count"] for r in reports)})
    return reports


def candidate_flags(raw_prediction, scm_target, propagated_prediction):
    scm = int(scm_target) == 0
    propagated = int(propagated_prediction) == 0
    return {"raw_flip": int(raw_prediction) == 0, "scm_accept": scm,
            "propagated_flip": propagated, "both": scm and propagated}


def attempt_flags(candidates):
    return {name: any(c["flags"][name] for c in candidates) for name in ENDPOINTS}


def attempt_path(output, fold, source_id, repeat):
    return output / "attempts" / f"f{fold}_s{source_id}_r{repeat}.json"


def validate_attempt(record, key, manifest_id):
    if (record["fold"], record["source_row_id"], record["repeat"]) != key:
        raise ValueError("Attempt identity mismatch")
    if record["manifest_id"] != manifest_id:
        raise ValueError("Attempt manifest mismatch")
    candidates = record["candidates"]
    if record["returned_proposals"] != len(candidates):
        raise ValueError("Proposal count mismatch")
    if record["flags"] != attempt_flags(candidates):
        raise ValueError("Attempt endpoint mismatch")
    for i, candidate in enumerate(candidates):
        if candidate["proposal_id"] != i:
            raise ValueError("Duplicate or missing proposal identity")
        expected = candidate_flags(candidate["raw_prediction"], candidate["scm_target"],
                                   candidate["propagated_prediction"])
        if candidate["flags"] != expected:
            raise ValueError("Candidate endpoint mismatch")
    if record["status"] not in ("ok", "no_cf"):
        raise ValueError("Nonterminal attempt in completed records")
    if (record["status"] == "no_cf") != (not candidates):
        raise ValueError("Status/proposal mismatch")


_CACHE = {}


def worker_context(output, manifest, fold):
    key = (str(output), fold)
    if key not in _CACHE:
        from dowhy import gcm
        from src.pipeline.kfold_dice import FoldDiceGenerator
        from src.pipeline.scm_analyzer import SCMAnalyzer
        gcm.config.set_default_n_jobs(1)
        directory = output / f"fold_{fold}"
        with open(directory / "training.pkl", "rb") as stream:
            training = pickle.load(stream)
        with open(directory / "validation.pkl", "rb") as stream:
            validation = pickle.load(stream)
        generator = FoldDiceGenerator(str(directory / "classifier.pkl"), "",
                                    manifest["config"]["dice"])
        generator.load_model_and_data(training_data=training)
        generator.setup_dice_explainer()
        scm_config = dict(manifest["config"]["scm"], model_dir=str(directory))
        analyzer = SCMAnalyzer(scm_config)
        analyzer.initialize_analyzer()
        _CACHE[key] = generator, analyzer, validation
    return _CACHE[key]


def known_no_cf(error):
    # DiCE 0.11 ExplainerBase._check_any_counterfactuals_computed.
    from raiutils.exceptions import UserConfigValidationException
    return (isinstance(error, UserConfigValidationException) and
            str(error) == "No counterfactuals found for any of the query points! Kindly check your configuration.")


def process_patient(output_text, manifest, fold, source_id, repeats):
    from src.pipeline.dice_cf_generator import derive_seed
    output = Path(output_text)
    completed = 0
    for repeat in repeats:
        path = attempt_path(output, fold, source_id, repeat)
        key = (fold, source_id, repeat)
        if path.exists():
            validate_attempt(read_record(path), key, digest(manifest))
            continue
        seed = derive_seed(repeat, source_id, manifest["config"]["dice"].get("seed_base", 42))
        proposals = None
        candidates = []
        started = time.perf_counter()
        try:
            with threadpool_limits(limits=1):
                generator, analyzer, validation = worker_context(output, manifest, fold)
                original = validation.loc[[source_id]]
                features = original.drop(columns="target").columns.tolist()
                # Fresh explainer avoids state carried across queries / resume boundaries.
                generator.setup_dice_explainer()
                no_cf_reason = None
                try:
                    result = generator.generate_counterfactuals(
                        original[features], seed=seed, strict=True)
                except Exception as error:
                    if not known_no_cf(error):
                        raise
                    result = None
                    no_cf_reason = str(error)
                if result is not None:
                    if len(result.cf_examples_list) != 1:
                        raise ValueError("Expected one DiCE query result")
                    proposals = result.cf_examples_list[0].final_cfs_df
                if proposals is not None:
                    for i in range(len(proposals)):
                        proposal = proposals.iloc[[i]].copy()
                        # DiCE categories can be strings; fitted XGB/SCM use numeric codes.
                        proposal[features] = proposal[features].apply(pd.to_numeric, errors="raise")
                        if not np.isfinite(proposal[features].to_numpy(dtype=float)).all():
                            raise ValueError("Non-finite raw proposal")
                        raw_prediction = int(generator.model.predict(proposal[features])[0])
                        propagated = analyzer.apply_scm_intervention(original, proposal, strict=True)
                        if propagated is None or len(propagated) != 1:
                            raise ValueError("Missing propagated vector")
                        covariates = pd.DataFrame([{c: propagated[f"cf_{c}"].iloc[0]
                                                   for c in features}])[features]
                        if not np.isfinite(covariates.to_numpy(dtype=float)).all():
                            raise ValueError("Non-finite propagated vector")
                        prediction = int(generator.model.predict(covariates)[0])
                        scm_target = int(propagated["target"].iloc[0])
                        candidates.append({
                            "proposal_id": i, "raw": frame_rows(proposal)[0],
                            "propagated": dict(frame_rows(covariates)[0], target=scm_target),
                            "raw_prediction": raw_prediction, "scm_target": scm_target,
                            "propagated_prediction": prediction,
                            "flags": candidate_flags(raw_prediction, scm_target, prediction),
                        })
                record = {
                    "manifest_id": digest(manifest), "fold": fold, "source_row_id": source_id,
                    "repeat": repeat, "seed": seed, "original": frame_rows(original)[0],
                    "status": "ok" if candidates else "no_cf", "no_cf_reason": no_cf_reason,
                    "requested_proposals": manifest["config"]["dice"]["total_cfs"],
                    "returned_proposals": len(candidates), "candidates": candidates,
                    "flags": attempt_flags(candidates),
                    "elapsed_seconds": time.perf_counter() - started,
                }
                validate_attempt(record, key, digest(manifest))
                write_record(path, record)
                completed += 1
                print(f"checkpoint fold={fold} source={source_id} repeat={repeat} "
                      f"proposals={len(candidates)}", flush=True)
        except Exception as error:
            write_record(output / "errors" / path.name, {
                "manifest_id": digest(manifest), "fold": fold, "source_row_id": source_id,
                "repeat": repeat, "seed": seed, "exception": repr(error),
                "traceback": traceback.format_exc(),
                "returned_raw": frame_rows(proposals) if proposals is not None else None,
                "evaluated_candidates": candidates,
            })
            raise
    return completed


def scan_attempts(output, manifest, folds):
    expected = {(fold, source, repeat)
                for fold, ready in folds.items() for source in ready["tp_ids"]
                for repeat in range(manifest["expected_repeats"])}
    records = {}
    for path in sorted((output / "attempts").glob("*.json")):
        record = read_record(path)
        key = record["fold"], record["source_row_id"], record["repeat"]
        if key not in expected or key in records:
            raise ValueError(f"Unexpected/duplicate attempt: {path}")
        if path != attempt_path(output, *key):
            raise ValueError(f"Noncanonical attempt file: {path}")
        validate_attempt(record, key, digest(manifest))
        records[key] = record
    errors = {}
    for path in (output / "errors").glob("*.json"):
        record = read_record(path)
        key = record["fold"], record["source_row_id"], record["repeat"]
        if (key not in expected or key in errors or
                record["manifest_id"] != digest(manifest) or
                path.name != attempt_path(output, *key).name):
            raise ValueError(f"Unexpected/duplicate error: {path}")
        errors[key] = record
    return expected, records, errors


def run(output, manifest, folds, workers, patient_limit, repeat_limit, retry_errors):
    expected, records, errors = scan_attempts(output, manifest, folds)
    unresolved = set(errors) - set(records)
    if unresolved and not retry_errors:
        raise ValueError("Unresolved audited errors; inspect errors then use --retry-errors")
    jobs = []
    for fold, ready in folds.items():
        ids = sorted(ready["tp_ids"])
        if patient_limit is not None:
            ids = ids[:patient_limit]
        repeats = range(min(repeat_limit or manifest["expected_repeats"],
                            manifest["expected_repeats"]))
        for source in ids:
            pending = [r for r in repeats if (fold, source, r) not in records]
            if pending:
                jobs.append((str(output), manifest, fold, source, pending))
    print(f"Expected full attempts={len(expected)}; completed={len(records)}; "
          f"pending selected={sum(len(j[-1]) for j in jobs)}; jobs={len(jobs)}", flush=True)
    if workers == 1:
        for job in jobs:
            process_patient(*job)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(process_patient, *job) for job in jobs]
            try:
                for future in as_completed(futures):
                    future.result()
            except BaseException:
                for future in futures:
                    future.cancel()
                # Running jobs finish their atomic checkpoints; no success summary on error.
                raise


def rate(successes, denominator):
    return successes / denominator if denominator else None


def summarize_records(records, expected, expected_repeats, bootstrap=2000):
    values = list(records.values())
    groups = defaultdict(list)
    for record in values:
        groups[(record["fold"], record["source_row_id"])].append(record)
    complete = {k: rows for k, rows in groups.items() if len(rows) == expected_repeats}
    candidate_count = sum(r["returned_proposals"] for r in values)
    fully_complete = len(records) == len(expected)
    endpoints = {}
    for name in ENDPOINTS:
        successes = sum(r["flags"][name] for r in values)
        candidate_successes = sum(c["flags"][name] for r in values for c in r["candidates"])
        means = np.array([np.mean([r["flags"][name] for r in rows])
                          for rows in complete.values()])
        ci = None
        if fully_complete and len(means) > 1 and bootstrap > 0:
            rng = np.random.default_rng(42)
            draws = [float(np.mean(rng.choice(means, size=len(means), replace=True)))
                     for _ in range(bootstrap)]
            ci = np.quantile(draws, [0.025, 0.975]).tolist()
        endpoints[name] = {
            "successful_attempts": int(successes),
            "completed_attempt_rate_descriptive": rate(successes, len(values)),
            "primary_pooled_record_mean": float(means.mean()) if fully_complete and len(means) else None,
            "record_cluster_bootstrap_95ci_conditional_on_fitted_folds": ci,
            "candidate_successes": int(candidate_successes),
            "candidate_conditional_rate": rate(candidate_successes, candidate_count),
        }
    patient_means = [
        {"fold": fold, "source_row_id": source, "completed_repeats": len(rows),
         "complete": len(rows) == expected_repeats,
         **{name: float(np.mean([r["flags"][name] for r in rows])) for name in ENDPOINTS}}
        for (fold, source), rows in sorted(groups.items())]
    return {"expected": len(expected), "completed": len(records),
            "complete": fully_complete, "expected_records": len({k[:2] for k in expected}),
            "records_with_completed_attempts": len(groups), "complete_records": len(complete),
            "no_cf_attempts": sum(r["status"] == "no_cf" for r in values),
            "returned_proposals": candidate_count, "endpoints": endpoints,
            "record_means": patient_means}


def summary(output, manifest, folds):
    from src.training.train_model import compute_metrics
    expected, records, errors = scan_attempts(output, manifest, folds)
    result = summarize_records(records, expected, manifest["expected_repeats"])
    unresolved = set(errors) - set(records)
    result.update({
        "manifest_id": digest(manifest), "error_attempts": len(unresolved),
        "attempted": len(set(records) | set(errors)),
        "pending": len(expected - set(records) - set(errors)),
        "historical_error_records": len(errors),
        "partial_files_detected": [str(p.relative_to(output)) for p in output.rglob("*.partial")],
        "estimand": "Held-out true-positive record-repeat recourse; false negatives excluded",
        "uncertainty": "Record-cluster percentile bootstrap conditional on fitted folds; "
                       "not model-training uncertainty; residual person/site dependence unknown",
    })
    all_oof = [r for ready in folds.values() for r in ready["oof"]]
    oof_ids = [r["source_row_id"] for r in all_oof]
    if len(oof_ids) != len(set(oof_ids)) or set(oof_ids) != set(manifest["eligible_ids"]):
        raise ValueError("Saved classifier OOF coverage is incomplete or duplicated")
    result["classifier_all_eligible_oof"] = compute_metrics(
        [r["target"] for r in all_oof], [r["prediction"] for r in all_oof],
        [r["score"] for r in all_oof])
    result["folds"] = {}
    for fold, ready in folds.items():
        subset = {k: v for k, v in records.items() if k[0] == fold}
        fold_result = summarize_records(subset, {k for k in expected if k[0] == fold},
                                        manifest["expected_repeats"], bootstrap=0)
        fold_result.pop("record_means")
        fold_result["classifier"] = ready["metrics"]
        result["folds"][fold] = fold_result
    result["fold_spread_descriptive"] = {}
    for name in ENDPOINTS:
        fold_rates = [f["endpoints"][name]["primary_pooled_record_mean"]
                      for f in result["folds"].values()]
        observed = [value for value in fold_rates if value is not None]
        result["fold_spread_descriptive"][name] = (
            {"min": min(observed), "max": max(observed)} if len(observed) == 5 else None
        )
    atomic_bytes(output / "summary.json", json_bytes(result))
    atomic_bytes(output / "oof_predictions.csv", pd.DataFrame(all_oof).to_csv(index=False).encode())
    atomic_bytes(output / "record_means.csv",
                 pd.DataFrame(result["record_means"]).to_csv(index=False).encode())
    print(json.dumps({k: v for k, v in result.items()
                      if k not in ("record_means", "folds")}, indent=2), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preflight", "prepare", "run", "summary"))
    parser.add_argument("--output", type=Path, default=ROOT / "kfold_runs" / "primary_seed42")
    parser.add_argument("--data", type=Path,
                        default=ROOT / "data" / "heart_statlog_cleveland_hungary_final.csv")
    parser.add_argument("--config", type=Path, default=ROOT / "pipeline_config.yaml")
    parser.add_argument("--workers", type=int, choices=range(1, 5), default=4)
    parser.add_argument("--patients-per-fold", type=int, help="Pilot cap; does not change full estimand")
    parser.add_argument("--repeats", type=int, help="Pilot prefix; resume without cap for full run")
    parser.add_argument("--retry-errors", action="store_true")
    args = parser.parse_args()
    if any(v is not None and v < 1 for v in (args.patients_per_fold, args.repeats)):
        parser.error("Pilot caps must be positive")
    output = args.output.resolve()
    with run_lock(output):
        manifest, data = make_manifest(args.data.resolve(), args.config.resolve())
        manifest_path = output / "manifest.json"
        if manifest_path.exists():
            assert_manifest(read_record(manifest_path), manifest)
        elif args.command not in ("prepare", "preflight"):
            raise ValueError("Run prepare first")
        else:
            existing = [p for p in output.iterdir() if p.name not in ("RUN.lock", "manifest.json.partial")]
            if existing:
                raise ValueError("Refusing to adopt output files without a manifest")
            write_record(manifest_path, manifest)
        if args.command in ("prepare", "preflight"):
            preflight(output, manifest, data)
        if args.command == "preflight":
            return
        if args.command == "prepare":
            prepare(output, manifest, data)
        folds = {p["fold"]: check_fold(output, manifest, p) for p in manifest["folds"]}
        if args.command == "run":
            preflight(output, manifest, data, prepared=True)
            try:
                run(output, manifest, folds, args.workers, args.patients_per_fold,
                    args.repeats, args.retry_errors)
            except BaseException:
                summary(output, manifest, folds)
                raise
        summary(output, manifest, folds)


if __name__ == "__main__":
    main()
