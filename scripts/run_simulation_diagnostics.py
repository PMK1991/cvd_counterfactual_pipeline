"""Run controlled simulation diagnostics without DiCE or changing frozen artifacts.

Example (from repository root, using the existing project environment):
  python scripts\\run_simulation_diagnostics.py --workers 4 --structural main

Outputs are refused if the destination exists. A completion.json is written only
after all requested cohorts, refits, inference, and checksum verifications pass.
Bootstrap intervals are conditional record-resampling intervals, NOT treatment
effect intervals, and do not account for model fitting or cohort selection.
"""

import os

for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_variable] = "1"

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import importlib.metadata
import inspect
import json
from pathlib import Path
import pickle
import platform
import subprocess
import sys
import time
import warnings

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import networkx as nx
import numpy as np
import pandas as pd
from dowhy import gcm
from dowhy.gcm import whatif
from threadpoolctl import threadpool_limits

from src.pipeline.scm_analyzer import SCMAnalyzer
from src.pipeline.simulation_diagnostics import (
    ABSOLUTE_GRID, PROPOSED_BP, RELATIVE_GRID, SYMPTOMS, action_grid, aggregate,
    canonical_hash, clip_samples, concatenate_plans, draw_noise_plan,
    evaluate_actions, legacy_seed, load_pickle, matched_seed, outcome, read_envelope,
    sha256, summarize, validate_partition,
)
from src.training.train_model import load_train_test_data
from src.training.train_scm import fit_one
from src.utils.dataLoader import DataLoader

DATA = ROOT / "data" / "heart_statlog_cleveland_hungary_final.csv"


def write_json(path, value):
    Path(path).write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )


def frame_hash(frame):
    return __import__("hashlib").sha256(
        pd.util.hash_pandas_object(frame, index=True).values.tobytes()
    ).hexdigest()


def mechanism_metadata(model):
    return {
        node: {
            "parents": sorted(model.graph.predecessors(node)),
            "mechanism": type(model.causal_mechanism(node)).__name__,
            "description": str(model.causal_mechanism(node)),
        } for node in model.graph
    }


def assert_scm(artifact, variant):
    model = artifact["causal_model"]
    if artifact["graph_structure"] != variant or set(model.graph.edges) != set(SCMAnalyzer.GRAPH_VARIANTS[variant]):
        raise ValueError("Unexpected SCM graph/artifact variant")
    if artifact["fit_seed"] != 42:
        raise ValueError("Unexpected frozen fit seed")
    return model


def build_cohorts():
    loader = DataLoader(str(DATA))
    clean = loader.load_clean_data()
    train_x, test_x, train_y, test_y = load_train_test_data()
    train = train_x.copy()
    train["target"] = train_y
    validation = test_x.copy()
    validation["target"] = test_y
    classifier_path = ROOT / "model" / "xgb_pipeline.pkl"
    classifier = load_pickle(classifier_path)
    selected = validation[(validation.target == 1) & (classifier.predict(test_x) == 1)]
    if len(clean) != 707 or len(selected) != 48:
        raise ValueError("Unexpected main cohort; refusing to substitute a new population")
    main_scm = ROOT / "model" / "scm_full.pkl"
    artifact = load_pickle(main_scm)
    assert_scm(artifact, "full")
    # The legacy main artifact does in fact contain a train-only provenance hash.
    if (
        artifact["fit_data"] != "train" or artifact["n_rows"] != len(train)
        or artifact["data_sha256_16"] != frame_hash(train.reset_index(drop=True))[:16]
    ):
        raise ValueError("Main SCM provenance does not match reconstructed training rows")
    validate_partition(train.index, validation.index, selected.index.tolist())
    specs = [{
        "name": "main48", "fold": -1, "training": train, "validation": validation,
        "selected": selected, "classifier_path": classifier_path, "scm_path": main_scm,
        "provenance": {
            "global_target_stratified_iqr_before_split": True,
            "train_provenance_hash_verified": True,
            "classifier_training_provenance": "Reconstructed source split; pickle has no training-row metadata",
            "selection": "Original globally filtered test split; actual target=1 AND frozen XGB prediction=1",
        },
    }]
    base = ROOT / "kfold_runs" / "primary_seed42"
    manifest = read_envelope(base / "manifest.json")
    if sha256(DATA) != manifest["data_sha256"]:
        raise ValueError("Fold manifest dataset has changed")
    seen = []
    for plan in manifest["folds"]:
        fold = plan["fold"]
        directory = base / f"fold_{fold}"
        ready = read_envelope(directory / "ready.json")
        if ready["manifest_id"] != canonical_hash(manifest) or ready["plan"] != plan:
            raise ValueError("Fold provenance manifest mismatch")
        for name, digest in ready["files"].items():
            if sha256(directory / name) != digest:
                raise ValueError(f"Changed frozen fold artifact: {directory / name}")
        training = pd.read_pickle(directory / "training.pkl")
        validation = pd.read_pickle(directory / "validation.pkl")
        if training.index.tolist() != plan["training_ids"] or validation.index.tolist() != plan["validation_ids"]:
            raise ValueError("Fold training/validation row identities disagree")
        frozen_classifier = load_pickle(directory / "classifier.pkl")
        prediction = frozen_classifier.predict(validation.drop(columns="target"))
        selected = validation[(validation.target == 1) & (prediction == 1)]
        if selected.index.tolist() != ready["tp_ids"]:
            raise ValueError("Saved held-out TP selection changed")
        validate_partition(training.index, validation.index, selected.index.tolist())
        artifact = load_pickle(directory / "scm_full.pkl")
        assert_scm(artifact, "full")
        if artifact["training_ids"] != plan["training_ids"] or artifact["manifest_id"] != ready["manifest_id"]:
            raise ValueError("Fold SCM training provenance mismatch")
        seen.extend(selected.index.tolist())
        specs.append({
            "name": f"fold{fold}", "fold": fold, "training": training,
            "validation": validation, "selected": selected,
            "classifier_path": directory / "classifier.pkl",
            "scm_path": directory / "scm_full.pkl",
            "provenance": {
                "global_target_stratified_iqr_before_split": False,
                "train_provenance_hash_verified": True, "manifest_id": ready["manifest_id"],
                "selection": "Held-out outer-fold target=1 AND own frozen fold XGB prediction=1",
                "iqr_policy": "Target-stratified training-fold filtering only; validation untouched",
            },
        })
    if len(seen) != 312 or len(set(seen)) != 312 or len(manifest["eligible_ids"]) != 746:
        raise ValueError("Unexpected held-out TP cohort or repeated record identifiers")
    return specs


def legacy_samples(model, original, cholesterol, proposed_bp, n):
    seed = legacy_seed(original, cholesterol, proposed_bp)
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        observed = pd.DataFrame([original] * n)[list(model.graph)].reset_index(drop=True)
        for col in ("target", "exang", "fbs", "cp", "restecg", "slope"):
            observed[col] = observed[col].astype("category")
        samples = gcm.interventional_samples(
            model, {"chol": lambda _: float(cholesterol)}, observed_data=observed
        )
    finally:
        np.random.set_state(state)
    return samples, seed


def verify_dowhy_equivalence(model, original, n, seed):
    plan = draw_noise_plan(model, n, seed)
    explicit = evaluate_actions(model, original, [original.chol], plan)
    observed = pd.DataFrame([original] * n)[list(model.graph)].reset_index(drop=True)
    for col in ("target", "exang", "fbs", "cp", "restecg", "slope"):
        observed[col] = observed[col].astype("category")
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        reference = gcm.interventional_samples(
            model, {"chol": lambda _: float(original.chol)}, observed_data=observed
        )
    finally:
        np.random.set_state(state)
    np.testing.assert_array_equal(explicit.to_numpy(float), reference[explicit.columns].to_numpy(float))
    return True


def personalization_audit(model, selected, config, cohort, variant):
    common = draw_noise_plan(model, config["draws"], matched_seed(config["seed"], 0, 77))
    conditional = {0: [], 1: []}
    perturbation = []
    factual_mismatch = []
    for source, original in selected.iterrows():
        samples = evaluate_actions(model, original, [original.chol], common)
        altered = original.copy()
        for i, symptom in enumerate(SYMPTOMS):
            altered[symptom] = float(original[symptom]) + 17 + i
        regenerated = evaluate_actions(model, altered, [original.chol], common)
        perturbation.append(float(np.max(np.abs(
            samples[list(SYMPTOMS)].to_numpy() - regenerated[list(SYMPTOMS)].to_numpy()
        ))))
        reconstructed = aggregate(clip_samples(samples)).iloc[0]
        for symptom in SYMPTOMS:
            factual_mismatch.append({
                "cohort": cohort, "variant": variant, "source_row_id": int(source),
                "symptom": symptom, "factual": float(original[symptom]),
                "control_reconstructed": float(reconstructed[symptom]),
                "difference": float(reconstructed[symptom] - original[symptom]),
            })
        for target in (0, 1):
            group_samples = clip_samples(evaluate_actions(
                model, original, [original.chol], common, forced_target=target
            ))
            conditional[target].append(group_samples[list(SYMPTOMS)].to_numpy())
    rows = []
    for target, profiles in conditional.items():
        profiles = np.stack(profiles)
        for i, symptom in enumerate(SYMPTOMS):
            means = profiles[:, :, i].mean(axis=1)
            rows.append({
                "cohort": cohort, "variant": variant, "forced_target": target,
                "symptom": symptom, "parents": ",".join(sorted(model.graph.predecessors(symptom))),
                "max_between_record_matched_draw_range": float(np.ptp(profiles[:, :, i], axis=0).max()),
                "between_record_conditional_mean_range": float(np.ptp(means)),
                "between_record_conditional_mean_sd": float(means.std()),
                "max_factual_symptom_perturbation_effect": max(perturbation),
            })
    return rows, factual_mismatch


def measure_model(model, classifier, selected, training, cohort, variant, config):
    features = list(classifier.feature_names_in_)
    raw, seed_rows = [], []
    support = (float(training.chol.min()), float(training.chol.max()))
    first = selected.iloc[0]
    verify_dowhy_equivalence(model, first, min(config["draws"], 128), config["seed"])
    for number, (source, original) in enumerate(selected.iterrows()):
        plans = [
            draw_noise_plan(model, config["draws"], matched_seed(config["seed"], int(source), rep))
            for rep in range(config["replicates"])
        ]
        noise = concatenate_plans(plans)
        actions = action_grid(float(original.chol))
        samples = evaluate_actions(model, original, [value for _, value in actions], noise)
        target_outside = ~samples.target.isin([0, 1])
        clipped = clip_samples(samples)
        draw_predictions = classifier.predict(clipped[features])
        n = config["draws"]
        stride = n * config["replicates"]
        controls = []
        for action_index, (name, value) in enumerate(actions):
            for rep in range(config["replicates"]):
                start = action_index * stride + rep * n
                block = clipped.iloc[start:start + n]
                result = outcome(block, classifier, features, draw_predictions[start:start + n])
                if action_index == 0:
                    controls.append(result)
                control = controls[rep]
                result.update(
                    cohort=cohort, variant=variant, source_row_id=int(source), action=name,
                    replicate=rep, noise_seed=matched_seed(config["seed"], int(source), rep),
                    draws=n, factual_cholesterol=float(original.chol), cholesterol=value,
                    direction="increase" if value > original.chol else "decrease" if value < original.chol else "unchanged",
                    out_of_support=int(value < support[0] or value > support[1]),
                    target_raw_outside_binary_fraction=float(target_outside.iloc[start:start+n].mean()),
                    control_p_target0=control["p_target0"], control_accept=control["accept"],
                    delta_p_target0=result["p_target0"] - control["p_target0"],
                    gained_acceptance=int(result["accept"] == 1 and control["accept"] == 0),
                    lost_acceptance=int(result["accept"] == 0 and control["accept"] == 1),
                    aggregation_disagreement=int(result["accept"] != result["xgb_aggregate_accept"]),
                )
                if name in ("control", "reduction_0pct") and result["delta_p_target0"] != 0:
                    raise AssertionError("Identity action has a nonzero paired difference")
                raw.append(result)
        # Same fixed action, only the otherwise unused proposed-BP seed input changes.
        for proposed_bp in PROPOSED_BP:
            legacy, old_seed = legacy_samples(model, original, 200, proposed_bp, n)
            probability = float((clip_samples(legacy).target == 0).mean())
            repeated, _ = legacy_samples(model, original, 200, proposed_bp, n)
            np.testing.assert_array_equal(legacy.to_numpy(float), repeated.to_numpy(float))
            matched = next(r for r in raw[-len(actions) * config["replicates"]:]
                           if r["action"] == "absolute_200" and r["replicate"] == 0)
            seed_rows.append({
                "cohort": cohort, "variant": variant, "source_row_id": int(source),
                "cholesterol": 200, "proposed_bp_seed_input_only": proposed_bp,
                "legacy_seed": old_seed, "legacy_p_target0": probability,
                "legacy_accept": int(probability >= .5),
                "matched_seed": matched["noise_seed"],
                "matched_p_target0": matched["p_target0"], "matched_accept": matched["accept"],
                "identical_legacy_call_max_absolute_difference": 0.0,
            })
        if (number + 1) % 20 == 0 or number + 1 == len(selected):
            print(f"{cohort}/{variant}: {number+1}/{len(selected)} records", flush=True)
    profiles, reconstruction = personalization_audit(model, selected, config, cohort, variant)
    return raw, seed_rows, profiles, reconstruction


def cohort_worker(spec, config, output):
    started = time.monotonic()
    with threadpool_limits(limits=1), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        classifier = load_pickle(spec["classifier_path"])
        classifier.named_steps["classifier"].set_params(n_jobs=1)
        frozen = load_pickle(spec["scm_path"])
        model = assert_scm(frozen, "full")
        models = [("frozen_full", model, frozen)]
        should_refit = config["structural"] == "all" or (
            config["structural"] == "main" and spec["name"] == "main48"
        )
        if should_refit:
            for variant in ("full", "minimal", "extended"):
                print(f"{spec['name']}: refitting {variant} on exact same {len(spec['training'])} training rows", flush=True)
                fitted = fit_one(variant, spec["training"], fit_seed=42)
                artifact = {
                    "causal_model": fitted, "graph_structure": variant, "fit_seed": 42,
                    "fit_data": "diagnostic_exact_same_training_rows",
                    "training_ids": spec["training"].index.tolist(),
                    "training_frame_sha256": frame_hash(spec["training"]),
                }
                path = Path(output) / "fitted" / f"{spec['name']}_{variant}.pkl"
                with path.open("wb") as stream:
                    pickle.dump(artifact, stream)
                models.append((f"refit_{variant}", fitted, artifact))
        provenance = {
            "cohort": spec["name"], "fold": spec["fold"],
            "n_training": len(spec["training"]), "n_validation": len(spec["validation"]),
            "n_selected": len(spec["selected"]),
            "training_ids": spec["training"].index.tolist(),
            "validation_ids": spec["validation"].index.tolist(),
            "selected_ids": spec["selected"].index.tolist(),
            "training_frame_sha256": frame_hash(spec["training"]),
            "selected_frame_sha256": frame_hash(spec["selected"]),
            "frozen_scm_metadata": {k: v for k, v in frozen.items() if k != "causal_model"},
            **spec["provenance"],
            "models": {},
        }
        for label, causal_model, artifact in models:
            provenance["models"][label] = mechanism_metadata(causal_model)
            raw, seed_rows, profiles, reconstruction = measure_model(
                causal_model, classifier, spec["selected"], spec["training"],
                spec["name"], label, config,
            )
            prefix = Path(output) / f"{spec['name']}_{label}"
            for suffix, rows in (
                ("raw", raw), ("bp_seed", seed_rows), ("personalization", profiles),
                ("reconstruction", reconstruction),
            ):
                pd.DataFrame(rows).to_csv(str(prefix) + f"_{suffix}.csv", index=False)
        provenance["elapsed_seconds"] = time.monotonic() - started
        write_json(Path(output) / f"{spec['name']}_provenance.json", provenance)
    return provenance


def combine_outputs(output, config):
    tables = {}
    for suffix in ("raw", "bp_seed", "personalization", "reconstruction"):
        tables[suffix] = pd.concat(
            [pd.read_csv(path) for path in sorted(output.glob(f"*_{suffix}.csv"))],
            ignore_index=True,
        )
        tables[suffix].to_csv(output / f"combined_{suffix}.csv", index=False)
    raw = tables["raw"].copy()
    raw["cohort"] = raw.cohort.where(~raw.cohort.str.startswith("fold"), "heldout312")
    record, summary, response = summarize(raw, config["bootstrap_repeats"])
    record.to_csv(output / "record_action_means.csv", index=False)
    summary.to_csv(output / "action_summary.csv", index=False)
    response.to_csv(output / "record_responsiveness.csv", index=False)
    seeds = tables["bp_seed"].copy()
    seeds["cohort"] = seeds.cohort.where(~seeds.cohort.str.startswith("fold"), "heldout312")
    seed_summary = seeds.groupby(["cohort", "variant", "source_row_id"]).agg(
        legacy_probability_range=("legacy_p_target0", lambda x: float(np.ptp(x))),
        legacy_acceptance_range=("legacy_accept", lambda x: int(np.ptp(x))),
        matched_probability_range=("matched_p_target0", lambda x: float(np.ptp(x))),
        matched_acceptance_range=("matched_accept", lambda x: int(np.ptp(x))),
    ).reset_index()
    seed_summary.to_csv(output / "bp_seed_record_ranges.csv", index=False)
    findings = []
    for (cohort, variant), group in raw.groupby(["cohort", "variant"]):
        n_records = group.source_row_id.nunique()
        expected = n_records * config["replicates"] * (1 + len(ABSOLUTE_GRID) + len(RELATIVE_GRID))
        if len(group) != expected or group.duplicated(["source_row_id", "action", "replicate"]).any():
            raise ValueError("Missing or duplicate raw diagnostic outcomes")
        rng = response[(response.cohort == cohort) & (response.variant == variant)]
        bp = seed_summary[(seed_summary.cohort == cohort) & (seed_summary.variant == variant)]
        controls = group[group.action == "control"]
        findings.append({
            "cohort": cohort, "variant": variant, "n_records": n_records,
            "raw_record_action_replicate_rows": len(group),
            "control_mean_simulated_p_target0": float(controls.p_target0.mean()),
            "control_acceptance_rate": float(controls["accept"].mean()),
            "invariant_absolute_probability_records": int(rng.invariant_absolute_probability.sum()),
            "invariant_absolute_acceptance_records": int(rng.invariant_absolute_acceptance.sum()),
            "mean_absolute_probability_response_range": float(rng.absolute_grid_probability_range.mean()),
            "max_absolute_probability_response_range": float(rng.absolute_grid_probability_range.max()),
            "legacy_bp_seed_mean_probability_range": float(bp.legacy_probability_range.mean()),
            "legacy_bp_seed_max_probability_range": float(bp.legacy_probability_range.max()),
            "legacy_bp_seed_acceptance_flip_records": int(bp.legacy_acceptance_range.sum()),
            "matched_bp_seed_max_probability_range": float(bp.matched_probability_range.max()),
            "fraction_raw_target_samples_outside_0_1_before_legacy_clip": float(group.target_raw_outside_binary_fraction.mean()),
        })
    write_json(output / "findings.json", findings)
    # Deterministic paired comparisons: fresh full vs minimal/extended, never
    # compare an old variant trained on unknown/different rows.
    structural = []
    full = record[record.variant == "refit_full"].set_index(["cohort", "source_row_id", "action"])
    for variant in ("refit_minimal", "refit_extended"):
        other = record[record.variant == variant].set_index(["cohort", "source_row_id", "action"])
        if other.empty:
            continue
        paired = other.join(full, lsuffix="_variant", rsuffix="_full", how="inner").reset_index()
        for (cohort, action), group in paired.groupby(["cohort", "action"]):
            difference = group.p_target0_variant - group.p_target0_full
            structural.append({
                "cohort": cohort, "variant": variant, "action": action, "n_records": len(group),
                "mean_p_target0_difference_from_refit_full": float(difference.mean()),
                "max_absolute_p_target0_difference": float(difference.abs().max()),
                "pooled_probability_acceptance_disagreement_rate": float((
                    group.accept_from_pooled_probability_variant != group.accept_from_pooled_probability_full
                ).mean()),
            })
    pd.DataFrame(structural).to_csv(output / "structural_comparison.csv", index=False)
    plot_summary(summary, output)
    return findings


def plot_summary(summary, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    absolute = summary[summary.action.str.startswith("absolute_")].copy()
    absolute["cholesterol"] = absolute.action.str.replace("absolute_", "", regex=False).astype(int)
    for (cohort, variant), group in absolute.groupby(["cohort", "variant"]):
        group = group.sort_values("cholesterol")
        label = f"{cohort}: {variant}"
        axes[0].plot(group.cholesterol, group.mean_p_target0, marker=".", label=label)
        axes[1].plot(group.cholesterol, group.acceptance_rate_over_replicates, marker=".", label=label)
    axes[0].set_ylabel("Mean simulated P(target=0)")
    axes[1].set_ylabel("Thresholded acceptance fraction")
    for ax in axes:
        ax.set_xlabel("Scenario cholesterol (mg/dL; NOT treatment targets)")
        ax.set_ylim(-.02, 1.02)
        ax.grid(alpha=.25)
    axes[0].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(output / "intervention_response.png", dpi=180)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "measurements" / "intervention_responsiveness_20261027")
    parser.add_argument("--workers", type=int, default=4, choices=range(1, 5))
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--replicates", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20261027)
    parser.add_argument("--bootstrap-repeats", type=int, default=2000)
    parser.add_argument("--structural", choices=("main", "all", "none"), default="main")
    args = parser.parse_args(argv)
    if min(args.draws, args.replicates, args.bootstrap_repeats) < 2:
        raise ValueError("At least two draws, noise replicates, and bootstrap repeats required")
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite diagnostic output: {args.output}")
    started = time.monotonic()
    config = vars(args).copy()
    config["output"] = str(args.output.resolve())
    args.output.mkdir(parents=True)
    (args.output / "fitted").mkdir()
    with threadpool_limits(limits=1), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        specs = build_cohorts()
    inputs = {str(DATA.relative_to(ROOT)): sha256(DATA)}
    for spec in specs:
        for key in ("classifier_path", "scm_path"):
            path = spec[key]
            inputs[str(path.relative_to(ROOT))] = sha256(path)
    fold_base = ROOT / "kfold_runs" / "primary_seed42"
    for path in [fold_base / "manifest.json", fold_base / "oof_predictions.csv"] + [
        path for directory in fold_base.glob("fold_*") for path in directory.iterdir() if path.is_file()
    ]:
        inputs[str(path.relative_to(ROOT))] = sha256(path)
    source_files = list((ROOT / "src").rglob("*.py")) + [
        Path(__file__), ROOT / "tests" / "test_simulation_diagnostics.py",
        ROOT / "scripts" / "run_kfold_recourse.py",
    ]
    sources = {str(path.relative_to(ROOT)): sha256(path) for path in source_files}
    metadata = {
        "schema": "simulation-diagnostics-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": config, "input_sha256": inputs, "source_sha256": sources,
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "python": platform.python_version(),
        "packages": {name: importlib.metadata.version(name) for name in (
            "numpy", "pandas", "scikit-learn", "dowhy", "xgboost", "networkx", "matplotlib", "threadpoolctl",
        )},
        "dowhy_intervention_source": inspect.getsource(whatif._interventional_samples),
        "semantics": {
            "primary": "Existing interventional_samples semantics, explicit shared newly sampled marginal noise; not abduction",
            "control": "do(chol=factual), not no-intervention observational copy",
            "probability": "Sample fraction target=0 AFTER existing target clipping; target=1 is its complement",
            "acceptance": "P(target=0)>=0.5 per 1000-draw replicate; exact mode tie favors 0",
            "pooled_acceptance": "Separate sensitivity: threshold after averaging probabilities across noise replicates",
            "matching": "Exact node noise arrays shared across actions; seed depends only on master/source-row/replicate",
            "legacy_seed_audit": "Fixed do(chol=200); proposed BP varies ONLY as unused legacy seed input",
            "profile_audit": "Shared noise across records; simulated target forced to 0/1 only in this audit",
            "uncertainty": "Paired conditional record bootstrap after averaging noise replicates; no pseudoreplicated patients",
            "classifier_agreement": "Shared-data INTERNAL CONSISTENCY ONLY, not independent causal validation",
            "structural_scope": args.structural,
            "scenarios": "Absolute grid can raise cholesterol; outside training min/max flagged (not a density/support guarantee)",
            "limitations": [
                "No clinical benefit, treatment recommendation, empirical causal identification or external validation",
                "Main cohort uses global target-conditioned outlier filtering; main classifier pickle lacks training-row metadata",
                "Outer folds are reused frozen fits; uncertainty excludes refitting and cohort selection",
                "Record identifiers are source CSV row indices, not real patient IDs; main and outer cohorts overlap",
                "Refit variants change both DAG and automatically selected mechanisms; not an isolated-edge causal test",
                "Numerically categorical integer columns may be assigned discrete regression mechanisms; audit records actual families",
            ],
        },
    }
    write_json(args.output / "config_and_provenance.json", metadata)
    cohort_metadata = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(cohort_worker, spec, config, str(args.output)) for spec in specs]
        for future in futures:
            cohort_metadata.append(future.result())
    findings = combine_outputs(args.output, config)
    for relative, digest in {**inputs, **sources}.items():
        if sha256(ROOT / relative) != digest:
            raise RuntimeError(f"Input or source changed while diagnostics ran: {relative}")
    outputs = {str(path.relative_to(args.output)): sha256(path)
               for path in args.output.rglob("*") if path.is_file()}
    completion = {
        "status": "complete", "elapsed_seconds": time.monotonic() - started,
        "cohorts": [{"cohort": c["cohort"], "n_selected": c["n_selected"]} for c in cohort_metadata],
        "output_sha256": outputs,
    }
    write_json(args.output / "completion.json", completion)
    print(json.dumps({"completion": completion["status"], "elapsed_seconds": completion["elapsed_seconds"],
                      "findings": findings}, indent=2), flush=True)


if __name__ == "__main__":
    main()
