"""Verify a completed run and measure the historical action-dependent seed sweep.

The primary experiment uses ten matched-noise replicates. This supplement uses
ONE historical seed per action, with proposed BP fixed at factual BP, to expose
action-dependent Monte Carlo variation rather than interpret it as responsiveness.
It also reports exact legacy ordinal/source-row mapping and tidy negative-result
summaries. Primary files are read-only; outputs go in a new supplement directory.
"""

import os

for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_variable] = "1"

import argparse
import json
from pathlib import Path
import sys
import time
import warnings

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import networkx as nx
from dowhy import gcm
from dowhy.gcm.causal_mechanisms import DiscreteAdditiveNoiseModel
from threadpoolctl import threadpool_limits

from scripts.run_simulation_diagnostics import build_cohorts, legacy_samples, write_json
from src.pipeline.simulation_diagnostics import (
    action_grid, clip_samples, draw_noise_plan, evaluate_actions, legacy_seed,
    load_pickle, sha256,
)


def exact_target_probability(model, original, cholesterol):
    """Finite-support integration of the fitted discrete BP/target noise.

    This is an additional numerical check, not a substitute for the primary
    Monte Carlo experiment. It is valid only for the inspected graph's target
    and (if affected) BP additive mechanisms; reject any unsupported mechanism.
    """
    target = model.causal_mechanism("target")
    if type(target) is not DiscreteAdditiveNoiseModel:
        raise ValueError("Exact integration requires the inspected discrete target ANM")
    noise, counts = np.unique(target.noise_model.data, return_counts=True)
    target_weights = counts / counts.sum()
    if "trestbps" in nx.descendants(model.graph, "chol"):
        bp = model.causal_mechanism("trestbps")
        if type(bp) is not DiscreteAdditiveNoiseModel:
            raise ValueError("Exact integration requires the inspected discrete BP ANM")
        bp_noise, bp_counts = np.unique(bp.noise_model.data, return_counts=True)
        bp_weights = bp_counts / bp_counts.sum()
        parents = pd.DataFrame([original] * len(bp_noise))
        parents["chol"] = cholesterol
        bp_values = bp.evaluate(
            parents[sorted(model.graph.predecessors("trestbps"))].to_numpy(),
            bp_noise.reshape(-1, 1),
        ).reshape(-1)
    else:
        bp_values, bp_weights = np.array([original.trestbps]), np.array([1.0])
    parents = pd.DataFrame([original] * len(bp_values))
    parents["chol"] = cholesterol
    parents["trestbps"] = bp_values
    prediction = target.evaluate(
        parents[sorted(model.graph.predecessors("target"))].to_numpy(),
        np.zeros((len(parents), 1), dtype=noise.dtype),
    ).reshape(-1)
    outputs = prediction[:, None] + noise[None, :]
    probability = float(np.sum((outputs <= 0) * bp_weights[:, None] * target_weights[None, :]))
    return probability


def target_mechanism_metadata(model):
    mechanism = model.causal_mechanism("target")
    regressor = mechanism.prediction_model.sklearn_model
    values, counts = np.unique(mechanism.noise_model.data, return_counts=True)
    result = {
        "mechanism": str(mechanism), "parents": sorted(model.graph.predecessors("target")),
        "noise_values": values.tolist(), "noise_counts": counts.tolist(),
        "noise_probabilities": (counts / counts.sum()).tolist(),
        "prediction_rounding": "np.round to int32 BEFORE adding discrete residual noise",
        "output_clipping": "Existing pipeline clips raw target to [0,1] AFTER propagation to symptoms",
    }
    if hasattr(regressor, "coef_"):
        result.update(
            linear_coefficients=np.asarray(regressor.coef_).reshape(-1).tolist(),
            linear_intercept=float(regressor.intercept_),
        )
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "measurements" / "intervention_responsiveness_20261027")
    args = parser.parse_args(argv)
    started = time.monotonic()
    sources = {
        str(path.relative_to(ROOT)): sha256(path) for path in (
            Path(__file__), ROOT / "tests" / "test_simulation_exact_integration.py",
        )
    }
    primary_completion_hash = sha256(args.input / "completion.json")
    completion = json.loads((args.input / "completion.json").read_text())
    if completion["status"] != "complete":
        raise ValueError("Primary run is not complete")
    for relative, digest in completion["output_sha256"].items():
        if sha256(args.input / relative) != digest:
            raise ValueError(f"Changed primary diagnostic artifact: {relative}")
    provenance = json.loads((args.input / "config_and_provenance.json").read_text())
    for relative, digest in provenance["input_sha256"].items():
        if sha256(ROOT / relative) != digest:
            raise ValueError(f"Changed frozen input: {relative}")
    out = args.input / "supplement"
    out.mkdir(exist_ok=False)
    raw = pd.read_csv(args.input / "combined_raw.csv")
    records = pd.read_csv(args.input / "record_action_means.csv")
    profiles = pd.read_csv(args.input / "combined_personalization.csv")
    reconstruction = pd.read_csv(args.input / "combined_reconstruction.csv")
    n = provenance["config"]["draws"]
    threshold_rows = []
    threshold_raw = raw.copy()
    threshold_raw["cohort"] = threshold_raw.cohort.where(
        ~threshold_raw.cohort.str.startswith("fold"), "heldout312"
    )
    for (cohort, variant, action), group in threshold_raw.groupby(["cohort", "variant", "action"]):
        pooled = group.groupby("source_row_id").p_target0.mean()
        for threshold in (.25, .4, .5, .6, .75):
            threshold_rows.append({
                "cohort": cohort, "variant": variant, "action": action,
                "threshold": threshold, "n_records": len(pooled),
                "acceptance_rate_over_replicates": float((group.p_target0 >= threshold).mean()),
                "pooled_probability_acceptance_rate": float((pooled >= threshold).mean()),
                "historical_threshold": int(threshold == .5),
            })
    pd.DataFrame(threshold_rows).to_csv(out / "threshold_sensitivity.csv", index=False)
    absolute = raw[raw.action.str.startswith("absolute_")].copy()
    absolute["cohort"] = absolute.cohort.where(~absolute.cohort.str.startswith("fold"), "heldout312")
    replicate_response = absolute.groupby(
        ["cohort", "variant", "source_row_id", "replicate"]
    ).agg(
        probability_range=("p_target0", lambda x: float(np.ptp(x))),
        threshold_range=("accept", lambda x: int(np.ptp(x))),
    ).reset_index()
    replicate_response.to_csv(out / "replicate_record_absolute_response.csv", index=False)
    invariant = replicate_response.groupby(["cohort", "variant", "source_row_id"]).agg(
        max_replicate_probability_range=("probability_range", "max"),
        any_replicate_threshold_change=("threshold_range", "max"),
    ).reset_index()
    invariant.groupby(["cohort", "variant"]).agg(
        n_records=("source_row_id", "count"),
        records_with_any_replicate_probability_change=(
            "max_replicate_probability_range", lambda x: int((x > 0).sum())
        ),
        records_with_any_replicate_threshold_change=("any_replicate_threshold_change", "sum"),
    ).reset_index().to_csv(out / "replicate_invariance_summary.csv", index=False)
    legacy_rows, observational_controls, exact_rows, mechanisms = [], [], [], {}
    with threadpool_limits(limits=1), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for spec in build_cohorts():
            model = load_pickle(spec["scm_path"])["causal_model"]
            first = spec["selected"].iloc[0]
            reference, reference_seed = legacy_samples(model, first, 200, first.trestbps, n)
            replay = evaluate_actions(
                model, first, [200], draw_noise_plan(model, n, reference_seed)
            )
            np.testing.assert_array_equal(replay, reference[replay.columns].to_numpy(float))
            primary = raw[(raw.cohort == spec["name"]) & (raw.variant == "frozen_full") & (raw.replicate == 0)]
            for ordinal, (source, original) in enumerate(spec["selected"].iterrows()):
                factual = pd.DataFrame([original])[list(model.graph)].reset_index(drop=True)
                unchanged = gcm.interventional_samples(model, {}, observed_data=factual)
                np.testing.assert_array_equal(factual, unchanged)
                cohort = "heldout312" if spec["name"].startswith("fold") else spec["name"]
                matched_control_means = records[
                    (records.cohort == cohort) & (records.variant == "frozen_full")
                    & (records.source_row_id == source) & (records.action == "control")
                ].iloc[0]
                observational_controls.append({
                    "cohort": cohort, "source_row_id": int(source),
                    "observed_copy_p_target0": float(original.target == 0),
                    "do_factual_cholesterol_mean_p_target0": float(matched_control_means.p_target0),
                    "do_factual_cholesterol_replicate_acceptance_rate": float(matched_control_means.acceptance_rate),
                    "do_factual_cholesterol_pooled_acceptance": int(matched_control_means.accept_from_pooled_probability),
                    "do_factual_cholesterol_aggregate_xgb_acceptance_rate": float(matched_control_means.xgb_aggregate_acceptance),
                })
                control_probability = None
                matched_control = primary[
                    (primary.source_row_id == source) & (primary.action == "control")
                ].iloc[0]
                for action, value in action_grid(float(original.chol)):
                    seed = legacy_seed(original, value, original.trestbps)
                    samples = evaluate_actions(
                        model, original, [value], draw_noise_plan(model, n, seed)
                    )
                    probability = float((clip_samples(samples).target == 0).mean())
                    if action == "control":
                        control_probability = probability
                    matched = primary[
                        (primary.source_row_id == source) & (primary.action == action)
                    ].iloc[0]
                    legacy_rows.append({
                        "cohort": spec["name"], "source_row_id": int(source),
                        "main_legacy_ordinal_0based": ordinal if spec["name"] == "main48" else -1,
                        "action": action, "cholesterol": value, "draws": n,
                        "legacy_seed": seed, "legacy_proposed_bp_seed_input": float(original.trestbps),
                        "legacy_p_target0": probability, "legacy_accept": int(probability >= .5),
                        "legacy_delta_p_target0": probability - control_probability,
                        "matched_replicate0_p_target0": float(matched.p_target0),
                        "matched_replicate0_accept": int(matched["accept"]),
                        "matched_replicate0_delta_p_target0": float(matched.delta_p_target0),
                        "matched_control_accept": int(matched_control["accept"]),
                    })
            if spec["name"] == "main48":
                mapping = spec["selected"].copy()
                mapping.insert(0, "source_row_id", mapping.index)
                mapping.insert(0, "legacy_ordinal_0based", range(len(mapping)))
                mapping.to_csv(out / "main_legacy_ordinal_source_row_mapping.csv", index=False)
            models = [("frozen_full", model)]
            for variant in ("full", "minimal", "extended"):
                artifact_path = args.input / "fitted" / f"{spec['name']}_{variant}.pkl"
                if artifact_path.exists():
                    models.append((f"refit_{variant}", load_pickle(artifact_path)["causal_model"]))
            for variant, fitted in models:
                mechanisms[f"{spec['name']}/{variant}"] = target_mechanism_metadata(fitted)
                cohort = "heldout312" if spec["name"].startswith("fold") else spec["name"]
                averaged = records[
                    (records.cohort == cohort) & (records.variant == variant)
                ].set_index(["source_row_id", "action"])
                for source, original in spec["selected"].iterrows():
                    control = exact_target_probability(fitted, original, original.chol)
                    for action, value in action_grid(float(original.chol)):
                        probability = exact_target_probability(fitted, original, value)
                        monte_carlo = averaged.loc[(source, action)]
                        exact_rows.append({
                            "cohort": cohort, "model_cohort": spec["name"], "variant": variant,
                            "source_row_id": int(source), "action": action,
                            "exact_fitted_model_p_target0": probability,
                            "exact_fitted_model_accept": int(probability >= .5),
                            "exact_control_delta": probability - control,
                            "monte_carlo_mean_p_target0": float(monte_carlo.p_target0),
                            "monte_carlo_minus_exact_probability": float(monte_carlo.p_target0 - probability),
                            "monte_carlo_minus_exact_delta": float(monte_carlo.delta_p_target0 - (probability - control)),
                        })
            print(f"Historical seed sweep completed: {spec['name']}", flush=True)
    legacy = pd.DataFrame(legacy_rows)
    pd.DataFrame(observational_controls).to_csv(out / "observational_copy_vs_do_factual_cholesterol.csv", index=False)
    legacy.to_csv(out / "legacy_action_seed_raw.csv", index=False)
    legacy["cohort"] = legacy.cohort.where(~legacy.cohort.str.startswith("fold"), "heldout312")
    comparisons = []
    for (cohort, action), group in legacy.groupby(["cohort", "action"]):
        comparisons.append({
            "cohort": cohort, "action": action, "n_records": len(group),
            "legacy_mean_p_target0": float(group.legacy_p_target0.mean()),
            "matched_rep0_mean_p_target0": float(group.matched_replicate0_p_target0.mean()),
            "legacy_acceptance_rate": float(group.legacy_accept.mean()),
            "matched_rep0_acceptance_rate": float(group.matched_replicate0_accept.mean()),
            "acceptance_disagreement_rate": float((group.legacy_accept != group.matched_replicate0_accept).mean()),
            "mean_absolute_probability_difference": float((group.legacy_p_target0 - group.matched_replicate0_p_target0).abs().mean()),
            "legacy_mean_control_delta": float(group.legacy_delta_p_target0.mean()),
            "matched_rep0_mean_control_delta": float(group.matched_replicate0_delta_p_target0.mean()),
            "mean_absolute_control_delta_difference": float((group.legacy_delta_p_target0 - group.matched_replicate0_delta_p_target0).abs().mean()),
        })
    pd.DataFrame(comparisons).to_csv(out / "legacy_vs_matched_action_summary.csv", index=False)
    exact = pd.DataFrame(exact_rows)
    exact.to_csv(out / "exact_fitted_model_target_probabilities.csv", index=False)
    exact_summary = exact.groupby(["cohort", "variant", "action"]).agg(
        n_records=("source_row_id", "count"),
        exact_mean_p_target0=("exact_fitted_model_p_target0", "mean"),
        exact_acceptance_rate=("exact_fitted_model_accept", "mean"),
        exact_mean_control_delta=("exact_control_delta", "mean"),
        mean_absolute_monte_carlo_error=("monte_carlo_minus_exact_probability", lambda x: float(x.abs().mean())),
        max_absolute_monte_carlo_error=("monte_carlo_minus_exact_probability", lambda x: float(x.abs().max())),
    ).reset_index()
    exact_summary.to_csv(out / "exact_fitted_model_action_summary.csv", index=False)
    exact_absolute = exact[exact.action.str.startswith("absolute_")]
    exact_absolute.groupby(["cohort", "variant", "source_row_id"]).agg(
        exact_probability_range=("exact_fitted_model_p_target0", lambda x: float(np.ptp(x))),
        exact_threshold_range=("exact_fitted_model_accept", lambda x: int(np.ptp(x))),
    ).reset_index().to_csv(out / "exact_record_responsiveness.csv", index=False)
    write_json(out / "target_mechanisms.json", mechanisms)
    reconstruction["cohort"] = reconstruction.cohort.where(
        ~reconstruction.cohort.str.startswith("fold"), "heldout312"
    )
    reconstruction_summary = reconstruction.groupby(["cohort", "variant", "symptom"]).agg(
        n_records=("source_row_id", "count"), mean_factual=("factual", "mean"),
        mean_control_reconstructed=("control_reconstructed", "mean"),
        mean_signed_difference=("difference", "mean"),
        mean_absolute_difference=("difference", lambda x: float(x.abs().mean())),
        exact_factual_reconstruction_fraction=("difference", lambda x: float((x == 0).mean())),
    ).reset_index()
    reconstruction_summary.to_csv(out / "control_factual_symptom_reconstruction.csv", index=False)
    profiles["cohort"] = profiles.cohort.where(~profiles.cohort.str.startswith("fold"), "heldout312")
    profile_summary = profiles.groupby(["cohort", "variant", "symptom"]).agg(
        parents=("parents", "first"),
        max_between_record_matched_draw_range=("max_between_record_matched_draw_range", "max"),
        max_between_record_conditional_mean_range=("between_record_conditional_mean_range", "max"),
        max_factual_symptom_perturbation_effect=("max_factual_symptom_perturbation_effect", "max"),
    ).reset_index()
    profile_summary.to_csv(out / "conditional_profile_summary.csv", index=False)
    # Quantify exact-training refit reproducibility separately from DAG sensitivity.
    frozen = records[(records.cohort == "main48") & (records.variant == "frozen_full")].set_index(["source_row_id", "action"])
    fresh = records[(records.cohort == "main48") & (records.variant == "refit_full")].set_index(["source_row_id", "action"])
    refit_comparison = None
    if not fresh.empty:
        paired = frozen.join(fresh, lsuffix="_frozen", rsuffix="_refit", how="inner")
        refit_comparison = {
            "n_record_actions": len(paired),
            "max_probability_difference": float((paired.p_target0_frozen - paired.p_target0_refit).abs().max()),
            "pooled_acceptance_disagreement_count": int((
                paired.accept_from_pooled_probability_frozen != paired.accept_from_pooled_probability_refit
            ).sum()),
        }
    results = {
        "status": "complete", "elapsed_seconds": time.monotonic() - started,
        "source_sha256": sources,
        "primary_completion_sha256": primary_completion_hash,
        "main48_heldout312_overlapping_source_records": len(
            set(raw[raw.cohort == "main48"].source_row_id)
            & set(raw[raw.cohort.str.startswith("fold")].source_row_id)
        ),
        "exact_training_full_refit_reproduction": refit_comparison,
        "exact_integration": "Finite empirical residual support integrated for discrete BP and target mechanisms; not causal identification",
        "legacy_seed_replay": "Explicit fresh noise replay; bitwise equality to original DoWhy calls verified for each cohort",
        "limitations": [
            "Historical seed sweep has one historical seed per action, not ten replicates",
            "Matched replicate 0 comparison changes the noise realization, not the structural equations",
            "Conditional profile audit forces raw target=0/1; primary samples are never forced",
            "Cross-DAG noise is matched across actions WITHIN each model, not matched residual draws across model families",
            "Reported probability is from clipped discrete regression outputs, not a calibrated causal disease probability",
            "Main 48 and outer-fold 312 are overlapping record cohorts, never pooled together as independent patients",
            "Exact integration applies only to the fitted discrete BP/target ANMs; it is not external probability calibration",
        ],
    }
    results["output_sha256"] = {
        path.name: sha256(path) for path in out.iterdir() if path.is_file()
    }
    for relative, digest in {**sources, **provenance["input_sha256"]}.items():
        if sha256(ROOT / relative) != digest:
            raise RuntimeError(f"Source or frozen input changed during supplement: {relative}")
    if sha256(args.input / "completion.json") != primary_completion_hash:
        raise RuntimeError("Primary run changed during supplement")
    write_json(out / "completion.json", results)
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
