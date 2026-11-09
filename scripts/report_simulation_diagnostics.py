"""Compact record-level response report from the completed simulation audit.

No SCM refitting, DiCE generation, manuscript editing, or new simulation occurs.
Monte Carlo and exactly integrated fitted-model probabilities remain separate.
All probability changes are action MINUS do(chol=factual), not observational
copies, treatment effects, or externally validated clinical risk reductions.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import inspect
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from dowhy.gcm.causal_mechanisms import ClassifierFCM, DiscreteAdditiveNoiseModel

from src.pipeline.simulation_diagnostics import load_pickle, sha256


QUANTILES = (0, .05, .25, .5, .75, .95, 1)


def describe(values):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("A finite nonempty record-level vector is required")
    result = {
        "n_records": len(values), "mean": float(values.mean()),
        "max_absolute": float(np.abs(values).max()),
        "negative_records": int((values < 0).sum()),
        "zero_records": int((values == 0).sum()),
        "positive_records": int((values > 0).sum()),
    }
    for quantile, value in zip(QUANTILES, np.quantile(values, QUANTILES)):
        result[f"q{int(quantile * 100):02d}"] = float(value)
    return result


def acceptance_status(probabilities):
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.ndim != 1 or len(probabilities) == 0 or not np.isfinite(probabilities).all():
        raise ValueError("Acceptance classification requires an action vector")
    accepted = probabilities >= .5
    return "always" if accepted.all() else "never" if not accepted.any() else "switches"


def extreme_counts(probabilities):
    probabilities = np.asarray(probabilities, dtype=float)
    if not np.isfinite(probabilities).all():
        raise ValueError("Non-finite probability")
    return {
        "p0_exactly_zero": int((probabilities == 0).sum()),
        "p0_exactly_one": int((probabilities == 1).sum()),
    }


def verify_manifest(base, manifest):
    if manifest["status"] != "complete":
        raise ValueError(f"Incomplete input experiment: {base}")
    for name, digest in manifest["output_sha256"].items():
        if sha256(base / name) != digest:
            raise ValueError(f"Changed diagnostic output: {base / name}")


def add_baselines(records, exact):
    keys = ["cohort", "variant", "source_row_id", "action"]
    if records.duplicated(keys).any() or exact.duplicated(keys).any():
        raise ValueError("Duplicate record/action outcomes")
    selected_exact = exact[keys + [
        "exact_fitted_model_p_target0", "exact_fitted_model_accept", "exact_control_delta",
    ]]
    merged = records.merge(selected_exact, on=keys, how="outer", validate="one_to_one", indicator=True)
    if not (merged["_merge"] == "both").all():
        raise ValueError("Monte Carlo/exact record coverage differs")
    merged = merged.drop(columns="_merge")
    base_keys = keys[:-1]
    control = merged[merged.action == "control"][base_keys + [
        "p_target0", "acceptance_rate", "accept_from_pooled_probability",
        "exact_fitted_model_p_target0", "exact_fitted_model_accept",
        "xgb_draw_fraction0", "xgb_aggregate_acceptance",
    ]].rename(columns={
        column: "control_recomputed_acceptance_rate" if column == "acceptance_rate" else f"control_{column}"
        for column in (
            "p_target0", "acceptance_rate", "accept_from_pooled_probability",
            "exact_fitted_model_p_target0", "exact_fitted_model_accept",
            "xgb_draw_fraction0", "xgb_aggregate_acceptance",
        )
    })
    merged = merged.merge(control, on=base_keys, validate="many_to_one")
    np.testing.assert_allclose(
        merged.control_acceptance_rate, merged.control_recomputed_acceptance_rate,
        rtol=0, atol=1e-14,
    )
    merged = merged.drop(columns="control_recomputed_acceptance_rate")
    np.testing.assert_allclose(
        merged.delta_p_target0,
        merged.p_target0 - merged.control_p_target0, rtol=0, atol=1e-14,
    )
    np.testing.assert_allclose(
        merged.exact_control_delta,
        merged.exact_fitted_model_p_target0 - merged.control_exact_fitted_model_p_target0,
        rtol=0, atol=1e-14,
    )
    merged["p_disease1"] = 1 - merged.p_target0
    merged["delta_p_disease1"] = -merged.delta_p_target0
    merged["exact_p_disease1"] = 1 - merged.exact_fitted_model_p_target0
    merged["exact_delta_p_disease1"] = -merged.exact_control_delta
    merged["net_replicate_acceptance_change"] = merged.acceptance_rate - merged.control_acceptance_rate
    merged["pooled_mc_acceptance_change"] = (
        merged.accept_from_pooled_probability - merged.control_accept_from_pooled_probability
    )
    merged["exact_acceptance_change"] = (
        merged.exact_fitted_model_accept - merged.control_exact_fitted_model_accept
    )
    return merged


def action_reports(records, raw):
    summaries, flat = [], []
    for (cohort, variant, action), group in records.groupby(["cohort", "variant", "action"]):
        draws = raw[(raw.cohort == cohort) & (raw.variant == variant) & (raw.action == action)]
        if draws.source_row_id.nunique() != len(group):
            raise ValueError("Record/replicate denominator mismatch")
        mc_delta, exact_delta = describe(group.delta_p_target0), describe(group.exact_control_delta)
        item = {
            "cohort": cohort, "variant": variant, "action": action, "n_records": len(group),
            "mc_mean_p0": float(group.p_target0.mean()),
            "mc_mean_p_disease1": float(group.p_disease1.mean()),
            "exact_mean_p0": float(group.exact_fitted_model_p_target0.mean()),
            "exact_mean_p_disease1": float(group.exact_p_disease1.mean()),
            "mc_record_probability_delta": mc_delta,
            "exact_record_probability_delta": exact_delta,
            "acceptance": {
                "mc_mean_replicate_rate": float(group.acceptance_rate.mean()),
                "mc_control_mean_replicate_rate": float(group.control_acceptance_rate.mean()),
                "mc_net_replicate_rate_change": float(group.net_replicate_acceptance_change.mean()),
                "mc_gained_replicate_rate": float(group.gained_rate.mean()),
                "mc_lost_replicate_rate": float(group.lost_rate.mean()),
                "pooled_mc_accepted_records": int(group.accept_from_pooled_probability.sum()),
                "pooled_mc_gained_records": int((group.pooled_mc_acceptance_change == 1).sum()),
                "pooled_mc_lost_records": int((group.pooled_mc_acceptance_change == -1).sum()),
                "exact_accepted_records": int(group.exact_fitted_model_accept.sum()),
                "exact_gained_records": int((group.exact_acceptance_change == 1).sum()),
                "exact_lost_records": int((group.exact_acceptance_change == -1).sum()),
                "exact_net_rate_change": float(group.exact_acceptance_change.mean()),
            },
            "extremes": {
                "mc_pooled_record": extreme_counts(group.p_target0),
                "exact_integrated_record": extreme_counts(group.exact_fitted_model_p_target0),
                "mc_record_replicate": {
                    "n_record_replicates": len(draws), **extreme_counts(draws.p_target0),
                },
            },
            "increase_records": int((group.direction == "increase").sum()),
            "decrease_records": int((group.direction == "decrease").sum()),
            "out_of_training_minmax_records": int(group.out_of_support.sum()),
        }
        summaries.append(item)
        row = {key: value for key, value in item.items() if not isinstance(value, dict)}
        for prefix, values in (
            ("mc_delta", mc_delta), ("exact_delta", exact_delta), ("acceptance", item["acceptance"]),
        ):
            row.update({f"{prefix}_{name}": value for name, value in values.items()})
        for prefix, values in item["extremes"].items():
            row.update({f"{prefix}_{name}": value for name, value in values.items()})
        flat.append(row)
    return summaries, pd.DataFrame(flat)


def stable_grid_reports(records, raw):
    output = []
    definitions = {
        "absolute_only": lambda names: names.str.startswith("absolute_"),
        "absolute_plus_factual_control": lambda names: names.str.startswith("absolute_") | (names == "control"),
        "relative_including_zero": lambda names: names.str.startswith("reduction_"),
        "all_actions_including_control": lambda names: np.ones(len(names), dtype=bool),
    }
    for grid, selector in definitions.items():
        chosen = records[selector(records.action)]
        chosen_raw = raw[selector(raw.action)]
        for (cohort, variant, source), group in chosen.groupby(["cohort", "variant", "source_row_id"]):
            rep = chosen_raw[
                (chosen_raw.cohort == cohort) & (chosen_raw.variant == variant)
                & (chosen_raw.source_row_id == source)
            ]
            rep_switches = rep.groupby("replicate")["accept"].agg(lambda x: int(np.ptp(x)))
            output.append({
                "cohort": cohort, "variant": variant, "source_row_id": int(source), "grid": grid,
                "n_actions": len(group),
                "exact_acceptance_status": acceptance_status(group.exact_fitted_model_p_target0),
                "pooled_mc_acceptance_status": acceptance_status(group.p_target0),
                "exact_probability_range": float(np.ptp(group.exact_fitted_model_p_target0)),
                "mc_probability_range": float(np.ptp(group.p_target0)),
                "all_replicates_all_actions_accepted": int((rep["accept"] == 1).all()),
                "all_replicates_all_actions_rejected": int((rep["accept"] == 0).all()),
                "any_replicate_switches_across_actions": int(rep_switches.any()),
                "all_replicates_switch_across_actions": int(rep_switches.all()),
            })
    per_record = pd.DataFrame(output)
    summary = []
    for (cohort, variant, grid), group in per_record.groupby(["cohort", "variant", "grid"]):
        row = {"cohort": cohort, "variant": variant, "grid": grid, "n_records": len(group)}
        for prefix, column in (
            ("exact", "exact_acceptance_status"), ("pooled_mc", "pooled_mc_acceptance_status"),
        ):
            for status in ("never", "always", "switches"):
                row[f"{prefix}_{status}_records"] = int((group[column] == status).sum())
        row.update(
            exact_probability_invariant_records=int((group.exact_probability_range == 0).sum()),
            mc_probability_invariant_records=int((group.mc_probability_range == 0).sum()),
            all_replicates_all_actions_accepted_records=int(group.all_replicates_all_actions_accepted.sum()),
            all_replicates_all_actions_rejected_records=int(group.all_replicates_all_actions_rejected.sum()),
            any_replicate_switches_records=int(group.any_replicate_switches_across_actions.sum()),
            all_replicates_switch_records=int(group.all_replicates_switch_across_actions.sum()),
        )
        summary.append(row)
    return per_record, pd.DataFrame(summary)


def bp_seed_reports(seed_raw):
    per_record = seed_raw.groupby(["cohort", "variant", "source_row_id"]).agg(
        legacy_probability_range=("legacy_p_target0", lambda x: float(np.ptp(x))),
        legacy_acceptance_switch=("legacy_accept", lambda x: int(np.ptp(x))),
        matched_probability_range=("matched_p_target0", lambda x: float(np.ptp(x))),
        identical_legacy_repeat_max_difference=("identical_legacy_call_max_absolute_difference", "max"),
    ).reset_index()
    summaries = []
    for (cohort, variant), group in per_record.groupby(["cohort", "variant"]):
        summaries.append({
            "cohort": cohort, "variant": variant, "n_records": len(group),
            "fixed_cholesterol": 200, "proposed_bp_seed_inputs": [100, 120, 140, 160, 180],
            "legacy_probability_range": describe(group.legacy_probability_range),
            "legacy_acceptance_switch_records": int(group.legacy_acceptance_switch.sum()),
            "matched_probability_max_range": float(group.matched_probability_range.max()),
            "identical_legacy_repeat_max_difference": float(group.identical_legacy_repeat_max_difference.max()),
        })
    return per_record, summaries


def structural_reports(records):
    rows = []
    reference = records[records.variant == "refit_full"].set_index(["cohort", "source_row_id", "action"])
    for variant in ("refit_minimal", "refit_extended"):
        changed = records[records.variant == variant].set_index(["cohort", "source_row_id", "action"])
        if changed.empty:
            continue
        if not changed.index.equals(reference.index):
            # Dataframe ordering is not a scientific matching criterion.
            if set(changed.index) != set(reference.index):
                raise ValueError("Structural model record/action coverage differs")
        pair = changed.join(reference, lsuffix="_variant", rsuffix="_full").reset_index()
        for (cohort, action), group in pair.groupby(["cohort", "action"]):
            row = {"cohort": cohort, "variant": variant, "reference": "refit_full", "action": action}
            for label, column in (
                ("mc_level_difference", "p_target0"),
                ("exact_level_difference", "exact_fitted_model_p_target0"),
                ("mc_control_delta_difference", "delta_p_target0"),
                ("exact_control_delta_difference", "exact_control_delta"),
            ):
                stats = describe(group[f"{column}_variant"] - group[f"{column}_full"])
                row.update({f"{label}_{key}": value for key, value in stats.items()})
            row["exact_accepted_label_disagreement_records"] = int((
                group.exact_fitted_model_accept_variant != group.exact_fitted_model_accept_full
            ).sum())
            row["pooled_mc_accepted_label_disagreement_records"] = int((
                group.accept_from_pooled_probability_variant != group.accept_from_pooled_probability_full
            ).sum())
            row["exact_baseline_adjusted_acceptance_change_difference_mean"] = float((
                group.exact_acceptance_change_variant - group.exact_acceptance_change_full
            ).mean())
            rows.append(row)
    return pd.DataFrame(rows)


def aggregation_reports(records, raw):
    rows = []
    for (cohort, variant, action), group in records.groupby(["cohort", "variant", "action"]):
        draws = raw[(raw.cohort == cohort) & (raw.variant == variant) & (raw.action == action)]
        row = {"cohort": cohort, "variant": variant, "action": action, "n_records": len(group)}
        for label, values in (
            ("xgb_draw_fraction_minus_scm_p0", group.xgb_draw_fraction0 - group.p_target0),
            ("xgb_draw_control_delta_minus_scm_control_delta", (
                group.xgb_draw_fraction0 - group.control_xgb_draw_fraction0 - group.delta_p_target0
            )),
        ):
            row.update({f"{label}_{key}": value for key, value in describe(values).items()})
        row.update(
            scm_mode_acceptance_rate=float(group.acceptance_rate.mean()),
            scm_mode_vs_half_probability_disagreement_rows=int((draws.mode_accept != draws["accept"]).sum()),
            xgb_fraction0_mean=float(group.xgb_draw_fraction0.mean()),
            xgb_of_median_mode_profile_acceptance_rate=float(group.xgb_aggregate_acceptance.mean()),
            xgb_aggregate_vs_scm_mode_disagreement_rate=float(draws.aggregation_disagreement.mean()),
            xgb_draw_majority_vs_scm_mode_disagreement_rate=float((
                (draws.xgb_draw_fraction0 >= .5).astype(int) != draws["accept"]
            ).mean()),
            draw_wise_shared_data_xgb_scm_label_agreement=float(draws.xgb_draw_scm_agreement.mean()),
            pooled_noise_threshold_acceptance_rate=float(group.accept_from_pooled_probability.mean()),
            raw_target_outside_binary_fraction=float(draws.target_raw_outside_binary_fraction.mean()),
        )
        rows.append(row)
    return pd.DataFrame(rows)


def verify_mechanisms(base):
    paths = [("main48/frozen_full", ROOT / "model" / "scm_full.pkl")]
    paths.extend(
        (f"fold{fold}/frozen_full", ROOT / "kfold_runs" / "primary_seed42" / f"fold_{fold}" / "scm_full.pkl")
        for fold in range(5)
    )
    paths.extend(
        (f"main48/refit_{variant}", base / "fitted" / f"main48_{variant}.pkl")
        for variant in ("full", "minimal", "extended")
    )
    reports = {}
    for name, path in paths:
        artifact = load_pickle(path)
        model = artifact["causal_model"]
        mechanism = model.causal_mechanism("target")
        if type(mechanism) is not DiscreteAdditiveNoiseModel:
            raise ValueError(f"Unexpected target mechanism changed: {name}")
        values, counts = np.unique(mechanism.noise_model.data, return_counts=True)
        reports[name] = {
            "artifact_sha256": sha256(path), "actual_target_type": type(mechanism).__name__,
            "description": str(mechanism),
            "target_parents_in_coefficient_order": sorted(model.graph.predecessors("target")),
            "coefficients": mechanism.prediction_model.sklearn_model.coef_.tolist(),
            "intercept": float(mechanism.prediction_model.sklearn_model.intercept_),
            "residual_values": values.tolist(), "residual_counts": counts.tolist(),
            "residual_probabilities": (counts / counts.sum()).tolist(),
            "raw_target_noise_is_nondegenerate": bool(len(values) > 1),
            "symptom_parents": {
                node: sorted(model.graph.predecessors(node))
                for node in ("cp", "restecg", "thalach", "exang", "slope", "oldpeak")
            },
        }
    return reports


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "measurements" / "intervention_responsiveness_20261027")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    base = args.input
    output = args.output or base / "compact_report"
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite report: {output}")
    provenance = json.loads((base / "config_and_provenance.json").read_text())
    primary = json.loads((base / "completion.json").read_text())
    supplement = json.loads((base / "supplement" / "completion.json").read_text())
    verify_manifest(base, primary)
    verify_manifest(base / "supplement", supplement)
    input_hashes = {
        str(path.relative_to(ROOT)): sha256(path) for path in base.rglob("*")
        if path.is_file()
    }
    source_paths = [Path(__file__), ROOT / "tests" / "test_simulation_report.py"]
    source_hashes = {str(path.relative_to(ROOT)): sha256(path) for path in source_paths}
    for relative, digest in {
        **provenance["input_sha256"], **provenance["source_sha256"], **supplement["source_sha256"],
    }.items():
        if sha256(ROOT / relative) != digest:
            raise ValueError(f"Historical source or input changed: {relative}")
    mechanisms = verify_mechanisms(base)
    records = pd.read_csv(base / "record_action_means.csv")
    exact = pd.read_csv(base / "supplement" / "exact_fitted_model_target_probabilities.csv")
    raw = pd.read_csv(base / "combined_raw.csv")
    raw["cohort"] = raw.cohort.where(~raw.cohort.str.startswith("fold"), "heldout312")
    merged = add_baselines(records, exact)
    if len(merged) != 7560 or len(raw) != 75600:
        raise ValueError("Unexpected report coverage")
    actions, action_flat = action_reports(merged, raw)
    stable_records, stable_summary = stable_grid_reports(merged, raw)
    seeds = pd.read_csv(base / "combined_bp_seed.csv")
    seeds["cohort"] = seeds.cohort.where(~seeds.cohort.str.startswith("fold"), "heldout312")
    bp_records, bp_summary = bp_seed_reports(seeds)
    structural = structural_reports(merged)
    aggregation = aggregation_reports(merged, raw)
    profile = pd.read_csv(base / "supplement" / "conditional_profile_summary.csv")
    reconstruction = pd.read_csv(base / "supplement" / "control_factual_symptom_reconstruction.csv")
    semantics = {
        "estimand": "Record-conditional response of the fitted structural simulation under do(chol=c), versus do(chol=factual). Not a treatment effect.",
        "probability_equations": [
            "T_raw = round(beta0 + beta_age*age + beta_chol*c + beta_fbs*fbs + beta_sex*sex + beta_bp*B_sim) + U_T.",
            "Full/extended: B_sim = round(g(age,sex,c)) + U_BP; minimal: B_sim = factual BP.",
            "Symptoms are generated using T_raw before clipping; final T_clip=clip(round(T_raw),0,1).",
            "P0=P(T_clip=0)=P(T_raw<=0), P_disease1=1-P0; delta0=P0(action)-P0(do-factual); delta1=-delta0.",
        ],
        "noise": "Fresh marginal residual noise, explicitly shared by node/record/replicate across actions. No factual-residual abduction.",
        "mc": "10 independent noise replicates x 1000 draws; record probabilities/deltas average replicates before record-level summaries.",
        "exact": "Finite empirical residual support integrated over BP and target mechanisms only; exact fitted-model expectation, not true clinical probability.",
        "acceptance": "Historical per-replicate target mode, equivalent to P0>=0.5; ties favor target 0. Pooled-MC and exact probability thresholds are separately labelled.",
        "grids": "Absolute-only excludes factual control; absolute-plus-control includes it; relative-including-zero has five reductions; all-actions includes all 15 scenarios.",
        "zero_one_counts": "Strict equality, not rounding/tolerance. Monte Carlo 0/1 counts and exact finite-support 0/1 counts are separate.",
        "target_mechanism_verification": "All 9 audited artifacts use nondegenerate DiscreteAdditiveNoiseModel target residuals, not ClassifierFCM.",
        "classifier_fcm_general_note": "DoWhy ClassifierFCM itself draws uniform noise and need not be deterministic; only degenerate conditional class probabilities give deterministic class output.",
        "classifier_fcm_noise_source_sha256": hashlib.sha256(inspect.getsource(ClassifierFCM.draw_noise_samples).encode()).hexdigest(),
        "profile_equality": "Full/minimal symptom parents contain only RAW target. Given identical raw target and symptom noise, profiles are identical; this is not automatically conditioning on the clipped label.",
        "uncertainty": "Existing 95% intervals are paired conditional record bootstrap, excluding fitting/selection uncertainty; not treatment-effect intervals.",
    }
    limitations = [
        "Main uses global target-conditioned outlier filtering; main classifier pickle has no training-row metadata. Main SCM's 565-row train hash was verified.",
        "Heldout312 uses each record's own saved outer-fold classifier/SCM, with disjoint training/validation identifiers and verified ready-manifest hashes.",
        "Main48 and heldout312 overlap in 47 source records; cohorts are never pooled as independent patients.",
        "Structural fresh full/minimal/extended refits cover main48 only; outer-fold structural refits were NOT RUN.",
        "Different DAGs can select different mechanisms; noise is matched across actions within each fit, not across distinct mechanism families.",
        "Absolute scenarios may increase cholesterol or be outside training min/max; min/max flags do not prove multivariate support.",
        "Shared-data classifier/SCM agreement is internal consistency, not independent causal validation or clinical mitigation evidence.",
        "Grid diagnostics weight records equally; they are not the repaired DiCE candidate-level acceptance estimand.",
        "Legacy BP-seed comparison holds do(chol=200) fixed. Proposed BP is a seed input only, not a BP intervention.",
        "Historical action-seed sweep has one historical seed per action, whereas primary matched-noise results use ten replicates.",
        "No observational realism, factual symptom preservation, external calibration, or causal identification was established.",
    ]
    report = {
        "schema": "simulation-diagnostics-compact-v1",
        "status": "complete_for_declared_reporting_scope",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha256": source_hashes, "input_sha256": input_hashes,
        "semantics": semantics, "limitations": limitations,
        "cohorts": [
            {"cohort": "main48", "n_records": 48, "variants": ["frozen_full", "refit_full", "refit_minimal", "refit_extended"]},
            {"cohort": "heldout312", "n_records": 312, "variants": ["frozen_full"]},
        ],
        "delta_quantiles": list(QUANTILES), "actions": actions,
        "stable_grid_acceptance": stable_summary.to_dict(orient="records"),
        "bp_seed_effect": bp_summary, "target_mechanisms": mechanisms,
        "personalization": profile.to_dict(orient="records"),
        "control_factual_reconstruction": reconstruction.to_dict(orient="records"),
        "structural_aggregation_tables": ["structural_sensitivity.csv", "aggregation_sensitivity.csv"],
        "structural_scope": "PARTIAL: main-only exact-training DAG refits",
        "clinical_or_independent_causal_validation": False,
    }
    output.mkdir(parents=True)
    merged.to_csv(output / "record_action_deltas.csv", index=False)
    action_flat.to_csv(output / "action_delta_summary.csv", index=False)
    stable_records.to_csv(output / "record_grid_acceptance.csv", index=False)
    stable_summary.to_csv(output / "grid_acceptance_summary.csv", index=False)
    bp_records.to_csv(output / "record_bp_seed_effect.csv", index=False)
    structural.to_csv(output / "structural_sensitivity.csv", index=False)
    aggregation.to_csv(output / "aggregation_sensitivity.csv", index=False)
    write_json(output / "report.json", report)
    for relative, digest in {**input_hashes, **source_hashes}.items():
        if sha256(ROOT / relative) != digest:
            raise RuntimeError(f"Source/input changed while reporting: {relative}")
    write_json(output / "completion.json", {
        "status": "complete", "schema": report["schema"],
        "record_action_rows": len(merged), "action_summary_rows": len(action_flat),
        "output_sha256": {path.name: sha256(path) for path in output.iterdir() if path.is_file()},
    })
    print(json.dumps({
        "report": str(output), "record_action_rows": len(merged),
        "action_summary_rows": len(action_flat),
        "absolute_grid_acceptance": stable_summary[stable_summary.grid == "absolute_only"].to_dict(orient="records"),
        "p0_exact_integrated_zero_one_counts": {
            "zero": int((merged.exact_fitted_model_p_target0 == 0).sum()),
            "one": int((merged.exact_fitted_model_p_target0 == 1).sum()),
        },
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
