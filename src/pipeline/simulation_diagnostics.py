"""Matched-noise diagnostics of the *existing* do(chol) simulation, not causal validation.

Non-descendants retain their observed values; affected non-root mechanisms receive
fresh marginal noise, NOT noise abducted from the record. Noise is sampled once
per node/record/replicate and reused explicitly across every action. Even the
discarded pre-intervention cholesterol draw is retained to reproduce DoWhy's RNG
consumption. This does not modify the published pipeline or its acceptance rule.
"""

import hashlib
import json
import pickle
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from dowhy.graph import get_ordered_predecessors
from dowhy.gcm.causal_mechanisms import AdditiveNoiseModel, DiscreteAdditiveNoiseModel
from sklearn.ensemble import HistGradientBoostingRegressor


CATEGORICAL = ("target", "exang", "fbs", "cp", "restecg", "slope")
SYMPTOMS = ("cp", "restecg", "thalach", "exang", "slope", "oldpeak")
ABSOLUTE_GRID = tuple(range(100, 301, 25))
RELATIVE_GRID = (0, 10, 20, 30, 40)
PROPOSED_BP = (100, 120, 140, 160, 180)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_hash(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def read_envelope(path):
    envelope = json.loads(Path(path).read_text(encoding="utf-8"))
    if set(envelope) != {"payload", "sha256"}:
        raise ValueError(f"Invalid provenance envelope: {path}")
    if canonical_hash(envelope["payload"]) != envelope["sha256"]:
        raise ValueError(f"Provenance checksum mismatch: {path}")
    return envelope["payload"]


def validate_partition(training_ids, validation_ids, selected_ids):
    train, valid, selected = map(set, (training_ids, validation_ids, selected_ids))
    if train & valid or not selected <= valid or len(selected) != len(selected_ids):
        raise ValueError("Training/validation overlap, duplicate or non-held-out selection")


def affected_order(model):
    affected = nx.descendants(model.graph, "chol") | {"chol"}
    return [node for node in nx.topological_sort(model.graph) if node in affected]


def matched_seed(master, source_id, replicate):
    # No proposed value (cholesterol OR BP), symptom, or iteration-order input.
    return int.from_bytes(
        hashlib.sha256(f"{master}:{source_id}:{replicate}".encode()).digest()[:4], "little"
    )


def legacy_seed(original, cholesterol, proposed_bp):
    return int(abs(hash((
        float(original["chol"]), float(original["trestbps"]),
        float(original["thalach"]), float(cholesterol), float(proposed_bp),
    )))) % (2**31)


def draw_noise_plan(model, n, seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        return {
            node: np.array(model.causal_mechanism(node).draw_noise_samples(n), copy=True)
            for node in affected_order(model) if model.graph.in_degree(node)
        }
    finally:
        np.random.set_state(state)


def concatenate_plans(plans):
    return {node: np.concatenate([plan[node] for plan in plans]) for node in plans[0]}


def evaluate_actions(model, original, cholesterol_values, noise, forced_target=None,
                     cache_repeated_parents=True):
    """Return action-major, draw-minor samples using explicit shared noise.

    The forced target argument is ONLY used by the conditional-profile audit.
    The primary measurement never forces target or abducts factual residuals.
    """
    cholesterol_values = np.asarray(cholesterol_values, dtype=float)
    if not np.isfinite(cholesterol_values).all() or len(cholesterol_values) == 0:
        raise ValueError("Actions must be finite and nonempty")
    n = len(next(iter(noise.values())))
    count = len(cholesterol_values)
    samples = {
        node: np.full(n * count, float(original[node])) for node in model.graph
    }
    for node in affected_order(model):
        if node == "chol":
            samples[node] = np.repeat(cholesterol_values, n)
        elif node == "target" and forced_target is not None:
            samples[node] = np.full(n * count, forced_target, dtype=float)
        else:
            mechanism = model.causal_mechanism(node)
            parents = get_ordered_predecessors(model.graph, node)
            parent_samples = np.column_stack([samples[p] for p in parents])
            node_noise = np.tile(noise[node], (count, 1))
            if (
                cache_repeated_parents
                and type(mechanism) in (AdditiveNoiseModel, DiscreteAdditiveNoiseModel)
                and isinstance(mechanism.prediction_model.sklearn_model, HistGradientBoostingRegressor)
            ):
                # Disease-only parents repeat massively. Cache ONLY deterministic
                # additive predictions; retain every original row's exact noise.
                unique, inverse = np.unique(parent_samples, axis=0, return_inverse=True)
                prediction = mechanism.evaluate(
                    unique, np.zeros((len(unique), 1), dtype=node_noise.dtype)
                ).reshape(-1)
                values = prediction[inverse] + node_noise.reshape(-1)
            else:
                values = mechanism.evaluate(parent_samples, node_noise).reshape(-1)
            samples[node] = values.astype(float)
    result = pd.DataFrame(samples)
    if not np.isfinite(result.to_numpy(dtype=float)).all():
        raise ValueError("Non-finite structural output")
    return result


def clip_samples(samples):
    samples = samples.copy()
    samples["oldpeak"] = samples["oldpeak"].clip(lower=0)
    for column, lo, hi in (
        ("exang", 0, 1), ("cp", 1, 4), ("slope", 1, 3),
        ("restecg", 0, 2), ("target", 0, 1),
    ):
        samples[column] = samples[column].round().clip(lo, hi)
    return samples


def aggregate(samples):
    return pd.DataFrame([{
        column: samples[column].mode().iloc[0] if column in CATEGORICAL
        else samples[column].median() for column in samples
    }])


def outcome(samples, classifier=None, features=None, draw_predictions=None):
    probability = float((samples.target == 0).mean())
    row = aggregate(samples)
    result = {
        "p_target0": probability, "p_target1": 1.0 - probability,
        "accept": int(probability >= 0.5),
        "mode_accept": int(row.target.iloc[0] == 0),
    }
    if result["accept"] != result["mode_accept"]:
        raise AssertionError("Binary mode tie policy no longer matches >= 0.5")
    for symptom in SYMPTOMS:
        result[f"aggregate_{symptom}"] = float(row[symptom].iloc[0])
    if classifier is not None:
        if draw_predictions is None:
            draw_predictions = classifier.predict(samples[features])
        if not np.isin(draw_predictions, (0, 1)).all():
            raise ValueError("Classifier returned a non-binary label")
        result.update(
            xgb_draw_fraction0=float((draw_predictions == 0).mean()),
            xgb_draw_scm_agreement=float((draw_predictions == samples.target).mean()),
            xgb_aggregate_accept=int(classifier.predict(row[features])[0] == 0),
        )
    return result


def action_grid(factual_chol):
    return (
        [("control", float(factual_chol))]
        + [(f"absolute_{value}", float(value)) for value in ABSOLUTE_GRID]
        + [(f"reduction_{value}pct", float(factual_chol * (1 - value / 100)))
           for value in RELATIVE_GRID]
    )


def record_bootstrap(values, bootstrap_repeats=2000, seed=913):
    """Conditional RECORD bootstrap: one seed-averaged value per selected record."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("Bootstrap requires one finite value per record")
    rng = np.random.default_rng(seed)
    means = values[rng.integers(len(values), size=(bootstrap_repeats, len(values)))].mean(axis=1)
    return {
        "mean": float(values.mean()), "lower": float(np.quantile(means, .025)),
        "upper": float(np.quantile(means, .975)), "n_records": len(values),
    }


def summarize(raw, bootstrap_repeats):
    keys = ["cohort", "variant", "source_row_id", "action"]
    record = raw.groupby(keys, sort=True).agg(
        p_target0=("p_target0", "mean"), delta_p_target0=("delta_p_target0", "mean"),
        acceptance_rate=("accept", "mean"), control_acceptance_rate=("control_accept", "mean"),
        gained_rate=("gained_acceptance", "mean"), lost_rate=("lost_acceptance", "mean"),
        mc_sd_probability=("p_target0", "std"), mc_sd_delta=("delta_p_target0", "std"),
        xgb_draw_fraction0=("xgb_draw_fraction0", "mean"),
        xgb_aggregate_acceptance=("xgb_aggregate_accept", "mean"),
        xgb_draw_scm_agreement=("xgb_draw_scm_agreement", "mean"),
        aggregation_disagreement=("aggregation_disagreement", "mean"),
        direction=("direction", "first"), out_of_support=("out_of_support", "first"),
        cholesterol=("cholesterol", "first"),
    ).reset_index()
    record["accept_from_pooled_probability"] = (record.p_target0 >= .5).astype(int)
    summaries = []
    for (cohort, variant, action), rows in record.groupby(["cohort", "variant", "action"]):
        summary = {
            "cohort": cohort, "variant": variant, "action": action,
            "n_records": len(rows), "increase_records": int((rows.direction == "increase").sum()),
            "decrease_records": int((rows.direction == "decrease").sum()),
            "out_of_training_range_records": int(rows.out_of_support.sum()),
            "mean_p_target0": float(rows.p_target0.mean()),
            "acceptance_rate_over_replicates": float(rows.acceptance_rate.mean()),
            "pooled_probability_acceptance_rate": float(rows.accept_from_pooled_probability.mean()),
            "gained_acceptance_rate": float(rows.gained_rate.mean()),
            "lost_acceptance_rate": float(rows.lost_rate.mean()),
            "between_record_sd_probability": float(rows.p_target0.std()),
            "mean_within_record_mc_sd_probability": float(rows.mc_sd_probability.mean()),
            "mean_within_record_mc_sd_delta": float(rows.mc_sd_delta.mean()),
            "xgb_draw_fraction0": float(rows.xgb_draw_fraction0.mean()),
            "xgb_aggregate_acceptance_rate": float(rows.xgb_aggregate_acceptance.mean()),
            "draw_wise_xgb_scm_agreement": float(rows.xgb_draw_scm_agreement.mean()),
            "aggregate_xgb_scm_disagreement_rate": float(rows.aggregation_disagreement.mean()),
        }
        for column in ("delta_p_target0", "gained_rate", "lost_rate"):
            ci = record_bootstrap(rows[column], bootstrap_repeats)
            summary[f"{column}_mean"] = ci["mean"]
            summary[f"{column}_conditional_record_bootstrap_lower"] = ci["lower"]
            summary[f"{column}_conditional_record_bootstrap_upper"] = ci["upper"]
        summaries.append(summary)
    response = []
    for (cohort, variant, source), rows in record.groupby(["cohort", "variant", "source_row_id"]):
        absolute = rows[rows.action.str.startswith("absolute_")].sort_values("cholesterol")
        slopes = np.diff(absolute.p_target0) / np.diff(absolute.cholesterol)
        response.append({
            "cohort": cohort, "variant": variant, "source_row_id": int(source),
            "absolute_grid_probability_range": float(np.ptp(absolute.p_target0)),
            "all_actions_probability_range": float(np.ptp(rows.p_target0)),
            "invariant_absolute_probability": int(np.ptp(absolute.p_target0) == 0),
            "invariant_absolute_acceptance": int(np.ptp(absolute.accept_from_pooled_probability) == 0),
            "min_finite_difference_per_mgdl": float(slopes.min()),
            "max_finite_difference_per_mgdl": float(slopes.max()),
            "probability_increases_with_cholesterol_steps": int((slopes > 0).sum()),
            "probability_decreases_with_cholesterol_steps": int((slopes < 0).sum()),
            "zero_response_steps": int((slopes == 0).sum()),
        })
    return record, pd.DataFrame(summaries), pd.DataFrame(response)


def load_pickle(path):
    # Only local, pre-existing trusted artifacts (or artifacts this run generated).
    with Path(path).open("rb") as stream:
        return pickle.load(stream)
