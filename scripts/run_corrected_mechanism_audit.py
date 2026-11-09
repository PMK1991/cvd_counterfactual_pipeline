"""Corrected-mechanism simulation audit (binary target, action-independent noise).

This script refits the full SCM graph on the SAME training rows used by the
original main split and by each outer fold, but with explicitly specified,
deterministic mechanism families instead of DoWhy auto-assignment:

  age, sex, fbs            EmpiricalDistribution (roots)
  chol                     AdditiveNoiseModel(LinearRegression)          [age, sex]
  trestbps                 AdditiveNoiseModel(LinearRegression)          [age, sex, chol]
  target                   ClassifierFCM(standardised LogisticRegression) [age, sex, chol, fbs, trestbps]
  cp, restecg, exang, slope ClassifierFCM(LogisticRegression)            [target]
  thalach, oldpeak         AdditiveNoiseModel(LinearRegression)          [target]

The target is therefore a binary classifier mechanism whose samples are always in
{0, 1}; no clipping is needed before symptom propagation.

Two simulation semantics are evaluated for every record and cholesterol action:

  C1  Interventional (rung 2), the corrected analogue of the original audit.
      P0(i, a) = P(target = 0 | do(chol = a), record i's non-descendants).
      Computed by exact integration over the fitted trestbps residual
      distribution; a matched-noise Monte Carlo check (10 x 1,000 draws, seeds
      keyed only by master seed / source row / replicate) is also reported.
      Neither depends on any proposed value, so no action-dependent seed exists.

  C2  Factual abduction (rung 3) under DoWhy's own noise models.
      The record's additive residuals (trestbps, thalach, oldpeak) are abducted
      and reused; the target's uniform noise is restricted to the interval
      consistent with the observed target = 1 (rank-preserving coupling), so
      P(target' = 0 | target = 1, do(chol = a)) = max(P0' - P0_obs, 0) / (1 - P0_obs).
      Categorical symptom noise is restricted to the observed category's
      interval; this coupling depends on the arbitrary order of nominal codes
      and is reported as a sensitivity assumption only.
      Under the unchanged-cholesterol control C2 reproduces the factual record
      exactly.

Both are fitted-model quantities on cross-sectional data. They are NOT
identified treatment effects, are NOT validated against outcomes, and the
classifier is not used as a validator anywhere in this script.

Usage (repository root):
  python scripts/run_corrected_mechanism_audit.py \
      --data data/heart_statlog_cleveland_hungary_final.csv \
      --provenance-dir measurements/intervention_responsiveness_20261103_structural_all \
      --output measurements/corrected_mechanism_audit_20261108
"""

import argparse
import hashlib
import json
from pathlib import Path
import warnings

import networkx as nx
import numpy as np
import pandas as pd
from dowhy import gcm
from dowhy.gcm.ml.classification import SklearnClassificationModel
from dowhy.gcm.ml.regression import SklearnRegressionModel
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

COHORTS = ("main48", "fold0", "fold1", "fold2", "fold3", "fold4")
EDGES = [  # identical to SCMAnalyzer.GRAPH_VARIANTS["full"]
    ("age", "target"), ("sex", "target"), ("chol", "target"), ("fbs", "target"),
    ("trestbps", "target"),
    ("target", "cp"), ("target", "restecg"), ("target", "thalach"),
    ("target", "exang"), ("target", "slope"), ("target", "oldpeak"),
    ("age", "chol"), ("age", "trestbps"), ("sex", "trestbps"), ("sex", "chol"),
    ("chol", "trestbps"),
]
TARGET_PARENTS = ["age", "chol", "fbs", "sex", "trestbps"]  # DoWhy ordered (sorted)
BP_PARENTS = ["age", "chol", "sex"]
CATEGORICAL_SYMPTOMS = ("cp", "restecg", "exang", "slope")
CONTINUOUS_SYMPTOMS = ("thalach", "oldpeak")
ABSOLUTE_GRID = tuple(range(100, 301, 25))
CORRIDOR = (150, 175, 200)
RELATIVE_GRID = (0, 10, 20, 30, 40)
MASTER_SEED = 20261108
FIT_SEED = 42


def logistic():
    return SklearnClassificationModel(
        make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))
    )


def linear():
    return SklearnRegressionModel(LinearRegression())


def fit_corrected(training):
    graph = nx.DiGraph(EDGES)
    model = gcm.InvertibleStructuralCausalModel(graph)
    for root in ("age", "sex", "fbs"):
        model.set_causal_mechanism(root, gcm.EmpiricalDistribution())
    model.set_causal_mechanism("chol", gcm.AdditiveNoiseModel(linear()))
    model.set_causal_mechanism("trestbps", gcm.AdditiveNoiseModel(linear()))
    model.set_causal_mechanism("target", gcm.ClassifierFCM(logistic()))
    for node in CATEGORICAL_SYMPTOMS:
        model.set_causal_mechanism(node, gcm.ClassifierFCM(logistic()))
    for node in CONTINUOUS_SYMPTOMS:
        model.set_causal_mechanism(node, gcm.AdditiveNoiseModel(linear()))
    data = training[list(graph.nodes)].copy()
    for column in ("target",) + CATEGORICAL_SYMPTOMS:
        data[column] = data[column].astype(int).astype(str)  # ClassifierFCM needs string labels
    state = np.random.get_state()
    np.random.seed(FIT_SEED)
    try:
        gcm.fit(model, data)
    finally:
        np.random.set_state(state)
    return model


def classes(mechanism):
    """Fitted class labels in the column order of estimate_probabilities."""
    k = len(mechanism.classifier_model.classes)
    return [int(c) for c in mechanism.get_class_names(np.arange(k))]


def bp_mechanism_parts(model):
    mech = model.causal_mechanism("trestbps")
    residuals = np.asarray(mech.noise_model._data).reshape(-1)  # fitted empirical residuals
    return mech, residuals


def bp_prediction(model, frame):
    mech = model.causal_mechanism("trestbps")
    return mech.prediction_model.predict(frame[BP_PARENTS].to_numpy(float)).reshape(-1)


def target_p0(model, frame):
    mech = model.causal_mechanism("target")
    probabilities = mech.estimate_probabilities(frame[TARGET_PARENTS].to_numpy(float))
    return probabilities[:, classes(mech).index(0)]


def exact_interventional_p0(model, record, cholesterol_values):
    """C1: exact integration over the fitted trestbps residual distribution."""
    _, residuals = bp_mechanism_parts(model)
    rows = []
    for value in cholesterol_values:
        base = pd.DataFrame([{**record, "chol": float(value)}])
        mean_bp = bp_prediction(model, base)[0]
        frame = pd.DataFrame({
            "age": record["age"], "sex": record["sex"], "fbs": record["fbs"],
            "chol": float(value), "trestbps": mean_bp + residuals,
        })
        rows.append(float(target_p0(model, frame).mean()))
    return np.asarray(rows)


def matched_noise_interventional_p0(model, record, cholesterol_values, source, replicates=10, draws=1000):
    """C1 Monte Carlo check: noise keyed by record and replicate only."""
    _, residuals = bp_mechanism_parts(model)
    out = np.zeros((len(cholesterol_values), replicates))
    for rep in range(replicates):
        seed = int.from_bytes(hashlib.sha256(f"{MASTER_SEED}:{source}:{rep}".encode()).digest()[:4], "little")
        rng = np.random.default_rng(seed)
        bp_noise = residuals[rng.integers(len(residuals), size=draws)]
        target_u = rng.random(draws)
        for k, value in enumerate(cholesterol_values):
            base = pd.DataFrame([{**record, "chol": float(value)}])
            bp = bp_prediction(model, base)[0] + bp_noise
            frame = pd.DataFrame({"age": record["age"], "sex": record["sex"], "fbs": record["fbs"],
                                  "chol": float(value), "trestbps": bp})
            p0 = target_p0(model, frame)
            out[k, rep] = float((target_u < p0).mean())  # DoWhy ClassifierFCM inverse-CDF rule
    return out


def counterfactual_p0(model, record, cholesterol_values):
    """C2: rank-preserving abduction of the target noise given observed target = 1."""
    bp_eps = record["trestbps"] - bp_prediction(model, pd.DataFrame([record]))[0]
    p_obs = target_p0(model, pd.DataFrame([record]))[0]
    rows = []
    for value in cholesterol_values:
        base = pd.DataFrame([{**record, "chol": float(value)}])
        bp = bp_prediction(model, base)[0] + bp_eps
        p_new = target_p0(model, pd.DataFrame([{**record, "chol": float(value), "trestbps": bp}]))[0]
        rows.append(max(p_new - p_obs, 0.0) / (1.0 - p_obs))
    return np.asarray(rows), float(p_obs)


def symptom_profiles(model, record, p_target0, forced=None):
    """Expected symptom profile. Interventional (factual=False) uses marginal noise;
    counterfactual reuses the record's own residual / category interval."""
    out = {}
    for node in CONTINUOUS_SYMPTOMS:
        mech = model.causal_mechanism(node)
        f = {t: float(mech.prediction_model.predict(np.array([[str(t)]], dtype=object)).reshape(-1)[0]) for t in (0, 1)}
        eps = record[node] - f[1]  # observed target = 1 for every selected record
        out[node] = {
            "interventional": p_target0 * f[0] + (1 - p_target0) * f[1] + float(np.mean(np.asarray(mech.noise_model._data))),
            "counterfactual": p_target0 * (f[0] + eps) + (1 - p_target0) * record[node],
        }
    for node in CATEGORICAL_SYMPTOMS:
        mech = model.causal_mechanism(node)
        labels = classes(mech)
        probs = {t: mech.estimate_probabilities(np.array([[str(t)]], dtype=object))[0] for t in (0, 1)}
        cdf1 = np.concatenate([[0], np.cumsum(probs[1])])
        cdf0 = np.concatenate([[0], np.cumsum(probs[0])])
        iv = p_target0 * probs[0] + (1 - p_target0) * probs[1]
        if int(record[node]) not in labels:  # category unseen in this training split
            out[node] = {"interventional_mode": labels[int(np.argmax(iv))],
                         "counterfactual_mode": int(record[node]), "counterfactual_p_unchanged": 1.0,
                         "unseen_category": 1}
            continue
        j = labels.index(int(record[node]))
        lo, hi = cdf1[j], cdf1[j + 1]
        overlap = np.array([max(0.0, min(hi, cdf0[k + 1]) - max(lo, cdf0[k])) for k in range(len(labels))])
        cf_given0 = overlap / (hi - lo) if hi > lo else np.eye(len(labels))[j]
        cf = p_target0 * cf_given0 + (1 - p_target0) * np.eye(len(labels))[j]
        iv = p_target0 * probs[0] + (1 - p_target0) * probs[1]
        out[node] = {
            "interventional_mode": labels[int(np.argmax(iv))],
            "counterfactual_mode": labels[int(np.argmax(cf))],
            "counterfactual_p_unchanged": float(cf[j]),
        }
    return out


def record_bootstrap(values, repeats=2000, seed=913):
    values = np.asarray(values, float)
    rng = np.random.default_rng(seed)
    means = values[rng.integers(len(values), size=(repeats, len(values)))].mean(axis=1)
    return float(values.mean()), float(np.quantile(means, .025)), float(np.quantile(means, .975))


def status(accepts):
    accepts = np.asarray(accepts)
    return "always" if accepts.all() else "never" if not accepts.any() else "switch"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--provenance-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mc-check", action="store_true", help="also run the matched-noise Monte Carlo check")
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=False)
    data = pd.read_csv(args.data)
    warnings.simplefilter("ignore")
    record_rows, profile_rows, mc_rows, mechanisms = [], [], [], {}
    for cohort in COHORTS:
        provenance = json.loads((args.provenance_dir / f"{cohort}_provenance.json").read_text())
        training = data.loc[provenance["training_ids"]]
        selected = data.loc[provenance["selected_ids"]]
        if (selected.target != 1).any():
            raise ValueError("Selected records must be observed target = 1")
        model = fit_corrected(training)
        mechanisms[cohort] = {node: str(model.causal_mechanism(node)) for node in model.graph}
        pipe = model.causal_mechanism("target").classifier_model.sklearn_model
        k = TARGET_PARENTS.index("chol")
        sign = 1 if classes(model.causal_mechanism("target"))[1] == 1 else -1
        mechanisms[cohort]["target_chol_log_odds_disease_per_10mgdl"] = float(
            sign * pipe[-1].coef_[0][k] / pipe[0].scale_[k] * 10)
        mechanisms[cohort]["n_training"] = len(training)
        support = (training.chol.min(), training.chol.max())
        for source, row in selected.iterrows():
            record = row.astype(float).to_dict()
            actions = ([("control", record["chol"])]
                       + [(f"absolute_{v}", float(v)) for v in ABSOLUTE_GRID]
                       + [(f"reduction_{v}pct", record["chol"] * (1 - v / 100)) for v in RELATIVE_GRID])
            values = [v for _, v in actions]
            c1 = exact_interventional_p0(model, record, values)
            c2, p_obs = counterfactual_p0(model, record, values)
            if args.mc_check:
                mc = matched_noise_interventional_p0(model, record, values, int(source))
                for (name, value), reps, exact in zip(actions, mc, c1):
                    mc_rows.append({"cohort": cohort, "source_row_id": int(source), "action": name,
                                    "exact_p0": exact, "mc_mean_p0": reps.mean(),
                                    "mc_replicate_accept_rate": float((reps >= .5).mean())})
            for (name, value), p1, p2 in zip(actions, c1, c2):
                record_rows.append({
                    "cohort": cohort, "source_row_id": int(source), "action": name,
                    "cholesterol": value, "factual_cholesterol": record["chol"],
                    "out_of_training_range": int(value < support[0] or value > support[1]),
                    "c1_interventional_p0": p1, "c1_accept": int(p1 >= .5),
                    "c2_counterfactual_p0": p2, "c2_accept": int(p2 >= .5),
                    "observed_record_model_p0": p_obs,
                })
            control_c1, control_c2 = c1[0], c2[0]
            for semantics, p0 in (("C1_interventional", control_c1), ("C2_counterfactual", control_c2)):
                prof = symptom_profiles(model, record, p0)
                for node in CONTINUOUS_SYMPTOMS:
                    key = "interventional" if semantics.startswith("C1") else "counterfactual"
                    profile_rows.append({"cohort": cohort, "semantics": semantics, "source_row_id": int(source),
                                         "symptom": node, "factual": record[node],
                                         "control_expected": prof[node][key]})
                for node in CATEGORICAL_SYMPTOMS:
                    key = "interventional_mode" if semantics.startswith("C1") else "counterfactual_mode"
                    profile_rows.append({"cohort": cohort, "semantics": semantics, "source_row_id": int(source),
                                         "symptom": node, "factual": record[node],
                                         "control_expected": float(prof[node][key])})
    records = pd.DataFrame(record_rows)
    records.to_csv(args.output / "record_actions.csv", index=False)
    pd.DataFrame(profile_rows).to_csv(args.output / "control_profiles.csv", index=False)
    if mc_rows:
        pd.DataFrame(mc_rows).to_csv(args.output / "matched_noise_check.csv", index=False)
    (args.output / "mechanisms.json").write_text(json.dumps(mechanisms, indent=2))

    main_ids = set(records[records.cohort == "main48"].source_row_id)
    records["cohort_group"] = np.where(records.cohort == "main48", "main48", "heldout312")
    held = records[records.cohort_group == "heldout312"]
    nonoverlap = held[~held.source_row_id.isin(main_ids)].copy()
    nonoverlap["cohort_group"] = "nonoverlap265"
    groups = pd.concat([records, nonoverlap])
    summary = []
    for group, frame in groups.groupby("cohort_group"):
        for semantics, pcol, acol in (("C1", "c1_interventional_p0", "c1_accept"),
                                      ("C2", "c2_counterfactual_p0", "c2_accept")):
            wide = frame.pivot_table(index="source_row_id", columns="action", values=pcol)
            acc = frame.pivot_table(index="source_row_id", columns="action", values=acol)
            row = {"cohort": group, "semantics": semantics, "n_records": len(wide),
                   "control_mean_p0": wide["control"].mean(),
                   "control_accept_records": int(acc["control"].sum())}
            for action in ("reduction_20pct", "absolute_200"):
                delta = wide[action] - wide["control"]
                mean, lo, hi = record_bootstrap(delta)
                row.update({f"{action}_delta_mean": mean, f"{action}_delta_lo": lo, f"{action}_delta_hi": hi,
                            f"{action}_pos": int((delta > 1e-12).sum()), f"{action}_zero": int((delta.abs() <= 1e-12).sum()),
                            f"{action}_neg": int((delta < -1e-12).sum()),
                            f"{action}_accept_records": int(acc[action].sum())})
            for label, cols in (("absolute", [f"absolute_{v}" for v in ABSOLUTE_GRID]),
                                ("corridor", [f"absolute_{v}" for v in CORRIDOR])):
                st = acc[cols].apply(lambda r: status(r.values), axis=1).value_counts()
                rng = wide[cols].max(axis=1) - wide[cols].min(axis=1)
                row.update({f"{label}_always": int(st.get("always", 0)), f"{label}_never": int(st.get("never", 0)),
                            f"{label}_switch": int(st.get("switch", 0)),
                            f"{label}_prob_invariant": int((rng <= 1e-12).sum())})
            summary.append(row)
    pd.DataFrame(summary).to_csv(args.output / "summary.csv", index=False)

    profiles = pd.DataFrame(profile_rows)
    profiles["cohort_group"] = np.where(profiles.cohort == "main48", "main48", "heldout312")
    prof_summary = []
    for (group, semantics, symptom), frame in profiles.groupby(["cohort_group", "semantics", "symptom"]):
        diff = frame.control_expected - frame.factual
        prof_summary.append({"cohort": group, "semantics": semantics, "symptom": symptom,
                             "mean_signed_difference": diff.mean(), "mean_absolute_difference": diff.abs().mean(),
                             "exact_reconstruction_fraction": float((diff.abs() < 1e-9).mean()),
                             "distinct_control_values": int(frame.control_expected.round(6).nunique())})
    pd.DataFrame(prof_summary).to_csv(args.output / "control_profile_summary.csv", index=False)
    if mc_rows:
        mc = pd.DataFrame(mc_rows)
        pd.DataFrame([{
            "max_abs_mc_minus_exact": float((mc.mc_mean_p0 - mc.exact_p0).abs().max()),
            "mean_abs_mc_minus_exact": float((mc.mc_mean_p0 - mc.exact_p0).abs().mean()),
            "pooled_threshold_disagreements": int(((mc.mc_mean_p0 >= .5) != (mc.exact_p0 >= .5)).sum()),
            "record_actions": len(mc)}]).to_csv(args.output / "matched_noise_check_summary.csv", index=False)
    print(pd.DataFrame(summary).T.to_string())


if __name__ == "__main__":
    main()
