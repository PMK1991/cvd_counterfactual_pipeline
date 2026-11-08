"""Derived summaries for the R3 round-2 revision (pure pandas; no model fitting).

Regenerates, from the saved outputs of run_simulation_diagnostics.py --structural all:
  1. the strict source-row non-overlap (265-record) sensitivity, and checks it
     against the previously saved strict_nonoverlap_findings.csv;
  2. acceptance status restricted to the DiCE cholesterol corridor (150/175/200
     mg/dL) using pooled matched-noise Monte Carlo probabilities;
  3. historical record-level acceptance counts (never / every repeat);
  4. a frozen-versus-refit comparison of automatically selected mechanisms;
  5. an INFERRED source composition of the pooled file (row-order hypothesis
     checked against source-specific signatures). Inferred, not verified.

Usage (repository root):
  python scripts/summarize_structural_supplement.py \
      --diagnostics measurements/intervention_responsiveness_20261103_structural_all \
      --audit measurements/independent_audit_20260919_1630 \
      --data data/heart_statlog_cleveland_hungary_final.csv \
      --output measurements/structural_supplement_20261108
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

CORRIDOR = ("absolute_150", "absolute_175", "absolute_200")
ABSOLUTE = tuple(f"absolute_{v}" for v in range(100, 301, 25))
# Inferred row order of the distributed pooled file (see section 5).
SOURCE_BLOCKS = (("Hungarian", 294), ("Switzerland", 123), ("Long Beach VA", 200),
                 ("Cleveland", 303), ("Statlog", 270))


def status_counts(table, actions):
    wide = table[table.action.isin(actions)].pivot_table(
        index="source_row_id", columns="action", values="accept_from_pooled_probability")
    lo, hi = wide.min(axis=1), wide.max(axis=1)
    prob = table[table.action.isin(actions)].pivot_table(index="source_row_id", columns="action", values="p_target0")
    return {"n": len(wide), "always": int((lo == 1).sum()), "never": int((hi == 0).sum()),
            "switch": int(((lo == 0) & (hi == 1)).sum()),
            "probability_invariant": int(((prob.max(axis=1) - prob.min(axis=1)) == 0).sum())}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--diagnostics", type=Path, required=True)
    p.add_argument("--audit", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args(argv)
    a.output.mkdir(parents=True, exist_ok=True)
    D = a.diagnostics
    main_ids = set(json.loads((D / "main48_provenance.json").read_text())["selected_ids"])

    # 1. strict non-overlap sensitivity
    raw = pd.read_csv(D / "combined_raw.csv", usecols=["cohort", "variant", "source_row_id", "action", "accept",
                                                       "p_target0", "target_raw_outside_binary_fraction"])
    raw = raw[raw.cohort.str.startswith("fold") & ~raw.source_row_id.isin(main_ids)]
    response = pd.read_csv(D / "record_responsiveness.csv")
    response = response[(response.cohort == "heldout312") & ~response.source_row_id.isin(main_ids)]
    seeds = pd.read_csv(D / "bp_seed_record_ranges.csv")
    seeds = seeds[(seeds.cohort == "heldout312") & ~seeds.source_row_id.isin(main_ids)]
    rows = []
    for variant, group in raw.groupby("variant"):
        controls = group[group.action == "control"]
        rr, bp = response[response.variant == variant], seeds[seeds.variant == variant]
        rows.append({
            "cohort": "heldout_nonoverlap265", "variant": variant, "n_records": group.source_row_id.nunique(),
            "removed_overlap_records": 312 - group.source_row_id.nunique(),
            "raw_record_action_replicate_rows": len(group),
            "control_mean_simulated_p_target0": controls.p_target0.mean(),
            "control_acceptance_rate": controls.accept.mean(),
            "invariant_absolute_probability_records": int(rr.invariant_absolute_probability.sum()),
            "invariant_absolute_acceptance_records": int(rr.invariant_absolute_acceptance.sum()),
            "mean_absolute_probability_response_range": rr.absolute_grid_probability_range.mean(),
            "max_absolute_probability_response_range": rr.absolute_grid_probability_range.max(),
            "legacy_bp_seed_mean_probability_range": bp.legacy_probability_range.mean(),
            "legacy_bp_seed_max_probability_range": bp.legacy_probability_range.max(),
            "legacy_bp_seed_acceptance_flip_records": int(bp.legacy_acceptance_range.sum()),
            "matched_bp_seed_max_probability_range": bp.matched_probability_range.max(),
            "fraction_raw_target_samples_outside_0_1_before_legacy_clip": group.target_raw_outside_binary_fraction.mean(),
        })
    strict = pd.DataFrame(rows)
    strict.to_csv(a.output / "strict_nonoverlap_findings_regenerated.csv", index=False)
    saved = D / "strict_nonoverlap_findings.csv"
    if saved.exists():
        old = pd.read_csv(saved).set_index("variant").sort_index()
        new = strict.set_index("variant").sort_index()[old.columns]
        num = old.select_dtypes("number").columns
        max_diff = float((old[num] - new[num]).abs().max().max())
        print("strict non-overlap regenerated vs saved, max abs difference:", max_diff)
        if max_diff > 1e-9:
            raise SystemExit("Regenerated non-overlap summary does not match the saved file")

    # 2. corridor and grid status (pooled matched-noise MC probabilities)
    record = pd.read_csv(D / "record_action_means.csv")
    status = []
    for (cohort, variant), group in record.groupby(["cohort", "variant"]):
        subsets = [(cohort, group)]
        if cohort == "heldout312":
            subsets.append(("heldout_nonoverlap265", group[~group.source_row_id.isin(main_ids)]))
        for name, frame in subsets:
            for grid, actions in (("corridor_150_200", CORRIDOR), ("absolute_100_300", ABSOLUTE)):
                status.append({"cohort": name, "variant": variant, "grid": grid, **status_counts(frame, actions)})
    status = pd.DataFrame(status)
    status.to_csv(a.output / "grid_status_pooled_mc.csv", index=False)

    # 3. historical record-level acceptance (original 100-repeat recourse outputs)
    hist = []
    for label, name in (("legacy", "legacy_record_totals_recomputed.csv"),
                        ("corrected_dice", "corrected_record_totals_recomputed.csv")):
        t = pd.read_csv(a.audit / name)
        hist.append({"analysis": label, "n_records": len(t),
                     "never_accepted": int((t.any_scm_accept == 0).sum()),
                     "accepted_in_every_repeat": int((t.any_scm_accept == t.any_scm_accept.max()).sum()),
                     "every_returned_proposal_accepted": int((t.scm_accept == t.returned).sum()),
                     "repeats": int(t.any_raw_flip.max())})
    pd.DataFrame(hist).to_csv(a.output / "historical_record_acceptance.csv", index=False)

    # 4. frozen vs refit mechanism selection
    mech = []
    for cohort in ["main48"] + [f"fold{i}" for i in range(5)]:
        models = json.loads((D / f"{cohort}_provenance.json").read_text())["models"]
        for node in models["frozen_full"]:
            f, r = models["frozen_full"][node]["description"], models["refit_full"][node]["description"]
            if f != r:
                mech.append({"cohort": cohort, "node": node, "frozen_full": f, "refit_full": r})
    pd.DataFrame(mech).to_csv(a.output / "frozen_vs_refit_mechanisms.csv", index=False)
    frozen_vs_refit = record[record.cohort == "heldout312"].pivot_table(
        index=["source_row_id", "action"], columns="variant", values="p_target0")
    fold_of = {}
    for i in range(5):
        for s in json.loads((D / f"fold{i}_provenance.json").read_text())["selected_ids"]:
            fold_of[s] = i
    diff = (frozen_vs_refit["frozen_full"] - frozen_vs_refit["refit_full"]).abs().reset_index()
    diff["fold"] = diff.source_row_id.map(fold_of)
    diff.groupby("fold")[0].max().rename("max_abs_p0_difference").to_csv(a.output / "frozen_vs_refit_p0_by_fold.csv")

    # 5. inferred source composition
    data = pd.read_csv(a.data)
    data["inferred_source"] = np.concatenate([[n] * k for n, k in SOURCE_BLOCKS])
    features = data.columns.drop("inferred_source")
    dup = data[features].duplicated(keep="first")
    eligible = (data.chol != 0) & ~dup
    held = []
    for i in range(5):
        held += json.loads((D / f"fold{i}_provenance.json").read_text())["selected_ids"]
    comp = pd.DataFrame({
        "distributed_records": data.inferred_source.value_counts(),
        "cholesterol_zero": data[data.chol == 0].inferred_source.value_counts(),
        "later_exact_duplicates": data[dup].inferred_source.value_counts(),
        "eligible_746": data[eligible].inferred_source.value_counts(),
        "target_prevalence_all": data.groupby("inferred_source").target.mean().round(3),
        "main_recourse_48": data.loc[sorted(main_ids)].inferred_source.value_counts(),
        "heldout_recourse_312": data.loc[held].inferred_source.value_counts(),
        "nonoverlap_265": data.loc[sorted(set(held) - main_ids)].inferred_source.value_counts(),
    }).reindex([n for n, _ in SOURCE_BLOCKS]).fillna(0)
    comp.to_csv(a.output / "inferred_source_composition.csv")
    # where do the originals of the later duplicates sit?
    key = data[features].astype(str).agg("|".join, axis=1)
    first = {k: i for i, k in key.drop_duplicates(keep="first").items()}
    originals = pd.Series([first[key[i]] for i in data.index[dup]])
    origin = data.loc[originals, "inferred_source"].value_counts()
    origin.rename("originals_of_later_duplicates").to_csv(a.output / "duplicate_origin_blocks.csv")
    # eligible Statlog-block rows: minimum number of differing fields vs any Cleveland-block row
    cle = data[data.inferred_source == "Cleveland"][features].to_numpy(float)
    st = data[eligible & (data.inferred_source == "Statlog")][features].to_numpy(float)
    mins = [int((np.abs(cle - r) > 1e-9).sum(axis=1).min()) for r in st]
    pd.Series(mins).value_counts().sort_index().rename("statlog_eligible_min_fields_differing_from_cleveland").to_csv(
        a.output / "statlog_vs_cleveland_near_duplicates.csv")
    print(comp.to_string())
    print(status.to_string())
    print(pd.DataFrame(hist).to_string())
    print(pd.DataFrame(mech).to_string())
    print(origin.to_string())


if __name__ == "__main__":
    main()
