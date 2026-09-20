"""Read-only historical replay and endpoint checks for the paired measurement."""
import argparse
from collections import Counter
import contextlib
import io
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from measure_original_dice_compat import (
    ROOT, ENDPOINTS, load_inputs, sha, write_json, SCMAnalyzer, threadpool_limits,
    pd, np, read_attempts, attempt_state, completion_status,
)


def historical(output):
    cfg, data, train, test, cohort, model = load_inputs()
    archive = ROOT / "fresh_cf_iterations_archive_published_20260607"
    comparison_dirs = {
        "published_archive": archive,
        "backup_cholonly": ROOT / "backups/pre_deterministic_seeding_20260727_152130/fresh_cf_iterations_cholonly",
        "backup_older_main": ROOT / "backups/pre_deterministic_seeding_20260727_152130/fresh_cf_iterations",
        "current_previous_seeded": ROOT / "fresh_cf_iterations",
    }
    versions = {}
    for name, directory in comparison_dirs.items():
        path = directory / "aggregated_results/all_iteration_metrics.csv"
        df = pd.read_csv(path)
        versions[name] = {
            "directory": str(directory), "metrics_sha256": sha(path),
            "returned": int(df.total_generated_cfs.sum()),
            "scm_accepted": int(df.total_successful_cfs.sum()),
            "pooled_rate_pct": float(100 * df.total_successful_cfs.sum() / df.total_generated_cfs.sum()),
            "mean_repeat_rate_pct": float(df.target_flip_rate_pct.mean()),
        }
    # Replay all proposals in the first historical repeat, not regenerated proposals.
    scm = SCMAnalyzer(cfg["scm"])
    scm.initialize_analyzer()
    replay = scm.analyze_iteration(str(archive / "iteration_000"))
    saved = pd.read_csv(archive / "iteration_000/successful/successful_counterfactuals.csv")
    cols = list(saved.columns)

    def canonical(frame):
        return Counter(tuple(round(float(v), 7) for v in row)
                       for row in frame[cols].to_numpy())

    result = {
        "historical_versions": versions,
        "selected_anchor": "published_archive equals backup_cholonly byte-for-byte; README's [33.1,36.7] interval matches this archive, not older main",
        "archive_run": "2026-06-07 14:20:09 to 15:51:41 local, 100 repeats, 48 records, 10 workers, unseeded DiCE",
        "older_main_note": "May 29/30 backup also rounds to 34.8%; it is a different run and formerly constrained proposed trestbps to [100,120]",
        "SCM_replay_repeat_0": {
            "saved_accepted": len(saved), "replayed_accepted": len(replay),
            "accepted_multiset_equal_to_7_decimals": canonical(saved) == canonical(replay)},
        "archive_raw_audit_note": "Re-scoring archived proposals only; NOT a corrected generation experiment",
    }
    all_props, all_propagated = [], []
    for i in range(100):
        directory = archive / f"iteration_{i:03d}"
        for path in sorted((directory / "counterfactuals").glob("*.csv")):
            proposal = pd.read_csv(path)
            proposal["iteration"] = i
            proposal["record_id"] = int(path.stem.split("_")[1])
            all_props.append(proposal)
        good = pd.read_csv(directory / "successful/successful_counterfactuals.csv")
        features = list(cohort.drop(columns="target"))
        frame = good[[f"cf_{c}" for c in features]].copy()
        frame.columns = features
        all_propagated.append(frame)
        if i % 20 == 0:
            print(f"Read historical repeat {i}", flush=True)
    proposals = pd.concat(all_props, ignore_index=True)
    propagated = pd.concat(all_propagated, ignore_index=True)
    predictions = model.predict(proposals[features])
    props_pred = model.predict(propagated[features])
    proposals["native_prediction"] = predictions
    proposals[["iteration", "record_id", "native_prediction"]].to_csv(
        output / "archived_raw_predictions.csv", index=False)
    result["archived_proposals"] = {
        "count": len(proposals), "native_raw_flips": int((predictions == 0).sum()),
        "native_raw_flip_rate_pct": float(100 * (predictions == 0).mean()),
        "bp_outside_100_120": int(((proposals.trestbps < 100) | (proposals.trestbps > 120)).sum()),
        "chol_min": float(proposals.chol.min()), "chol_max": float(proposals.chol.max()),
        "archived_SCM_accepted": len(propagated),
        "native_propagated_flips_among_SCM_accepted": int((props_pred == 0).sum()),
        "archived_same_proposal_joint_rate_pct": float(100 * (props_pred == 0).sum() / len(proposals)),
    }
    write_json(output / "historical_validation.json", result)
    print(json.dumps(result, indent=2))


def check(output):
    records = read_attempts(output)
    rows = [{
        "arm": r["arm"], "iteration": r["iteration"], "record_id": r["record_id"],
        "status": r["status"], "error": r["error"], "returned": r["returned"],
        "desired_class": r.get("desired_class"), "seconds": r["seconds"],
        "state": attempt_state(r),
        "scm_errors": sum(c["scm_error"] is not None for c in r["candidates"]),
        **{e: r["counts"][e] for e in ENDPOINTS},
        **{"any_" + e: int(r["any"][e]) for e in ENDPOINTS},
    } for r in records]
    df = pd.DataFrame(rows, columns=["arm", "iteration", "record_id", "status", "error", "returned",
                                    "desired_class", "seconds", "state", "scm_errors",
                                    *ENDPOINTS, *("any_" + e for e in ENDPOINTS)])
    df = df.astype({column: "int64" for column in ("returned", *ENDPOINTS)})
    result = {"validated_attempts": len(df), "arms": {}, "paired_attempt_contrasts": {}}
    plan = json.loads((output / "run_plan.json").read_text())
    result.update(completion_status(records, plan))
    manifest = json.loads((output / "manifest.json").read_text())
    provenance_path = output / "artifact_provenance.json"
    if not provenance_path.exists():
        raise RuntimeError("Missing audit artifact provenance; use a new audited directory.")
    provenance = json.loads(provenance_path.read_text())
    if not provenance.get("current_artifact_hashes"):
        raise RuntimeError("Empty artifact provenance; use a new audited directory.")
    result["changed_sources"] = [p for p, h in manifest["source_hashes"].items()
                                 if not (ROOT / p).is_file() or sha(ROOT / p) != h]
    result["changed_artifact_references"] = [
        p for p, h in provenance["current_artifact_hashes"].items()
        if not (ROOT / p).is_file() or sha(ROOT / p) != h]
    for arm in ("legacy", "corrected"):
        observed = df[df.arm == arm]
        group = observed[observed.state == "completed"]
        repeat = group.groupby("iteration")[["returned", *ENDPOINTS]].sum()
        valid_repeat = repeat[repeat.returned > 0]
        result["arms"][arm] = {
            "statuses": observed.status.value_counts().to_dict(),
            "errors": observed.error.dropna().value_counts().to_dict(),
            "scm_errors": int(observed.scm_errors.sum()),
            "returned_histogram": group.returned.value_counts().sort_index().to_dict(),
            "opposite_target_histogram": group.desired_class.value_counts().to_dict(),
            "mean_repeat_SCM_acceptance_pct": (
                float((valid_repeat.scm_accept / valid_repeat.returned).mean() * 100)
                if len(valid_repeat) and result["complete"] else None),
            "valid_repeat_SCM_rates": len(valid_repeat),
            "undefined_repeat_SCM_rates": len(repeat) - len(valid_repeat),
            "repeat_rate_interpretation": "Conditional mean over positive-denominator repeats; only emitted for a complete error-free plan",
            "seconds_total": float(observed.seconds.sum()),
            "seconds_max": float(observed.seconds.max()) if len(observed) else None,
            "no_cf_count": int((group.status == "no_cf").sum()),
        }
    if result["complete"] and not result["changed_sources"] and not result["changed_artifact_references"]:
        num = df.groupby(["arm", "record_id"])[["any_" + e for e in ENDPOINTS]].mean()
        ids = sorted(df.record_id.unique())
        boot = np.random.default_rng(20260912).integers(0, len(ids), (10000, len(ids)))
        for e in ENDPOINTS:
            old = num.loc["legacy"].reindex(ids)["any_" + e].to_numpy()
            new = num.loc["corrected"].reindex(ids)["any_" + e].to_numpy()
            diff = new - old
            result["paired_attempt_contrasts"][e] = {
                "delta_pp": float(100 * diff.mean()),
                "paired_record_bootstrap_95pct_pp": (100 * np.quantile(diff[boot].mean(axis=1), [.025, .975])).tolist(),
            }
        per_record = df.groupby(["arm", "record_id"])[["returned", *ENDPOINTS, *("any_" + e for e in ENDPOINTS)]].sum()
        per_record.to_csv(output / "record_totals.csv")
    write_json(output / "endpoint_validation.json", result)
    print(json.dumps(result, indent=2))
    if not result["complete"]:
        raise RuntimeError("Measurement is incomplete or contains failed attempts; see endpoint_validation.json")
    if result["changed_sources"] or result["changed_artifact_references"]:
        raise RuntimeError("Source or artifact reference changed during measurement")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("historical", "check"))
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    with threadpool_limits(1):
        if args.mode == "historical":
            historical(args.output)
        else:
            check(args.output)
