"""Re-run the corrected-mechanism audit locally and compare it with the reference summaries.

Usage (repository root, in the project's Python environment):
  python scripts/verify_corrected_audit.py

Writes a fresh run to a new timestamped folder next to the reference, compares
summary.csv, control_profile_summary.csv, matched_noise_check_summary.csv and the
fitted chol coefficients, prints PASS/FAIL, and saves verification_report.json in
the new run folder. Counts must match exactly; continuous values within --tol.
"""
import argparse
import datetime as dt
import importlib.metadata as md
import json
import platform
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def compare(ref, new, tol):
    issues, max_abs = [], 0.0
    if list(ref.columns) != list(new.columns) or len(ref) != len(new):
        return [f"shape/columns differ: {ref.shape} vs {new.shape}"], float("nan")
    for col in ref.columns:
        a, b = ref[col], new[col]
        if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
            integral = (a.dropna() % 1 == 0).all() and (b.dropna() % 1 == 0).all()
            diff = (a - b).abs().max()
            if integral and diff != 0:
                issues.append(f"{col}: count differs (max {diff})")
            elif not integral:
                max_abs = max(max_abs, float(diff))
                if diff > tol:
                    issues.append(f"{col}: max abs difference {diff:.3g} > {tol}")
        elif not a.astype(str).equals(b.astype(str)):
            issues.append(f"{col}: labels differ")
    return issues, max_abs


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, default=ROOT / "data" / "heart_statlog_cleveland_hungary_final.csv")
    p.add_argument("--provenance-dir", type=Path,
                   default=ROOT / "measurements" / "intervention_responsiveness_20261103_structural_all")
    p.add_argument("--reference", type=Path, default=ROOT / "measurements" / "corrected_mechanism_audit_20261108")
    p.add_argument("--tol", type=float, default=1e-6)
    a = p.parse_args()
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = a.reference.parent / f"corrected_mechanism_audit_rerun_{stamp}"
    cmd = [sys.executable, str(ROOT / "scripts" / "run_corrected_mechanism_audit.py"), "--data", str(a.data),
           "--provenance-dir", str(a.provenance_dir), "--output", str(out), "--mc-check"]
    print("Running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    report = {"output": str(out), "python": platform.python_version(), "platform": platform.platform(),
              "packages": {}, "files": {}}
    for pkg in ("dowhy", "scikit-learn", "numpy", "pandas", "scipy", "networkx"):
        try:
            report["packages"][pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            report["packages"][pkg] = None
    ok = True
    for name in ("summary.csv", "control_profile_summary.csv", "matched_noise_check_summary.csv"):
        issues, max_abs = compare(pd.read_csv(a.reference / name), pd.read_csv(out / name), a.tol)
        report["files"][name] = {"issues": issues, "max_abs_continuous_difference": max_abs}
        ok &= not issues
    ref_m = json.loads((a.reference / "mechanisms.json").read_text())
    new_m = json.loads((out / "mechanisms.json").read_text())
    key = "target_chol_log_odds_disease_per_10mgdl"
    coef = {c: abs(ref_m[c][key] - new_m[c][key]) for c in ref_m}
    report["files"]["mechanisms.json"] = {"max_abs_coefficient_difference": max(coef.values())}
    ok &= max(coef.values()) <= a.tol
    report["result"] = "PASS" if ok else "FAIL"
    (out / "verification_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print("RESULT:", report["result"])
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
