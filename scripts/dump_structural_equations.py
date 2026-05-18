# conda-env: mtech-env
"""
Dump SCM structural equations.

Builds the SCM with the same config used by the pipeline, fits it on the
cleaned dataset, and writes both a machine-readable JSON dump and a
human-readable Markdown summary of the per-node mechanisms.

Usage:
    python scripts/dump_structural_equations.py \
        --output_dir fresh_cf_iterations/ablation_filtered/aggregated_results
"""

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import yaml

from src.pipeline.scm_analyzer import SCMAnalyzer


def _rewrite_feature_names(feature_names, parents):
    """Map sklearn's x0/x1/... placeholders back to parent variable names."""
    if not feature_names or not parents:
        return feature_names
    out = []
    for fname in feature_names:
        renamed = fname
        # Replace longest indices first so x10 doesn't collide with x1
        for i in sorted(range(len(parents)), reverse=True):
            renamed = renamed.replace(f"x{i}", parents[i])
        out.append(renamed)
    return out


def _extract_fit(mechanism, parents=None):
    """Return a JSON-serialisable summary of the prediction model."""
    pred = getattr(mechanism, "prediction_model", None)
    if pred is None:
        return None
    sk = getattr(pred, "sklearn_model", pred)
    name = type(sk).__name__

    info: dict = {"estimator": name}

    # Pipeline → look at final step
    if name == "Pipeline":
        steps = [s[0] for s in sk.steps]
        info["pipeline_steps"] = steps
        final = sk.steps[-1][1]
        info["final_estimator"] = type(final).__name__
        if hasattr(final, "coef_"):
            try:
                feature_names = sk[:-1].get_feature_names_out().tolist()
            except Exception:
                feature_names = None
            feature_names = _rewrite_feature_names(feature_names, parents)
            coefs = np.atleast_1d(final.coef_).ravel().tolist()
            if feature_names and len(feature_names) == len(coefs):
                info["feature_names"] = feature_names
                info["coefficients"] = dict(zip(feature_names, coefs))
            else:
                info["coefficients"] = coefs
            if hasattr(final, "intercept_"):
                info["intercept"] = float(np.atleast_1d(final.intercept_)[0])
        return info

    if name == "LinearRegression":
        coefs = np.atleast_1d(sk.coef_).ravel().tolist()
        feature_names = None
        if hasattr(sk, "feature_names_in_"):
            feature_names = sk.feature_names_in_.tolist()
        if not feature_names and parents and len(parents) == len(coefs):
            feature_names = list(parents)
        if feature_names and len(feature_names) == len(coefs):
            info["coefficients"] = dict(zip(feature_names, coefs))
        else:
            info["coefficients"] = coefs
        if hasattr(sk, "intercept_"):
            info["intercept"] = float(np.atleast_1d(sk.intercept_)[0])
        return info

    if name == "HistGradientBoostingRegressor":
        info["max_iter"] = getattr(sk, "max_iter", None)
        info["learning_rate"] = getattr(sk, "learning_rate", None)
        return info

    return info


def _extract_noise_summary(mechanism):
    nm = getattr(mechanism, "noise_model", None)
    if nm is None:
        return None
    data = getattr(nm, "data", None)
    if data is None:
        return type(nm).__name__
    arr = np.asarray(data).ravel()
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "median": float(np.median(arr)),
        "max": float(arr.max()),
    }


def dump_mechanisms(causal_model):
    rows = []
    graph = causal_model.graph
    for node in graph.nodes:
        parents = list(graph.predecessors(node))
        mech = causal_model.causal_mechanism(node)
        mech_type = type(mech).__name__
        entry = {"node": node, "parents": parents, "mechanism": mech_type}
        if not parents:
            # Root node: only a stochastic model
            continue_summary = type(mech).__name__
            entry["mechanism"] = continue_summary
        else:
            entry["fit"] = _extract_fit(mech, parents)
            noise_model = getattr(mech, "noise_model", None)
            entry["noise_model"] = type(noise_model).__name__ if noise_model else None
            entry["noise_summary"] = _extract_noise_summary(mech)
        rows.append(entry)
    return rows


def render_markdown(rows, graph_variant):
    lines = []
    lines.append(f"# SCM Structural Equations (graph variant: `{graph_variant}`)")
    lines.append("")
    lines.append(
        "Mechanisms assigned by DoWhy's `gcm.auto.assign_causal_mechanisms` "
        "(default `AssignmentQuality.GOOD`) and fitted on the full cleaned "
        "dataset (n=1190 rows, before train/test split). Each non-root node's "
        "mechanism is `X = f(parents) + noise`, where `f` is the listed sklearn "
        "estimator and the noise distribution is the empirical residual."
    )
    lines.append("")
    lines.append(
        "Graph: core 3 layers (`risk-factors → target → symptoms`) + "
        "risk-factor cross-links. Symptom-to-symptom cross-links "
        "(`thalach → exang`, `exang → cp`) are excluded so that symptoms "
        "depend on each other only through `target`."
    )
    lines.append("")

    roots = [r for r in rows if not r["parents"]]
    if roots:
        lines.append("## Root nodes (empirical distributions)")
        lines.append("")
        lines.append("| Node | Distribution |")
        lines.append("|------|--------------|")
        for r in roots:
            lines.append(f"| `{r['node']}` | {r['mechanism']} |")
        lines.append("")

    def section(title, nodes):
        if not nodes:
            return
        lines.append(f"## {title}")
        lines.append("")
        for r in nodes:
            parents_str = ", ".join(r["parents"])
            lines.append(f"### `{r['node']} = f({parents_str}) + ε`")
            fit = r.get("fit") or {}
            est = fit.get("estimator", "?")
            if est == "Pipeline":
                steps = "→".join(fit.get("pipeline_steps", []))
                lines.append(f"Estimator: `Pipeline({steps})` — wrapped in `{r['mechanism']}`.")
                if "coefficients" in fit and isinstance(fit["coefficients"], dict):
                    intercept = fit.get("intercept", 0.0)
                    coef_terms = " ".join(
                        f"{v:+.4f}·{name}" for name, v in fit["coefficients"].items()
                    )
                    lines.append("```")
                    lines.append(f"{r['node']} ≈ {intercept:+.4f} {coef_terms} + ε")
                    lines.append("ε: empirical residual")
                    lines.append("```")
            elif est == "LinearRegression" and isinstance(fit.get("coefficients"), dict):
                intercept = fit.get("intercept", 0.0)
                coef_terms = " ".join(
                    f"{v:+.4f}·{name}" for name, v in fit["coefficients"].items()
                )
                lines.append(f"Estimator: `LinearRegression`.")
                lines.append("```")
                lines.append(f"{r['node']} ≈ {intercept:+.4f} {coef_terms} + ε")
                lines.append("```")
            elif est == "HistGradientBoostingRegressor":
                mi = fit.get("max_iter")
                lr = fit.get("learning_rate")
                lines.append(
                    f"Estimator: `HistGradientBoostingRegressor` (max_iter={mi}, learning_rate={lr}). No closed form."
                )
            else:
                lines.append(f"Estimator: `{est}`.")
            ns = r.get("noise_summary")
            if isinstance(ns, dict):
                lines.append(
                    f"Noise: empirical residual (n={ns['n']}, mean={ns['mean']:.4f}, "
                    f"std={ns['std']:.4f}, min={ns['min']:.4f}, median={ns['median']:.4f}, "
                    f"max={ns['max']:.4f})."
                )
            lines.append("")

    risk_nodes = [r for r in rows if r["node"] in ("chol", "trestbps") and r["parents"]]
    disease_nodes = [r for r in rows if r["node"] == "target"]
    symptom_nodes = [
        r for r in rows
        if r["node"] in ("cp", "restecg", "thalach", "exang", "slope", "oldpeak")
    ]

    section("Risk-factor layer", risk_nodes)
    section("Disease layer", disease_nodes)
    section("Symptom layer", symptom_nodes)

    lines.append("## Notes")
    lines.append("")
    lines.append(
        "- Tree-based mechanisms (`HistGradientBoostingRegressor`) have no closed-form "
        "equation. Access them at runtime via "
        "`cm.causal_mechanism(node).prediction_model.sklearn_model` and "
        "`.noise_model.data`."
    )
    lines.append(
        "- `DiscreteAdditiveNoiseModel` rounds the sum to the nearest valid category; "
        "physiological clamps (oldpeak ≥ 0, cp ∈ [1,4], …) are applied post-sampling in "
        "`scm_analyzer.apply_scm_intervention`."
    )
    lines.append(
        "- DoWhy's auto-assignment is data-driven: refitting on a different sample (or "
        "after changing the graph) may swap `LinearRegression` ↔ "
        "`HistGradientBoostingRegressor`. The closed-form coefficients above are "
        "specific to this fit."
    )
    lines.append(
        "- Machine-readable dump in `structural_equations.json` carries per-node "
        "parents, estimator class, coefficients (when applicable), and a summary of "
        "the empirical noise distribution."
    )
    lines.append(
        "- To reproduce the legacy DAG with symptom cross-links, set "
        "`scm.graph_structure: full_with_symptom_links` in `pipeline_config.yaml`."
    )
    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="pipeline_config.yaml",
        help="Pipeline config to read scm.graph_structure / data path from",
    )
    parser.add_argument(
        "--output_dir", required=True,
        help="Directory to write structural_equations.{md,json} into",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    scm_cfg = dict(cfg.get("scm", {}))
    scm_cfg.setdefault("train_data_path", cfg["dice"]["data_path"])
    graph_variant = scm_cfg.get("graph_structure", "full")

    analyzer = SCMAnalyzer(config=scm_cfg)
    analyzer.initialize_analyzer()
    rows = dump_mechanisms(analyzer.causal_model)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "structural_equations.json"
    md_path = out_dir / "structural_equations.md"

    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    md_path.write_text(render_markdown(rows, graph_variant), encoding="utf-8")

    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
