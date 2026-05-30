# conda-env: mtech-env
"""
Train + serialize the SCM validator(s).

Mirrors the role of train_model.py for the structural causal model: fits one
DoWhy InvertibleStructuralCausalModel per graph variant and pickles it to
model/scm_<variant>.pkl, so the pipeline LOADS the fitted SCM at inference
instead of re-fitting it inside every worker on every run.

The build logic (graph edges, categorical casting, seeded fit) is taken from
SCMAnalyzer so the serialized artifact is byte-for-byte the model the pipeline
would otherwise have fit on the fly.

By default the SCM is fitted ONLY on the rows used to train the ML model (the
train_model.py train split), so the validator never sees the test-set patients
it later validates. Pass --fit-data full to reproduce the old whole-dataset fit.

Usage:
  python src/training/train_scm.py                       # 'full' graph, fitted on the ML train split (default)
  python src/training/train_scm.py --variants full minimal extended
  python src/training/train_scm.py --all                 # every variant (needed for the graph_structure sensitivity sweep)
  python src/training/train_scm.py --fit-data full       # legacy: fit on the entire cleaned dataset

Author: PMK
"""
import argparse
import hashlib
import pickle
import platform
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx
import sklearn
import dowhy
from dowhy import gcm

# Make the project root importable when run as a script
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.pipeline.scm_analyzer import SCMAnalyzer  # reuse graph variants
from src.training.train_model import load_train_test_data  # single source of truth for the split
from src.utils.dataLoader import DataLoader

DATA_PATH = ROOT / "data" / "heart_statlog_cleveland_hungary_final.csv"
MODEL_DIR = ROOT / "model"

# Same columns SCMAnalyzer casts to category before fitting
CAT_COLS = ["target", "exang", "fbs", "cp", "restecg", "slope"]


def _load_fit_data(fit_data: str) -> pd.DataFrame:
    """Return the dataframe the SCM is fitted on.

    'train' -> ONLY the rows used to train the ML model. Calls train_model.py's
               own load_train_test_data() so the SCM trains on the EXACT same
               split (no risk of the two re-deriving different partitions), and
               the validator never sees the test-set patients (leakage-free).
    'full'  -> entire cleaned dataset (legacy behaviour, reproduces old numbers).
    """
    if fit_data == "train":
        X_train, _X_test, y_train, _y_test = load_train_test_data()
        df = X_train.copy()
        df["target"] = y_train
        return df.reset_index(drop=True)

    loader = DataLoader(str(DATA_PATH))
    return loader.load_clean_data().reset_index(drop=True)


def fit_one(variant: str, df: pd.DataFrame, fit_seed: int = 42):
    """Build, auto-assign, and fit the SCM for a single graph variant.

    This is the single source of truth for SCM fitting (the pipeline only loads
    the resulting artifact). Seed-around-fit keeps the fit reproducible without
    perturbing the global RNG stream."""
    edges = SCMAnalyzer.GRAPH_VARIANTS[variant]
    graph = nx.DiGraph(edges)
    model = gcm.InvertibleStructuralCausalModel(graph)

    data = df[list(graph.nodes)].copy()
    for col in CAT_COLS:
        data[col] = data[col].astype("category")

    state = np.random.get_state() if fit_seed is not None else None
    if fit_seed is not None:
        np.random.seed(fit_seed)
    try:
        gcm.auto.assign_causal_mechanisms(model, data)
        gcm.fit(model, data)
    finally:
        if state is not None:
            np.random.set_state(state)
    return model


def main():
    ap = argparse.ArgumentParser(description="Fit and serialize SCM validator(s).")
    ap.add_argument(
        "--variants", nargs="+", default=["full"],
        choices=list(SCMAnalyzer.GRAPH_VARIANTS),
        help="Graph variants to fit (default: full).",
    )
    ap.add_argument("--all", action="store_true",
                    help="Fit every graph variant (covers the sensitivity sweep).")
    ap.add_argument("--fit-data", choices=["full", "train"], default="train",
                    help="train (default) = fit on the ML model's train split; full = entire dataset (legacy).")
    ap.add_argument("--fit-seed", type=int, default=42)
    ap.add_argument("--out-dir", default=str(MODEL_DIR))
    args = ap.parse_args()

    variants = list(SCMAnalyzer.GRAPH_VARIANTS) if args.all else args.variants
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = _load_fit_data(args.fit_data)
    data_hash = hashlib.sha256(
        pd.util.hash_pandas_object(df, index=True).values.tobytes()
    ).hexdigest()[:16]

    for variant in variants:
        print(f"Fitting SCM '{variant}' on {len(df)} rows ({args.fit_data} data) …")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = fit_one(variant, df, args.fit_seed)

        artifact = {
            "causal_model": model,
            "graph_structure": variant,
            "fit_seed": args.fit_seed,
            "fit_data": args.fit_data,
            "n_rows": int(len(df)),
            "data_sha256_16": data_hash,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "versions": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "sklearn": sklearn.__version__,
                "dowhy": dowhy.__version__,
            },
        }
        out = out_dir / f"scm_{variant}.pkl"
        with open(out, "wb") as f:
            pickle.dump(artifact, f)
        print(f"  saved {out}  ({out.stat().st_size / 1024:.0f} KB)")

    print("Done. The pipeline will load these instead of re-fitting per worker.")
    print("Note: pickle is version-sensitive — regenerate if dowhy/sklearn versions change.")


if __name__ == "__main__":
    main()
