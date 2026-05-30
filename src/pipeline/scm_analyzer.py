# conda-env: mtech-env
"""
SCM Analyzer Module

This module validates DiCE-generated counterfactuals using a pre-fitted
DoWhy structural causal model. The SCM is fitted OFFLINE by
``src/training/train_scm.py`` and serialized to ``model/scm_<variant>.pkl``;
this module only LOADS the matching artifact at inference and runs
interventional sampling. There is no in-process fitting fallback — if no
matching artifact is present, initialization raises.

Author: PMK
Date: 2026-01-26
"""

import warnings
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
import logging

from dowhy import gcm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Repo root (…/src/pipeline/scm_analyzer.py -> parents[2]) so a relative
# model_dir resolves regardless of the worker's current working directory.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]


class SCMAnalyzer:
    """
    SCM-based Counterfactual Analyzer

    Validates counterfactuals using structural causal models (DoWhy).
    Thread-safe for concurrent execution.
    """

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._default_config()
        self.causal_model = None
        self._warn_on_deprecated_config()

        logger.info("Initialized SCMAnalyzer")

    # Config keys that drove the removed in-process fitting fallback.
    _DEPRECATED_CONFIG_KEYS = (
        'train_data_path',
        'use_pretrained_scm',
        'require_pretrained_scm',
    )

    def _warn_on_deprecated_config(self) -> None:
        """Warn if a config still sets keys from the removed in-process-fit path."""
        present = [k for k in self._DEPRECATED_CONFIG_KEYS if k in self.config]
        if present:
            logger.warning(
                "Ignoring deprecated SCM config key(s) %s: the SCM is now fitted "
                "offline by train_scm.py and only loaded at inference.",
                ", ".join(present),
            )

    def _default_config(self) -> Dict:
        """Default SCM configuration"""
        return {
            'n_samples': 1000,
            'graph_structure': 'full',          # 'minimal', 'full', 'full_with_symptom_links', 'extended'
            'intervention_targets': 'both',     # 'both', 'chol_only', or 'trestbps_only'
            'fit_seed': 42,                     # Must match the artifact built by train_scm.py --fit-seed
            'model_dir': 'model',               # Where train_scm.py writes scm_<variant>.pkl
        }

    # ------------------------------------------------------------------
    # Graph structure variants
    # ------------------------------------------------------------------

    _CORE_EDGES = [
        # Risk Factors → Disease
        ('age', 'target'),
        ('sex', 'target'),
        ('chol', 'target'),
        ('fbs', 'target'),
        ('trestbps', 'target'),

        # Disease → Symptoms
        ('target', 'cp'),
        ('target', 'restecg'),
        ('target', 'thalach'),
        ('target', 'exang'),
        ('target', 'slope'),
        ('target', 'oldpeak'),
    ]

    # Cross-layer risk-factor edges from nb_cvd_scm.ipynb (upstream of target)
    _RISK_FACTOR_CROSSLINKS = [
        ('age', 'chol'),        # age affects lipid levels
        ('age', 'trestbps'),    # age affects blood pressure
        ('sex', 'trestbps'),    # sex-based BP differences
        ('sex', 'chol'),        # sex-based lipid differences
        ('chol', 'trestbps'),   # dyslipidemia raises BP
    ]

    # Symptom-to-symptom edges; bypass the disease node and violate
    # conditional independence of symptoms given target. Only included in
    # `full_with_symptom_links`.
    _SYMPTOM_CROSSLINKS = [
        ('thalach', 'exang'),
        ('exang', 'cp'),
    ]

    # Extended edges: additional physiologically plausible relationships
    _EXTENDED_EDGES = [
        ('age', 'thalach'),     # age affects max heart rate
        ('sex', 'thalach'),     # sex-based HR differences
        ('trestbps', 'oldpeak'),  # BP affects ST depression
    ]

    GRAPH_VARIANTS = {
        'minimal': _CORE_EDGES,
        'full': _CORE_EDGES + _RISK_FACTOR_CROSSLINKS,
        'full_with_symptom_links': _CORE_EDGES + _RISK_FACTOR_CROSSLINKS + _SYMPTOM_CROSSLINKS,
        'extended': _CORE_EDGES + _RISK_FACTOR_CROSSLINKS + _EXTENDED_EDGES,
    }

    # ------------------------------------------------------------------
    # Pre-fitted artifact loading (serialization refactor)
    # ------------------------------------------------------------------

    def _artifact_path(self) -> Path:
        """Resolve model/scm_<graph_structure>.pkl, relative to repo root."""
        variant = self.config.get('graph_structure', 'full')
        model_dir = Path(self.config.get('model_dir', 'model'))
        if not model_dir.is_absolute():
            model_dir = _PROJECT_ROOT / model_dir
        return model_dir / f"scm_{variant}.pkl"

    def _load_pretrained(self) -> gcm.InvertibleStructuralCausalModel:
        """Load the pre-fitted SCM artifact produced by ``src/training/train_scm.py``.

        The SCM is fitted OFFLINE (train_scm.py) and only loaded here at
        inference, so the validator is never re-fit per worker or per run. The
        artifact is accepted only if its ``graph_structure`` and ``fit_seed``
        match this config and its graph edges match the requested variant;
        provenance (``fit_data``, ``n_rows``, hash) is logged. Any problem —
        missing file, unpicklable artifact, or metadata/graph mismatch — raises
        with a clear message. There is no in-process fallback fit.
        """
        variant = self.config.get('graph_structure', 'full')
        hint = "Run: python src/training/train_scm.py --all"

        if variant not in self.GRAPH_VARIANTS:
            raise ValueError(
                f"Unknown graph_structure '{variant}'. "
                f"Known variants: {sorted(self.GRAPH_VARIANTS)}."
            )

        path = self._artifact_path()
        if not path.exists():
            raise FileNotFoundError(f"No pre-fitted SCM at {path}. {hint}")

        try:
            with open(path, 'rb') as f:
                artifact = pickle.load(f)
        except Exception as e:
            raise RuntimeError(
                f"Failed to unpickle SCM artifact {path} ({e}). Pickle is "
                f"version-sensitive — regenerate it. {hint}"
            ) from e

        if not isinstance(artifact, dict):
            raise ValueError(
                f"SCM artifact {path} is not a dict (got {type(artifact).__name__}). {hint}"
            )

        cfg_seed = self.config.get('fit_seed', 42)
        art_variant = artifact.get('graph_structure')
        art_seed = artifact.get('fit_seed')
        if art_variant != variant:
            raise ValueError(
                f"SCM artifact {path} graph_structure '{art_variant}' "
                f"!= config '{variant}'. {hint}"
            )
        if art_seed != cfg_seed:
            raise ValueError(
                f"SCM artifact {path} fit_seed {art_seed} != config {cfg_seed}. {hint}"
            )

        model = artifact.get('causal_model')
        if model is None or not hasattr(model, 'graph'):
            raise ValueError(
                f"SCM artifact {path} has no usable 'causal_model'. {hint}"
            )

        expected_edges = set(self.GRAPH_VARIANTS[variant])
        if set(model.graph.edges) != expected_edges:
            raise ValueError(
                f"SCM artifact {path} graph edges do not match variant "
                f"'{variant}'. The artifact is stale — {hint}"
            )

        self._warn_on_version_skew(artifact.get('versions', {}))
        logger.info(
            "Loaded pre-fitted SCM from %s (variant=%s, fit_data=%s, n_rows=%s, "
            "fit_seed=%s, data_sha256_16=%s).",
            path, art_variant, artifact.get('fit_data'), artifact.get('n_rows'),
            art_seed, artifact.get('data_sha256_16'),
        )
        return model

    @staticmethod
    def _warn_on_version_skew(art_versions: Dict) -> None:
        """Warn if the artifact's dowhy/sklearn major.minor differs from current."""
        try:
            import sklearn
            import dowhy
            current = {'dowhy': dowhy.__version__, 'sklearn': sklearn.__version__}
        except Exception:  # pragma: no cover
            return
        for pkg, cur in current.items():
            old = art_versions.get(pkg)
            if old and old.split('.')[:2] != cur.split('.')[:2]:
                logger.warning(
                    "SCM artifact was pickled under %s %s but %s is installed; "
                    "behaviour may differ.", pkg, old, cur,
                )

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def initialize_analyzer(self) -> None:
        """Load the pre-fitted SCM artifact (``train_scm.py`` output).

        Called once per worker. The SCM is fitted OFFLINE by train_scm.py; this
        only loads it. Raises if no matching artifact exists — there is no
        in-process fit fallback.
        """
        if self.causal_model is None:
            self.causal_model = self._load_pretrained()
        logger.info("SCMAnalyzer ready")

    # ------------------------------------------------------------------
    # Intervention
    # ------------------------------------------------------------------

    def apply_scm_intervention(
        self,
        original: pd.DataFrame,
        cf_suggestion: pd.DataFrame,
    ) -> Optional[pd.DataFrame]:
        """
        Apply SCM intervention to validate a counterfactual.

        Intervenes on chol (and optionally trestbps) using the
        CF-suggested values, then propagates through the causal graph.
        Uses a fixed random seed derived from patient features for
        deterministic, reproducible results per patient.

        Returns a single-row DataFrame with orig_* and cf_* columns
        compatible with MetricsCalculator, or None on failure.
        """
        if self.causal_model is None:
            raise ValueError("Must call initialize_analyzer() first")

        try:
            chol_val = cf_suggestion['chol'].values[0]
            bp_val = cf_suggestion['trestbps'].values[0]

            # Ensure original is a single-row DataFrame with graph columns
            graph_nodes = list(self.causal_model.graph.nodes)
            orig_row = original.copy()

            # Add target column if missing (high-risk patients)
            if 'target' not in orig_row.columns:
                orig_row['target'] = 1

            orig_row = orig_row[graph_nodes].iloc[0:1].copy()

            # Cast categorical columns to match training data dtypes
            for col in ['target', 'exang', 'fbs', 'cp', 'restecg', 'slope']:
                if col in orig_row.columns:
                    orig_row[col] = orig_row[col].astype('category')

            # Build intervention dict based on configured targets
            intervention_targets = self.config.get('intervention_targets', 'both')
            intervention_dict = {}
            if intervention_targets in ('both', 'chol_only') and pd.notna(chol_val):
                intervention_dict['chol'] = lambda _: float(chol_val)
            if intervention_targets in ('both', 'trestbps_only') and pd.notna(bp_val):
                intervention_dict['trestbps'] = lambda _: float(bp_val)

            if not intervention_dict:
                return None

            # Fix random seed based on patient features + intervention values
            # for deterministic, reproducible SCM results per patient-CF pair
            seed_val = int(abs(hash((
                float(original['chol'].values[0]),
                float(original['trestbps'].values[0]),
                float(original['thalach'].values[0]),
                float(chol_val), float(bp_val),
            )))) % (2**31)
            np.random.seed(seed_val)

            n_samples = self.config.get('n_samples', 1)

            # DoWhy: observed_data and num_samples_to_draw are mutually exclusive.
            # To draw multiple samples, duplicate the observed row.
            if n_samples > 1:
                obs_data = orig_row.loc[orig_row.index.repeat(n_samples)].reset_index(drop=True)
            else:
                obs_data = orig_row

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cf_samples = gcm.interventional_samples(
                    self.causal_model,
                    intervention_dict,
                    observed_data=obs_data,
                )

            # Apply physiological constraints
            cf_samples['oldpeak'] = cf_samples['oldpeak'].clip(lower=0)
            cf_samples['exang'] = cf_samples['exang'].clip(0, 1).round()
            cf_samples['cp'] = cf_samples['cp'].round().clip(1, 4)
            cf_samples['slope'] = cf_samples['slope'].round().clip(1, 3)
            cf_samples['restecg'] = cf_samples['restecg'].round().clip(0, 2)
            cf_samples['target'] = cf_samples['target'].round().clip(0, 1)

            # Aggregate: single sample → take directly; multiple → majority vote
            if n_samples == 1:
                cf_row = cf_samples.iloc[0:1].reset_index(drop=True)
            else:
                # For target: majority vote; for others: median (continuous) or mode (categorical)
                agg = {}
                for col in cf_samples.columns:
                    if col in ('target', 'exang', 'fbs', 'cp', 'restecg', 'slope'):
                        col_mode = cf_samples[col].mode()
                        agg[col] = col_mode.iloc[0] if len(col_mode) else cf_samples[col].iloc[0]
                    else:
                        agg[col] = cf_samples[col].median()
                cf_row = pd.DataFrame([agg])

            # Build result with orig_*/cf_* columns for MetricsCalculator
            result = {}
            orig_vals = orig_row.iloc[0]
            cf_vals = cf_row.iloc[0]

            for col in graph_nodes:
                result[f'orig_{col}'] = orig_vals[col]
                result[f'cf_{col}'] = cf_vals[col]

            result['target'] = cf_vals['target']

            return pd.DataFrame([result])

        except Exception as e:
            logger.debug(f"SCM intervention failed: {e}")
            return None

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate_counterfactual(
        self,
        cf_result: Optional[pd.DataFrame],
        original_target: int,
    ) -> bool:
        """True if the SCM counterfactual flipped target from 1 → 0."""
        if cf_result is None or len(cf_result) == 0:
            return False
        cf_target = cf_result['target'].values[0]
        return original_target == 1 and cf_target == 0

    # ------------------------------------------------------------------
    # Iteration-level analysis
    # ------------------------------------------------------------------

    def analyze_iteration(
        self,
        iteration_dir: str,
        output_dir: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Complete SCM analysis for one iteration directory.

        Reads original/counterfactual CSVs written by DiceCFGenerator,
        validates each via SCM, and returns a DataFrame of successful CFs.
        """
        if self.causal_model is None:
            self.initialize_analyzer()

        cf_pairs = self.load_counterfactuals_for_iteration(iteration_dir)

        successful_cfs: List[pd.DataFrame] = []

        for pair in cf_pairs:
            cf_result = self.apply_scm_intervention(
                pair['original'], pair['cf_suggestion']
            )

            original_target = pair['original']['target'].values[0] if 'target' in pair['original'].columns else 1

            if self.validate_counterfactual(cf_result, original_target):
                cf_result['patient_id'] = pair['patient_id']
                successful_cfs.append(cf_result)

        if successful_cfs:
            result_df = pd.concat(successful_cfs, ignore_index=True)
            logger.info(f"Found {len(result_df)} successful CFs in iteration")
        else:
            result_df = pd.DataFrame()
            logger.warning("No successful CFs found in iteration")

        if output_dir:
            self.save_successful_cfs(result_df, iteration_dir, output_dir)

        return result_df

    # ------------------------------------------------------------------
    # I/O helpers
    # ------------------------------------------------------------------

    def load_counterfactuals_for_iteration(
        self, iteration_dir: str
    ) -> List[Dict]:
        """Load all original/CF pairs for a specific iteration directory."""
        iter_path = Path(iteration_dir)
        orig_dir = iter_path / "original"
        cf_dir = iter_path / "counterfactuals"

        if not orig_dir.exists() or not cf_dir.exists():
            logger.warning(f"Missing directories in {iteration_dir}")
            return []

        cf_pairs = []
        for orig_file in orig_dir.glob("patient_*.csv"):
            patient_id = orig_file.stem.replace("patient_", "")
            orig_df = pd.read_csv(orig_file)

            for cf_file in cf_dir.glob(f"patient_{patient_id}_cf_*.csv"):
                cf_df = pd.read_csv(cf_file)
                cf_pairs.append({
                    'patient_id': patient_id,
                    'original': orig_df,
                    'cf_suggestion': cf_df,
                    'cf_file': str(cf_file),
                })

        logger.info(f"Loaded {len(cf_pairs)} CF pairs from {iteration_dir}")
        return cf_pairs

    def save_successful_cfs(
        self,
        successful_cfs: pd.DataFrame,
        iteration_dir: str,
        output_dir: Optional[str] = None,
    ) -> str:
        if output_dir is None:
            out = Path(iteration_dir) / "successful"
        else:
            out = Path(output_dir)

        out.mkdir(parents=True, exist_ok=True)
        output_file = out / "successful_counterfactuals.csv"
        successful_cfs.to_csv(output_file, index=False)
        logger.info(f"Saved {len(successful_cfs)} successful CFs to {output_file}")
        return str(output_file)


if __name__ == "__main__":
    print("SCM Analyzer Module")
    print("This module should be imported and used by the pipeline orchestrator")
