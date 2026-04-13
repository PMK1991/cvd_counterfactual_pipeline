# conda-env: mtech-env
"""
SCM Analyzer Module

This module handles SCM-based counterfactual validation using DoWhy.
Builds a causal graph for CVD, fits a probabilistic causal model,
and validates DiCE-generated counterfactuals via interventional sampling.

Author: PMK
Date: 2026-01-26
"""

import warnings
import pandas as pd
import numpy as np
import networkx as nx
from pathlib import Path
from typing import List, Dict, Optional
import logging

from dowhy import gcm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SCMAnalyzer:
    """
    SCM-based Counterfactual Analyzer

    Validates counterfactuals using structural causal models (DoWhy).
    Thread-safe for concurrent execution.
    """

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._default_config()
        self.causal_model = None

        logger.info("Initialized SCMAnalyzer")

    def _default_config(self) -> Dict:
        """Default SCM configuration"""
        return {
            'n_samples': 1,
            'train_data_path': 'data/heart_statlog_cleveland_hungary_final.csv',
            'graph_structure': 'full',          # 'minimal', 'full', or 'extended'
            'intervention_targets': 'both',     # 'both', 'chol_only', or 'trestbps_only'
        }

    # ------------------------------------------------------------------
    # Causal model construction
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Graph structure variants
    # ------------------------------------------------------------------

    # Core 3-layer edges (always present)
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

    # Cross-layer domain-knowledge edges from nb_cvd_scm.ipynb
    _CROSS_LAYER_EDGES = [
        ('age', 'chol'),        # age affects lipid levels
        ('age', 'trestbps'),    # age affects blood pressure
        ('sex', 'trestbps'),    # sex-based BP differences
        ('sex', 'chol'),        # sex-based lipid differences
        ('chol', 'trestbps'),   # dyslipidemia raises BP
        ('thalach', 'exang'),   # high HR triggers exercise angina
        ('exang', 'cp'),        # exercise angina manifests as chest pain
    ]

    # Extended edges: additional physiologically plausible relationships
    _EXTENDED_EDGES = [
        ('age', 'thalach'),     # age affects max heart rate
        ('sex', 'thalach'),     # sex-based HR differences
        ('trestbps', 'oldpeak'),  # BP affects ST depression
    ]

    GRAPH_VARIANTS = {
        'minimal': _CORE_EDGES,
        'full': _CORE_EDGES + _CROSS_LAYER_EDGES,
        'extended': _CORE_EDGES + _CROSS_LAYER_EDGES + _EXTENDED_EDGES,
    }

    def _build_causal_model(self) -> gcm.InvertibleStructuralCausalModel:
        """
        Build, auto-assign mechanisms, and fit a DoWhy invertible
        structural causal model for the CVD dataset.

        Graph structure is selected via config['graph_structure']:
          'minimal'  — Core 3-layer only (Risk Factors → target → Symptoms)
          'full'     — Core + cross-layer edges from nb_cvd_scm.ipynb (default)
          'extended' — Full + additional physiological edges

        Core 3-layer structure from nb_cvd_scm.ipynb:
          Layer 1 — Risk Factors (root nodes): age, sex, chol, fbs, trestbps
          Layer 2 — Disease: target
          Layer 3 — Symptoms: cp, restecg, thalach, exang, slope, oldpeak
        """
        graph_name = self.config.get('graph_structure', 'full')
        edges = self.GRAPH_VARIANTS.get(graph_name)
        if edges is None:
            logger.warning(f"Unknown graph_structure '{graph_name}', falling back to 'full'")
            edges = self.GRAPH_VARIANTS['full']

        logger.info(f"Building causal graph for CVD (variant: {graph_name}, {len(edges)} edges) …")

        causal_graph = nx.DiGraph(edges)

        causal_model = gcm.InvertibleStructuralCausalModel(causal_graph)

        # Load training data
        train_path = self.config.get(
            'train_data_path', 'data/heart_statlog_cleveland_hungary_final.csv'
        )
        logger.info(f"Loading training data from {train_path} …")
        train_data = pd.read_csv(train_path)

        # Keep only the columns present in the graph
        graph_nodes = list(causal_graph.nodes)
        train_data = train_data[graph_nodes]

        # Cast categorical columns to category dtype (matching nb_cvd_scm.ipynb)
        for col in ['target', 'exang', 'fbs', 'cp', 'restecg', 'slope']:
            train_data[col] = train_data[col].astype('category')

        # Auto-assign and fit
        logger.info("Auto-assigning causal mechanisms …")
        gcm.auto.assign_causal_mechanisms(causal_model, train_data)

        logger.info("Fitting causal model …")
        gcm.fit(causal_model, train_data)

        logger.info("Causal model built and fitted successfully")
        return causal_model

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def initialize_analyzer(self) -> None:
        """Build and fit the causal model (call once per worker)."""
        try:
            if self.causal_model is None:
                self.causal_model = self._build_causal_model()
            logger.info("SCMAnalyzer ready")
        except Exception as e:
            logger.error(f"Error initializing analyzer: {e}")
            raise

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
                obs_data = pd.concat([orig_row] * n_samples, ignore_index=True)
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
                        agg[col] = cf_samples[col].mode().iloc[0]
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
