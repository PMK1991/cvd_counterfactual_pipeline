# conda-env: mtech-env
"""
Unfiltered CF Scorer Module

The "no SCM" arm of the SCM-filtered vs. unfiltered ablation
(Reviewer 3, Comment 4b). Instead of validating DiCE-generated
counterfactuals through the structural causal model, this scores each
DiCE-proposed counterfactual directly with the deployed prediction
model. A counterfactual is "successful" when the model predicts class 0
(low risk) for it.

This is the ablation counterpart to SCMAnalyzer. It deliberately reuses
the same on-disk CF layout (iteration_NNN/original/patient_*.csv and
iteration_NNN/counterfactuals/patient_*_cf_*.csv) and emits the same
orig_*/cf_*/target/patient_id schema, so MetricsCalculator works
unchanged across both arms and the two retention rates are directly
comparable.

Author: PMK
Date: 2026-01-26
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Graph nodes kept identical to SCMAnalyzer so both arms emit
# byte-compatible columns for MetricsCalculator.
GRAPH_NODES = [
    'age', 'sex', 'chol', 'fbs', 'trestbps', 'target',
    'cp', 'restecg', 'thalach', 'exang', 'slope', 'oldpeak',
]


class UnfilteredScorer:
    """
    Unfiltered (model-only) counterfactual scorer.

    Keeps DiCE's raw counterfactuals that the prediction model itself
    accepts as a class flip, with no causal-validation layer. The cf_*
    values are DiCE's proposed feature values directly (DiCE varies all
    features), in contrast to the SCM arm where cf_* symptom values are
    the SCM's interventionally-propagated estimates.

    Thread-safe: holds no mutable state beyond the fitted model reference.
    """

    def __init__(self, model):
        if model is None:
            raise ValueError("UnfilteredScorer requires a fitted prediction model")
        self.model = model
        logger.info("Initialized UnfilteredScorer")

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _predict_class(self, features: pd.DataFrame) -> int:
        """Return the model's predicted class for a single-row feature frame."""
        pred = self.model.predict(features)
        return int(np.asarray(pred).ravel()[0])

    def score_counterfactual(
        self,
        original: pd.DataFrame,
        cf_suggestion: pd.DataFrame,
    ) -> Optional[pd.DataFrame]:
        """
        Score one DiCE counterfactual with the prediction model.

        Returns a single-row DataFrame with orig_*/cf_* columns for every
        graph feature plus a `target` column equal to the model's
        predicted class for the CF, or None if the CF cannot be scored.
        No SCM intervention is run.
        """
        try:
            orig_row = original.iloc[0:1].copy()
            cf_row = cf_suggestion.iloc[0:1].copy()

            # Model expects features only — drop the outcome column if DiCE
            # carried it into final_cfs_df.
            cf_features = cf_row.drop(columns=['target'], errors='ignore')
            cf_class = self._predict_class(cf_features)

            result: Dict[str, float] = {}
            orig_vals = orig_row.iloc[0]
            cf_vals = cf_row.iloc[0]

            for col in GRAPH_NODES:
                if col == 'target':
                    continue
                if col in orig_row.columns:
                    result[f'orig_{col}'] = orig_vals[col]
                if col in cf_row.columns:
                    result[f'cf_{col}'] = cf_vals[col]

            # `target` carries the model's verdict on the CF (0 = success).
            result['target'] = cf_class
            return pd.DataFrame([result])

        except Exception as e:
            logger.debug(f"Unfiltered scoring failed: {e}")
            return None

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate_counterfactual(
        self,
        scored: Optional[pd.DataFrame],
        original_target: int,
    ) -> bool:
        """True if the model accepts the CF as a class flip from 1 → 0."""
        if scored is None or len(scored) == 0:
            return False
        cf_target = scored['target'].values[0]
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
        Complete unfiltered analysis for one iteration directory.

        Mirrors SCMAnalyzer.analyze_iteration but applies model scoring in
        place of SCM validation. Reads the same original/counterfactual
        CSVs written by DiceCFGenerator and returns a DataFrame of
        model-accepted CFs.
        """
        cf_pairs = self.load_counterfactuals_for_iteration(iteration_dir)

        successful_cfs: List[pd.DataFrame] = []

        for pair in cf_pairs:
            scored = self.score_counterfactual(
                pair['original'], pair['cf_suggestion']
            )

            original_target = (
                pair['original']['target'].values[0]
                if 'target' in pair['original'].columns else 1
            )

            if self.validate_counterfactual(scored, original_target):
                scored['patient_id'] = pair['patient_id']
                successful_cfs.append(scored)

        if successful_cfs:
            result_df = pd.concat(successful_cfs, ignore_index=True)
            logger.info(f"Kept {len(result_df)} model-accepted CFs in iteration")
        else:
            result_df = pd.DataFrame()
            logger.warning("No model-accepted CFs found in iteration")

        if output_dir:
            self.save_successful_cfs(result_df, iteration_dir, output_dir)

        return result_df

    # ------------------------------------------------------------------
    # I/O helpers (kept self-contained — same layout as SCMAnalyzer)
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
            out = Path(iteration_dir) / "successful_unfiltered"
        else:
            out = Path(output_dir)

        out.mkdir(parents=True, exist_ok=True)
        output_file = out / "successful_counterfactuals_unfiltered.csv"
        successful_cfs.to_csv(output_file, index=False)
        logger.info(f"Saved {len(successful_cfs)} model-accepted CFs to {output_file}")
        return str(output_file)


if __name__ == "__main__":
    print("Unfiltered CF Scorer Module")
    print("This module should be imported and used by the pipeline orchestrator")
