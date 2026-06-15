# conda-env: mtech-env
"""
Recourse Analyzer Module

The "improvement-focused causal recourse" arm. Where :class:`SCMAnalyzer`
keeps only counterfactuals whose SCM intervention flips the disease label
(``target`` 1 -> 0) and discards the rest, this retains EVERY SCM-scored row
and tags it with a boolean ``flipped`` column. Downstream analysis then looks
at the NON-flip subset to measure whether interventions still improved the
patient's downstream symptoms even when the binary diagnosis did not reverse.

It deliberately reuses the same on-disk CF layout
(``iteration_NNN/original/patient_*.csv`` and
``iteration_NNN/counterfactuals/patient_*_cf_*.csv``) and the same
``orig_*/cf_*/target/patient_id`` schema as ``SCMAnalyzer``, so
``MetricsCalculator`` works unchanged. The SCM step is deterministic (per
patient-CF seed + offline-loaded artifact), so re-scoring a completed run's
DiCE proposals reproduces exactly the rows the original run computed and
discarded — no new DiCE generation required.

Author: PMK
Date: 2026-06-06
"""

import pandas as pd
from typing import List
import logging

from src.pipeline.scm_analyzer import SCMAnalyzer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RecourseAnalyzer(SCMAnalyzer):
    """SCM analyzer that retains all scored CFs, flipped and non-flipped.

    Inherits artifact loading, intervention, and I/O from :class:`SCMAnalyzer`;
    only the iteration-level retention policy differs.
    """

    def analyze_iteration_all(self, iteration_dir: str) -> pd.DataFrame:
        """Score every CF in an iteration and return all rows.

        Mirrors :meth:`SCMAnalyzer.analyze_iteration` but keeps non-flip rows
        too, adding a ``flipped`` boolean (True when the SCM moved ``target``
        from 1 to 0). Rows whose intervention failed (None) are skipped.
        """
        if self.causal_model is None:
            self.initialize_analyzer()

        cf_pairs = self.load_counterfactuals_for_iteration(iteration_dir)

        scored_rows: List[pd.DataFrame] = []
        for pair in cf_pairs:
            cf_result = self.apply_scm_intervention(
                pair['original'], pair['cf_suggestion']
            )
            if cf_result is None or len(cf_result) == 0:
                continue

            original_target = (
                pair['original']['target'].values[0]
                if 'target' in pair['original'].columns else 1
            )

            cf_result['patient_id'] = pair['patient_id']
            cf_result['flipped'] = self.validate_counterfactual(
                cf_result, original_target
            )
            scored_rows.append(cf_result)

        if scored_rows:
            result_df = pd.concat(scored_rows, ignore_index=True)
            n_flip = int(result_df['flipped'].sum())
            logger.info(
                "Scored %d CFs in iteration (%d flipped, %d non-flip)",
                len(result_df), n_flip, len(result_df) - n_flip,
            )
        else:
            result_df = pd.DataFrame()
            logger.warning("No scorable CFs found in iteration")

        return result_df


if __name__ == "__main__":
    print("Recourse Analyzer Module")
    print("This module should be imported and used by the pipeline orchestrator")
