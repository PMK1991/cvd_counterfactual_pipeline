import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from scripts.summarize_simulation_diagnostics import exact_target_probability
from src.pipeline.simulation_diagnostics import (
    clip_samples, draw_noise_plan, evaluate_actions, load_pickle,
)


ROOT = Path(__file__).resolve().parents[1]


class ExactIntegrationTests(unittest.TestCase):
    def test_exact_factual_control_is_identical_on_repetition(self):
        model = load_pickle(ROOT / "model" / "scm_full.pkl")["causal_model"]
        row = pd.read_csv(ROOT / "data" / "heart_statlog_cleveland_hungary_final.csv").iloc[1]
        self.assertEqual(
            exact_target_probability(model, row, row.chol),
            exact_target_probability(model, row, row.chol),
        )

    def test_exact_probability_matches_large_sample_without_clinical_interpretation(self):
        row = pd.read_csv(ROOT / "data" / "heart_statlog_cleveland_hungary_final.csv").iloc[1]
        with threadpool_limits(limits=1):
            for variant in ("full", "minimal", "extended"):
                model = load_pickle(ROOT / "model" / f"scm_{variant}.pkl")["causal_model"]
                noise = draw_noise_plan(model, 30000, 20261027)
                samples = clip_samples(evaluate_actions(model, row, [100, 200, 300], noise))
                for action, value in enumerate((100, 200, 300)):
                    probability = exact_target_probability(model, row, value)
                    simulated = float((samples.iloc[action*30000:(action+1)*30000].target == 0).mean())
                    self.assertLessEqual(abs(probability - simulated), .015)
                    self.assertGreaterEqual(probability, 0)
                    self.assertLessEqual(probability, 1)


if __name__ == "__main__":
    unittest.main()
