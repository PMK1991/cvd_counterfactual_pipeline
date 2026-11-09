import unittest
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from dowhy import gcm
from threadpoolctl import threadpool_limits

from src.pipeline.simulation_diagnostics import (
    SYMPTOMS, aggregate, clip_samples, draw_noise_plan, evaluate_actions,
    legacy_seed, load_pickle, matched_seed, outcome, record_bootstrap,
    summarize, validate_partition,
)


ROOT = Path(__file__).resolve().parents[1]


class SimulationDiagnosticsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.limits = threadpool_limits(limits=1)
        cls.model = load_pickle(ROOT / "model" / "scm_full.pkl")["causal_model"]
        cls.original = pd.read_csv(
            ROOT / "data" / "heart_statlog_cleveland_hungary_final.csv"
        ).iloc[1]

    @classmethod
    def tearDownClass(cls):
        cls.limits.restore_original_limits()

    def test_explicit_noise_matches_dowhy_exactly(self):
        n, seed = 120, 438
        noise = draw_noise_plan(self.model, n, seed)
        explicit = evaluate_actions(self.model, self.original, [200], noise)
        observed = pd.DataFrame([self.original] * n)[list(self.model.graph)].reset_index(drop=True)
        state = np.random.get_state()
        np.random.seed(seed)
        try:
            reference = gcm.interventional_samples(
                self.model, {"chol": lambda _: 200.0}, observed_data=observed
            )
        finally:
            np.random.set_state(state)
        np.testing.assert_array_equal(explicit, reference[explicit.columns].to_numpy(float))

    def test_control_and_repeated_actions_have_exact_zero_difference(self):
        noise = draw_noise_plan(self.model, 80, 52)
        before = {k: v.copy() for k, v in noise.items()}
        samples = evaluate_actions(self.model, self.original, [self.original.chol] * 2, noise)
        np.testing.assert_array_equal(samples.iloc[:80], samples.iloc[80:].to_numpy())
        for key in noise:
            np.testing.assert_array_equal(before[key], noise[key])
        a = outcome(clip_samples(samples.iloc[:80]))
        b = outcome(clip_samples(samples.iloc[80:]))
        self.assertEqual(a["p_target0"] - b["p_target0"], 0)

    def test_cached_additive_predictions_are_bitwise_equal_to_direct_evaluation(self):
        for variant in ("full", "minimal", "extended"):
            model = load_pickle(ROOT / "model" / f"scm_{variant}.pkl")["causal_model"]
            noise = draw_noise_plan(model, 321, 42)
            cached = evaluate_actions(model, self.original, [100, 200, 300], noise)
            direct = evaluate_actions(
                model, self.original, [100, 200, 300], noise, cache_repeated_parents=False
            )
            np.testing.assert_array_equal(cached, direct)

    def test_noise_arrays_are_shared_not_only_seeds(self):
        noise = draw_noise_plan(self.model, 75, 43)
        samples = evaluate_actions(self.model, self.original, [100, 300], noise)
        for node in nx.descendants(self.model.graph, "chol"):
            parents = sorted(self.model.graph.predecessors(node))
            for offset in (0, 75):
                block = samples.iloc[offset:offset+75]
                expected = self.model.causal_mechanism(node).evaluate(
                    block[parents].to_numpy(), noise[node]
                ).reshape(-1)
                np.testing.assert_array_equal(block[node], expected)

    def test_probability_is_not_threshold_and_tie_favors_zero(self):
        sample = pd.DataFrame([self.original] * 10)
        sample["target"] = [0] * 5 + [1] * 5
        result = outcome(sample)
        self.assertEqual(result["p_target0"], .5)
        self.assertEqual(result["accept"], 1)
        self.assertEqual(result["mode_accept"], 1)
        sample["target"] = [0] * 4 + [1] * 6
        self.assertEqual(outcome(sample)["accept"], 0)

    def test_bp_changes_only_legacy_noise(self):
        seeds = [legacy_seed(self.original, 200, bp) for bp in (100, 120, 140, 160, 180)]
        self.assertGreater(len(set(seeds)), 1)
        plans = [draw_noise_plan(self.model, 500, seed) for seed in seeds]
        self.assertTrue(any(not np.array_equal(plans[0]["target"], p["target"]) for p in plans[1:]))
        original_seed = matched_seed(47, 5, 0)
        for bp in (100, 120, 140, 160, 180):
            self.assertEqual(original_seed, matched_seed(47, 5, 0))
        # Proposal data cannot enter the matched-noise simulation API.
        noise = draw_noise_plan(self.model, 100, original_seed)
        a = evaluate_actions(self.model, self.original, [200], noise)
        b = evaluate_actions(self.model, self.original, [200], noise)
        np.testing.assert_array_equal(a, b)

    def test_group_only_symptom_parents_and_no_factual_symptom_retention(self):
        for symptom in SYMPTOMS:
            self.assertEqual(list(self.model.graph.predecessors(symptom)), ["target"])
        noise = draw_noise_plan(self.model, 100, 18)
        altered = self.original.copy()
        for symptom in SYMPTOMS:
            altered[symptom] += 13
        a = evaluate_actions(self.model, self.original, [200], noise)
        b = evaluate_actions(self.model, altered, [200], noise)
        np.testing.assert_array_equal(a[list(SYMPTOMS)], b[list(SYMPTOMS)])
        altered["age"] += 20
        a = evaluate_actions(self.model, self.original, [200], noise, forced_target=0)
        b = evaluate_actions(self.model, altered, [200], noise, forced_target=0)
        np.testing.assert_array_equal(a[list(SYMPTOMS)], b[list(SYMPTOMS)])

    def test_disjoint_fold_provenance_rejects_leakage(self):
        validate_partition([1, 2], [3, 4], [3])
        with self.assertRaises(ValueError):
            validate_partition([1, 2], [2, 4], [4])
        with self.assertRaises(ValueError):
            validate_partition([1], [3], [2])
        with self.assertRaises(ValueError):
            validate_partition([1], [3], [3, 3])

    def test_bootstrap_denominator_is_record_not_replicate(self):
        rows = []
        for source, value in ((10, 0.1), (20, 0.9)):
            for rep in range(10):
                for action, chol in (("control", 200), ("absolute_100", 100), ("absolute_300", 300)):
                    rows.append(dict(
                        cohort="test", variant="full", source_row_id=source, replicate=rep,
                        action=action, p_target0=value, delta_p_target0=value if action != "control" else 0,
                        accept=int(value >= .5), control_accept=0, gained_acceptance=0,
                        lost_acceptance=0, xgb_draw_fraction0=0, xgb_aggregate_accept=0,
                        xgb_draw_scm_agreement=0, aggregation_disagreement=0,
                        direction="unchanged", out_of_support=0, cholesterol=chol,
                    ))
        record, summary, _ = summarize(pd.DataFrame(rows), 100)
        self.assertEqual(len(record), 6)
        self.assertTrue((summary.n_records == 2).all())
        self.assertEqual(record_bootstrap([.1, .9], 100)["n_records"], 2)
        self.assertAlmostEqual(summary[summary.action == "absolute_100"].delta_p_target0_mean.iloc[0], .5)


if __name__ == "__main__":
    unittest.main()
