"""Synthetic paired-measurement checks; never generate experimental proposals."""
import contextlib
import copy
from concurrent.futures import Future
import inspect
import io
import json
from pathlib import Path
import pickle
import shutil
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import uuid

import numpy as np
import pandas as pd
from raiutils.exceptions import UserConfigValidationException

from scripts import measure_original_dice_compat as measurement
from scripts import validate_original_dice_measurement as validation


NO_CF = "No counterfactuals found for any of the query points! Kindly check your configuration."


def attempt(arm, record_id=0, iteration=0, returned=1, status=None, scm_error=None):
    candidates = [{
        "proposal_id": i, "raw": {"chol": 180}, "raw_prediction": 0,
        "scm_target": 0, "propagated_prediction": 0, "scm_error": scm_error,
        "flags": {e: scm_error is None or e == "raw_flip" for e in measurement.ENDPOINTS},
    } for i in range(returned)]
    counts = {e: sum(c["flags"][e] for c in candidates) for e in measurement.ENDPOINTS}
    status = status or ("ok" if returned else "no_cf")
    return {
        "arm": arm, "record_id": record_id, "source_row_id": 100 + record_id,
        "iteration": iteration, "requested": 5, "returned": returned, "status": status,
        "error": "RuntimeError: failed" if status == "generation_error" else None,
        "saved_factual": {"chol": 240, "target": 1}, "desired_class": 0,
        "candidates": candidates, "counts": counts,
        "any": {e: counts[e] > 0 for e in measurement.ENDPOINTS}, "seconds": .1,
    }


class ImmediatePool:
    def __init__(self, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def submit(self, function, task):
        future = Future()
        try:
            future.set_result(function(task))
        except Exception as exc:
            future.set_exception(exc)
        return future


class MeasurementTests(unittest.TestCase):
    def setUp(self):
        self.scratch = measurement.ROOT / "tests" / ("measurement_unit_" + uuid.uuid4().hex)
        self.scratch.mkdir()
        self.output = self.scratch / "output"
        self.output.mkdir()
        self.quiet = contextlib.redirect_stdout(io.StringIO())
        self.quiet.__enter__()
        self.old_state = measurement._STATE
        self.addCleanup(setattr, measurement, "_STATE", self.old_state)
        self.addCleanup(self.quiet.__exit__, None, None, None)
        self.addCleanup(shutil.rmtree, self.scratch)

    def read(self, name):
        return json.loads((self.output / name).read_text())

    def save(self, record):
        path = (self.output / "attempts" / record["arm"] /
                f"repeat_{record['iteration']:03d}_record_{record['record_id']:02d}.json")
        measurement.write_json(path, record)
        return path

    def plan(self, repeats=1, ids=(0,)):
        measurement.write_json(self.output / "run_plan.json",
                               {"repeats": repeats, "record_ids": list(ids)})
        measurement.write_json(self.output / "manifest.json", {"source_hashes": {}, "records": list(ids)})
        artifact = self.scratch / "artifact.bin"
        artifact.write_bytes(b"synthetic artifact")
        measurement.write_json(self.output / "artifact_provenance.json",
                               {"current_artifact_hashes": {str(artifact): measurement.sha(artifact)}})

    def pair(self, **kwargs):
        for arm in ("legacy", "corrected"):
            self.save(attempt(arm, **kwargs))

    def worker(self, result=None, error=None, factual_target=None, scm_error=None):
        cohort = pd.DataFrame({"chol": [240], "slope": [2], "target": [1]}, index=[100])
        example_factual = cohort.drop(columns="target")
        if factual_target is not None:
            example_factual["target"] = factual_target
        if result == "success":
            result = SimpleNamespace(cf_examples_list=[SimpleNamespace(
                test_instance_df=example_factual, desired_class=0,
                final_cfs_df=pd.DataFrame({"chol": [180], "slope": [2], "target": [0]}))])
        gens = {arm: Mock() for arm in ("legacy", "corrected")}
        for gen in gens.values():
            gen.generate_counterfactuals.return_value = result
            gen.generate_counterfactuals.side_effect = error
        model = Mock()
        model.predict.return_value = np.array([0])
        model.predict_proba.return_value = np.array([[.8, .2]])
        scm = Mock()
        scm.apply_scm_intervention.return_value = pd.DataFrame(
            {"cf_chol": [180], "cf_slope": [2], "target": [0]})
        scm.apply_scm_intervention.side_effect = scm_error
        scm.validate_counterfactual.return_value = True
        measurement._STATE = {"dice": {"total_cfs": 5}}, cohort, model, gens, scm
        return gens, scm

    def test_pinned_legacy_really_supports_strict_and_propagates_errors(self):
        cls, digest = measurement.legacy_class()
        self.assertEqual(len(digest), 64)
        self.assertIn("strict", inspect.signature(cls.generate_counterfactuals).parameters)
        gen = cls.__new__(cls)
        gen.dice_exp = Mock()
        gen.config = copy.deepcopy(measurement.DiceCFGenerator("unused", "unused").config)
        gen.dice_exp.generate_counterfactuals.side_effect = RuntimeError("unexpected")
        with self.assertRaisesRegex(RuntimeError, "unexpected"):
            gen.generate_counterfactuals(pd.DataFrame({"chol": [240]}), strict=True)

    def test_factual_target_added_without_leaking_label_to_query(self):
        gens, scm = self.worker("success")
        measurement.measure_pair((str(self.output), 0, 0))
        for arm, gen in gens.items():
            args, kwargs = gen.generate_counterfactuals.call_args
            self.assertNotIn("target", args[0])
            self.assertTrue(kwargs["strict"])
            saved = self.read(f"attempts/{arm}/repeat_000_record_00.json")
            self.assertEqual(saved["saved_factual"]["target"], 1)
            self.assertEqual(saved["status"], "ok")
        self.assertEqual(scm.apply_scm_intervention.call_args.args[0].target.iloc[0], 1)

    def test_predicted_factual_label_is_not_ground_truth(self):
        self.worker("success", factual_target=0)
        measurement.measure_pair((str(self.output), 0, 0))
        saved = self.read("attempts/legacy/repeat_000_record_00.json")
        self.assertEqual(saved["saved_factual"]["target"], 1)
        self.assertEqual(saved["counts"]["scm_accept"], 1)

    def test_exact_known_no_cf_is_terminal_for_both_strict_arms(self):
        self.worker(error=UserConfigValidationException(NO_CF))
        measurement.measure_pair((str(self.output), 0, 0))
        for r in measurement.read_attempts(self.output):
            self.assertEqual(r["status"], "no_cf")
            self.assertEqual(r["saved_factual"]["target"], 1)
            self.assertIsNone(r["error"])

    def test_other_exception_types_and_messages_remain_errors(self):
        self.plan()
        for index, error in enumerate((RuntimeError(NO_CF), UserConfigValidationException("Other failure"))):
            with self.subTest(error=error):
                self.worker(error=error)
                measurement.measure_pair((str(self.output), index, 0))
                r = self.read(f"attempts/corrected/repeat_{index:03d}_record_00.json")
                self.assertEqual(r["status"], "generation_error")
                self.assertEqual(measurement.attempt_state(r), "failed")

    def test_zero_proposals_have_null_candidate_but_zero_attempt_rates(self):
        self.plan()
        self.pair(returned=0)
        measurement.summarize(self.output)
        validation.check(self.output)
        summary, checked = self.read("summary.json"), self.read("endpoint_validation.json")
        self.assertTrue(summary["complete"])
        for arm in ("legacy", "corrected"):
            self.assertIsNone(summary["arms"][arm]["candidate_rates"]["joint"])
            self.assertEqual(summary["arms"][arm]["attempt_rates"]["joint"], 0)
            self.assertIsNone(checked["arms"][arm]["mean_repeat_SCM_acceptance_pct"])
            self.assertEqual(checked["arms"][arm]["undefined_repeat_SCM_rates"], 1)
            self.assertEqual(checked["arms"][arm]["valid_repeat_SCM_rates"], 0)
        for contrast in summary["contrasts"].values():
            self.assertIsNone(contrast["delta_pp"])
            self.assertIsNone(contrast["paired_record_bootstrap_95pct_pp"])
            self.assertEqual(contrast["excluded_bootstrap_draws"], 10000)
        self.assertEqual(checked["paired_attempt_contrasts"]["joint"]["delta_pp"], 0)

    def test_only_one_arm_has_zero_proposals(self):
        self.plan()
        self.save(attempt("legacy", returned=0))
        self.save(attempt("corrected"))
        measurement.summarize(self.output)
        validation.check(self.output)
        self.assertIsNone(self.read("summary.json")["contrasts"]["joint"]["delta_pp"])
        self.assertEqual(self.read("endpoint_validation.json")["paired_attempt_contrasts"]["joint"]["delta_pp"], 100)

    def test_bootstrap_joint_mask_excludes_zero_denominators_transparently(self):
        self.plan(ids=(0, 1))
        self.pair(record_id=0, returned=0)
        self.pair(record_id=1)
        measurement.summarize(self.output)
        contrast = self.read("summary.json")["contrasts"]["joint"]
        self.assertGreater(contrast["valid_bootstrap_draws"], 0)
        self.assertGreater(contrast["excluded_bootstrap_draws"], 0)
        self.assertEqual(contrast["valid_bootstrap_draws"] + contrast["excluded_bootstrap_draws"], 10000)
        self.assertEqual(contrast["paired_record_bootstrap_95pct_pp"], [0, 0])
        self.assertIn("BOTH", contrast["interval_interpretation"])

    def test_repeat_mean_is_explicitly_conditional_on_nonzero_repeats(self):
        self.plan(repeats=2)
        self.pair(iteration=0, returned=0)
        self.pair(iteration=1)
        validation.check(self.output)
        arm = self.read("endpoint_validation.json")["arms"]["legacy"]
        self.assertEqual(arm["mean_repeat_SCM_acceptance_pct"], 100)
        self.assertEqual(arm["valid_repeat_SCM_rates"], 1)
        self.assertEqual(arm["undefined_repeat_SCM_rates"], 1)

    def test_bootstrap_mask_requires_both_arms_not_independent_drops(self):
        self.plan(ids=(0, 1))
        for arm in ("legacy", "corrected"):
            for rid in (0, 1):
                self.save(attempt(arm, record_id=rid, returned=int((arm == "legacy") == (rid == 0))))
        measurement.summarize(self.output)
        boot = np.random.default_rng(20260912).integers(0, 2, (10000, 2))
        valid = (boot == 0).any(axis=1) & (boot == 1).any(axis=1)
        contrast = self.read("summary.json")["contrasts"]["joint"]
        self.assertEqual(contrast["valid_bootstrap_draws"], int(valid.sum()))
        self.assertEqual(contrast["paired_record_bootstrap_95pct_pp"], [0, 0])

    def test_incomplete_error_pending_and_scm_error_plans_suppress_contrasts(self):
        for case in ("empty", "missing_arm", "missing_repeat", "errors_only", "mixed_error", "pending", "scm_error"):
            with self.subTest(case=case):
                shutil.rmtree(self.output)
                self.output.mkdir()
                self.plan(repeats=2 if case == "missing_repeat" else 1)
                if case == "missing_arm":
                    self.save(attempt("legacy"))
                elif case == "missing_repeat":
                    self.pair()
                elif case == "errors_only":
                    self.pair(returned=0, status="generation_error")
                elif case == "mixed_error":
                    self.save(attempt("legacy"))
                    self.save(attempt("corrected", returned=0, status="generation_error"))
                elif case == "pending":
                    self.pair(returned=0, status="pending")
                elif case == "scm_error":
                    self.pair(scm_error="RuntimeError: SCM failed")
                measurement.summarize(self.output)
                with self.assertRaisesRegex(RuntimeError, "incomplete|failed"):
                    validation.check(self.output)
                summary, checked = self.read("summary.json"), self.read("endpoint_validation.json")
                self.assertFalse(summary["complete"])
                self.assertFalse(checked["complete"])
                self.assertEqual(summary["contrasts"], {})
                self.assertEqual(checked["paired_attempt_contrasts"], {})
                if case in ("errors_only", "scm_error"):
                    self.assertEqual(summary["failed_attempts"], 2)
                    self.assertEqual(summary["arms"]["legacy"]["completed_attempts"], 0)
                    self.assertIsNone(summary["arms"]["legacy"]["attempt_rates"]["joint"])
                    self.assertEqual(checked["arms"]["legacy"]["no_cf_count"], 0)
                if case == "pending":
                    self.assertEqual(summary["pending_attempts"], 2)

    def test_positive_denominator_bootstrap_matches_previous_algorithm_exactly(self):
        self.plan(ids=(0, 1, 2))
        for arm in ("legacy", "corrected"):
            for rid in range(3):
                r = attempt(arm, record_id=rid, returned=rid + 1)
                if arm == "legacy":
                    r["candidates"][0]["scm_target"] = 1
                    r["candidates"][0]["flags"]["scm_accept"] = False
                    r["candidates"][0]["flags"]["joint"] = False
                    for endpoint in ("scm_accept", "joint"):
                        r["counts"][endpoint] -= 1
                        r["any"][endpoint] = r["counts"][endpoint] > 0
                self.save(r)
        measurement.summarize(self.output)
        summary = self.read("summary.json")
        boot = np.random.default_rng(20260912).integers(0, 3, (10000, 3))
        den = np.array([1, 2, 3])
        old, new = np.array([0, 1, 2]), den
        differences = new[boot].sum(axis=1) / den[boot].sum(axis=1) - old[boot].sum(axis=1) / den[boot].sum(axis=1)
        contrast = summary["contrasts"]["joint"]
        self.assertEqual(contrast["delta_pp"], 100 * (new.sum() / den.sum() - old.sum() / den.sum()))
        self.assertEqual(contrast["paired_record_bootstrap_95pct_pp"],
                         (100 * np.quantile(differences, [.025, .975])).tolist())
        self.assertEqual(contrast["excluded_bootstrap_draws"], 0)

    def test_resume_preserves_error_checkpoints_and_fails_instead_of_skipping(self):
        self.plan()
        self.worker()
        self.pair(returned=0, status="generation_error")
        before = {p: p.read_bytes() for p in self.output.rglob("*.json")}
        with self.assertRaisesRegex(RuntimeError, "preserved"):
            measurement.run(self.output, 1, "all", 1)
        with self.assertRaisesRegex(RuntimeError, "preserved"):
            measurement.measure_pair((str(self.output), 0, 0))
        self.assertEqual(before, {p: p.read_bytes() for p in self.output.rglob("*.json")})

    def test_valid_no_cf_resume_does_not_generate_again(self):
        self.plan()
        gens, _ = self.worker()
        self.pair(returned=0)
        with patch.object(measurement, "ProcessPoolExecutor", ImmediatePool):
            measurement.run(self.output, 1, "all", 1)
        measurement.measure_pair((str(self.output), 0, 0))
        for gen in gens.values():
            gen.generate_counterfactuals.assert_not_called()

    def test_scm_worker_error_is_not_a_completed_attempt(self):
        self.plan()
        self.worker("success", scm_error=RuntimeError("SCM failure"))
        measurement.measure_pair((str(self.output), 0, 0))
        measurement.summarize(self.output)
        result = self.read("summary.json")
        self.assertEqual(result["failed_attempts"], 2)
        self.assertEqual(result["contrasts"], {})
        with self.assertRaisesRegex(RuntimeError, "failed"):
            validation.check(self.output)

    def test_unexpected_keys_and_duplicate_plan_ids_are_not_complete(self):
        self.plan()
        self.pair()
        self.pair(record_id=1)
        measurement.summarize(self.output)
        self.assertEqual(self.read("summary.json")["unexpected_attempts"], 2)
        with self.assertRaisesRegex(RuntimeError, "incomplete"):
            validation.check(self.output)
        with self.assertRaisesRegex(ValueError, "unique"):
            measurement.run(self.output, 1, "0,0", 1)

    def test_missing_provenance_cannot_be_backfilled_by_check_or_run(self):
        self.plan()
        self.pair()
        (self.output / "artifact_provenance.json").unlink()
        for function in (validation.check, measurement.check_sources):
            with self.assertRaisesRegex(RuntimeError, "Missing audit artifact provenance"):
                function(self.output)
        self.assertFalse((self.output / "artifact_provenance.json").exists())

    def test_changed_artifact_blocks_run_and_validation(self):
        self.plan()
        self.pair()
        (self.scratch / "artifact.bin").write_bytes(b"changed")
        with self.assertRaisesRegex(RuntimeError, "artifact"):
            measurement.check_sources(self.output)
        with self.assertRaisesRegex(RuntimeError, "artifact"):
            validation.check(self.output)
        self.assertEqual(self.read("endpoint_validation.json")["paired_attempt_contrasts"], {})

    def test_checkpoint_identity_counts_and_cohort_label_are_checked(self):
        self.plan()
        r = attempt("legacy")
        path = self.save(r)
        path.rename(path.with_name("repeat_999_record_00.json"))
        with self.assertRaisesRegex(ValueError, "identity"):
            measurement.read_attempts(self.output)
        path.with_name("repeat_999_record_00.json").unlink()
        r["counts"]["joint"] = 0
        self.save(r)
        with self.assertRaisesRegex(ValueError, "endpoint"):
            measurement.read_attempts(self.output)
        r = attempt("legacy")
        self.save(r)
        pd.DataFrame([{"record_id": 0, "source_row_id": 100, "target": 0}]).to_csv(
            self.output / "cohort.csv", index=False)
        with self.assertRaisesRegex(ValueError, "target differs"):
            measurement.read_attempts(self.output)

    def test_fresh_audit_run_check_produces_provenance_without_sidecar_setup(self):
        root = self.scratch / "repo"
        root.mkdir()
        archive = root / "fresh_cf_iterations_archive_published_20260607"
        metrics_dir = archive / "aggregated_results"
        original_dir = archive / "iteration_000" / "original"
        metrics_dir.mkdir(parents=True)
        original_dir.mkdir(parents=True)
        metrics = {"iteration": 0, "total_generated_cfs": 1, "total_successful_cfs": 1,
                   "total_patients": 1, "total_requested_cfs": 5, "target_flip_rate_pct": 100}
        pd.DataFrame([metrics]).to_csv(metrics_dir / "all_iteration_metrics.csv", index=False)
        measurement.write_json(archive / "iteration_000" / "metrics.json", metrics)
        cohort = pd.DataFrame({"chol": [240], "slope": [2], "target": [1]}, index=[100])
        cohort.drop(columns="target").to_csv(original_dir / "patient_0.csv", index=False)
        (root / "data.csv").write_text("synthetic")
        (root / "classifier.pkl").write_bytes(b"synthetic")
        (root / "scm_full.pkl").write_bytes(pickle.dumps({"causal_model": None, "graph_structure": "full"}))
        (root / "code.py").write_text("# synthetic source\n")
        cfg = {"dice": {"model_path": "classifier.pkl", "data_path": "data.csv", "total_cfs": 5},
               "scm": {"model_dir": ".", "graph_structure": "full"}}
        model = Mock()
        model.predict_proba.side_effect = lambda frame: np.tile([.2, .8], (len(frame), 1))
        gen = SimpleNamespace(
            dice_data=SimpleNamespace(prepare_query_instance=lambda frame: frame),
            dice_exp=SimpleNamespace(label_encode=lambda frame: frame,
                                     label_decode=lambda frame: pd.DataFrame(frame, columns=["chol", "slope"]),
                                     predict_fn_scores=model.predict_proba,
                                     predict_fn_for_sparsity=model.predict_proba),
            dice_model=SimpleNamespace(get_output=lambda frame, **kwargs: model.predict_proba(frame)),
        )
        self.worker("success")
        with (patch.object(measurement, "ROOT", root),
              patch.object(validation, "ROOT", root),
              patch.object(measurement, "SOURCE_PATHS", ("code.py",)),
              patch.object(measurement, "load_inputs", return_value=(cfg, cohort, cohort, cohort, cohort, model)),
              patch.object(measurement, "generators", return_value={"legacy": gen, "corrected": gen}),
              patch.object(measurement, "NativeNumericClassifier", return_value=model),
              patch.object(measurement, "legacy_class", return_value=(None, "synthetic")),
              patch.object(measurement, "protected_inventory", return_value={}),
              patch.object(measurement.subprocess, "check_output", return_value="synthetic"),
              patch.object(measurement, "ProcessPoolExecutor", ImmediatePool)):
            measurement.audit(self.output)
            self.assertIsNone(self.read("audit.json")["rows"][0]["archive_factual_target"])
            self.assertEqual(self.read("audit.json")["rows"][0]["cohort_factual_target"], 1)
            hashes = self.read("artifact_provenance.json")["current_artifact_hashes"]
            self.assertEqual(len(hashes), 4)
            measurement.run(self.output, 1, "all", 1)
            validation.check(self.output)
            self.assertTrue(self.read("endpoint_validation.json")["complete"])
            before = (self.output / "artifact_provenance.json").read_bytes()
            with self.assertRaisesRegex(RuntimeError, "not empty"):
                measurement.audit(self.output)
            self.assertEqual((self.output / "artifact_provenance.json").read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
