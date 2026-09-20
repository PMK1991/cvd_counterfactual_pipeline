"""Run with python -m unittest discover -s tests -p test_kfold_recourse.py -v."""
import copy
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

from scripts import run_kfold_recourse as cv


def fixture_data():
    rng = np.random.default_rng(42)
    n = 100
    return pd.DataFrame({
        "age": rng.integers(35, 75, n), "sex": rng.integers(0, 2, n),
        "cp": rng.integers(1, 5, n), "trestbps": rng.integers(100, 160, n),
        "chol": rng.integers(180, 280, n), "fbs": rng.integers(0, 2, n),
        "restecg": rng.integers(0, 3, n), "thalach": rng.integers(100, 170, n),
        "exang": rng.integers(0, 2, n), "oldpeak": rng.random(n) * 3,
        "slope": rng.integers(1, 4, n), "target": [0, 1] * (n // 2),
    }, index=np.arange(1000, 1000 + n))


def candidate(raw, scm, prediction, index=0):
    return {"proposal_id": index, "raw_prediction": raw, "scm_target": scm,
            "propagated_prediction": prediction,
            "flags": cv.candidate_flags(raw, scm, prediction)}


def record(candidates, source=1000, repeat=0):
    return {"manifest_id": cv.digest({"expected_repeats": 2}), "fold": 0,
            "source_row_id": source, "repeat": repeat,
            "returned_proposals": len(candidates), "candidates": candidates,
            "flags": cv.attempt_flags(candidates), "status": "ok" if candidates else "no_cf"}


class ScratchTest(unittest.TestCase):
    def setUp(self):
        self.scratch = cv.ROOT / "kfold_runs" / ("unit_" + uuid.uuid4().hex)
        self.scratch.mkdir(parents=True)

    def tearDown(self):
        cv._CACHE.clear()
        shutil.rmtree(self.scratch)


class FoldIsolationTests(ScratchTest):
    def test_prepare_same_training_rows_for_classifier_and_scm_and_reuse(self):
        import networkx as nx
        from src.training.train_model import BASELINE_XGB_PARAMS
        data = fixture_data()
        manifest = {
            "classifier_params": dict(BASELINE_XGB_PARAMS, n_estimators=2, random_state=42, n_jobs=1),
            "config": {"scm": {"graph_structure": "full", "fit_seed": 42}},
            "folds": cv.fold_plan(data),
        }
        seen = []
        def fit(variant, training, seed):
            seen.append(training.copy())
            return SimpleNamespace(graph=nx.DiGraph())
        with patch("src.training.train_scm.fit_one", side_effect=fit):
            cv.prepare(self.scratch, manifest, data)
            cv.prepare(self.scratch, manifest, data)
        self.assertEqual(len(seen), 5)
        for plan, scm_training in zip(manifest["folds"], seen):
            directory = self.scratch / f"fold_{plan['fold']}"
            with open(directory / "training.pkl", "rb") as stream:
                saved_training = pickle.load(stream)
            pd.testing.assert_frame_equal(saved_training, scm_training)
            self.assertEqual(saved_training.index.tolist(), plan["training_ids"])
            cv.check_fold(self.scratch, manifest, plan)
        path = self.scratch / "fold_0" / "classifier.pkl"
        path.write_bytes(path.read_bytes() + b"corruption")
        with self.assertRaises(ValueError):
            cv.check_fold(self.scratch, manifest, manifest["folds"][0])

    def test_complete_disjoint_oof(self):
        data = fixture_data()
        plans = cv.fold_plan(data)
        cv.validate_plan(data, plans)
        self.assertEqual(sorted(i for p in plans for i in p["validation_ids"]),
                         sorted(data.index))
        for plan in plans:
            self.assertFalse(set(plan["training_ids"]) & set(plan["validation_ids"]))
        broken = copy.deepcopy(plans)
        broken[0]["validation_ids"].append(plans[1]["validation_ids"][0])
        with self.assertRaises(ValueError):
            cv.validate_plan(data, broken)

    def test_training_only_target_specific_sequential_iqr(self):
        data = fixture_data()
        data.loc[data.target == 1, "chol"] += 500
        training = data.iloc[:80].copy()
        validation = data.iloc[80:].copy()
        training.loc[training.index[0], "chol"] = 9999
        filtered, fences = cv.filter_training(training)
        self.assertNotIn(training.index[0], filtered.index)
        self.assertGreater(fences[2]["lower"], fences[0]["upper"])
        self.assertEqual(fences[1]["before"], fences[0]["after"])
        validation.loc[:, "chol"] = 99999
        validation.loc[:, "target"] = 1 - validation.target
        again, other_fences = cv.filter_training(training)
        pd.testing.assert_frame_equal(filtered, again)
        self.assertEqual(fences, other_fences)
        self.assertEqual(len(validation), 20)

    def test_heldout_changes_do_not_affect_fitted_scaler_or_dice_reference(self):
        from src.pipeline.dice_cf_generator import DiceCFGenerator
        from src.training.train_model import build_xgb_pipeline
        data = fixture_data()
        plan = cv.fold_plan(data)[0]
        training, _ = cv.filter_training(data.loc[plan["outer_train_ids"]])
        changed = data.copy()
        changed.loc[plan["validation_ids"], "chol"] = 100000
        changed.loc[plan["validation_ids"], "target"] = 1 - changed.loc[plan["validation_ids"], "target"]
        training_after, _ = cv.filter_training(changed.loc[plan["outer_train_ids"]])
        pd.testing.assert_frame_equal(training, training_after)
        classifier = build_xgb_pipeline(training.drop(columns="target"),
                                        {"n_estimators": 2, "random_state": 42, "n_jobs": 1})
        classifier.fit(training.drop(columns="target"), training.target)
        means = classifier["preprocessor"].named_transformers_["num"]["scaler"].mean_
        np.testing.assert_allclose(means, training[["age", "trestbps", "chol", "thalach", "oldpeak"]].mean())
        model_path = self.scratch / "model.pkl"
        model_path.write_bytes(pickle.dumps(classifier))
        gen = DiceCFGenerator(str(model_path), "MUST_NOT_READ")
        with patch("src.utils.dataLoader.DataLoader.load_data", side_effect=AssertionError("leakage")):
            gen.load_model_and_data(training_data=training)
        self.assertEqual(set(gen.dice_data.data_df.index), set(training.index))
        self.assertNotIn("source_row_id", gen.dice_data.feature_names)
        self.assertEqual(gen.dice_data.data_df.chol.max(), training.chol.max())

    def test_fixed_eligibility_deduplicates_before_folds(self):
        data = fixture_data().reset_index(drop=True)
        data.loc[0, "chol"] = 0
        data.loc[1, "trestbps"] = 0
        data = pd.concat([data, data.iloc[[2]]], ignore_index=True)
        path = self.scratch / "data.csv"
        data.to_csv(path, index=False)
        eligible = cv.eligible_data(path)
        self.assertEqual(len(eligible), 98)
        self.assertNotIn(100, eligible.index)
        self.assertIn(2, eligible.index)


class EndpointTests(unittest.TestCase):
    def test_disagreement_and_both_must_be_same_candidate(self):
        rows = [candidate(0, 0, 1), candidate(0, 1, 0, 1)]
        flags = cv.attempt_flags(rows)
        self.assertTrue(flags["scm_accept"])
        self.assertTrue(flags["propagated_flip"])
        self.assertFalse(flags["both"])
        self.assertTrue(cv.attempt_flags(rows + [candidate(0, 0, 0, 2)])["both"])

    def test_no_cf_in_primary_denominator(self):
        records = {
            (0, 1000, 0): record([candidate(0, 0, 0)]),
            (0, 1000, 1): record([], repeat=1),
            (0, 1001, 0): record([], source=1001),
            (0, 1001, 1): record([], source=1001, repeat=1),
        }
        summary = cv.summarize_records(records, set(records), 2, bootstrap=20)
        endpoint = summary["endpoints"]["both"]
        self.assertEqual(endpoint["primary_pooled_record_mean"], 0.25)
        self.assertEqual(endpoint["candidate_conditional_rate"], 1)
        self.assertEqual(summary["no_cf_attempts"], 3)
        self.assertEqual(summary["completed"], 4)
        self.assertEqual(summary["record_means"][0]["both"], 0.5)

    def test_partial_run_never_presented_as_primary(self):
        records = {(0, 1000, 0): record([])}
        summary = cv.summarize_records(records, {(0, 1000, 0), (0, 1000, 1)}, 2)
        self.assertIsNone(summary["endpoints"]["both"]["primary_pooled_record_mean"])
        self.assertIsNone(summary["endpoints"]["both"]["record_cluster_bootstrap_95ci_conditional_on_fitted_folds"])
        self.assertFalse(summary["complete"])


class FoldDiceCompatibilityTests(ScratchTest):
    def test_every_real_oof_tp_matches_actual_genetic_prediction_path(self):
        from src.pipeline.kfold_dice import FoldDiceGenerator, preflight_generator
        from src.training.train_model import build_xgb_pipeline
        data = cv.eligible_data(cv.ROOT / "data" / "heart_statlog_cleveland_hungary_final.csv")
        config = cv.yaml.safe_load((cv.ROOT / "pipeline_config.yaml").read_text())["dice"]
        checked = set()
        class StopBeforeSearch(Exception):
            pass
        for plan in cv.fold_plan(data):
            training = data.loc[plan["training_ids"]]
            validation = data.loc[plan["validation_ids"]]
            model = build_xgb_pipeline(training.drop(columns="target"),
                                       {"random_state": 42, "n_jobs": 1})
            model.fit(training.drop(columns="target"), training.target)
            generator = FoldDiceGenerator("", "", config).initialize_for_training(model, training)
            report = preflight_generator(generator, validation)
            features = generator.dice_data.feature_names
            predictions = model.predict(validation[features])
            expected = set(validation[(validation.target == 1) & (predictions == 1)].index)
            self.assertEqual({r["source_row_id"] for r in report["checked_tps"]}, expected)
            self.assertEqual(report["max_probability_difference"], 0)
            for source in expected:
                query = validation.loc[[source], features]
                # Execute DiCE 0.11's actual generation entry path, including
                # prepare -> label_encode -> predict_fn_scores -> desired class.
                with patch.object(generator.dice_exp, "do_param_initializations",
                                  side_effect=StopBeforeSearch) as stop:
                    with self.assertRaises(StopBeforeSearch):
                        generator.dice_exp._generate_counterfactuals(
                            query, total_CFs=5, desired_class=0,
                            permitted_range=config["permitted_range"])
                np.testing.assert_array_equal(generator.dice_exp.test_pred, model.predict_proba(query))
                self.assertEqual(stop.call_args.args[3], 0)
                self.assertNotIn(source, checked)
                checked.add(source)
        self.assertGreater(len(checked), 0)

    def test_predeclared_unknown_factual_category_without_reference_leakage(self):
        from src.pipeline.kfold_dice import FoldDiceGenerator, CATEGORICAL_SCHEMA
        from src.training.train_model import build_xgb_pipeline
        data = cv.eligible_data(cv.ROOT / "data" / "heart_statlog_cleveland_hungary_final.csv")
        plan = cv.fold_plan(data)[0]
        training = data.loc[plan["training_ids"]]
        original = data.loc[[517]].drop(columns="target")
        self.assertNotIn(0, training.slope.unique())
        self.assertEqual(original.slope.iloc[0], 0)
        model = build_xgb_pipeline(training.drop(columns="target"),
                                   {"random_state": 42, "n_jobs": 1})
        model.fit(training.drop(columns="target"), training.target)
        self.assertEqual(model.predict(original)[0], 1)
        generator = FoldDiceGenerator("", "").initialize_for_training(model, training)
        reference = generator.dice_data.data_df
        self.assertEqual(reference.index.tolist(), training.index.tolist())
        for column in training:
            np.testing.assert_array_equal(pd.to_numeric(reference[column]), training[column])
        self.assertEqual(CATEGORICAL_SCHEMA["slope"], (0, 1, 2, 3))
        self.assertNotIn("0", generator.dice_data.get_features_range()[0]["slope"])
        generator.dice_exp.setup("all", {"chol": [150, 200]}, original, "inverse_mad")
        allowed = generator.dice_exp.get_valid_feature_range()["slope"]
        decoded_allowed = generator.dice_exp.labelencoder["slope"].inverse_transform(allowed)
        self.assertNotIn("0", decoded_allowed)
        encoded = generator.dice_exp.label_encode(
            generator.dice_data.prepare_query_instance(original)).to_numpy()
        self.assertEqual(generator.dice_exp.label_decode(encoded).slope.iloc[0], "0")
        changed = original.copy()
        changed["slope"] = 99
        with self.assertRaises(ValueError):
            generator.dice_data.prepare_query_instance(changed)


class OrdinaryDiceCompatibilityTests(ScratchTest):
    def setUp(self):
        super().setUp()
        from src.training.train_model import build_xgb_pipeline
        self.training = fixture_data()
        self.features = self.training.columns.drop("target").tolist()
        self.model = build_xgb_pipeline(
            self.training[self.features], {"n_estimators": 5, "n_jobs": 1})
        self.model.fit(self.training[self.features], self.training.target)
        self.model_path = self.scratch / "classifier.pkl"
        self.model_path.write_bytes(pickle.dumps(self.model))

    def load_generator(self):
        from src.pipeline.dice_cf_generator import DiceCFGenerator
        generator = DiceCFGenerator(str(self.model_path), "reference.csv")
        # Exercise the ordinary loader without changing its historical cleaning.
        with patch("src.utils.dataLoader.DataLoader") as loader:
            loader.return_value.load_data.return_value = self.training.copy()
            loader.return_value.remove_outliers_iqr.return_value = self.training.copy()
            generator.load_model_and_data()
            loader.assert_called_once_with("reference.csv")
            loader.return_value.remove_outliers_iqr.assert_called_once()
        generator.setup_dice_explainer()
        return generator

    def test_ordinary_loader_uses_shared_adapter_and_matches_native_predictions(self):
        from src.pipeline.dice_compat import (
            CATEGORICAL_SCHEMA, NativeNumericClassifier, SchemaPublicData, SchemaDiceGenetic,
        )
        generator = self.load_generator()
        self.assertIsInstance(generator.dice_data, SchemaPublicData)
        self.assertIsInstance(generator.dice_exp, SchemaDiceGenetic)
        adapter = generator.dice_model.model
        self.assertIsInstance(adapter, NativeNumericClassifier)
        numeric = self.training[self.features]
        text = numeric.copy()
        for column in CATEGORICAL_SCHEMA:
            text[column] = text[column].astype(str)
        for values in (numeric, text, text.to_numpy()):
            with self.subTest(input_type=type(values)):
                np.testing.assert_array_equal(adapter.predict(values), self.model.predict(numeric))
                np.testing.assert_array_equal(adapter.predict_proba(values),
                                              self.model.predict_proba(numeric))
        np.testing.assert_array_equal(adapter.classes_, self.model.classes_)
        np.testing.assert_array_equal(
            generator.dice_model.get_output(text), self.model.predict_proba(numeric))
        _, tree, predictions = generator.dice_exp.build_KD_tree(
            generator.dice_data.data_df.copy(), None, 0, "target_pred")
        np.testing.assert_array_equal(predictions, self.model.predict(numeric))

    def test_unknown_factual_round_trip_preserves_reference_and_sampling_domain(self):
        generator = self.load_generator()
        original = self.training.iloc[[0]][self.features].copy()
        original["slope"] = 0
        original["oldpeak"] = 0.123456789123
        reference_before = generator.dice_data.data_df.copy(deep=True)
        ranges_before = copy.deepcopy(generator.dice_data.get_features_range()[0])
        exp = generator.dice_exp
        exp.setup("all", {"chol": [150, 200]}, original, "inverse_mad")
        prepared = generator.dice_data.prepare_query_instance(original)
        encoded = exp.label_encode(prepared.copy()).to_numpy()
        decoded = exp.label_decode(encoded)
        self.assertEqual(decoded.slope.iloc[0], "0")
        self.assertEqual(decoded.oldpeak.iloc[0], original.oldpeak.iloc[0])
        np.testing.assert_array_equal(exp.predict_fn_scores(encoded),
                                      self.model.predict_proba(original))
        allowed = exp.labelencoder["slope"].inverse_transform(
            exp.get_valid_feature_range()["slope"])
        self.assertNotIn("0", allowed)
        self.assertEqual(generator.dice_data.get_features_range()[0], ranges_before)
        pd.testing.assert_frame_equal(generator.dice_data.data_df, reference_before)
        self.assertEqual(reference_before.index.tolist(), self.training.index.tolist())
        for column in self.training:
            np.testing.assert_array_equal(
                pd.to_numeric(reference_before[column]), self.training[column])
        _, tree, _ = exp.build_KD_tree(reference_before.copy(), None, 0, "target_pred")
        dummies = pd.get_dummies(prepared)
        self.assertEqual(dummies.columns.tolist(),
                         list(generator.dice_data.get_all_dummy_colnames()))
        if tree is not None:
            tree.query(dummies, k=1)

    def test_ordinary_generation_keeps_native_opposite_target_and_search_settings(self):
        generator = self.load_generator()
        original = self.training.iloc[[0]][self.features].copy()
        original["slope"] = 0
        config_before = copy.deepcopy(generator.config)

        class StopBeforeSearch(Exception):
            pass

        with patch.object(generator.dice_exp, "do_param_initializations",
                          side_effect=StopBeforeSearch) as stop:
            with self.assertRaises(StopBeforeSearch):
                generator.generate_counterfactuals(original, seed=42, strict=True)
        np.testing.assert_array_equal(generator.dice_exp.test_pred,
                                      self.model.predict_proba(original))
        self.assertEqual(stop.call_args.args[3], 1 - self.model.predict(original)[0])
        self.assertEqual(generator.config, config_before)

    def test_invalid_categories_and_nonfinite_predictions_are_rejected(self):
        generator = self.load_generator()
        original = self.training.iloc[[0]][self.features].copy()
        for value in (99, 1.5, "invalid"):
            with self.subTest(slope=value):
                invalid = original.copy()
                invalid["slope"] = value
                with self.assertRaises(ValueError):
                    generator.dice_data.prepare_query_instance(invalid)
                with self.assertRaises(ValueError):
                    generator.dice_model.model.predict_proba(invalid)
        for value in (np.nan, np.inf):
            invalid = original.copy()
            invalid["chol"] = value
            with self.assertRaisesRegex(ValueError, "Non-finite"):
                generator.dice_model.model.predict_proba(invalid)

    def test_unsupported_method_is_explicit_not_uncorrected_fallback(self):
        generator = self.load_generator()
        generator.config["method"] = "random"
        with self.assertRaisesRegex(ValueError, "genetic only"):
            generator.setup_dice_explainer()

    def test_shared_source_is_included_in_future_manifest_hashes(self):
        self.assertIn("src/pipeline/dice_compat.py", cv.SOURCE_FILES)


class PersistenceTests(ScratchTest):
    def test_atomic_checkpoint_and_corruption_detection(self):
        path = self.scratch / "record.json"
        cv.write_record(path, {"version": 1})
        with patch.object(cv.os, "replace", side_effect=OSError("interrupted")):
            with self.assertRaises(OSError):
                cv.write_record(path, {"version": 2})
        self.assertEqual(cv.read_record(path), {"version": 1})
        self.assertTrue(path.with_name("record.json.partial").exists())
        cv.write_record(path, {"version": 2})
        self.assertFalse(path.with_name("record.json.partial").exists())
        envelope = json.loads(path.read_text())
        envelope["payload"]["version"] = 3
        path.write_text(json.dumps(envelope))
        with self.assertRaises(ValueError):
            cv.read_record(path)

    def test_manifest_rejects_code_config_data_and_environment_changes(self):
        original = {"source_sha256": "a", "config": {"repeats": 100},
                    "data_sha256": "b", "packages": {"numpy": "1"}}
        cv.assert_manifest(original, copy.deepcopy(original))
        for key in original:
            changed = copy.deepcopy(original)
            changed[key] = "changed"
            with self.assertRaises(ValueError):
                cv.assert_manifest(original, changed)

    def test_exclusive_lock_no_stale_lock_override(self):
        with cv.run_lock(self.scratch):
            with self.assertRaises(FileExistsError):
                with cv.run_lock(self.scratch):
                    pass
        self.assertFalse((self.scratch / "RUN.lock").exists())

    def test_resume_no_double_count_duplicate_and_wrong_manifest_rejected(self):
        manifest = {"expected_repeats": 2}
        folds = {0: {"tp_ids": [1000]}}
        path = cv.attempt_path(self.scratch, 0, 1000, 0)
        cv.write_record(path, record([]))
        expected, records, _ = cv.scan_attempts(self.scratch, manifest, folds)
        self.assertEqual(len(records), 1)
        self.assertEqual(len(expected), 2)
        duplicate = path.with_name("duplicate.json")
        duplicate.write_bytes(path.read_bytes())
        with self.assertRaises(ValueError):
            cv.scan_attempts(self.scratch, manifest, folds)
        duplicate.unlink()
        changed = record([])
        changed["manifest_id"] = "wrong"
        cv.write_record(path, changed)
        with self.assertRaises(ValueError):
            cv.scan_attempts(self.scratch, manifest, folds)

    def test_proposal_count_and_same_proposal_validation(self):
        bad = record([candidate(0, 0, 1), candidate(0, 1, 0, 1)])
        bad["flags"]["both"] = True
        with self.assertRaises(ValueError):
            cv.validate_attempt(bad, (0, 1000, 0), bad["manifest_id"])
        bad = record([candidate(0, 0, 0)])
        bad["returned_proposals"] = 2
        with self.assertRaises(ValueError):
            cv.validate_attempt(bad, (0, 1000, 0), bad["manifest_id"])

    def test_worker_no_cf_checkpoints_resume_and_audits_unexpected_error(self):
        data = fixture_data()
        generator = Mock()
        generator.generate_counterfactuals.return_value = Mock(
            cf_examples_list=[Mock(final_cfs_df=None)])
        manifest = {"expected_repeats": 2, "config": {"dice": {"total_cfs": 5}}}
        with patch.object(cv, "worker_context", return_value=(generator, Mock(), data)):
            cv.process_patient(str(self.scratch), manifest, 0, 1000, [0])
            cv.process_patient(str(self.scratch), manifest, 0, 1000, [0])
            self.assertEqual(generator.generate_counterfactuals.call_count, 1)
            checkpoint = cv.read_record(cv.attempt_path(self.scratch, 0, 1000, 0))
            self.assertEqual(checkpoint["status"], "no_cf")
            generator.generate_counterfactuals.side_effect = RuntimeError("unexpected")
            with self.assertRaises(RuntimeError):
                cv.process_patient(str(self.scratch), manifest, 0, 1000, [1])
            self.assertFalse(cv.attempt_path(self.scratch, 0, 1000, 1).exists())
            error = cv.read_record(self.scratch / "errors" / "f0_s1000_r1.json")
            self.assertIn("unexpected", error["traceback"])

    def test_worker_persists_raw_and_propagated_disagreement(self):
        data = fixture_data()
        original = data.loc[[1001]]
        proposals = pd.concat([original, original], ignore_index=True)
        proposals.chol = [180, 190]
        generator = Mock()
        generator.generate_counterfactuals.return_value = Mock(
            cf_examples_list=[Mock(final_cfs_df=proposals)])
        generator.model.predict.side_effect = [[0], [1], [0], [0]]
        analyzer = Mock()
        propagated = []
        for target in (0, 1):
            values = {f"cf_{col}": value for col, value in original.iloc[0].items()}
            values["target"] = target
            values["cf_target"] = target
            propagated.append(pd.DataFrame([values]))
        analyzer.apply_scm_intervention.side_effect = propagated
        manifest = {"expected_repeats": 2, "config": {"dice": {"total_cfs": 5}}}
        with patch.object(cv, "worker_context", return_value=(generator, analyzer, data)):
            cv.process_patient(str(self.scratch), manifest, 0, 1001, [0])
        saved = cv.read_record(cv.attempt_path(self.scratch, 0, 1001, 0))
        self.assertEqual(saved["returned_proposals"], 2)
        self.assertTrue(saved["flags"]["scm_accept"])
        self.assertTrue(saved["flags"]["propagated_flip"])
        self.assertFalse(saved["flags"]["both"])
        self.assertEqual(saved["candidates"][0]["raw"]["chol"], 180)
        self.assertEqual(saved["candidates"][1]["propagated"]["target"], 1)

    def test_strict_generation_propagates_unexpected_errors(self):
        from src.pipeline.dice_cf_generator import DiceCFGenerator
        gen = DiceCFGenerator("", "")
        gen.dice_exp = Mock()
        gen.dice_exp.generate_counterfactuals.side_effect = RuntimeError("unexpected")
        query = fixture_data().drop(columns="target").iloc[[0]]
        with self.assertRaises(RuntimeError):
            gen.generate_counterfactuals(query, strict=True)
        self.assertIsNone(gen.generate_counterfactuals(query))

    def test_strict_scm_propagates_errors_legacy_returns_none(self):
        from src.pipeline.scm_analyzer import SCMAnalyzer
        analyzer = SCMAnalyzer()
        analyzer.causal_model = Mock()
        with self.assertRaises(KeyError):
            analyzer.apply_scm_intervention(pd.DataFrame(), pd.DataFrame(), strict=True)
        self.assertIsNone(analyzer.apply_scm_intervention(pd.DataFrame(), pd.DataFrame()))


if __name__ == "__main__":
    unittest.main()
