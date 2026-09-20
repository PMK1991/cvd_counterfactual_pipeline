"""Training-only DiCE initialization and preflight for the outer-CV experiment."""
import copy
import pickle

import numpy as np
import pandas as pd
from src.pipeline.dice_cf_generator import DiceCFGenerator
# Re-export the original adapter names for existing imports.
from src.pipeline.dice_compat import (
    CATEGORICAL_SCHEMA, NativeNumericClassifier, SchemaPublicData, SchemaDiceGenetic,
)


class FoldDiceGenerator(DiceCFGenerator):
    def initialize_for_training(self, model, training_data):
        self.model = model
        self._prepare_dice_data(training_data)
        self.setup_dice_explainer()
        return self

    def load_model_and_data(self, training_data=None):
        if training_data is None:
            raise ValueError("Fold DiCE requires explicit training-only reference rows")
        with open(self.model_path, "rb") as stream:
            model = pickle.load(stream)
        self.initialize_for_training(model, training_data)

    def generate_counterfactuals(self, patient_data, seed=None, strict=False):
        return super().generate_counterfactuals(
            patient_data, seed=seed, strict=strict, desired_class=0)


def preflight_generator(generator, validation):
    """Exercise actual genetic/KD prediction and query encoding for EVERY OOF TP."""
    features = generator.dice_data.feature_names
    classifier = generator.model
    probabilities = classifier.predict_proba(validation[features])
    predictions = classifier.predict(validation[features])
    selected = validation[(validation.target == 1) & (predictions == 1)]
    exp = generator.dice_exp
    reference = exp.data_interface.data_df
    _, tree, kd_predictions = exp.build_KD_tree(reference.copy(), None, 0, "target_pred")
    native_reference = generator.dice_model.model.native_frame(reference[features])
    np.testing.assert_array_equal(kd_predictions, classifier.predict(native_reference))
    checked = []
    for source_id, original in selected.iterrows():
        query = validation.loc[[source_id], features]
        exp.setup(generator.config.get("features_to_vary") or "all",
                  copy.deepcopy(generator.config["permitted_range"]), query, "inverse_mad")
        prepared = exp.data_interface.prepare_query_instance(query)
        encoded = exp.label_encode(prepared.copy()).to_numpy()
        actual_scores = exp.predict_fn_scores(encoded)
        expected_scores = classifier.predict_proba(query)
        np.testing.assert_allclose(actual_scores, expected_scores, rtol=0, atol=1e-7)
        np.testing.assert_array_equal(exp.predict_fn(encoded), classifier.predict(query))
        np.testing.assert_allclose(exp.model.get_output(prepared), expected_scores, rtol=0, atol=1e-7)
        decoded = exp.label_decode(encoded)
        for column in CATEGORICAL_SCHEMA:
            if int(decoded[column].iloc[0]) != int(original[column]):
                raise ValueError(f"Factual category changed during round trip: {column}")
        dummies = pd.get_dummies(prepared)
        if dummies.columns.tolist() != list(exp.data_interface.get_all_dummy_colnames()):
            raise ValueError("Query/reference KD-tree columns misaligned")
        if tree is not None:
            tree.query(dummies, k=1)
        checked.append({"source_row_id": int(source_id),
                        "native_probability": float(expected_scores[0, 1]),
                        "genetic_probability": float(actual_scores[0, 1]),
                        "slope": int(original["slope"])})
    return {"validation_rows": len(validation), "checked_tp_count": len(checked),
            "checked_tps": checked,
            "reference_rows": len(reference),
            "proposal_categorical_ranges": {
                c: sorted(str(v) for v in reference[c].unique()) for c in CATEGORICAL_SCHEMA},
            "max_probability_difference": max(
                (abs(r["native_probability"] - r["genetic_probability"]) for r in checked),
                default=0.0)}
