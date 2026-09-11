"""Opt-in DiCE numeric-model/schema compatibility for the outer-CV experiment."""
import copy
import pickle

import numpy as np
import pandas as pd
from dice_ml import Model
from dice_ml.data_interfaces.public_data_interface import PublicData
from dice_ml.explainer_interfaces.dice_genetic import DiceGenetic
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder

from src.pipeline.dice_cf_generator import DiceCFGenerator
from src.training.train_model import NUMERICAL_FEATURES

# Harmonized input coding, declared independently of the fold observations.
# slope=0 is retained as an unknown factual code, not an ordinal clinical state.
CATEGORICAL_SCHEMA = {
    "sex": (0, 1), "cp": (1, 2, 3, 4), "fbs": (0, 1),
    "restecg": (0, 1, 2), "exang": (0, 1), "slope": (0, 1, 2, 3),
}


class NativeNumericClassifier(ClassifierMixin, BaseEstimator):
    """Normalize every DiCE prediction, including search, KD-tree and post-hoc."""
    def __init__(self, model, feature_names):
        self.model = model
        self.feature_names = feature_names

    @property
    def classes_(self):
        return self.model.classes_

    def native_frame(self, values):
        if isinstance(values, pd.DataFrame):
            frame = values.loc[:, self.feature_names].copy()
        else:
            frame = pd.DataFrame(values, columns=self.feature_names)
        frame = frame.apply(pd.to_numeric, errors="raise")
        if not np.isfinite(frame.to_numpy(dtype=float)).all():
            raise ValueError("Non-finite DiCE model input")
        for column, domain in CATEGORICAL_SCHEMA.items():
            if not frame[column].isin(domain).all():
                raise ValueError(f"Undeclared categorical code for {column}")
            frame[column] = frame[column].astype("int64")
        return frame

    def predict(self, values):
        return self.model.predict(self.native_frame(values))

    def predict_proba(self, values):
        return self.model.predict_proba(self.native_frame(values))


class SchemaPublicData(PublicData):
    """Schema governs encoding only; empirical ranges/MADs/rows remain training-only."""
    def __init__(self, training):
        super().__init__({"dataframe": training.copy(),
                          "continuous_features": NUMERICAL_FEATURES,
                          "outcome_name": "target"})
        if set(self.categorical_feature_names) != set(CATEGORICAL_SCHEMA):
            raise ValueError("Unexpected feature schema")
        self.data_df = self.schema_categories(self.data_df)

    def _set_feature_dtypes(self, data_df, categorical_feature_names, continuous_feature_names):
        # DiCE's float32 cast can move original values across fitted scaler/tree
        # thresholds. Keep native float64 precision through query encode/decode.
        for column in continuous_feature_names:
            data_df[column] = pd.to_numeric(data_df[column], errors="raise")
            if pd.api.types.is_float_dtype(data_df[column]):
                data_df[column] = data_df[column].astype("float64")
        return self.schema_categories(data_df)

    @staticmethod
    def schema_categories(frame):
        for column, values in CATEGORICAL_SCHEMA.items():
            codes = pd.to_numeric(frame[column], errors="raise")
            if not codes.isin(values).all():
                raise ValueError(f"Undeclared categorical code for {column}")
            strings = codes.astype("int64").astype(str)
            frame[column] = pd.Categorical(strings, categories=[str(v) for v in values])
        return frame

    def fit_label_encoders(self):
        return {column: LabelEncoder().fit([str(v) for v in values])
                for column, values in CATEGORICAL_SCHEMA.items()}

    def prepare_query_instance(self, query_instance):
        return self.schema_categories(super().prepare_query_instance(query_instance))


class SchemaDiceGenetic(DiceGenetic):
    def check_query_instance_validity(self, features_to_vary, permitted_range,
                                      query_instance, feature_ranges_orig):
        # Accept factual schema codes absent from this training fold without
        # broadening self.feature_range (the proposal sampling domain).
        factual_ranges = copy.deepcopy(feature_ranges_orig)
        factual_ranges.update({c: [str(v) for v in values]
                               for c, values in CATEGORICAL_SCHEMA.items()})
        super().check_query_instance_validity(
            features_to_vary, permitted_range, query_instance, factual_ranges)


class FoldDiceGenerator(DiceCFGenerator):
    def initialize_for_training(self, model, training_data):
        self.model = model
        self.dice_data = SchemaPublicData(training_data)
        adapter = NativeNumericClassifier(model, self.dice_data.feature_names)
        self.dice_model = Model(model=adapter, backend="sklearn")
        self.setup_dice_explainer()
        return self

    def load_model_and_data(self, training_data=None):
        if training_data is None:
            raise ValueError("Fold DiCE requires explicit training-only reference rows")
        with open(self.model_path, "rb") as stream:
            model = pickle.load(stream)
        self.initialize_for_training(model, training_data)

    def setup_dice_explainer(self):
        if self.config["method"] != "genetic":
            raise ValueError("The fold schema adapter is validated for DiCE genetic only")
        self.dice_exp = SchemaDiceGenetic(self.dice_data, self.dice_model)

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
