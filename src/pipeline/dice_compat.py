"""Shared DiCE compatibility for numeric CVD classifiers and categorical schemas."""
import copy

import numpy as np
import pandas as pd
from dice_ml.data_interfaces.public_data_interface import PublicData
from dice_ml.explainer_interfaces.dice_genetic import DiceGenetic
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder

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
    """Schema governs encoding only; empirical ranges/MADs/rows use reference data."""
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
        # Accept factual schema codes absent from the reference without
        # broadening self.feature_range (the proposal sampling domain).
        factual_ranges = copy.deepcopy(feature_ranges_orig)
        factual_ranges.update({c: [str(v) for v in values]
                               for c, values in CATEGORICAL_SCHEMA.items()})
        super().check_query_instance_validity(
            features_to_vary, permitted_range, query_instance, factual_ranges)
