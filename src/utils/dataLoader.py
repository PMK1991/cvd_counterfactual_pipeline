"""Data loading helpers for the combined heart disease dataset.

The primary CSV contains 1190 harmonized rows from Cleveland (303),
Hungarian (294), Switzerland (123), Long Beach VA (200), and Statlog
(270). The distributed file already contains a binary `target` column:
0 means no CVD diagnosis and 1 means CVD diagnosis; this matches the
standard collapse of original multi-class labels where target > 0 maps
to 1.
"""

import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

MODEL_TEST_SIZE = 0.2
MODEL_RANDOM_STATE = 42


class DataLoader:
    def __init__(self, file_path):
        self.file_path = file_path
        self._step_counts = []

    def _reset_step_counts(self):
        self._step_counts = []

    def _record_step(self, step_name, rows_in, rows_out):
        self._step_counts.append({
            'step_name': step_name,
            'rows_in': int(rows_in),
            'rows_out': int(rows_out),
            'dropped': int(rows_in - rows_out),
        })

    def record_step(self, step_name, rows_in, rows_out):
        self._record_step(step_name, rows_in, rows_out)

    def get_step_counts(self):
        return list(self._step_counts)

    def load_data(self):
        try:
            self._reset_step_counts()
            df = pd.read_csv(self.file_path)
            self._record_step('raw', len(df), len(df))
            print("Data loaded successfully.")

            rows_in = len(df)
            df = df.dropna()
            self._record_step('dropna', rows_in, len(df))

            rows_in = len(df)
            df = df[df['chol'] > 0]
            self._record_step('chol_gt_0', rows_in, len(df))

            rows_in = len(df)
            df = df[df['trestbps'] > 0]
            self._record_step('trestbps_gt_0', rows_in, len(df))

            rows_in = len(df)
            df = df.drop_duplicates()
            self._record_step('drop_duplicates', rows_in, len(df))

            return df
        except Exception as e:
            print(f"Error loading data: {e}")
            return None

    def remove_outliers_iqr(self, df):
        # Separate the DataFrame based on the target
        df_target_0 = df[df['target'] == 0]
        df_target_1 = df[df['target'] == 1]

        # Function to remove outliers using the 1.5 IQR rule
        def remove_outliers_iqr(df, column):
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

        # Remove outliers for 'chol' and 'trestbps' in target 0
        df_target_0 = remove_outliers_iqr(df_target_0, 'chol')
        df_target_0 = remove_outliers_iqr(df_target_0, 'trestbps')

        # Remove outliers for 'chol' and 'trestbps' in target 1
        df_target_1 = remove_outliers_iqr(df_target_1, 'chol')
        df_target_1 = remove_outliers_iqr(df_target_1, 'trestbps')

        rows_in = len(df)

        # Combine the filtered DataFrames back together
        df_filtered = pd.concat([df_target_0, df_target_1])
        self._record_step('remove_outliers_iqr', rows_in, len(df_filtered))

        return df_filtered

    def load_clean_data(self):
        df = self.load_data()
        if df is not None:
            df = self.remove_outliers_iqr(df)
        return df

    def test_set_high_risk(
        self,
        test_size=MODEL_TEST_SIZE,
        random_state=MODEL_RANDOM_STATE,
    ):
        if test_size != MODEL_TEST_SIZE or random_state != MODEL_RANDOM_STATE:
            raise AssertionError(
                "Cohort selection must match train_model.py split "
                f"(test_size={MODEL_TEST_SIZE}, random_state={MODEL_RANDOM_STATE})"
            )

        df = self.load_clean_data()
        if df is None:
            return None

        X = df.drop(columns='target')
        y = df['target']
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        self._record_step('train_test_split_test_set', len(df), len(X_test))

        test_df = X_test.copy()
        test_df['target'] = y_test

        rows_in = len(test_df)
        high_risk = test_df[test_df['target'] == 1].copy()
        self._record_step('test_set_high_risk', rows_in, len(high_risk))

        return high_risk

    def remove_outliers_std(self, df):
        # Separate the DataFrame based on the target
        df_target_0 = df[df['target'] == 0]
        df_target_1 = df[df['target'] == 1]

        # Function to remove outliers using mean +/- 3 standard deviations
        def remove_outliers_std(df, column):
            mean = df[column].mean()
            std = df[column].std()
            upper_bound = mean + 3 * std
            lower_bound = mean - 3 * std
            return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

        # Remove outliers for 'chol' and 'trestbps' in target 0
        df_target_0 = remove_outliers_std(df_target_0, 'chol')
        df_target_0 = remove_outliers_std(df_target_0, 'trestbps')

        # Remove outliers for 'chol' and 'trestbps' in target 1
        df_target_1 = remove_outliers_std(df_target_1, 'chol')
        df_target_1 = remove_outliers_std(df_target_1, 'trestbps')

        # Combine the filtered DataFrames back together
        df_filtered = pd.concat([df_target_0, df_target_1])

        return df_filtered

    def remove_outliers_anomaly(self, df):
        # Separate the DataFrame based on the target
        df_target_0 = df[df['target'] == 0]
        df_target_1 = df[df['target'] == 1]

        # Function to perform anomaly detection using Isolation Forest
        def detect_anomalies(df, columns):
            iso_forest = IsolationForest(contamination=0.05)  # Adjust contamination as needed
            df['anomaly'] = iso_forest.fit_predict(df[columns])
            return df

        # Perform anomaly detection for 'chol' and 'trestbps' in target 0
        df_target_0 = detect_anomalies(df_target_0, ['chol', 'trestbps'])

        # Perform anomaly detection for 'chol' and 'trestbps' in target 1
        df_target_1 = detect_anomalies(df_target_1, ['chol', 'trestbps'])

        # Combine the DataFrames back together
        df_anomaly_detected = pd.concat([df_target_0, df_target_1])

        # Filter out the anomalies
        df_filtered = df_anomaly_detected[df_anomaly_detected['anomaly'] == 1]

        return df_filtered
