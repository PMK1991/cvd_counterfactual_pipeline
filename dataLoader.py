import pandas as pd
from sklearn.ensemble import IsolationForest

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

class DataLoader:
    def __init__(self, file_path):
        self.file_path = file_path

    def load_data(self):
        try:
            df = pd.read_csv(self.file_path)
            print("Data loaded successfully.")
            df = df.dropna()
            df = df[df['chol'] > 0]
            df = df[df['trestbps'] > 0]
            df = df.drop_duplicates()
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

        # Combine the filtered DataFrames back together
        df_filtered = pd.concat([df_target_0, df_target_1])

        return df_filtered

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