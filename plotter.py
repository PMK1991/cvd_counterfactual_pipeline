import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

class Plotter:
    def __init__(self, data):
        self.data = data

    def box_plot(self, target_column, value_column, target_labels, title="Box Plot", xlabel="Target", ylabel="Values"):
        # Create lists of values for each target category
        values = [self.data[self.data[target_column] == label][value_column] for label in target_labels]
        
        plt.figure(figsize=(10, 6))
        plt.boxplot(values, labels=target_labels)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.show()

    def percentile_plot(self, value_column, target_column, title="Percentile Plot", xlabel="Percentile", ylabel="Values"):
        # Calculate percentiles for the value column
        percentiles = np.percentile(self.data[value_column], np.arange(0, 101, 1))

        # Separate percentiles by target
        percentiles_target_0 = np.percentile(self.data[self.data[target_column] == 0][value_column], np.arange(0, 101, 1))
        percentiles_target_1 = np.percentile(self.data[self.data[target_column] == 1][value_column], np.arange(0, 101, 1))

        # Create the percentile plot
        plt.figure(figsize=(10, 6))
        plt.plot(np.arange(0, 101, 1), percentiles_target_0, label=f'{target_column} 0')
        plt.plot(np.arange(0, 101, 1), percentiles_target_1, label=f'{target_column} 1')
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend()
        plt.show()

    def cross_tab_plot(self, col1, col2, title="Cross Tab Plot", xlabel="X-axis", ylabel="Y-axis"):
        # Create a crosstab for the specified columns
        cross_tab = pd.crosstab(self.data[col1], self.data[col2])

        # Create the bar plot
        cross_tab.plot(kind='bar', figsize=(10, 6))
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend(title=col2)
        plt.show()

