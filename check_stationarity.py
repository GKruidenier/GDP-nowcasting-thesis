import pandas as pd
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
import numpy as np

def preprocess_data(verbose=False):
    """
    Load, preprocess, and stationarize the FRED_MD and FRED_QD datasets.

    Returns:
        final_monthly_stationary (pd.DataFrame): Preprocessed and stationarized monthly dataset.
        final_quarterly_stationary (pd.DataFrame): Preprocessed and stationarized quarterly dataset.
    """
    # Load datasets
    fred_md = pd.read_csv("FRED_md_cleaned_transformed.csv", index_col=0, parse_dates=True, date_format="%Y-%m-%d")
    fred_qd = pd.read_csv("FRED_qd_cleaned_transformed.csv", index_col=0, parse_dates=True, date_format="%Y-%m-%d")

    fred_md = fred_md.loc[:'2019-12-31']
    fred_qd = fred_qd.loc[:'2019-12-31']

    print(min(fred_md.index), max(fred_md.index))
    print(min(fred_qd.index), max(fred_qd.index))

    # Function to check stationarity using the Augmented Dickey-Fuller test
    def check_stationarity(df):
        results = {}
        for col in df.columns:
            series = df[col].dropna()
            adf_result = adfuller(series)
            results[col] = {
                "ADF Statistic": adf_result[0],
                "p-value": adf_result[1],
                "Stationary": adf_result[1] <= 0.05
            }
        return pd.DataFrame(results).T.sort_values("p-value")

    # Check stationarity for monthly and quarterly datasets
    stationarity_monthly = check_stationarity(fred_md[3:])
    stationarity_quarterly = check_stationarity(fred_qd[1:])

    # Apply differencing to non-stationary monthly features
    non_stationary_monthly = stationarity_monthly[stationarity_monthly["Stationary"] == False].index.tolist()
    non_stationary_quarterly = stationarity_quarterly[stationarity_quarterly["Stationary"] == False].index.tolist()

    # Apply differencing to non-stationary monthly features
    fred_md_stationary = fred_md.copy()
    fred_md_stationary[non_stationary_monthly] = fred_md_stationary[non_stationary_monthly].diff()
    fred_md_stationary = fred_md_stationary.iloc[3:]  # Drop the first row after differencing

    # Apply differencing to non-stationary quarterly features
    fred_qd_stationary = fred_qd.copy()
    fred_qd_stationary[non_stationary_quarterly] = fred_qd_stationary[non_stationary_quarterly].diff()
    fred_qd_stationary = fred_qd_stationary.iloc[1:]  # Drop the first row after differencing

    stationarity_monthly_after_1_diff = check_stationarity(fred_md_stationary)
    stationarity_quarterly_after_1_diff = check_stationarity(fred_qd_stationary)

    non_stationary_monthly_after_1_diff = stationarity_monthly_after_1_diff[stationarity_monthly_after_1_diff["Stationary"] == False].index.tolist()
    non_stationary_quarterly_after_1_diff = stationarity_quarterly_after_1_diff[stationarity_quarterly_after_1_diff["Stationary"] == False].index.tolist()
    
    print("Stationarity before differencing:")
    print(stationarity_monthly.loc[non_stationary_monthly])
    print(stationarity_quarterly.loc[non_stationary_quarterly])
    print("Stationarity after 1st differencing:")
    print(stationarity_monthly_after_1_diff.loc[non_stationary_monthly])
    print(stationarity_quarterly_after_1_diff.loc[non_stationary_quarterly])

    if any(non_stationary_monthly_after_1_diff) or any(non_stationary_quarterly_after_1_diff):
        print("Non-stationary monthly features:")
        print(non_stationary_monthly_after_1_diff)

        print("Non-stationary quarterly features:")
        print(non_stationary_quarterly_after_1_diff)
    
        print(stationarity_monthly.loc[non_stationary_monthly_after_1_diff])
        print(stationarity_quarterly.loc[non_stationary_quarterly_after_1_diff])

        # Plot the stationarized data
        fred_md_stationary.loc[:, non_stationary_monthly_after_1_diff].plot(subplots=True, figsize=(12, 20), title="Monthly Stationary Features After Differencing")
        plt.tight_layout()
        plt.show()

        fred_md_stationary.loc[:, non_stationary_quarterly_after_1_diff].plot(subplots=True, figsize=(12, 20), title="Quarterly Stationary Features After Differencing")
        plt.tight_layout()
        plt.show()
        
        raise ValueError("Found non-stationary features. Please check the plots.")

    # Split the data into training and test sets
    test_size_quarterly = int(len(fred_qd_stationary) * 0.15)
    test_size_monthly = test_size_quarterly * 3

    train_monthly = fred_md_stationary.iloc[:-test_size_monthly]
    test_monthly = fred_md_stationary.iloc[-test_size_monthly:]

    train_quarterly = fred_qd_stationary.iloc[:-test_size_quarterly]
    test_quarterly = fred_qd_stationary.iloc[-test_size_quarterly:]

    # Save the datasets to new files
    train_monthly.to_csv("FRED_md_train_cleaned_transformed.csv")
    test_monthly.to_csv("FRED_md_test_cleaned_transformed.csv")

    train_quarterly.to_csv("FRED_qd_train_cleaned_transformed.csv")
    test_quarterly.to_csv("FRED_qd_test_cleaned_transformed.csv")

    return train_monthly, train_quarterly

def load_data(split):
    """
    Load the training or test data for monthly and quarterly datasets.

    Returns:
        train_monthly (pd.DataFrame): Training dataset for monthly features.
        train_quarterly (pd.DataFrame): Training dataset for quarterly features.
    """

    assert split in ["train", "test"], "Invalid split. Choose 'train' or 'test'."

    train_monthly = pd.read_csv(f"FRED_md_{split}_cleaned_transformed.csv", index_col=0, parse_dates=True, date_format="%Y-%m-%d")
    train_quarterly = pd.read_csv(f"FRED_qd_{split}_cleaned_transformed.csv", index_col=0, parse_dates=True, date_format="%Y-%m-%d")
    return train_monthly, train_quarterly

def load_train_data(): return load_data("train")
def load_test_data(): return load_data("test")

if __name__ == "__main__":
    import sys
    verbose = "--verbose" in sys.argv
    final_monthly_stationary, final_quarterly_stationary = preprocess_data(verbose)

    if verbose:
        # Output dataset shapes and preview
        final_monthly_shape = final_monthly_stationary.shape
        final_quarterly_shape = final_quarterly_stationary.shape
        final_monthly_stationary.head(), final_quarterly_stationary.head(), final_monthly_shape, final_quarterly_shape

        # Plotting
        final_monthly_stationary.plot(subplots=True, figsize=(12, 20), title="Monthly Stationary Features")
        plt.tight_layout()
        plt.show()

        final_quarterly_stationary.plot(figsize=(10, 4), title="Quarterly Stationary GDP (GDPC1)")
        plt.tight_layout()
        plt.show()
