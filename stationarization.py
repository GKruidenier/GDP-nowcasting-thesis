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
    fred_md = pd.read_csv("FRED_md_cleaned.csv", index_col=0, parse_dates=True, date_format="%Y-%m-%d")
    fred_qd = pd.read_csv("FRED_qd_cleaned.csv", index_col=0, parse_dates=True, date_format="%Y-%m-%d")

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
    stationarity_monthly = check_stationarity(fred_md)
    stationarity_quarterly = check_stationarity(fred_qd)

    # Apply differencing to non-stationary monthly features
    non_stationary_monthly = stationarity_monthly[stationarity_monthly["Stationary"] == False].index.tolist() + ["UEMP15T26"]
    fred_md_stationary = fred_md.copy()
    fred_md_stationary[non_stationary_monthly] = fred_md_stationary[non_stationary_monthly].diff()
    fred_md_stationary = fred_md_stationary.iloc[3:]  # Drop the first row after differencing

    # Apply differencing to non-stationary quarterly features
    non_stationary_quarterly = stationarity_quarterly[stationarity_quarterly["Stationary"] == False].index.tolist()
    fred_qd_stationary = fred_qd.copy()
    fred_qd_stationary[non_stationary_quarterly] = fred_qd_stationary[non_stationary_quarterly].diff()
    fred_qd_stationary = fred_qd_stationary.iloc[1:]  # Drop the first row after differencing

    # Re-check stationarity
    stationarity_monthly_after_first_diff = check_stationarity(fred_md_stationary)
    stationarity_quarterly_after_first_diff = check_stationarity(fred_qd_stationary)

    if verbose:
        print("Non-stationary monthly features after differencing:")
        print(stationarity_monthly_after_first_diff[stationarity_monthly_after_first_diff["Stationary"] == False])

        print("Non-stationary quarterly features after differencing:")
        print(stationarity_quarterly_after_first_diff[stationarity_quarterly_after_first_diff["Stationary"] == False])

        print("Number of nonstationary features after differencing. Monthly:", (stationarity_monthly_after_first_diff["Stationary"] == False).sum(), "and Quarterly:", (stationarity_quarterly_after_first_diff["Stationary"] == False).sum())

        # Plot the stationarized data
        fred_md_stationary.loc[:, stationarity_monthly_after_first_diff["Stationary"] == False].plot(subplots=True, figsize=(12, 20), title="Monthly Stationary Features After Differencing")
        plt.tight_layout()
        plt.show()

        fred_qd_stationary.loc[:, stationarity_quarterly_after_first_diff["Stationary"] == False].plot(subplots=True, figsize=(12, 20), title="Quarterly Stationary Features After Differencing")
        plt.tight_layout()
        plt.show()

    # Apply second-order differencing to non-stationary monthly features
    still_non_stationary_monthly = stationarity_monthly_after_first_diff[stationarity_monthly_after_first_diff["Stationary"] == False].index.tolist()
    
    if verbose:
        print("Still not stationary monthly:", still_non_stationary_monthly)

    # still_non_stationary_monthly = stationarity_monthly_after["Stationary"] == False
    fred_md_stationary[still_non_stationary_monthly] = fred_md_stationary[still_non_stationary_monthly].diff()
    fred_md_stationary = fred_md_stationary.iloc[3:]  # Drop the first row after second differencing

    # Apply second-order differencing to non-stationary quarterly features
    still_non_stationary_quarterly = stationarity_quarterly_after_first_diff[stationarity_quarterly_after_first_diff["Stationary"] == False].index.tolist()
    # still_non_stationary_quarterly = stationarity_quarterly_after["Stationary"] == False
    fred_qd_stationary[still_non_stationary_quarterly] = fred_qd_stationary[still_non_stationary_quarterly].diff()
    fred_qd_stationary = fred_qd_stationary.iloc[1:]  # Drop the first row after second differencing

    # Re-check stationarity after second-order differencing
    stationarity_monthly_after_second_diff = check_stationarity(fred_md_stationary)
    stationarity_quarterly_after_second_diff = check_stationarity(fred_qd_stationary)

    if verbose:
        print("Non-stationary monthly features after second differencing:")
        print(stationarity_monthly_after_second_diff[stationarity_monthly_after_second_diff["Stationary"] == False].index.tolist())

        print("Non-stationary quarterly features after second differencing:")
        print(stationarity_quarterly_after_second_diff[stationarity_quarterly_after_second_diff["Stationary"] == False].index.tolist())

        print("Number of nonstationary features after second order differencing. Monthly:", (stationarity_monthly_after_second_diff["Stationary"] == False).sum(), "and Quarterly:", (stationarity_quarterly_after_second_diff["Stationary"] == False).sum())

        # Plot the second-order stationarized data
        fred_md_stationary.loc[:, stationarity_monthly_after_first_diff["Stationary"] == False].plot(subplots=True, figsize=(12, 20), title="Monthly Stationary Features After Second Differencing")
        plt.tight_layout()
        plt.show()

        fred_qd_stationary.loc[:, stationarity_quarterly_after_first_diff["Stationary"] == False].plot(subplots=True, figsize=(12, 20), title="Quarterly Stationary Features After Second Differencing")
        plt.tight_layout()
        plt.show()
        
    # Ensure all features are stationary
    assert stationarity_monthly_after_second_diff["Stationary"].all(), "Not all monthly features are stationary after differencing."
    assert stationarity_quarterly_after_second_diff["Stationary"].all(), "Not all quarterly features are stationary after differencing."

    assert len(fred_md_stationary) == len(fred_qd_stationary) * 3, "Monthly data should be exactly three times as long as quarterly data."

    # Split the data into training and test sets
    test_size_quarterly = int(len(fred_qd_stationary) * 0.15)
    test_size_monthly = test_size_quarterly * 3

    train_monthly = fred_md_stationary.iloc[:-test_size_monthly]
    test_monthly = fred_md_stationary.iloc[-test_size_monthly:]

    train_quarterly = fred_qd_stationary.iloc[:-test_size_quarterly]
    test_quarterly = fred_qd_stationary.iloc[-test_size_quarterly:]

    # Save the datasets to new files
    train_monthly.to_csv("FRED_md_train.csv")
    test_monthly.to_csv("FRED_md_test.csv")

    train_quarterly.to_csv("FRED_qd_train.csv")
    test_quarterly.to_csv("FRED_qd_test.csv")

    return train_monthly, train_quarterly

def load_data(split):
    """
    Load the training or test data for monthly and quarterly datasets.

    Returns:
        train_monthly (pd.DataFrame): Training dataset for monthly features.
        train_quarterly (pd.DataFrame): Training dataset for quarterly features.
    """

    assert split in ["train", "test"], "Invalid split. Choose 'train' or 'test'."

    train_monthly = pd.read_csv(f"FRED_md_{split}.csv", index_col=0, parse_dates=True, date_format="%Y-%m-%d")
    train_quarterly = pd.read_csv(f"FRED_qd_{split}.csv", index_col=0, parse_dates=True, date_format="%Y-%m-%d")
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
