from sklearn.metrics import mean_squared_error, root_mean_squared_error
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import layers, Input, Model 
from tensorflow.keras.layers import LSTM, Dense, Concatenate, TimeDistributed
import os
from statsmodels.tsa.arima.model import ARIMA, ARIMAResults
import random
from statsmodels.tools.eval_measures import bic

import datetime
from check_stationarity import load_train_data
from feature_selection import combine_qd_and_md_as_qd, remove_highly_correlated_features, lasso_feature_selection, tree_based_feature_selection, rfe_feature_selection


md_train_stationary, qd_train_stationary = load_train_data()

def stepwise_selection(data, target, max_lag=5):
    """
    Perform stepwise regression to select lagged predictors based on BIC.
    
    Parameters:
        data (pd.DataFrame): DataFrame containing potential predictors.
        target (pd.Series): Target variable for ARIMAX.
        max_lag (int): Maximum number of lags to consider for each predictor.
    
    Returns:
        list: Selected predictors with their optimal lag structures.
    """
    selected_predictors = []
    current_bic = float('inf')
    
    for predictor in data.columns:
        for lag in range(1, max_lag + 1):
            lagged_data = data[predictor].shift(lag).dropna()
            combined_data = pd.concat([target, lagged_data], axis=1).dropna()
            model = ARIMA(combined_data.iloc[:, 0], exog=combined_data.iloc[:, 1:], order=(1, 0, 0))
            result = model.fit()
            new_bic = bic(result.llf, result.nobs, result.df_model)
            
            if new_bic < current_bic:
                current_bic = new_bic
                selected_predictors.append((predictor, lag))
    
    return selected_predictors

def train_arimax_with_exogenous(md_train, qd_train, exogenous_data):
    """
    Train an ARIMAX model with selected exogenous variables.
    
    Parameters:
        md_train (pd.Series): Monthly data (target variable).
        qd_train (pd.DataFrame): Quarterly data (predictors).
        exogenous_data (pd.DataFrame): Additional macroeconomic indicators.
    
    Returns:
        ARIMAResults: Fitted ARIMAX model.
    """
    # Combine quarterly and exogenous data
    combined_data = combine_qd_and_md_as_qd(qd_train, exogenous_data)
    
    # Perform stepwise selection of exogenous variables
    selected_predictors = stepwise_selection(combined_data, md_train)
    
    # Prepare final exogenous dataset
    exog = pd.DataFrame()
    for predictor, lag in selected_predictors:
        exog[f"{predictor}_lag{lag}"] = combined_data[predictor].shift(lag)
    exog = exog.dropna()
    
    # Align target and exogenous data
    exog = exog.loc[md_train.index.intersection(exog.index)]  # Align indices
    md_train_aligned = md_train.loc[exog.index]
    
    # Train ARIMAX model
    model = ARIMA(md_train_aligned, exog=exog, order=(1, 0, 0))
    fitted_model = model.fit()
    
    return fitted_model

def train_arimax_with_expanding_window(md_train, qd_train, exogenous_data, initial_window=12, step=1):
    """
    Train an ARIMAX model using an expanding window approach.
    
    Parameters:
        md_train (pd.Series): Monthly data (target variable).
        qd_train (pd.DataFrame): Quarterly data (predictors).
        exogenous_data (pd.DataFrame): Additional macroeconomic indicators.
        initial_window (int): Initial size of the training window.
        step (int): Step size for expanding the window.
    
    Returns:
        list: List of fitted ARIMAX models for each expanding window.
    """
    # Combine quarterly and exogenous data
    combined_data = combine_qd_and_md_as_qd(qd_train, exogenous_data)
    
    # Prepare storage for models
    models = []
    
    for end_idx in range(initial_window, len(md_train), step):
        # Define the expanding window
        md_train_window = md_train.iloc[:end_idx]
        combined_data_window = combined_data.iloc[:end_idx]
        
        # Perform stepwise selection of exogenous variables
        selected_predictors = stepwise_selection(combined_data_window, md_train_window)
        
        # Prepare final exogenous dataset
        exog = pd.DataFrame()
        for predictor, lag in selected_predictors:
            exog[f"{predictor}_lag{lag}"] = combined_data_window[predictor].shift(lag)
        exog = exog.dropna()
        
        # Align target and exogenous data
        exog = exog.loc[md_train_window.index.intersection(exog.index)]  # Align indices
        md_train_aligned = md_train_window.loc[exog.index]
        
        # Train ARIMAX model
        model = ARIMA(md_train_aligned, exog=exog, order=(1, 0, 0))
        fitted_model = model.fit()
        models.append(fitted_model)
    
    return models

def arimax_train_predict(md_train: pd.DataFrame, qd_train: pd.DataFrame, initial_window=12, step=1):
    md = md_train.groupby(md_train.index.map(lambda date: date - pd.DateOffset(months=date.month % 3))).sum()[1:]
    print("Monthly data shape:", md.shape)
    print("Quarterly data shape:", qd_train.shape)
    print("Monthly index:", md.index)
    print("Quarterly index:", qd_train.index)

    data = pd.concat([md, qd_train], axis=1)

    endogenous_data = data[["GDPC1"]]
    exogenous_data = data.drop(columns=["GDPC1"])

    print("Endogenous data shape:", endogenous_data.shape)
    print("Exogenous data shape:", exogenous_data.shape)

    train_size = int(len(endogenous_data) * 0.8)

    endogenous_data_train = endogenous_data[:train_size]
    exogenous_data_train = exogenous_data[:train_size]
    endogenous_data_val = endogenous_data[train_size:]
    exogenous_data_val = exogenous_data[train_size:]

    model = ARIMA(endog=endogenous_data_train, exog=exogenous_data_train, order=(5, 0, 3))
    trained_model: ARIMAResults = model.fit()

    forecasts = []

    # copy the training data to use for recursive forecasting
    history = []  # convert to a list for appending new values

    # recursive forecasting
    for i in range(1, len(endogenous_data_val)):
        true_value = endogenous_data_val.iloc[i]
        forecast = trained_model.forecast(steps=i, exog=exogenous_data_val[:i], endog=endogenous_data_val[:i])[0]

        # print(len(exogenous_data_val))
        # print(len(endogenous_data_val))
        # forecast = trained_model.forecast(steps=1, exog=exogenous_data_val.iloc[i])[0]
        forecasts.append(forecast)

        # trained_model = trained_model.append([endogenous_data_val.iloc[i]], [exogenous_data_val.iloc[i]], refit=False)

        # append the true value to the history for the next iteration
        history.append(true_value)
    
    rmse = root_mean_squared_error(forecasts, history)
    rmse_baseline = root_mean_squared_error(np.mean(history).repeat(len(history)), history)
    print("RMSE:", rmse)
    print("RMSE Baseline:", rmse_baseline)

def main():
    # Example usage
    fitted_arimax = train_arimax_with_exogenous(md_train_stationary, qd_train_stationary, exogenous_data=pd.DataFrame())
    print(fitted_arimax.summary())

    fitted_models = train_arimax_with_expanding_window(md_train_stationary, qd_train_stationary, exogenous_data=pd.DataFrame(), initial_window=24, step=6)
    for i, model in enumerate(fitted_models):
        print(f"Model {i + 1} Summary:")
        print(model.summary())

if __name__ == "__main__":
    md_train_stationary, qd_train_stationary = load_train_data()
    arimax_train_predict(md_train_stationary, qd_train_stationary)