import optuna
from functools import partial
from sklearn.metrics import mean_squared_error
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import layers, Input, Model
from tensorflow.keras.layers import LSTM, GRU
import os
from statsmodels.tsa.arima.model import ARIMA, ARIMAResults
import random
import subprocess

import datetime

from check_stationarity import load_train_data, load_test_data

from baselines_123 import train_and_evaluate_model

def optuna_log(message):
    """
    Log messages to a file with a timestamp.
    """
    with open(f"optuna_log_{log_id}.txt", "a") as f:
        f.write(message)

def objective(trial, md_train_stationary, qd_train_stationary, md_test_stationary, qd_test_stationary, model, trial_context, optimize_feature_selection=True):
    # Define the hyperparameter search space
    start = datetime.datetime.now()
    optuna_log(f"\nTrial {trial.number} at {start.strftime("%Y-%m-%d %H:%M:%S")}:\n")

    assert model in ["arimax", "dfm"]

    if optimize_feature_selection:
        feature_selection_method = trial.suggest_categorical("feature_selection_method", ["lasso"])
        max_features = 80 if model == "arimax" else 20
        n_features = trial.suggest_int("n_features", 9, max_features)
    else:
        feature_selection_method = "none"
        n_features = 1

    add_recession_feature = trial.suggest_categorical("add_recession_feature", [True, False])

    if model == "arimax":
        model_params = {
            'endog_lags': trial.suggest_int("endog_lags", 1, 8),
            'error_lags': trial.suggest_int("error_lags", 1, 8),
        }
    elif model == "dfm":
        model_params = {
            'factors': trial.suggest_int("factors", 1, 5),
            'factor_orders': trial.suggest_int("factor_orders", 1, 5),
        }

    param_dict = {
        "feature_selection_method": feature_selection_method,
        "n_features": n_features,
        "add_recession_feature": add_recession_feature,
    } | model_params

    optuna_log(f"Parameters: {", ".join(f"{k}={v}" for k, v in param_dict.items())}\n")

    try:
        # val_losses = train_and_evaluate_model(
        #     md_train_stationary=md_train_stationary,
        #     qd_train_stationary=qd_train_stationary,
        #     feature_selection_method=feature_selection_method,
        #     n_features=n_features,
        #     dropout_rate=dropout_rate,
        #     learning_rate=learning_rate,
        #     optimizer=optimizer,
        #     batch_size=batch_size,
        #     hidden_units=hidden_units,
        #     sequence_length=sequence_length,
        #     instantiate_model=model,
        #     create_datapoints=create_datapoints,
        #     add_recession_feature=add_recession_feature,
        #     max_epochs=40,
        # )
        # Use the average of the lowest validation losses as the objective value
        # mean = sum(val_losses) / len(val_losses)

        val_scores, _ = train_and_evaluate_model(
            md_train_stationary, qd_train_stationary,
            feature_selection_method=feature_selection_method,
            n_features=n_features,
            add_recession_feature=add_recession_feature,
            model_name=model,
            model_params=model_params,
            verbose=False,
            md_test_stationary=md_test_stationary,
            qd_test_stationary=qd_test_stationary,
        )

        mean_val = np.mean(val_scores)

        end = datetime.datetime.now()
        duration = end - start
        minutes = duration.seconds // 60
        seconds = duration.seconds % 60

        optuna_log(f"Results: mean_val={mean_val}; took {minutes}m{seconds}s\n")

        if mean_val < trial_context["best_score"]:
            trial_context["best_score"] = mean_val
            optuna_log(f"New best score: {mean_val}\n")

        return mean_val
    except Exception as e:
        # Handle any exceptions during training (e.g., invalid hyperparameters)
        print(f"Trial failed with exception: {e}")
        optuna_log(f"Trial failed with exception: {e}\n")
        return float("inf")

if __name__ == "__main__":
    import sys

    log_idx = sys.argv.index("--log-id")
    log_id = sys.argv[log_idx + 1]

    model_idx = sys.argv.index("--model")
    model_id = sys.argv[model_idx + 1]

    try:
        trials_idx = sys.argv.index("--trials")
        n_trials = int(sys.argv[trials_idx + 1])
    except ValueError:
        n_trials = 50

    # Load the training data
    md_train_stationary, qd_train_stationary = load_train_data()
    md_test_stationary, qd_test_stationary = load_test_data()

    # Create the Optuna study with Bayesian optimization (TPE)
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler())

    optuna_log("\n----------------------------\n\n")

    optimize_feature_selection = True

    trial_context = {
        "best_score": float("inf"),
    }

    # Optimize the objective function
    study.optimize(
        partial(objective,
                md_train_stationary=md_train_stationary, qd_train_stationary=qd_train_stationary,
                md_test_stationary=md_test_stationary, qd_test_stationary=qd_test_stationary,
                model=model_id,
                optimize_feature_selection=optimize_feature_selection,
                trial_context=trial_context),
        n_trials=n_trials,
    )

    # Print the best hyperparameters
    print("Best trial:")
    print(f"  Value: {study.best_trial.value}")
    print(f"  Params: ")
    for key, value in study.best_trial.params.items():
        print(f"    {key}: {value}")

    optuna_log("\nBest trial:\n")
    optuna_log(f"  Value: {study.best_trial.value}\n")
    optuna_log(f"  Params: \n")
    for key, value in study.best_trial.params.items():
        optuna_log(f"    {key}: {value}\n")
