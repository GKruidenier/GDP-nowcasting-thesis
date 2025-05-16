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

from check_stationarity import load_train_data

from lstm_123 import instantiate_model_duplicate_qd, instantiate_model_multilayer, instantiate_model_alternating_rnn, create_datapoints_lstm_1_2_and_3
from univariate_RNN import instantiate_univariate_model, create_datapoints_univariate
from multivariate_RNN import instantiate_multivariate_model, create_datapoints_multivariate

def optuna_log(message):
    """
    Log messages to a file with a timestamp.
    """
    with open(f"optuna_log_{log_id}.txt", "a") as f:
        f.write(message)

def objective(trial, md_train_stationary, qd_train_stationary, model, create_datapoints, trial_context, optimize_feature_selection=True):
    # Define the hyperparameter search space
    start = datetime.datetime.now()
    optuna_log(f"\nTrial {trial.number} at {start.strftime("%Y-%m-%d %H:%M:%S")}:\n")

    if optimize_feature_selection:
        feature_selection_method = trial.suggest_categorical("feature_selection_method", ["lasso"])
        n_features = trial.suggest_int("n_features", 5, 40)
    else:
        feature_selection_method = "none"
        n_features = 1

    dropout_rate = trial.suggest_float("dropout_rate", 0.2, 0.5)
    optimizer = trial.suggest_categorical("optimizer", ["adam"])
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    hidden_units = trial.suggest_categorical("hidden_units", [16, 32, 64])
    learning_rate = trial.suggest_float("learning_rate", 3e-3, 1e-2, log=True)
    add_recession_feature = trial.suggest_categorical("add_recession_feature", [True, False])
    sequence_length = 12  # Fixed sequence length for LSTM

    param_dict = {
        "feature_selection_method": feature_selection_method,
        "n_features": n_features,
        "dropout_rate": dropout_rate,
        "optimizer": optimizer,
        "batch_size": batch_size,
        "hidden_units": hidden_units,
        "sequence_length": sequence_length,
        "learning_rate": learning_rate,
        "add_recession_feature": add_recession_feature,
    }
    optuna_log(f"Parameters: {", ".join(f"{k}={v}" for k, v in param_dict.items())}\n")

    # Call the LSTM_model_1 function with the sampled hyperparameters
    # if True:
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
        command = ["./thesis-env/Scripts/python.exe", "train_with_hyper_parameters.py", "--model", model_id, "--rnn", rnn_name, "--feature-selection-method", feature_selection_method, "--n-features", str(n_features), "--dropout-rate", str(dropout_rate), "--optimizer", optimizer, "--batch-size", str(batch_size), "--hidden-units", str(hidden_units), "--learning-rate", str(learning_rate), "--add-recession-feature", str(add_recession_feature)]
        print(" ".join(command))
        train_process = subprocess.run(command, capture_output=True, text=True)
        print("subrpocess stdout:", train_process.stdout)
        print("subrpocess stderr:", train_process.stderr)
        print("subrpocess exit code:", train_process.returncode)
        stdout = train_process.stdout.split("\n")
        mean_train = float(stdout[-3].strip())
        mean_val = float(stdout[-2].strip())

        end = datetime.datetime.now()
        duration = end - start
        minutes = duration.seconds // 60
        seconds = duration.seconds % 60

        optuna_log(f"Results: mean_val={mean_val}; mean_train={mean_train}; took {minutes}m{seconds}s\n")

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

    rnn_idx = sys.argv.index("--rnn")
    rnn_name = sys.argv[rnn_idx + 1]
    
    try:
        trials_idx = sys.argv.index("--trials")
        n_trials = int(sys.argv[trials_idx + 1])
    except ValueError:
        n_trials = 50

    # Load the training data
    md_train_stationary, qd_train_stationary = load_train_data()

    # Create the Optuna study with Bayesian optimization (TPE)
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler())

    optuna_log("\n----------------------------\n\n")

    optimize_feature_selection = True
    if model_id == '1':
        model = instantiate_model_duplicate_qd
        create_datapoints = create_datapoints_lstm_1_2_and_3
    elif model_id == '2':
        model = instantiate_model_multilayer
        create_datapoints = create_datapoints_lstm_1_2_and_3
    elif model_id == '3':
        model = instantiate_model_alternating_rnn
        create_datapoints = create_datapoints_lstm_1_2_and_3
    elif model_id == 'univariate':
        model = instantiate_univariate_model
        create_datapoints = create_datapoints_univariate
        optimize_feature_selection = False
    elif model_id == 'multivariate':
        model = instantiate_multivariate_model
        create_datapoints = create_datapoints_multivariate

    if rnn_name.lower() == 'lstm':
        rnn = LSTM
    elif rnn_name.lower() == 'gru':
        rnn = GRU

    trial_context = {
        "best_score": float("inf"),
    }

    # Optimize the objective function
    study.optimize(
        partial(objective, md_train_stationary=md_train_stationary, qd_train_stationary=qd_train_stationary, model=partial(model, Rnn=rnn), create_datapoints=create_datapoints, optimize_feature_selection=optimize_feature_selection, trial_context=trial_context),
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
