import optuna
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

from torch import nn
from torch.nn import LSTM, GRU

from statsmodels.tsa.arima.model import ARIMA, ARIMAResults

import os
import random
from functools import partial
import datetime
import traceback

from check_stationarity import load_train_data
from torch_train_evaluate import DuplicateQdModel, MultilayerModel, AlternatingRnnModel, create_datapoints_mixed_frequency, train_and_evaluate_model
from torch_multivariate_RNN import MultivariateModel, create_datapoints_multivariate
from torch_univariate_RNN import UnivariateModel, create_datapoints_univariate

# from torch_univariate_RNN import instantiate_univariate_model, create_datapoints_univariate
# from torch_multivariate_RNN import instantiate_multivariate_model, create_datapoints_multivariate

def optuna_log(message):
    """
    Log messages to a file with a timestamp.
    """
    with open(f"optuna_log_{log_id}.txt", "a") as f:
        f.write(message)

def objective(trial, md_train_stationary, qd_train_stationary, model, create_datapoints, trial_context, optimize_feature_selection=True, model_description=None):
    # Define the hyperparameter search space
    start = datetime.datetime.now()
    optuna_log(f"\nTrial {trial.number} at {start.strftime("%Y-%m-%d %H:%M:%S")}:\n")

    if optimize_feature_selection:
        feature_selection_method = trial.suggest_categorical("feature_selection_method", ["lasso"])
        n_features = trial.suggest_int("n_features", 9, 40)
    else:
        feature_selection_method = "none"
        n_features = 1

    dropout_rate = trial.suggest_float("dropout_rate", 0.2, 0.5)
    optimizer = trial.suggest_categorical("optimizer", ["adam"])
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    hidden_units = trial.suggest_categorical("hidden_units", [16, 32, 64])
    learning_rate = trial.suggest_float("learning_rate", 3e-3, 1e-2, log=True)
    add_recession_feature = False # trial.suggest_categorical("add_recession_feature", [True, False])
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
        results_for_hyperparameters = train_and_evaluate_model(
            md_train_stationary=md_train_stationary,
            qd_train_stationary=qd_train_stationary,
            md_test_stationary=None,
            qd_test_stationary=None,
            feature_selection_method=feature_selection_method,
            n_features=n_features,
            dropout_rate=dropout_rate,
            learning_rate=learning_rate,
            optimizer_name=optimizer,
            batch_size=batch_size,
            hidden_units=hidden_units,
            sequence_length=sequence_length,
            instantiate_model=model,
            create_datapoints=create_datapoints,
            add_recession_feature=add_recession_feature,
            max_epochs=40,
            model_description=model_description if model_description is not None else "Trial {trial.number}",
            store_results=False,
        )
        # Use the average of the lowest validation losses as the objective value
        mean_val = np.mean(results_for_hyperparameters["val_losses"])
        mean_train = np.mean(results_for_hyperparameters["train_losses"])

        end = datetime.datetime.now()
        duration = end - start
        minutes = duration.seconds // 60
        seconds = duration.seconds % 60

        optuna_log(f"Results: mean_val={mean_val}; mean_train={mean_train}; took {minutes}m{seconds}s\n")

        if mean_val < trial_context["best_score"]:
            trial_context["best_score"] = mean_val
            optuna_log(f"New best score: {mean_val}\n")

        return mean_val
    except Exception as e: # Handle any exceptions during training (e.g., invalid hyperparameters)

        # Print message
        print(f"\033[91mTrial failed with exception:\033[0m {e}")
        optuna_log(f"Trial failed with exception: {e}\n")

        # Print stack trace
        traceback.print_exc()
        optuna_log(traceback.format_exc())

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
        n_trials = 45

    # Load the training data
    md_train_stationary, qd_train_stationary = load_train_data()

    # Create the Optuna study with Bayesian optimization (TPE)
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler())

    optuna_log("\n----------------------------\n\n")

    optimize_feature_selection = True
    if model_id == '1':
        model = DuplicateQdModel
        create_datapoints = create_datapoints_mixed_frequency
    elif model_id == '2':
        model = MultilayerModel
        create_datapoints = create_datapoints_mixed_frequency
    elif model_id == '3':
        model = AlternatingRnnModel
        create_datapoints = create_datapoints_mixed_frequency
    elif model_id == 'univariate':
        model = UnivariateModel
        create_datapoints = create_datapoints_univariate
        optimize_feature_selection = False
    elif model_id == 'multivariate':
        model = MultivariateModel
        print("Using multivariate model")
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
        partial(objective, md_train_stationary=md_train_stationary, qd_train_stationary=qd_train_stationary, model=partial(model, Rnn=rnn), create_datapoints=create_datapoints, optimize_feature_selection=optimize_feature_selection, trial_context=trial_context, model_description=f"{model_id} {rnn_name}"),
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
