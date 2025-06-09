#!/usr/bin/env python3
"""
PyTorch re‑implementation of the original TensorFlow/Keras script.

Major changes
-------------
* Replaced all TensorFlow/Keras layers with native PyTorch `nn` modules.
* Training loop is now explicit (no `model.fit`); includes early‑stopping logic.
* Metrics are computed manually each epoch with NumPy helpers.
* Random seed control now uses `torch.manual_seed` in addition to NumPy & `random`.
* Model factory functions return `nn.Module` instances.
* Utility functions (`create_datapoints_*`, plotting, feature selection, etc.) stay
  intact where they did not rely on TensorFlow.

The public interface – i.e. the various `configs` dict keys such as
`lstm1`, `gru2`, etc. – is preserved so that the script can still be called with
`--eval <model>` from the command line.

Nota bene:
* Upsampling of quarterly series to monthly frequency is implemented with a
  learnable‑free nearest‑neighbour up‑sample (`nn.Upsample`).
* `TimeDistributed(Dense)` was replaced by a linear layer applied per time step.
  Because PyTorch tensors are contiguous, we reshape and apply `nn.Linear`.
* Early stopping uses patience on *validation* loss; best weights are cached
  locally and restored when patience is exceeded.
* CUDA is supported – the script auto‑detects `torch.cuda.is_available()` and
  moves the model & tensors to GPU if requested via `--cuda` flag.

Feel free to adjust hyper‑parameters or extend the models.
"""

from __future__ import annotations

import argparse
import datetime
import math
import os
import random
from functools import partial
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error
from sklearn.preprocessing import StandardScaler

from feature_selection import remove_highly_correlated_features, lasso_feature_selection, tree_based_feature_selection, rfe_feature_selection, pca_feature_selection, combine_qd_with_summed_md, expert_selection
from tqdm import tqdm
from torch_models import BaseRnn

# -----------------------------------------------------------------------------
# Random‑seed helper
# -----------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# -----------------------------------------------------------------------------
# Model definitions
# -----------------------------------------------------------------------------

class DuplicateQdModel(BaseRnn):
    def __init__(
        self,
        seq_len: int,
        n_monthly: int,
        n_quarterly: int,
        hidden: int,
        dropout: float,
        Rnn: type[nn.RNNBase] = nn.LSTM,
        n_layers: int = 1,
    ) -> None:
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=3, mode="nearest")
        self.rnn = Rnn(n_monthly + n_quarterly + 1, hidden, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden, 1)

    def forward(
        self,
        x_monthly: torch.Tensor,   # (B, Q*3, n_monthly)
        x_quarterly: torch.Tensor, # (B, Q,   n_quarterly)
        x_shift: torch.Tensor,     # (B, Q*3, 1)
    ) -> torch.Tensor:           # (B, 3,   1)
        B = x_monthly.size(0)
        q_up = self.upsample(x_quarterly.transpose(1, 2)).transpose(1, 2)
        x = torch.cat((x_monthly, q_up, x_shift), dim=-1)
        x, _ = self.rnn(x)
        x = self.dropout(self._last_three_months(x))
        x = self._time_distributed(x, self.fc)

        assert x.shape == (B, 3, 1), f"Expected output shape ({B = }, 3, 1), got {x.shape}"
        x = x.squeeze(-1)
        assert x.shape == (B, 3), f"Expected final output shape ({B = }, 3), got {x.shape}"

        return x


class MultilayerModel(BaseRnn):
    def __init__(
        self,
        seq_len: int,
        n_monthly: int,
        n_quarterly: int,
        hidden: int,
        dropout: float,
        Rnn: type[nn.RNNBase] = nn.LSTM,
    ) -> None:
        super().__init__()
        self.quarter_rnn = Rnn(n_quarterly, hidden, batch_first=True)
        self.upsample = nn.Upsample(scale_factor=3, mode="nearest")
        self.monthly_rnn = Rnn(n_monthly + hidden + 1, hidden, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden, 1)

    def forward(
        self,
        x_monthly: torch.Tensor,   # (B, Q*3, n_monthly)
        x_quarterly: torch.Tensor, # (B, Q,   n_quarterly)
        x_shift: torch.Tensor,     # (B, Q*3, 1)
    ) -> torch.Tensor:           # (B, 3,   1)
        B = x_monthly.size(0)
        q_feat, _ = self.quarter_rnn(x_quarterly)
        q_up = self.upsample(q_feat.transpose(1, 2)).transpose(1, 2)
        x = torch.cat((x_monthly, q_up, x_shift), dim=-1)
        x, _ = self.monthly_rnn(x)
        x = self.dropout(self._last_three_months(x))
        x = self._time_distributed(x, self.fc)

        assert x.shape == (B, 3, 1), f"Expected output shape ({B = }, 3, 1), got {x.shape}"
        x = x.squeeze(-1)
        assert x.shape == (B, 3), f"Expected final output shape ({B = }, 3), got {x.shape}"

        return x


class AlternatingRnnModel(BaseRnn):
    """Implements the custom quarter‑month interleaved loop in plain PyTorch."""

    def __init__(
        self,
        seq_len: int,
        n_monthly: int,
        n_quarterly: int,
        hidden: int,
        dropout: float,
        Rnn: type[nn.RNNBase] = nn.LSTM,
    ) -> None:
        super().__init__()
        self.hidden = hidden
        self.rnn_quarter = Rnn(n_quarterly, hidden, batch_first=True)
        self.rnn_month   = Rnn(n_monthly + 1, hidden, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden, 1)
        self.seq_len = seq_len

        self.rnn_type = "LSTM" if isinstance(self.rnn_quarter, nn.LSTM) else "GRU" if isinstance(self.rnn_quarter, nn.GRU) else None
        if self.rnn_type is None:
            raise ValueError("Unsupported RNN type. Use torch.nn.LSTM or torch.nn.GRU.")

    def forward(self, x_monthly: torch.Tensor, x_quarterly: torch.Tensor, x_shift: torch.Tensor) -> torch.Tensor:  # noqa: D401
        B = x_monthly.size(0)

        zeros = torch.zeros((1, B, self.hidden), device=x_monthly.device)

        if self.rnn_type == "LSTM":
            hidden = (zeros, zeros)
        else:  # GRU
            hidden = zeros

        all_monthly = torch.cat((x_monthly, x_shift), dim=-1)

        for i in range(self.seq_len // 3):
            _output, hidden = self.rnn_quarter(x_quarterly[:, i : i + 1, :], hidden)

            # # Transpose to batch-first format
            # if not isinstance(hidden, tuple):
            #     hidden = hidden.transpose(0, 1)
            # else:
            #     hidden = tuple(h.transpose(0, 1) for h in hidden)

            if not isinstance(hidden, tuple):
                assert hidden.shape == (1, B, self.hidden), f"Expected quarterly hidden shape (1, {B = }, {self.hidden}), got {hidden.shape}"
            else:
                for i, h in enumerate(hidden):
                    assert h.shape == (1, B, self.hidden), f"Expected quarterly hidden[{i}] shape (1, {B = }, {self.hidden}), got {h.shape}"

            # assert hidden.shape == (B, 3, self.hidden), f"Expected hidden shape ({B = }, 3, {self.hidden}), got {hidden.shape}"

            m_slice = all_monthly[:, i * 3 : i * 3 + 3, :]
            output, hidden = self.rnn_month(m_slice, hidden)

            # # Transpose to batch-first format
            # if not isinstance(hidden, tuple):
            #     hidden = hidden.transpose(0, 1)
            # else:
            #     hidden = tuple(o.transpose(0, 1) for o in hidden)
            
            # assert hidden.shape == (B, 3, self.hidden), f"Expected hidden shape ({B = }, 3, {self.hidden}), got {hidden.shape}"
            
            if not isinstance(hidden, tuple):
                assert hidden.shape == (1, B, self.hidden), f"Expected monthly hidden shape (1, {B = }, {self.hidden}), got {hidden.shape}"
            else:
                for i, h in enumerate(hidden):
                    assert h.shape == (1, B, self.hidden), f"Expected monthly hidden[{i}] shape (1, {B = }, {self.hidden}), got {h.shape}"

        # Output is now (B, 3, hidden)

        x = self.dropout(output)
        x = self._time_distributed(output, self.fc)

        assert x.shape == (B, 3, 1), f"Expected output shape ({B = }, 3, 1), got {x.shape}"
        x = x.squeeze(-1)
        assert x.shape == (B, 3), f"Expected final output shape ({B = }, 3), got {x.shape}"

        return x


# -----------------------------------------------------------------------------
# Data helpers (unchanged from original except TensorFlow → torch tensors)
# -----------------------------------------------------------------------------

def create_datapoints_mixed_frequency(
    monthly: pd.DataFrame,
    quarterly: pd.DataFrame,
    sequence_length: int,
):
    seq_len_q = sequence_length // 3
    X_md, X_qd, X_shift, y, y_dates = [], [], [], [], []

    for q in range(len(quarterly) - seq_len_q):
        m = q * 3 + 2

        X_md   .append( monthly.iloc[m : m + sequence_length].values       )
        X_qd   .append( quarterly.iloc[q : q + seq_len_q].values           )
        X_shift.append( (np.arange(0, sequence_length) % 3).reshape(-1, 1) )
        y      .append( quarterly.iloc[q + seq_len_q][["GDPC1"]]           )
        y_dates.append( quarterly.index[q + seq_len_q]                     )

    return [np.asarray(d, dtype=np.float32) for d in [X_md, X_qd, X_shift, y]] + [y_dates]

# -----------------------------------------------------------------------------
# Training helpers
# -----------------------------------------------------------------------------

def _torchify(*arrays, device):
    return [torch.from_numpy(a).to(device) for a in arrays]


class EarlyStopping:
    def __init__(self, patience: int = 5):
        self.patience = patience
        self.counter = 0
        self.best_loss = float("inf")
        self.best_state: Dict[str, torch.Tensor] | None = None

    def step(self, loss: float, model: nn.Module) -> bool:
        if loss < self.best_loss:
            self.best_loss = loss
            self.best_state = {k: v.clone() for k, v in model.state_dict().items()}
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience

    def restore(self, model: nn.Module) -> None:
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


# -----------------------------------------------------------------------------
# Main training function (truncated for brevity)
# -----------------------------------------------------------------------------

def train_and_evaluate_model(
    md_train_stationary: pd.DataFrame,
    qd_train_stationary: pd.DataFrame,
    feature_selection_method: str,
    n_features: int,
    dropout_rate: float,
    optimizer_name: str,
    batch_size: int,
    hidden_units: int,
    sequence_length: int,
    learning_rate: float,
    add_recession_feature: bool = False,
    instantiate_model: Callable[..., BaseRnn] = DuplicateQdModel,
    create_datapoints=create_datapoints_mixed_frequency,
    max_epochs: int = 40,
    verbose: bool = False,
    model_description: str | None = None,
    md_test_stationary: pd.DataFrame | None = None,
    qd_test_stationary: pd.DataFrame | None = None,
    device: torch.device | None = None,
    fast_mode: bool = False,
    store_results=True,
) -> Dict[str, np.ndarray]:
    start_of_train_run = datetime.datetime.now()
    start_of_train_run_str = start_of_train_run.strftime("%Y-%m-%d %Hh%Mm%Ss")

    if model_description is None:
        model_description = instantiate_model.__name__

    if store_results:
        results_dir = Path("results") / f"train_eval {start_of_train_run_str} {model_description}"
        results_dir.mkdir(exist_ok=True, parents=True)
        text_results = results_dir / "results.txt"

        def append_test_results(text: str) -> None:
            with text_results.open("a") as f:
                f.write(text + "\n")

        append_test_results(f"Training started at {start_of_train_run_str}")

        if verbose:
            print(f"Results will be stored in {results_dir}")

    md_train_stationary = md_train_stationary.copy()
    qd_train_stationary = qd_train_stationary.copy()

    if md_test_stationary is not None:
        md_test_stationary = md_test_stationary.copy()
    if qd_test_stationary is not None:
        qd_test_stationary = qd_test_stationary.copy()

    if fast_mode:
        print("Running in fast mode")
        md_train_stationary = md_train_stationary.iloc[:300]
        qd_train_stationary = qd_train_stationary.iloc[:100]
        max_epochs = 5

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    has_test_data = md_test_stationary is not None and qd_test_stationary is not None

    base_seed = 2571267 # + 12093871 # + 31
    set_seed(base_seed)

    param_dict = {
        "feature_selection_method": feature_selection_method,
        "n_features": n_features,
        "dropout_rate": dropout_rate,
        "optimizer": optimizer_name,
        "batch_size": batch_size,
        "hidden_units": hidden_units,
        "sequence_length": sequence_length,
        "learning_rate": learning_rate,
        "max_epochs": max_epochs,
        "add_recession_feature": add_recession_feature,
        "fast_mode": fast_mode,
        "model": model_description
    }
    if verbose:
        print("Parameters:", param_dict)

    if store_results:
        append_test_results("Parameters:\n" + "\n".join(f"{k}: {v}" for k, v in param_dict.items()))

    # Model 1 preprocessing: repeating GDP values
    # if feature_selection_method == "filter_highly_correlated":
    #     selected_features = remove_highly_correlated_features(combine_qd_and_md_as_qd(qd_train_stationary, md_train_stationary))
    # elif feature_selection_method == 'lasso':
    #     # Perform LASSO feature selection
    #     selected_features, _ = lasso_feature_selection(combine_qd_and_md_as_qd(qd_train_stationary, md_train_stationary), n_features=n_features)
    # elif feature_selection_method == "random_forest":
    #     # Perform Random Forest feature selection
    #     selected_features, _ = tree_based_feature_selection(combine_qd_and_md_as_qd(qd_train_stationary, md_train_stationary), n_features=n_features)
    # elif feature_selection_method == "rfe":
    #     # Perform Random Forest feature selection
    #     selected_features, _ = rfe_feature_selection(combine_qd_and_md_as_qd(qd_train_stationary, md_train_stationary), n_features=n_features)
    # elif feature_selection_method == "none":
    #     # Use all features
    #     selected_features = combine_qd_and_md_as_qd(qd_train_stationary, md_train_stationary).columns.tolist()
    selected_features = None
    if feature_selection_method == "expert_selection":
        # Use expert selection
        selected_features = expert_selection()
    if feature_selection_method == "filter_highly_correlated":
        selected_features = remove_highly_correlated_features(combine_qd_with_summed_md(qd_train_stationary, md_train_stationary))
    elif feature_selection_method == 'lasso':
        # Perform LASSO feature selection
        selected_features, _ = lasso_feature_selection(combine_qd_with_summed_md(qd_train_stationary, md_train_stationary), n_features=n_features)
    elif feature_selection_method == "random_forest":
        # Perform Random Forest feature selection
        selected_features, _ = tree_based_feature_selection(combine_qd_with_summed_md(qd_train_stationary, md_train_stationary), n_features=n_features)
    elif feature_selection_method == "rfe":
        # Perform Random Forest feature selection
        selected_features, _ = rfe_feature_selection(combine_qd_with_summed_md(qd_train_stationary, md_train_stationary), n_features=n_features)
    elif feature_selection_method == "pca":
        train_val_quarterly, train_val_monthly = pca_feature_selection(qd_train_stationary, md_train_stationary, n_features= n_features)
    elif feature_selection_method == "none":
        # Use all features
        selected_features = combine_qd_with_summed_md(qd_train_stationary, md_train_stationary).columns.tolist()

    if verbose:
        print("Selected features:", selected_features)

    # Separate the selected features into monthly and quarterly features
    if selected_features is not None:
        monthly_features = [feature for feature in selected_features if feature in md_train_stationary.columns]
        quarterly_features = [feature for feature in selected_features if feature in qd_train_stationary.columns]

        if len(monthly_features) == 0:
            monthly_features.append("HOUSTW")

        if "GDPC1" not in quarterly_features:
            quarterly_features.append("GDPC1")

        # Create separate DataFrames for monthly and quarterly selected features
        train_val_monthly = md_train_stationary[monthly_features].copy()
        train_val_quarterly = qd_train_stationary[quarterly_features].copy()
        test_monthly = md_test_stationary[monthly_features].copy() if has_test_data else None
        test_quarterly = qd_test_stationary[quarterly_features].copy() if has_test_data else None

    if add_recession_feature:
        # Define recession periods from NBER dates
        recessions = [
            ('1960-04-01', '1961-02-01'),
            ('1969-12-01', '1970-11-01'),
            ('1973-11-01', '1975-03-01'),
            ('1980-01-01', '1980-07-01'),
            ('1981-07-01', '1982-11-01'),
            ('1990-07-01', '1991-03-01'),
            ('2001-03-01', '2001-11-01'),
            ('2007-12-01', '2009-06-01'),
            ('2020-02-01', '2020-04-01')
        ]
        # Convert recession dates to datetime
        recessions = [(pd.to_datetime(start), pd.to_datetime(end)) for start, end in recessions]

        # Add recession binary feature to monthly data
        recession_binary = []
        for date in train_val_monthly.index:
            is_recession = any(start <= date <= end for start, end in recessions)
            recession_binary.append(1 if is_recession else 0)

        train_val_monthly['Recession'] = recession_binary

        test_recession_binary = []
        if has_test_data:
            for date in test_monthly.index:
                is_recession = any(start <= date <= end for start, end in recessions)
                test_recession_binary.append(1 if is_recession else 0)

            test_monthly['Recession'] = test_recession_binary

    if verbose:
        # Print the separated features for verification
        print("Selected Monthly Features:")
        print(train_val_monthly.columns)

        print("Selected Quarterly Features:")
        print(train_val_quarterly.columns)

    if store_results:
        append_test_results(f"Tried to select {n_features} features using {feature_selection_method} method")
        append_test_results(f"Selected Features: {selected_features}, n={len(selected_features)}")
        append_test_results(f"Selected Monthly Features: {list(train_val_monthly.columns)}, n={len(train_val_monthly.columns)}")
        append_test_results(f"Selected Quarterly Features: {list(train_val_quarterly.columns)}, n={len(train_val_quarterly.columns)}")

    if not fast_mode:
        initial_train_size = 20
        val_size = 10
        full_train_val_size = 100
    else:
        initial_train_size = 40
        val_size = 30
        full_train_val_size = 100


    lowest_validation_losses = []
    lowest_train_losses = []
    validation_predictions_per_split = []

    loss_plot_dir = f"results/loss_plot {start_of_train_run_str}"
    if verbose:
        os.makedirs(loss_plot_dir, exist_ok=True)

    number_of_quarters = len(train_val_quarterly)
    if verbose:
        print("n:", number_of_quarters)

    best_epoch = {}
    test_predictions = []

    val_losses_per_split = []
    train_losses_per_split = []
    test_losses = []

    val_predictions = []
    train_predictions = []
    test_predictions = []

    for train_proportion in range(initial_train_size, full_train_val_size, val_size):
        set_seed(base_seed + (hash(train_proportion) + hash(91647)) % 100_000_000)

        should_eval_test_set = has_test_data and train_proportion + val_size >= full_train_val_size
        best_epoch[train_proportion] = []
        validation_predictions_current_split = []

        # Split training and validation sets
        end_train_data_idx = int(number_of_quarters * train_proportion / full_train_val_size)
        end_val_data_idx = int(number_of_quarters * (train_proportion + val_size) / full_train_val_size)
        if verbose:
            print(f"end_train_data_idx: {end_train_data_idx}, end_val_data_idx: {end_val_data_idx}, number of quarters in val: {end_val_data_idx - end_train_data_idx}")

        train_monthly = train_val_monthly.iloc[:end_train_data_idx * 3]
        val_monthly = train_val_monthly.iloc[end_train_data_idx * 3:end_val_data_idx * 3]

        train_quarterly = train_val_quarterly.iloc[:end_train_data_idx]
        val_quarterly = train_val_quarterly.iloc[end_train_data_idx:end_val_data_idx]

        month_start_training_data = train_monthly.index[0].strftime("%Y-%m")
        month_end_training_data = train_monthly.index[-1].strftime("%Y-%m")
        month_end_validation_data = val_monthly.index[-1].strftime("%Y-%m")

        if verbose:
            print(f"Processing train-val split: start_train = {month_start_training_data}; end_train & start_val = {month_end_training_data}; val_end = {month_end_validation_data}")

            # Confirm shapes
            print(f"Train proportion: {train_proportion}%")
            print("Monthly:", train_monthly.shape, val_monthly.shape)
            print("Quarterly:", train_quarterly.shape, val_quarterly.shape)

        # Initialize the scaler
        scaler_md = StandardScaler()
        scaler_qd = StandardScaler()
        scaler_gdp = StandardScaler()

        # Fit the scaler on the training data and transform training, validation data
        train_monthly_scaled = pd.DataFrame(scaler_md.fit_transform(train_monthly),
                                            index=train_monthly.index,
                                            columns=train_monthly.columns)

        val_monthly_scaled = pd.DataFrame(scaler_md.transform(val_monthly),
                                            index=val_monthly.index,
                                            columns=val_monthly.columns)

        if should_eval_test_set:
            test_monthly_scaled = pd.DataFrame(scaler_md.transform(test_monthly),
                                                index=test_monthly.index,
                                                columns=test_monthly.columns)

        train_quarterly_scaled = pd.concat(
            [
                pd.DataFrame(
                    scaler_qd.fit_transform(train_quarterly.drop(columns=["GDPC1"])),
                    index=train_quarterly.index,
                    columns=[c for c in train_quarterly.columns if c != "GDPC1"]
                ),
                pd.DataFrame(
                    scaler_gdp.fit_transform(train_quarterly[["GDPC1"]]),
                    index=train_quarterly.index,
                    columns=["GDPC1"]
                ),
            ],
            axis=1
        )

        val_quarterly_scaled = pd.concat(
            [
                pd.DataFrame(
                    scaler_qd.transform(val_quarterly.drop(columns=["GDPC1"])),
                    index=val_quarterly.index,
                    columns=[c for c in val_quarterly.columns if c != "GDPC1"]
                ),
                pd.DataFrame(
                    scaler_gdp.transform(val_quarterly[["GDPC1"]]),
                    index=val_quarterly.index,
                    columns=["GDPC1"]
                ),
            ],
            axis=1
        )

        if should_eval_test_set:
            test_quarterly_scaled = pd.concat(
                [
                    pd.DataFrame(
                        scaler_qd.transform(test_quarterly.drop(columns=["GDPC1"])),
                        index=test_quarterly.index,
                        columns=[c for c in test_quarterly.columns if c != "GDPC1"]
                    ),
                    pd.DataFrame(
                        scaler_gdp.transform(test_quarterly[["GDPC1"]]),
                        index=test_quarterly.index,
                        columns=["GDPC1"]
                    ),
                ],
                axis=1
            )

        # Confirm shapes
        if verbose:
            print("Monthly Scaled:", train_monthly_scaled.shape, val_monthly_scaled.shape)
            print("Quarterly Scaled:", train_quarterly_scaled.shape, val_quarterly_scaled.shape)

            # plot_scaled_data(train_monthly_scaled, val_monthly_scaled, train_quarterly_scaled, val_quarterly_scaled)

        x_train_monthly, x_train_quarterly, x_train_shift, y_train, _ = create_datapoints(train_monthly_scaled, train_quarterly_scaled, sequence_length)
        x_train_monthly, x_train_quarterly, x_train_shift, y_train = _torchify(x_train_monthly, x_train_quarterly, x_train_shift, y_train, device=device)
        if verbose:
            print("x_train_md shape:", x_train_monthly.shape)
            print("x_train_qd shape:", x_train_quarterly.shape)
            print("y_train shape:", y_train.shape)

        x_val_monthly, x_val_quarterly, x_val_shift, y_val, val_dates = create_datapoints(val_monthly_scaled, val_quarterly_scaled, sequence_length)
        x_val_monthly, x_val_quarterly, x_val_shift, y_val = _torchify(x_val_monthly, x_val_quarterly, x_val_shift, y_val, device=device)

        if should_eval_test_set:
            x_test_monthly, x_test_quarterly, x_test_shift, y_test, test_dates = create_datapoints(test_monthly_scaled, test_quarterly_scaled, sequence_length)
            x_test_monthly, x_test_quarterly, x_test_shift, y_test = _torchify(x_test_monthly, x_test_quarterly, x_test_shift, y_test, device=device)

        # x_test_monthly, x_test_quarterly, x_test_shift, y_test, test_dates = create_datapoints(val_monthly_scaled, val_quarterly_scaled, sequence_length)

        if verbose:
            print("x_val_md shape:", x_val_monthly.shape)
            print("x_val_qd shape:", x_val_quarterly.shape)
            print("y_val shape:", y_val.shape)

        if not fast_mode:
            training_repetition_count = 5
            median_idx = 2
        else:
            training_repetition_count = 3
            median_idx = 1
        val_losses_current_split = [] # we are gonna train it 5 times and take the average of the second and third best model for increased stability
        val_predictions_current_split = []
        train_losses_current_split = []
        train_predictions_current_split = []

        for training_repetition_idx in range(training_repetition_count):
            # ...  (feature selection & preprocessing identical to original – omitted)

            # Example: build one model & train on entire dataset --------------------------------

            set_seed(base_seed + (hash(training_repetition_idx) + hash(89520)) % 100_000_000)

            model = instantiate_model(
                sequence_length,
                x_train_monthly.shape[2],
                x_train_quarterly.shape[2],
                hidden_units,
                dropout_rate,
            ).to(device)

            criterion = nn.MSELoss()
            if optimizer_name.lower() == "adam":
                optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            elif optimizer_name.lower() == "sgd":
                optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
            elif optimizer_name.lower() == "adagrad":
                optimizer = optim.Adagrad(model.parameters(), lr=learning_rate)
            elif optimizer_name.lower() == "adamw":
                optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
            elif optimizer_name.lower() == "rmsprop":
                optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, momentum=0.9)
            else:
                raise ValueError(f"Unknown optimizer: {optimizer_name}")

            early_stop = EarlyStopping(patience=5)

            n_batches = math.ceil(x_train_monthly.size(0) / batch_size)
            if verbose:
                print(f"{n_batches=}")

            for epoch in (list(range(max_epochs))):
                model.train()
                idx = torch.randperm(x_train_monthly.size(0))
                epoch_loss = 0.0
                for i in range(n_batches):
                    batch_indices = idx[i * batch_size:(i + 1) * batch_size]
                    m_b, q_b, s_b, y_b = (t[batch_indices] for t in (x_train_monthly, x_train_quarterly, x_train_shift, y_train))
                    optimizer.zero_grad()
                    pred = model(m_b, q_b, s_b)

                    # print(f"{y_t.shape=}, {pred.shape=}")
                    y_b = y_b.repeat((1, pred.shape[-1]))
                    loss = criterion(pred, y_b)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()

                (val_loss,), val_pred = model.compare_predictions(
                    x_qd=x_val_quarterly,
                    x_md=x_val_monthly,
                    x_shift=x_val_shift,
                    y_true=y_val,
                    metrics=[mean_squared_error],
                )
                if verbose:
                    print(f"Epoch {epoch+1:02d}/{max_epochs} – val_loss: {val_loss:.4f}")

                if early_stop.step(val_loss, model):
                    if verbose:
                        print("Early stopping triggered.")
                    break

            early_stop.restore(model)
            if verbose:
                print("Last epoch idx:", epoch)

            # Final evaluation
            model.eval()

            evaluation = {}

            with torch.no_grad():
                for split_name, x_monthly, x_quarterly, x_shift, y in [
                    ("train", x_train_monthly, x_train_quarterly, x_train_shift, y_train),
                    ("val", x_val_monthly, x_val_quarterly, x_val_shift, y_val),
                ] + ([] if not should_eval_test_set else [("test", x_test_monthly, x_test_quarterly, x_test_shift, y_test)]):
                    (mse, rmse, mae), pred = model.compare_predictions(x_monthly, x_quarterly, x_shift, y, [mean_squared_error, root_mean_squared_error, mean_absolute_error])

                    if verbose:
                        print(f"{split_name} MSE:", mse)
                        print(f"{split_name} RMSE:", rmse)
                        print(f"{split_name} MAE :", mae)

                    evaluation[split_name] = {
                        'mse': mse,
                        'rmse': rmse,
                        'mae': mae,
                        'predictions': pred,
                        'predictions_unscaled': scaler_gdp.inverse_transform(pred)
                    }
            
            val_losses_current_split.append(evaluation["val"]['mse'])
            val_predictions_current_split.append(evaluation["val"]['predictions'])

            train_losses_current_split.append(evaluation["train"]['mse'])
            train_predictions_current_split.append(evaluation["train"]['predictions'])

            if should_eval_test_set:
                test_losses.append(evaluation["test"]['mse'])
                test_predictions.append(evaluation["test"]['predictions'])
        
        # End of training repetitions for this split
        
        val_losses_per_split.append(val_losses_current_split)
        train_losses_per_split.append(train_losses_current_split)

    end_of_train_run = datetime.datetime.now()
    elapsed_time = end_of_train_run - start_of_train_run
    elapsed_time_str = str(elapsed_time).split(".")[0]  # Remove microseconds

    print(f"Training completed in {elapsed_time_str}.")

    return {
        'val_losses': np.array(val_losses_per_split),
        'train_losses': np.array(train_losses_per_split),
        'test_losses': np.array(test_losses)
    }

    # -------------------------------------------------------------------------
    # The rest of the original function – cross‑validation, plotting, logging –
    # can be adapted in similar fashion but is omitted here for brevity.
    # -------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# Entry‑point CLI wrapper (subset of original functionality)
# -----------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", required=True, help="Model key to evaluate (e.g. lstm1, gru2, ...)")
    parser.add_argument("--cuda", action="store_true", help="Force CUDA if available")
    parser.add_argument("--fast-mode", "--fast", action="store_true", help="Run in fast mode for quick testing")
    args = parser.parse_args()

    from check_stationarity import load_train_data, load_test_data
    from multivariate_RNN import instantiate_multivariate_model, create_datapoints_multivariate
    from univariate_RNN import instantiate_univariate_model, create_datapoints_univariate

    md_train_stationary, qd_train_stationary = load_train_data()
    md_test_stationary, qd_test_stationary = load_test_data()

    configs = {
        # Only LSTM & GRU configs shown; same hyper‑parameters as before
        "lstm1": dict(
            feature_selection_method="lasso",
            # feature_selection_method="expert_selection",
            n_features=18,
            dropout_rate=0.2586783399110758,
            optimizer_name="adam",
            batch_size=16,
            hidden_units=32,
            learning_rate=0.008420118343459052,
            add_recession_feature=False,
            instantiate_model=partial(DuplicateQdModel, rnn_cls=nn.LSTM),
            model_description="LSTM1",
            create_datapoints=create_datapoints_mixed_frequency,
        ),
        # ... (other entries identical, swapping rnn_cls between nn.GRU & nn.LSTM)
    }

    if args.eval not in configs:
        raise SystemExit(f"Unknown model key '{args.eval}'. Available: {', '.join(configs)}")

    cfg = configs[args.eval]
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")

    train_and_evaluate_model(
        md_train_stationary,
        qd_train_stationary,
        md_test_stationary=md_test_stationary,
        qd_test_stationary=qd_test_stationary,
        sequence_length=12,
        verbose=False,
        device=device,
        fast_mode=args.fast_mode,
        **cfg,
    )


if __name__ == "__main__":
    main()
