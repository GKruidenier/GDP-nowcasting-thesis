# import tensorflow as tf; tf.compat.v1.enable_eager_execution()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import root_mean_squared_error, mean_squared_error, mean_absolute_error

import tensorflow as tf
import keras
from keras import layers
from keras.layers import LSTM, Dense, Concatenate, TimeDistributed
from keras.callbacks import EarlyStopping

import os
import datetime
from functools import partial
import random
import math

# from stationarization import load_train_data
from check_stationarity import load_train_data, load_test_data
from feature_selection import remove_highly_correlated_features, lasso_feature_selection, tree_based_feature_selection, rfe_feature_selection, pca_feature_selection, combine_qd_with_summed_md

should_plot = False
fast_mode = False

metrics = [
    keras.metrics.RootMeanSquaredError(name='rmse'),
    keras.metrics.MeanAbsoluteError(name='mae')
]

# Set the random seed for reproducibility
def set_seed(seed):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)

def create_datapoints_lstm_1_2_and_3(monthly, quarterly, sequence_length):
    sequence_length_quarterly = sequence_length // 3

    x_monthly = []
    x_quarterly = []
    x_shift = []
    y = []
    y_dates = []

    for q in range(len(quarterly) - sequence_length_quarterly):
        m = q * 3 + 2
        m_end = m + sequence_length
        q_end = q + sequence_length_quarterly

        # assert monthly.index[m] == quarterly.index[q], f"Monthly start index {monthly.index[m]} does not match Quarterly start index {quarterly.index[q]}"
        # assert monthly.index[m_end] == quarterly.index[q_end], f"Monthly end index {monthly.index[m_end]} does not match Quarterly end index {quarterly.index[q_end]}"

        x_monthly.append(
            monthly.iloc[m:m_end].values
        )
        x_quarterly.append(
            quarterly.iloc[q:q_end].values
        )
        x_shift.append(
            np.arange(0, sequence_length).reshape(-1, 1) % 3
        )
        y.append(
            quarterly.iloc[q_end][["GDPC1"]]
        )
        y_dates.append(
            quarterly.index[q_end]
        )


    return np.array(x_monthly), np.array(x_quarterly), np.array(x_shift), np.array(y), y_dates

def instantiate_model_duplicate_qd(sequence_length, n_monthly_features, n_quarterly_features, n_hidden_units, optimizer_name, learning_rate, droupout_rate, Rnn=LSTM):
    monthly_input = layers.Input(shape=(sequence_length, n_monthly_features))
    quarterly_input = layers.Input(shape=(sequence_length // 3, n_quarterly_features))  # Quarterly is 3x slower
    shift_input = layers.Input(shape=(sequence_length, 1), dtype=tf.int32)

    quarterly_upsampled = layers.UpSampling1D(size=3)(quarterly_input)
    combined_input = Concatenate(axis=-1)([monthly_input, quarterly_upsampled, shift_input])

    rnn = Rnn(n_hidden_units, return_sequences=True)(combined_input)

    last_three_months = layers.Lambda(lambda x: x[:, -3:, :])(rnn)

    dropout = TimeDistributed(layers.Dropout(droupout_rate))(last_three_months)
    output = TimeDistributed(layers.Dense(1))(dropout)

    model = tf.keras.Model(inputs=[monthly_input, quarterly_input, shift_input], outputs=output)

    if optimizer_name == 'adam':
        optimizer_instance = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer_name == 'RMSprop':
        optimizer_instance = tf.keras.optimizers.SGD(learning_rate=learning_rate)

    model.compile(optimizer=optimizer_instance, loss="mse", metrics=metrics)

    return model


def instantiate_model_multilayer(sequence_length, n_monthly_features, n_quarterly_features, n_hidden_units, optimizer_name, learning_rate, droupout_rate, Rnn=LSTM):
    # Define model
    monthly_input = layers.Input(shape=(sequence_length, n_monthly_features))
    quarterly_input = layers.Input(shape=(sequence_length // 3, n_quarterly_features))  # Quarterly is 3x slower
    shift_input = layers.Input(shape=(sequence_length, 1), dtype=tf.int32)

    quarterly_rnn = Rnn(n_hidden_units, return_sequences=True)(quarterly_input)

    quarterly_upsampled = layers.UpSampling1D(size=3)(quarterly_rnn)
    monthly_augmented_input = layers.Concatenate()([monthly_input, quarterly_upsampled, shift_input])

    monthly_rnn = Rnn(n_hidden_units, return_sequences=True)(monthly_augmented_input)
    last_three_months = layers.Lambda(lambda x: x[:, -3:, :])(monthly_rnn)

    dropout = TimeDistributed(layers.Dropout(droupout_rate))(last_three_months)
    output = TimeDistributed(layers.Dense(1))(dropout)

    model = tf.keras.Model(inputs=[monthly_input, quarterly_input, shift_input], outputs=output)

    if optimizer_name == 'adam':
        optimizer_instance = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer_name == 'RMSprop':
        optimizer_instance = tf.keras.optimizers.SGD(learning_rate=learning_rate)

    model.compile(optimizer=optimizer_instance, loss="mse", metrics=metrics)

    return model

def instantiate_model_alternating_rnn(sequence_length, n_monthly_features, n_quarterly_features, n_hidden_units, optimizer_name, learning_rate, droupout_rate, Rnn=LSTM):
    # Define model
    monthly_input = layers.Input(shape=(sequence_length, n_monthly_features))
    quarterly_input = layers.Input(shape=(sequence_length // 3, n_quarterly_features))  # Quarterly is 3x slower
    shift_input = layers.Input(shape=(sequence_length, 1), dtype=tf.int32)

    all_monthly_inputs = layers.Concatenate(axis=-1)([monthly_input, shift_input])

    quarterly_rnn = Rnn(n_hidden_units, return_sequences=False, return_state=True)
    monthly_rnn = Rnn(n_hidden_units, return_sequences=True, return_state=True)

    def get_initial_state(inputs):
        batch_size = tf.shape(inputs)[0]
        h0 = tf.zeros((batch_size, n_hidden_units))
        c0 = tf.zeros((batch_size, n_hidden_units))
        return h0, c0

    # Later inside your model logic:
    hidden_state, cell_state = layers.Lambda(lambda x: get_initial_state(x))(monthly_input)

    for i in range(sequence_length // 3):
        q = i
        m = i * 3
        o = quarterly_rnn(quarterly_input[:, q:q+1, :], initial_state=[hidden_state, cell_state])
        if len(o) == 3:
            _hq, hidden_state, cell_state = o
        elif len(o) == 2:
            hidden_state, cell_state = o

        o = monthly_rnn(all_monthly_inputs[:, m:m+3, :], initial_state=[hidden_state, cell_state])
        if len(o) == 3:
            rnn_output, hidden_state, cell_state = o
        elif len(o) == 2:
            rnn_output, cell_state = o
            hidden_state = rnn_output[:, -1, :]

        # o = monthly_rnn(all_monthly_inputs[:, m:m+1, :], initial_state=[hidden_state, cell_state])
        # if len(o) == 3:
        #     h1, hidden_state, cell_state = o
        # elif len(o) == 2:
        #     hidden_state, cell_state = o
        #     h1 = hidden_state

        # o = monthly_rnn(all_monthly_inputs[:, m+1:m+2, :], initial_state=[hidden_state, cell_state])
        # if len(o) == 3:
        #     h2, hidden_state, cell_state = o
        # elif len(o) == 2:
        #     hidden_state, cell_state = o
        #     h2 = hidden_state

        # o = monthly_rnn(all_monthly_inputs[:, m+2:m+3, :], initial_state=[hidden_state, cell_state])
        # if len(o) == 3:
        #     h3, hidden_state, cell_state = o
        # elif len(o) == 2:
        #     hidden_state, cell_state = o
        #     h3 = hidden_state

        # _hq, hidden_state, cell_state = quarterly_rnn(quarterly_input[:, q:q+1, :], initial_state=[hidden_state, cell_state])
        # h1, hidden_state, cell_state = monthly_rnn(all_monthly_inputs[:, m+0:m+1, :], initial_state=[hidden_state, cell_state])
        # h2, hidden_state, cell_state = monthly_rnn(all_monthly_inputs[:, m+1:m+2, :], initial_state=[hidden_state, cell_state])
        # h3, hidden_state, cell_state = monthly_rnn(all_monthly_inputs[:, m+2:m+3, :], initial_state=[hidden_state, cell_state])

    # last_three_months = layers.Lambda(lambda hidden_states: tf.stack(hidden_states, axis = 1))([h1, h2, h3])
    last_three_months = rnn_output

    dropout = TimeDistributed(layers.Dropout(droupout_rate))(last_three_months)
    output = TimeDistributed(layers.Dense(1))(dropout)

    model = tf.keras.Model(inputs=[monthly_input, quarterly_input, shift_input], outputs=output)

    if optimizer_name == 'adam':
        optimizer_instance = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer_name == 'RMSprop':
        optimizer_instance = tf.keras.optimizers.SGD(learning_rate=learning_rate)

    model.compile(optimizer=optimizer_instance, loss="mse", metrics=metrics)

    return model

def train_and_evaluate_model(md_train_stationary, qd_train_stationary, feature_selection_method, n_features, dropout_rate, optimizer, batch_size, hidden_units, sequence_length, learning_rate, add_recession_feature=True, instantiate_model=instantiate_model_multilayer, create_datapoints=create_datapoints_lstm_1_2_and_3, max_epochs=40, verbose=True, model_description=None, md_test_stationary=None, qd_test_stationary=None):
    if fast_mode:
        md_train_stationary = md_train_stationary.iloc[:300]
        qd_train_stationary = qd_train_stationary.iloc[:100]
        max_epochs = 5

    has_test_data = md_test_stationary is not None and qd_test_stationary is not None

    start_of_train_run = datetime.datetime.now().strftime("%Y-%m-%d %Hh%Mm%Ss")

    base_seed = 2571267 + 12093871
    set_seed(base_seed)

    param_dict = {
        "feature_selection_method": feature_selection_method,
        "n_features": n_features,
        "dropout_rate": dropout_rate,
        "optimizer": optimizer,
        "batch_size": batch_size,
        "hidden_units": hidden_units,
        "sequence_length": sequence_length,
        "learning_rate": learning_rate,
        "max_epochs": max_epochs,
        "add_recession_feature": add_recession_feature,
        "fast_mode": fast_mode,
        "model": model_description if model_description is not None else instantiate_model.__name__,
        "base_seed": base_seed,
    }
    if verbose:
        print("Parameters:", param_dict)

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
        train_val_monthly = md_train_stationary[monthly_features]
        train_val_quarterly = qd_train_stationary[quarterly_features]
        test_monthly = md_test_stationary[monthly_features] if has_test_data else None
        test_quarterly = qd_test_stationary[quarterly_features] if has_test_data else None

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

    loss_plot_dir = f"results/loss_plot {start_of_train_run} {model_description}"
    if verbose:
        os.makedirs(loss_plot_dir, exist_ok=True)
        descrption_file = loss_plot_dir + "/" + model_description + ".txt"
        with open(descrption_file, "w") as f:
            f.write("\n".join([f"{k}={v}" for k, v in param_dict.items()]))
            f.write("\n")

    number_of_quarters = len(train_val_quarterly)
    if verbose:
        print("n:", number_of_quarters)

    best_epoch = {}
    test_predictions = []
    train_predictions = []

    for train_proportion in range(initial_train_size, full_train_val_size, val_size):
        set_seed(base_seed + (hash(train_proportion) + hash(91647)) % 100_000_000)

        should_eval_test_set = has_test_data and train_proportion + val_size >= full_train_val_size
        best_epoch[train_proportion] = []
        validation_predictions_current_split = []

        tf.keras.backend.clear_session()
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

            plot_scaled_data(train_monthly_scaled, val_monthly_scaled, train_quarterly_scaled, val_quarterly_scaled)

        x_train_monthly, x_train_quarterly, x_train_shift, y_train, train_dates = create_datapoints(train_monthly_scaled, train_quarterly_scaled, sequence_length)
        if verbose:
            print("x_train_md shape:", x_train_monthly.shape)
            print("x_train_qd shape:", x_train_quarterly.shape)
            print("y_train shape:", y_train.shape)

        x_val_monthly, x_val_quarterly, x_val_shift, y_val, val_dates = create_datapoints(val_monthly_scaled, val_quarterly_scaled, sequence_length)

        if should_eval_test_set:
            x_test_monthly, x_test_quarterly, x_test_shift, y_test, test_dates = create_datapoints(test_monthly_scaled, test_quarterly_scaled, sequence_length)

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
        val_losses = [] # we are gonna train it 5 times and take the average of the second and third best model for increased stability
        train_losses = []

        for training_repetition_idx in range(training_repetition_count):
            set_seed(base_seed + (hash(training_repetition_idx) + hash(89520)) % 100_000_000)
            model = instantiate_model(
                sequence_length,
                x_train_monthly.shape[2],
                x_train_quarterly.shape[2],
                hidden_units,
                optimizer,
                learning_rate,
                dropout_rate,
            )

            if verbose:
                pass
                # model.summary()

            # Add EarlyStopping callback
            early_stopping = EarlyStopping(
                monitor="val_loss",         # Stop/restore based on the validation loss
                patience=5,                 # Stop training if no improvement for 5 epochs
                restore_best_weights=True   # Restore the best weights after stopping
            )

            initial_val_loss = model.evaluate([x_val_monthly, x_val_quarterly, x_val_shift], y_val, verbose=0)
            if verbose:
                print(f"Initial validation Loss: {initial_val_loss}")
            initial_train_loss = model.evaluate([x_train_monthly, x_train_quarterly, x_train_shift], y_train, verbose=0)
            if verbose:
                print(f"Initial train Loss: {initial_train_loss}")

            # print("x_train_md type:", type(x_train_monthly), "shape:", x_train_monthly.shape)
            # print("x_train_qd type:", type(x_train_quarterly), "shape:", x_train_quarterly.shape)
            # print("y_train type:", type(y_train), "shape:", y_train.shape)
            # print("x_val_md type:", type(x_val_monthly), "shape:", x_val_monthly.shape)
            # print("x_val_qd type:", type(x_val_quarterly), "shape:", x_val_quarterly.shape)
            # print("y_val type:", type(y_val), "shape:", y_val.shape)
            x_train_monthly = tf.convert_to_tensor(x_train_monthly, dtype=tf.float32)
            x_train_quarterly = tf.convert_to_tensor(x_train_quarterly, dtype=tf.float32)
            y_train = tf.convert_to_tensor(y_train, dtype=tf.float32)
            x_val_monthly = tf.convert_to_tensor(x_val_monthly, dtype=tf.float32)
            x_val_quarterly = tf.convert_to_tensor(x_val_quarterly, dtype=tf.float32)
            y_val = tf.convert_to_tensor(y_val, dtype=tf.float32)

            history = model.fit(
                [x_train_monthly, x_train_quarterly, x_train_shift],
                y_train,
                validation_data=([x_val_monthly, x_val_quarterly, x_val_shift], tf.repeat(y_val, 3, -1)),
                epochs=max_epochs,
                batch_size=batch_size,
                verbose=0,#1 if verbose else 0,
                shuffle=True,
                callbacks=[early_stopping],
            )

            plot_loss(loss_plot_dir, month_start_training_data, month_end_training_data, month_end_validation_data, history, verbose=verbose)

            # Evaluate the model on validation data
            val_loss = np.mean(model.evaluate([x_val_monthly, x_val_quarterly, x_val_shift], y_val, verbose=0))
            train_loss = np.mean(model.evaluate([x_train_monthly, x_train_quarterly, x_train_shift], y_train, verbose=0))
            val_losses.append(val_loss)
            train_losses.append(train_loss)

            if should_eval_test_set:
                # Evaluate the model on test data
                test_prediction = model.predict([x_test_monthly, x_test_quarterly, x_test_shift], verbose=0)
                if len(test_prediction.shape) == 3:
                    test_prediction = test_prediction[..., 0]
                test_prediction_repetitions = test_prediction.shape[1] // y_test.shape[1]

                print(f"Test prediction shape: {test_prediction.shape}")
                print(f"Test ground truth shape: {y_test.shape}")

                rmse = np.sqrt(mean_squared_error(np.repeat(y_test, test_prediction_repetitions, axis=-1), test_prediction))
                mae = mean_absolute_error(np.repeat(y_test, test_prediction_repetitions, axis=-1), test_prediction)
                test_predictions.append({
                    'predictions': scaler_gdp.inverse_transform(test_prediction.reshape(-1, 1)),
                    'ground_truth': scaler_gdp.inverse_transform(y_test),
                    'rmse': rmse,
                    'mae': mae,
                    'dates': test_dates,
                })

                # Evaluate the model on training data
                train_prediction = model.predict([x_train_monthly, x_train_quarterly, x_train_shift], verbose=0)
                if len(train_prediction.shape) == 3:
                    train_prediction = train_prediction[..., 0]
                train_prediction_repetitions = train_prediction.shape[1] // y_train.shape[1]

                print(f"Train prediction shape: {train_prediction.shape}")
                print(f"Train ground truth shape: {y_train.shape}")

                rmse = np.sqrt(mean_squared_error(np.repeat(y_train, train_prediction_repetitions, axis=-1), train_prediction))
                mae = mean_absolute_error(np.repeat(y_train, train_prediction_repetitions, axis=-1), train_prediction)
                train_predictions.append({
                    'predictions': scaler_gdp.inverse_transform(train_prediction.reshape(-1, 1)),
                    'ground_truth': scaler_gdp.inverse_transform(y_train),
                    'rmse': rmse,
                    'mae': mae,
                    'dates': train_dates,
                })

            # test_predictions = scaler_gdp.inverse_transform(model.predict([x_val_monthly, x_val_quarterly, x_val_shift], verbose=0).reshape(-1, 1))
            # test_ground_truth = scaler_gdp.inverse_transform(y_val)

            if verbose:
                print(f"Validation Loss: {val_loss}")
                print(f"Train Loss: {train_loss}")

            validation_predictions_current_split.append((
                val_loss,
                scaler_gdp.inverse_transform(model.predict([x_val_monthly, x_val_quarterly, x_val_shift], verbose=0).reshape(-1, 1)),
                scaler_gdp.inverse_transform(y_val),
                val_dates,
                train_loss,
            ))

            plot_y_train_val(y_train, y_val)

            best_epoch[train_proportion].append({
                'val_loss': val_loss,
                'epoch': early_stopping.stopped_epoch,
            })

        print(f"{val_losses=}")
        val_losses.sort()
        # lowest_validation_losses.append(sum(val_losses[1:-1]) / (training_repetition_count - 2))
        # lowest_validation_losses.append(np.sum(val_losses[1:-1], axis=0) / (training_repetition_count - 2))
        lowest_validation_losses.append(val_losses[median_idx])
        train_losses.sort()
        # lowest_train_losses.append(sum(train_losses[1:-1]) / (training_repetition_count - 2))
        lowest_train_losses.append(train_losses[median_idx])


        best_validation_run = 0
        best_validation_loss = float("inf")

        for i, val_loss in enumerate(val_losses):
            if val_loss < best_validation_loss:
                best_validation_loss = val_loss
                best_validation_run = i

        validation_predictions_current_split.sort(key=lambda x: x[0])
        validation_predictions_per_split.append(validation_predictions_current_split)

    median_test_predictions = test_predictions[best_validation_run]

    test_predictions.sort(key=lambda x: x['rmse'])
    train_predictions.sort(key=lambda x: x['rmse'])
    print(test_predictions)
    print(train_predictions)
    # median_test_predictions = test_predictions[median_idx]
    # median_test_predictions = test_predictions[1]
    # best_test_predictions = test_predictions[0]

    # index_of_best = next(i for i, test_prediction in enumerate(test_predictions) if test_prediction['rmse'] == median_test_predictions['rmse'])
    # print("Best test predictions index:", index_of_best)

    # if best_test_predictions['rmse'] == median_test_predictions['rmse']:
    #     print("Same!")

    from collections import defaultdict
    total_results = defaultdict(lambda: 0)

    for idx_medians in range(1, 4):
        validation_predictions_median = [v[idx_medians] for v in validation_predictions_per_split]
        median_test_predictions = test_predictions[idx_medians]
        medain_train_predictions = train_predictions[idx_medians]
        validation_predictions_diff_log = np.array([p for _, ps, _, _, _ in validation_predictions_median for p in ps]).flatten()
        validation_ground_truth_diff_log = np.array([y for _, _, ys, _, _ in validation_predictions_median for y in ys]).flatten()
        test_predictions_diff_log: np.ndarray = median_test_predictions['predictions'].flatten()
        test_ground_truth_diff_log: np.ndarray = median_test_predictions['ground_truth'].flatten()
        train_predictions_diff_log: np.ndarray = medain_train_predictions['predictions'].flatten()
        train_ground_truth_diff_log: np.ndarray = medain_train_predictions['ground_truth'].flatten()

        prediction_repetitions = validation_predictions_diff_log.shape[0] // validation_ground_truth_diff_log.shape[0]

        assert prediction_repetitions * validation_ground_truth_diff_log.shape[0] == validation_predictions_diff_log.shape[0], f"Length of val predictions is not an integer multiple of ground truth. Predictions counts: {validation_predictions_diff_log.shape[0]}, ground truth count: {validation_ground_truth_diff_log.shape[0]}"
        assert prediction_repetitions * test_ground_truth_diff_log.shape[0] == test_predictions_diff_log.shape[0], f"Length of test predictions is not {prediction_repetitions} times that of the ground truth. Predictions counts: {test_predictions_diff_log.shape[0]}, ground truth count: {test_ground_truth_diff_log.shape[0]}"
        assert prediction_repetitions * train_ground_truth_diff_log.shape[0] == train_predictions_diff_log.shape[0], f"Length of train predictions is not {prediction_repetitions} times that of the ground truth. Predictions counts: {train_predictions_diff_log.shape[0]}, ground truth count: {train_ground_truth_diff_log.shape[0]}"

        val_dates = [d for _, _, _, dates, _ in validation_predictions_median for d in dates]
        repeated_val_dates = [d + pd.DateOffset(months=month_offset) for _, _, _, dates, _ in validation_predictions_median for d in dates for month_offset in range(prediction_repetitions)]
        test_dates = median_test_predictions['dates']
        repeated_test_dates = [d + pd.DateOffset(months=month_offset) for d in test_dates for month_offset in range(prediction_repetitions)]
        train_dates = medain_train_predictions['dates']
        repeated_train_dates = [d + pd.DateOffset(months=month_offset) for d in train_dates for month_offset in range(prediction_repetitions)]

        # print(f"{repeated_val_dates=}")
        # print(f"{[dates for _, _, _, dates in validation_predictions_per_split]=}")

        # qd = pd.read_csv("FRED_qd_train_cleaned_transformed.csv", index_col=0, parse_dates=True, date_format="%Y-%m-%d")
        # if fast_mode:
        #     qd = qd.iloc[:100]
        # qd.index = pd.to_datetime(qd.index, format="%m/%d/%Y")
        # gdp_qd = qd["GDPC1"].copy()
        # gdp_qd: pd.Series = gdp_qd.loc[flattened_val_dates]
        # print("gdp_qd shape:", gdp_qd.values.shape, "val_dates length:", len(flattened_val_dates))

        # gdp_cumulative = gdp_qd.values.cumsum()

        if verbose:
            qd = pd.read_csv("FRED_QD.csv", index_col=0, parse_dates=True, date_format="%Y-%m-%d")[2:]
            qd.index = pd.to_datetime(qd.index, format="%m/%d/%Y")

            gdp_qd_val = qd["GDPC1"].copy()
            gdp_qd_val: pd.Series = gdp_qd_val.loc[[date - pd.DateOffset(months=3) for date in val_dates]]
            gdp_log_val = np.log(gdp_qd_val.values)

            gdp_qd_test = qd["GDPC1"].copy()
            gdp_qd_test: pd.Series = gdp_qd_test.loc[[date - pd.DateOffset(months=3) for date in test_dates]]
            gdp_log_test = np.log(gdp_qd_test.values)

            gdp_qd_train = qd["GDPC1"].copy()
            gdp_qd_train: pd.Series = gdp_qd_train.loc[[date - pd.DateOffset(months=3) for date in train_dates]]
            gdp_log_train = np.log(gdp_qd_train.values)

            validation_mean_diff_log = np.mean(validation_ground_truth_diff_log).repeat(validation_ground_truth_diff_log.shape[0])
            validation_predictions_usd = np.exp(validation_predictions_diff_log + gdp_log_val.repeat(prediction_repetitions))
            validation_ground_truth_usd = np.exp(validation_ground_truth_diff_log + gdp_log_val)
            validation_mean_predictions_usd = np.exp(validation_mean_diff_log + gdp_log_val)

            test_mean_diff_log = np.mean(test_ground_truth_diff_log).repeat(test_ground_truth_diff_log.shape[0])
            test_predictions_usd = np.exp(test_predictions_diff_log + gdp_log_test.repeat(prediction_repetitions))
            test_ground_truth_usd = np.exp(test_ground_truth_diff_log + gdp_log_test)
            test_mean_predictions_usd = np.exp(test_mean_diff_log + gdp_log_test)

            train_mean_diff_log = np.mean(train_ground_truth_diff_log).repeat(train_ground_truth_diff_log.shape[0])
            train_predictions_usd = np.exp(train_predictions_diff_log + gdp_log_train.repeat(prediction_repetitions))
            train_ground_truth_usd = np.exp(train_ground_truth_diff_log + gdp_log_train)
            train_mean_predictions_usd = np.exp(train_mean_diff_log + gdp_log_train)

            val_df = pd.DataFrame({
                "predictions_usd": validation_predictions_usd,
                "ground_truth_usd": validation_ground_truth_usd.repeat(prediction_repetitions),
                "predictions_diff_log": validation_predictions_diff_log,
                "ground_truth_diff_log": validation_ground_truth_diff_log.repeat(prediction_repetitions),
            }, index=repeated_val_dates)

            val_df.to_csv(f"{loss_plot_dir}/val idx={idx_medians} {model_description}.csv")

            test_df = pd.DataFrame({
                "predictions_usd": test_predictions_usd,
                "ground_truth_usd": test_ground_truth_usd.repeat(prediction_repetitions),
                "predictions_diff_log": test_predictions_diff_log,
                "ground_truth_diff_log": test_ground_truth_diff_log.repeat(prediction_repetitions),
            }, index=repeated_test_dates)
            test_df.to_csv(f"{loss_plot_dir}/test idx={idx_medians} {model_description}.csv")

            train_df = pd.DataFrame({
                "predictions_usd": train_predictions_usd,
                "ground_truth_usd": train_ground_truth_usd.repeat(prediction_repetitions),
                "predictions_diff_log": train_predictions_diff_log,
                "ground_truth_diff_log": train_ground_truth_diff_log.repeat(prediction_repetitions),
            }, index=repeated_train_dates)
            train_df.to_csv(f"{loss_plot_dir}/train idx={idx_medians} {model_description}.csv")

            if idx_medians == 2:

                print(f"{len(val_dates)=}")
                print(f"{len(repeated_val_dates)=}")
                print(f"{validation_ground_truth_diff_log.shape=}")

                # Plot validation predictions in billion USD
                plt.figure(figsize=(12, 6))
                plt.plot(repeated_val_dates, validation_predictions_usd, label="Predicted GDP", color="green")
                plt.plot(val_dates, validation_ground_truth_usd, label="Actual GDP", color="orange")
                # plt.plot(val_dates, validation_mean_predictions_usd, label="Assuming mean GDP growth", color="blue")
                plt.title("Validation Predictions vs Actual GDP in billion USD for " + model_description)
                plt.xlabel("Date")
                plt.ylabel("GDP (billion USD)")
                plt.legend()
                plt.xticks(rotation=45)
                plt.grid(True)
                plt.savefig(f"{loss_plot_dir}/validation_predictions_full.png")
                plt.xlim(pd.Timestamp("2001-01-01"), pd.Timestamp("2013-01-01"))
                plt.ylim(14000, 18000)
                plt.savefig(f"{loss_plot_dir}/validation_predictions_2001_onward.png")
                plt.close()

                # Plot validation predictions in delta-log
                plt.figure(figsize=(12, 6))
                plt.plot(repeated_val_dates, validation_predictions_diff_log, label=r"Predicted $\Delta \log(GDP)$", color="green")
                plt.plot(val_dates, validation_ground_truth_diff_log, label=r"Actual $\Delta \log(GDP)$", color="orange")
                # plt.plot(val_dates, validation_mean_diff_log, label=r"Mean $\Delta \log(GDP)$", color="blue")
                plt.title(r"Validation Predictions vs Actual $\Delta \log(GDP) $ for " + model_description)
                plt.xlabel("Date")
                plt.ylabel(r"$\Delta \log(GDP)$")
                plt.legend()
                plt.xticks(rotation=45)
                plt.grid(True)
                plt.savefig(f"{loss_plot_dir}/validation_predictions_full_untransformed.png")
                plt.close()

                # Plot validation predictions in percentage growth
                plt.figure(figsize=(12, 6))
                plt.plot(repeated_val_dates, (validation_predictions_usd / gdp_qd_val.values.repeat(prediction_repetitions)) - 1, label=r"Predicted $\% \Delta GDP$", color="green")
                plt.plot(val_dates, (validation_ground_truth_usd / gdp_qd_val.values) - 1, label=r"Actual $\% \Delta GDP$", color="orange")
                # plt.plot(val_dates, (validation_mean_predictions_usd / gdp_qd_val.values) - 1, label=r"Mean $\% \Delta GDP$", color="blue")
                plt.title(r"Validation predictions VS Actual Percentage GDP Growth for " + model_description)
                plt.xlabel("Date")
                plt.ylabel(r"$\% \Delta GDP$")
                plt.legend()
                plt.xticks(rotation=45)
                plt.grid(True)
                plt.savefig(f"{loss_plot_dir}/validation_predictions_percentage_growth.png")
                plt.close()

                # Plot test predictions in billion USD
                plt.figure(figsize=(12, 6))
                plt.plot(repeated_test_dates, test_predictions_usd, label="Predicted GDP", color="green")
                plt.plot(test_dates, test_ground_truth_usd, label="Actual GDP", color="orange")
                # plt.plot(test_dates, test_mean_predictions_usd, label="Assuming mean GDP growth", color="blue")
                plt.title("Test Predictions vs Actual GDP in billion USD for " + model_description)
                plt.xlabel("Date")
                plt.ylabel("GDP (billion USD)")
                plt.legend()
                plt.xticks(rotation=45)
                plt.grid(True)
                plt.savefig(f"{loss_plot_dir}/test_predictions_full.png")
                plt.close()

                # Plot test predictions in delta-log
                plt.figure(figsize=(12, 6))
                plt.plot(repeated_test_dates, test_predictions_diff_log, label=r"Predicted $\Delta \log(GDP)$", color="green")
                plt.plot(test_dates, test_ground_truth_diff_log, label=r"Actual $\Delta \log(GDP)$", color="orange")
                # plt.plot(test_dates, test_mean_diff_log, label=r"Mean $\Delta \log(GDP)$", color="blue")
                plt.title(r"Test Predictions vs Actual $\Delta \log(GDP)$ for " + model_description)
                plt.xlabel("Date")
                plt.ylabel(r"$\Delta \log(GDP)$")
                plt.legend()
                plt.xticks(rotation=45)
                plt.grid(True)
                plt.savefig(f"{loss_plot_dir}/test_predictions_full_untransformed.png")
                plt.close()

                # Plot test predictions in percentage growth
                plt.figure(figsize=(12, 6))
                plt.plot(repeated_test_dates, (test_predictions_usd / gdp_qd_test.values.repeat(prediction_repetitions)) - 1, label=r"Predicted $\% \Delta GDP$", color="green")
                plt.plot(test_dates, (test_ground_truth_usd / gdp_qd_test.values) - 1, label=r"Actual $\% \Delta GDP$", color="orange")
                # plt.plot(test_dates, (test_mean_predictions_usd / gdp_qd_test.values) - 1, label=r"Mean $\% \Delta GDP$", color="blue")
                plt.title(r"Test predicitons VS Actual Percentage GDP Growth for " + model_description)
                plt.xlabel("Date")
                plt.ylabel(r"$\% \Delta GDP$")
                plt.legend()
                plt.xticks(rotation=45)
                plt.grid(True)
                plt.savefig(f"{loss_plot_dir}/test_predictions_percentage_growth.png")
                plt.close()

                # Plot train predictions in billion USD
                plt.figure(figsize=(12, 6))
                plt.plot(repeated_train_dates, train_predictions_usd, label="Predicted GDP", color="green")
                plt.plot(train_dates, train_ground_truth_usd, label="Actual GDP", color="orange")
                # plt.plot(train_dates, train_mean_predictions_usd, label="Assuming mean GDP growth", color="blue")
                plt.title("Train Predictions vs Actual GDP in billion USD for " + model_description)
                plt.xlabel("Date")
                plt.ylabel("GDP (billion USD)")
                plt.legend()
                plt.xticks(rotation=45)
                plt.grid(True)
                plt.savefig(f"{loss_plot_dir}/train_predictions_full.png")
                plt.close()

                # Plot train predictions in delta-log
                plt.figure(figsize=(12, 6))
                plt.plot(repeated_train_dates, train_predictions_diff_log, label=r"Predicted $\Delta \log(GDP)$", color="green")
                plt.plot(train_dates, train_ground_truth_diff_log, label=r"Actual $\Delta \log(GDP)$", color="orange")
                # plt.plot(train_dates, train_mean_diff_log, label=r"Mean $\Delta \log(GDP)$", color="blue")
                plt.title(r"Train Predictions vs Actual $\Delta \log(GDP)$ for " + model_description)
                plt.xlabel("Date")
                plt.ylabel(r"$\Delta \log(GDP)$")
                plt.legend()
                plt.xticks(rotation=45)
                plt.grid(True)
                plt.savefig(f"{loss_plot_dir}/train_predictions_full_untransformed.png")
                plt.close()

                # Plot train predictions in percentage growth
                plt.figure(figsize=(12, 6))
                plt.plot(repeated_train_dates, (train_predictions_usd / gdp_qd_train.values.repeat(prediction_repetitions)) - 1, label=r"Predicted $\% \Delta GDP$", color="green")
                plt.plot(train_dates, (train_ground_truth_usd / gdp_qd_train.values) - 1, label=r"Actual $\% \Delta GDP$", color="orange")
                # plt.plot(train_dates, (train_mean_predictions_usd / gdp_qd_train.values) - 1, label=r"Mean $\% \Delta GDP$", color="blue")
                plt.title(r"Train redictoins VS Actual Percentage GDP Growth for " + model_description)
                plt.xlabel("Date")
                plt.ylabel(r"$\% \Delta GDP$")
                plt.legend()
                plt.xticks(rotation=45)
                plt.grid(True)
                plt.savefig(f"{loss_plot_dir}/train_predictions_percentage_growth.png")
                plt.close()

            # Calcuate metrics for val, test, and train

            lowest_validation_losses = np.array(lowest_validation_losses)
            results = {
                # "Lowest validation losses per fold": lowest_validation_losses.tolist(),
                "Average": np.mean(lowest_validation_losses),
                "RMSE": np.sqrt(np.mean(lowest_validation_losses)),
                "Unpreprocessed MAE": mean_absolute_error(validation_ground_truth_usd.repeat(prediction_repetitions), validation_predictions_usd),
                "Unpreprocessed RMSE": root_mean_squared_error(validation_ground_truth_usd.repeat(prediction_repetitions), validation_predictions_usd),
                "Mean-predict Unpreprocessed MAE": mean_absolute_error(validation_ground_truth_usd, validation_mean_predictions_usd),
                "Mean-predict Unpreprocessed RMSE": root_mean_squared_error(validation_ground_truth_usd, validation_mean_predictions_usd),
                "Transformed MAE": mean_absolute_error(validation_ground_truth_diff_log.repeat(prediction_repetitions), validation_predictions_diff_log),
                "Transformed RMSE": root_mean_squared_error(validation_ground_truth_diff_log.repeat(prediction_repetitions), validation_predictions_diff_log),
                "Mean-predict Transformed MAE": mean_absolute_error(validation_ground_truth_diff_log, validation_mean_diff_log),
                "Mean-predict Transformed RMSE": root_mean_squared_error(validation_ground_truth_diff_log, validation_mean_diff_log),

                "Test RMSE": median_test_predictions["rmse"],
                "Test MAE": median_test_predictions["mae"],
                "Unpreprocessed Test MAE": mean_absolute_error(test_ground_truth_usd.repeat(prediction_repetitions), test_predictions_usd),
                "Unpreprocessed Test RMSE": root_mean_squared_error(test_ground_truth_usd.repeat(prediction_repetitions), test_predictions_usd),
                "Mean-predict Unpreprocessed Test MAE": mean_absolute_error(test_ground_truth_usd, test_mean_predictions_usd),
                "Mean-predict Unpreprocessed Test RMSE": root_mean_squared_error(test_ground_truth_usd, test_mean_predictions_usd),
                "Transformed Test MAE": mean_absolute_error(test_ground_truth_diff_log.repeat(prediction_repetitions), test_predictions_diff_log),
                "Transformed Test RMSE": root_mean_squared_error(test_ground_truth_diff_log.repeat(prediction_repetitions), test_predictions_diff_log),
                "Mean-predict Test MAE": mean_absolute_error(test_ground_truth_diff_log, test_mean_diff_log),
                "Mean-predict Test RMSE": root_mean_squared_error(test_ground_truth_diff_log, test_mean_diff_log),

                "Train RMSE": medain_train_predictions["rmse"],
                "Train MAE": medain_train_predictions["mae"],
                "Unpreprocessed Train MAE": mean_absolute_error(train_ground_truth_usd.repeat(prediction_repetitions), train_predictions_usd),
                "Unpreprocessed Train RMSE": root_mean_squared_error(train_ground_truth_usd.repeat(prediction_repetitions), train_predictions_usd),
                "Mean-predict Unpreprocessed Train MAE": mean_absolute_error(train_ground_truth_usd, train_mean_predictions_usd),
                "Mean-predict Unpreprocessed Train RMSE": root_mean_squared_error(train_ground_truth_usd, train_mean_predictions_usd),
                "Transformed Train MAE": mean_absolute_error(train_ground_truth_diff_log.repeat(prediction_repetitions), train_predictions_diff_log),
                "Transformed Train RMSE": root_mean_squared_error(train_ground_truth_diff_log.repeat(prediction_repetitions), train_predictions_diff_log),
                "Mean-predict Train MAE": mean_absolute_error(train_ground_truth_diff_log, train_mean_diff_log),
                "Mean-predict Train RMSE": root_mean_squared_error(train_ground_truth_diff_log, train_mean_diff_log),
            }

            for k, v in results.items():
                total_results[k] += v

    # Compute average results
    average_results = {}
    for key, value in total_results.items():
        average_results[key] = value / 3

    # Define the ordered keys and section breaks
    ordered_keys = [
        # "Lowest validation losses per fold",
        "Average",
        "RMSE",
        "Unpreprocessed MAE",
        "Unpreprocessed RMSE",
        "Mean-predict Unpreprocessed MAE",
        "Mean-predict Unpreprocessed RMSE",
        "Transformed MAE",
        "Transformed RMSE",
        "Mean-predict Transformed MAE",
        "Mean-predict Transformed RMSE",
        "",  # <-- Empty line for visual separation
        "Test RMSE",
        "Test MAE",
        "Unpreprocessed Test MAE",
        "Unpreprocessed Test RMSE",
        "Mean-predict Unpreprocessed Test MAE",
        "Mean-predict Unpreprocessed Test RMSE",
        "Transformed Test MAE",
        "Transformed Test RMSE",
        "Mean-predict Test MAE",
        "Mean-predict Test RMSE",
        "",  # <-- Empty line for visual separation
        "Train RMSE",
        "Train MAE",
        "Unpreprocessed Train MAE",
        "Unpreprocessed Train RMSE",
        "Mean-predict Unpreprocessed Train MAE",
        "Mean-predict Unpreprocessed Train RMSE",
        "Transformed Train MAE",
        "Transformed Train RMSE",
        "Mean-predict Train MAE",
        "Mean-predict Train RMSE",
    ]

    # Write to file
    output_file = f"{loss_plot_dir}/avg_results {start_of_train_run} {model_description}.txt"
    with open(output_file, "w") as f:
        for key in ordered_keys:
            if key == "":
                f.write("\n")  # Write empty line
            elif key in average_results:
                value = average_results[key]
                if isinstance(value, (list, np.ndarray)):
                    f.write(f"{key}: {np.array(value).tolist()}\n")
                else:
                    f.write(f"{key}: {value}\n")


        f.write("\n")

        for k, v in param_dict.items():
            f.write(f"{k}: {v}\n")

        f.write("\n")

        f.write(f"Selected Monthly Features: {", ".join(train_val_monthly.columns)}\n")
        f.write(f"Selected Quarterly Features: {", ".join(train_val_quarterly.columns)}\n")

        for train_proportion, stats, in best_epoch.items():
            f.write("\n")
            f.write(f"Train proportion: {train_proportion}%\n")

            sorted_stats = sorted(stats, key=lambda stat: stat['val_loss'])
            for stat in sorted_stats:
                f.write(f"Best Epoch: {stat['epoch']}, Validation Loss: {stat['val_loss']}\n")
    
    return lowest_validation_losses, lowest_train_losses

def plot_loss(loss_plot_dir, month_start_training_data, month_end_training_data, month_end_validation_data, history, verbose=True):
    loss_plot_filename = f"{loss_plot_dir}/from_{month_start_training_data}_to_{month_end_training_data}_to_{month_end_validation_data}.png"
    plt.plot(history.history['loss'], label=f'Training Loss ({month_start_training_data} to {month_end_training_data})')
    plt.plot(history.history['val_loss'], label=f'Validation Loss ({month_end_training_data} to {month_end_validation_data})')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'Training and Validation Loss from {month_start_training_data} to {month_end_training_data} to {month_end_validation_data}')
    plt.legend()
    if verbose:
        plt.savefig(loss_plot_filename)
    if should_plot:
        plt.show()
    else:
        plt.close()

def plot_y_train_val(y_train, y_val):
    if should_plot:
        print("y_val:")
        print(y_val)

        print("\ny_train:")
        print(y_train)
                # Visualize the distributions of y_train and y_val
        plt.figure(figsize=(12, 6))

                # Plot y_train distribution
        plt.subplot(1, 2, 1)
        sns.histplot(y_train.flatten(), kde=True, bins=30, color="blue", label="y_train")
        plt.title("Distribution of y_train")
        plt.xlabel("Values")
        plt.ylabel("Frequency")
        plt.legend()

                # Plot y_val distribution
        plt.subplot(1, 2, 2)
        sns.histplot(y_val.flatten(), kde=True, bins=30, color="orange", label="y_val")
        plt.title("Distribution of y_val")
        plt.xlabel("Values")
        plt.ylabel("Frequency")
        plt.legend()

        plt.tight_layout()
        plt.show()

def plot_scaled_data(train_monthly_scaled, val_monthly_scaled, train_quarterly_scaled, val_quarterly_scaled):
    if should_plot:
            # Plot standardized monthly data
        plt.figure(figsize=(12, 8))
        for column in train_monthly_scaled.columns:
            plt.plot(train_monthly_scaled.index, train_monthly_scaled[column], label=column)
        plt.title("Standardized Monthly Data (Training Set)")
        plt.xlabel("Date")
        plt.ylabel("Standardized Values")
        plt.legend(loc="upper right", bbox_to_anchor=(1.3, 1))
        plt.grid(True)
        plt.show()

            # Plot standardized quarterly data
        plt.figure(figsize=(12, 8))
        plt.plot(train_quarterly_scaled.index, train_quarterly_scaled["GDPC1"], label="Standardized GDP (Training Set)", color="blue")
        plt.title("Standardized Quarterly Data (Training Set)")
        plt.xlabel("Date")
        plt.ylabel("Standardized GDP")
        plt.legend()
        plt.grid(True)
        plt.show()

            # Plot standardized monthly validation data
        plt.figure(figsize=(12, 8))
        for column in val_monthly_scaled.columns:
            plt.plot(val_monthly_scaled.index, val_monthly_scaled[column], label=column)
        plt.title("Standardized Monthly Data (Validation Set)")
        plt.xlabel("Date")
        plt.ylabel("Standardized Values")
        plt.legend(loc="upper right", bbox_to_anchor=(1.3, 1))
        plt.grid(True)
        plt.show()

            # Plot standardized quarterly validation data
        plt.figure(figsize=(12, 8))
        plt.plot(val_quarterly_scaled.index, val_quarterly_scaled["GDPC1"], label="Standardized GDP (Validation Set)", color="orange")
        plt.title("Standardized Quarterly Data (Validation Set)")
        plt.xlabel("Date")
        plt.ylabel("Standardized GDP")
        plt.legend()
        plt.grid(True)
        plt.show()

if __name__ == "__main__":
    # Parse command-line argument
    import sys
    
    model_name = None
    for i, arg in enumerate(sys.argv):
        if arg == "--eval" and i + 1 < len(sys.argv):
            model_name = sys.argv[i + 1].lower()
            break

    if model_name is None:
        print("Usage: python lstm_123.py --eval <model>")
        print("Available models: gru1, gru2, gru3, lstm1, lstm2, lstm3, multivariate_gru, multivariate_lstm, univariate_gru, univariate_lstm")
        sys.exit(1)

    md_train_stationary, qd_train_stationary = load_train_data()
    md_test_stationary, qd_test_stationary = load_test_data()

    from multivariate_RNN import instantiate_multivariate_model, create_datapoints_multivariate
    from univariate_RNN import instantiate_univariate_model, create_datapoints_univariate

    configs = {
        "gru1": dict(
            feature_selection_method='lasso',
            n_features=14,
            dropout_rate=0.26473126503458194,
            optimizer='adam',
            batch_size=32,
            hidden_units=64,
            learning_rate=0.007922026556168792,
            add_recession_feature=True,
            instantiate_model=partial(instantiate_model_duplicate_qd, Rnn=tf.keras.layers.GRU),
            model_description="GRU-repeat",
            create_datapoints=create_datapoints_lstm_1_2_and_3,
        ),
        "gru2": dict(
            feature_selection_method='lasso',
            n_features=13,
            dropout_rate=0.3795713685036741,
            optimizer='adam',
            batch_size=64,
            hidden_units=64,
            learning_rate=0.0039072830004885945,
            add_recession_feature=True,
            instantiate_model=partial(instantiate_model_multilayer, Rnn=tf.keras.layers.GRU),
            model_description="GRU-bilayer",
            create_datapoints=create_datapoints_lstm_1_2_and_3,
        ),
        "gru3": dict(
            feature_selection_method='lasso',
            n_features=14,
            dropout_rate=0.4541694231988055,
            optimizer='adam',
            batch_size=16,
            hidden_units=64,
            learning_rate=0.0038072480424024917,
            add_recession_feature=True,
            instantiate_model=partial(instantiate_model_alternating_rnn, Rnn=tf.keras.layers.GRU),
            model_description="GRU-alternate",
            create_datapoints=create_datapoints_lstm_1_2_and_3,
        ),
        # Best trial:
        # Value: 0.4417503618945678
        # Params:
        #     feature_selection_method: lasso
        #     n_features: 12
        #     dropout_rate: 0.4739811144153839
        #     optimizer: adam
        #     batch_size: 32
        #     hidden_units: 32
        #     learning_rate: 0.008794942657558699
        #     add_recession_feature: False

        "multivariate_gru": dict(
            feature_selection_method='lasso',
            n_features=12,
            dropout_rate=0.4739811144153839,
            optimizer='adam',
            batch_size=32,
            hidden_units=32,
            learning_rate=0.008794942657558699,
            add_recession_feature=False,
            instantiate_model=partial(instantiate_multivariate_model, Rnn=tf.keras.layers.GRU),
            model_description="Quarterly GRU",
            create_datapoints=create_datapoints_multivariate,
        ),
        "univariate_gru": dict(
            feature_selection_method='none',  # If you use PCA, keep this; otherwise, adjust as needed
            n_features=100,                  # Keep as before unless you have a new value
            dropout_rate=0.23168946271367874,
            optimizer='adam',
            batch_size=32,
            hidden_units=16,
            learning_rate=0.006457314522932592,
            add_recession_feature=False,
            instantiate_model=partial(instantiate_univariate_model, Rnn=tf.keras.layers.GRU),
            model_description="Univariate GRU",
            create_datapoints=create_datapoints_univariate,
        ),
        "lstm1": dict(
            feature_selection_method='lasso',
            n_features=18,
            dropout_rate=0.2586783399110758,
            optimizer='adam',
            batch_size=16,
            hidden_units=32,
            learning_rate=0.008420118343459052,
            add_recession_feature=False,
            instantiate_model=partial(instantiate_model_duplicate_qd, Rnn=tf.keras.layers.LSTM),
            model_description="LSTM-repeat",
            create_datapoints=create_datapoints_lstm_1_2_and_3,
        ),
        "lstm2": dict(
            feature_selection_method='lasso',
            n_features=25,
            dropout_rate=0.4620799007751671,
            optimizer='adam',
            batch_size=32,
            hidden_units=64,
            learning_rate=0.005104323173367261,
            add_recession_feature=True,
            instantiate_model=partial(instantiate_model_multilayer, Rnn=tf.keras.layers.LSTM),
            model_description="LSTM-bilayer",
            create_datapoints=create_datapoints_lstm_1_2_and_3,
        ),
        "lstm3": dict(
            feature_selection_method='lasso',
            n_features=19,
            dropout_rate=0.24053423834450846,
            optimizer='adam',
            batch_size=32,
            hidden_units=64,
            learning_rate=0.0066183323272097205,
            add_recession_feature=True,
            instantiate_model=partial(instantiate_model_alternating_rnn, Rnn=tf.keras.layers.LSTM),
            model_description="LSTM-alternate",
            create_datapoints=create_datapoints_lstm_1_2_and_3,
        ),

        # Best trial:
        #   Value: 0.45764124952256685
        #   Params:
        #     feature_selection_method: lasso
        #     n_features: 17
        #     dropout_rate: 0.2668934106559545
        #     optimizer: adam
        #     batch_size: 64
        #     hidden_units: 32
        #     learning_rate: 0.005028757569779784
        #     add_recession_feature: False


        "multivariate_lstm": dict(
            feature_selection_method='lasso',
            n_features=17,
            dropout_rate=0.2668934106559545,
            optimizer='adam',
            batch_size=64,
            hidden_units=32,
            learning_rate=0.005028757569779784,
            add_recession_feature=False,
            instantiate_model=partial(instantiate_multivariate_model, Rnn=tf.keras.layers.LSTM),
            model_description="Quarterly LSTM",
            create_datapoints=create_datapoints_multivariate,
        ),
        "univariate_lstm": dict(
            feature_selection_method='none',
            n_features=100,
            dropout_rate=0.23168946271367874,
            optimizer='adam',
            batch_size=32,
            hidden_units=16,
            learning_rate=0.006457314522932592,
            add_recession_feature=False,
            instantiate_model=partial(instantiate_univariate_model, Rnn=tf.keras.layers.LSTM),
            model_description="Univariate LSTM",
            create_datapoints=create_datapoints_univariate,
        ),
    }

    if model_name not in configs:
        print("Unknown model:", model_name)
        print("Available models:", ", ".join(configs.keys()))
        sys.exit(1)

    params = configs[model_name]
    train_and_evaluate_model(
        md_train_stationary, qd_train_stationary,
        md_test_stationary=md_test_stationary, qd_test_stationary=qd_test_stationary,
        sequence_length=12,
        verbose=True,
        **params
    )