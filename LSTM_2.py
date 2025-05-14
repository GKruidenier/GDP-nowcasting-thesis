# import tensorflow as tf; tf.compat.v1.enable_eager_execution()

from sklearn.metrics import mean_squared_error
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

import datetime
import math

from stationarization import load_train_data
from feature_selection import combine_qd_and_md_as_qd, remove_highly_correlated_features, lasso_feature_selection, tree_based_feature_selection, rfe_feature_selection
from tensorflow.keras.callbacks import EarlyStopping

should_plot = False

# Set the random seed for reproducibility
def set_seed(seed):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)

def create_datapoints_lstm_2_and_3(monthly, quarterly, sequence_length):
    x_monthly = []
    x_quarterly = []
    x_shift = []
    y = []
    y_dates = []

    for m in range(len(monthly) - sequence_length - 3):  # -3 to account for the next quarter
        month_offset = m % 3
        q = m // 3
        q_end = math.ceil((m + sequence_length) / 3)

        x_monthly.append(
            monthly.iloc[m:m + sequence_length].values
        )
        x_shift.append(month_offset)

        x_quarterly.append(
            quarterly.iloc[q:q_end].values
        )
        y.append(
            quarterly.iloc[q_end][["GDPC1"]]
        )
        y_dates.append(
            monthly.index[m + sequence_length + month_offset]
        )

    return np.array(x_monthly), np.array(x_quarterly), np.array(x_shift), np.array(y), y_dates

def instantiate_model_lstm_2(sequence_length, n_monthly_features, n_quarterly_features, n_hidden_units, optimizer_name, learning_rate):
    # Define model
    monthly_input = layers.Input(shape=(sequence_length, n_monthly_features))
    quarterly_input = layers.Input(shape=(sequence_length // 3 + 1, n_quarterly_features))  # Quarterly is 3x slower
    shift_input = layers.Input(shape=(1,), dtype=tf.int32)

    quarterly_lstm = layers.LSTM(n_hidden_units, return_sequences=True)(quarterly_input)
    quarterly_upsampled = layers.UpSampling1D(size=3)(quarterly_lstm)

    # A custom layer that effectively peforms the slicing operation `quarterly_upsampled[:, shift_input : shift_input + sequence_length]`.
    class SlicingLayer(tf.keras.layers.Layer):
        def call(self, inputs):
            quarterly_upsampled, shift_input = inputs

            indices = tf.reshape(tf.range(sequence_length + 2, dtype=tf.int32), (1, sequence_length + 2))
            shift_input = tf.cast(shift_input, tf.int32)
            filter = tf.logical_and(
                indices >= shift_input,
                indices < shift_input + sequence_length
            )

            filtered = tf.boolean_mask(quarterly_upsampled, filter)

            return tf.reshape(filtered, (-1, sequence_length, n_hidden_units))

        def compute_output_shape(self, input_shape):
            return (input_shape[0][0], sequence_length, n_hidden_units)

    quarterly_sliced = SlicingLayer()([quarterly_upsampled, shift_input])
    monthly_augmented_input = layers.Concatenate()([monthly_input, quarterly_sliced])
    monthly_lstm = layers.LSTM(n_hidden_units, return_sequences=False)(monthly_augmented_input)

    output = layers.Dense(1)(monthly_lstm)

    model = tf.keras.Model(inputs=[monthly_input, quarterly_input, shift_input], outputs=output)

    if optimizer_name == 'adam':
        optimizer_instance = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer_name == 'RMSprop':
        optimizer_instance = tf.keras.optimizers.SGD(learning_rate=learning_rate)

    model.compile(optimizer=optimizer_instance, loss="mse")

    model.summary()

    return model

def create_datapoints_lstm_3(monthly, quarterly, sequence_length):
    x_monthly = []
    x_quarterly = []
    x_shift = []
    y = []
    y_dates = []

    for m in range(len(monthly) - sequence_length - 3):  # -3 to account for the next quarter
        month_offset = m % 3
        q = m // 3
        q_end = math.ceil((m + sequence_length) / 3)

        x_monthly.append(
            monthly.iloc[m:m + sequence_length].values
        )
        x_shift.append(month_offset)

        x_quarterly.append(
            quarterly.iloc[q:q_end].values
        )
        y.append(
            quarterly.iloc[q_end][["GDPC1"]]
        )
        y_dates.append(
            monthly.index[m + sequence_length - 1]
        )

    return np.array(x_monthly), np.array(x_quarterly), np.array(x_shift), np.array(y), y_dates

def instantiate_model_lstm_3(sequence_length, n_monthly_features, n_quarterly_features, n_hidden_units, optimizer_name, learning_rate):
    # Define model
    monthly_input = layers.Input(shape=(sequence_length, n_monthly_features))
    quarterly_input = layers.Input(shape=(sequence_length // 3 + 1, n_quarterly_features))  # Quarterly is 3x slower
    shift_input = layers.Input(shape=(1,), dtype=tf.int32)

    quarterly_lstm = layers.LSTM(n_hidden_units, return_sequences=True)(quarterly_input)
    quarterly_upsampled = layers.UpSampling1D(size=3)(quarterly_lstm)

    # A custom layer that effectively peforms the slicing operation `quarterly_upsampled[:, shift_input : shift_input + sequence_length]`.
    class SlicingLayer(tf.keras.layers.Layer):
        def call(self, inputs):
            quarterly_upsampled, shift_input = inputs

            indices = tf.reshape(tf.range(sequence_length + 2, dtype=tf.int32), (1, sequence_length + 2))
            shift_input = tf.cast(shift_input, tf.int32)
            filter = tf.logical_and(
                indices >= shift_input,
                indices < shift_input + sequence_length
            )

            filtered = tf.boolean_mask(quarterly_upsampled, filter)

            return tf.reshape(filtered, (-1, sequence_length, n_hidden_units))

        def compute_output_shape(self, input_shape):
            return (input_shape[0][0], sequence_length, n_hidden_units)

    quarterly_sliced = SlicingLayer()([quarterly_upsampled, shift_input])
    monthly_augmented_input = layers.Concatenate()([monthly_input, quarterly_sliced])
    monthly_lstm = layers.LSTM(n_hidden_units, return_sequences=False)(monthly_augmented_input)

    output = layers.Dense(1)(monthly_lstm)

    model = tf.keras.Model(inputs=[monthly_input, quarterly_input, shift_input], outputs=output)

    if optimizer_name == 'adam':
        optimizer_instance = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer_name == 'RMSprop':
        optimizer_instance = tf.keras.optimizers.SGD(learning_rate=learning_rate)

    model.compile(optimizer=optimizer_instance, loss="mse")

    model.summary()

    return model


def LSTM_model_2(md_train_stationary, qd_train_stationary, feature_selection_method, n_features, dropout_rate, optimizer, batch_size, hidden_units, sequence_length, learning_rate, instantiate_model=instantiate_model_lstm_2, create_datapoints=create_datapoints_lstm_2_and_3):
    start_of_train_run = datetime.datetime.now().strftime("%Y-%m-%d %Hh%Mm%Ss")

    base_seed = 2571267
    set_seed(base_seed)

    param_dict = {
        "feature_selection_method": feature_selection_method,
        "n_features": n_features,
        "dropout_rate": dropout_rate,
        "optimizer": optimizer,
        "batch_size": batch_size,
        "hidden_units": hidden_units,
        "sequence_length": sequence_length,
    }
    print("Parameters:", param_dict)

    # Model 1 preprocessing: repeating GDP values
    if feature_selection_method == "filter_highly_correlated":
        selected_features = remove_highly_correlated_features(combine_qd_and_md_as_qd(qd_train_stationary, md_train_stationary))
    if feature_selection_method == 'lasso':
        # Perform LASSO feature selection
        selected_features, _ = lasso_feature_selection(combine_qd_and_md_as_qd(qd_train_stationary, md_train_stationary), n_features=n_features)
    elif feature_selection_method == "random_forest":
        # Perform Random Forest feature selection
        selected_features, _ = tree_based_feature_selection(combine_qd_and_md_as_qd(qd_train_stationary, md_train_stationary), n_features=n_features)
    elif feature_selection_method == "rfe":
        # Perform Random Forest feature selection
        selected_features, _ = rfe_feature_selection(combine_qd_and_md_as_qd(qd_train_stationary, md_train_stationary), n_features=n_features)
    elif feature_selection_method == "none":
        # Use all features
        selected_features = combine_qd_and_md_as_qd(qd_train_stationary, md_train_stationary).columns.tolist()

    print("Selected features:", selected_features)

    # Separate the selected features into monthly and quarterly features
    monthly_features = [feature for feature in selected_features if feature in md_train_stationary.columns]
    quarterly_features = [feature for feature in selected_features if feature in qd_train_stationary.columns] + ["GDPC1"]

    # Create separate DataFrames for monthly and quarterly selected features
    train_val_monthly = md_train_stationary[monthly_features]
    train_val_quarterly = qd_train_stationary[quarterly_features]

    # Print the separated features for verification
    print("Selected Monthly Features:")
    print(train_val_monthly.columns)

    print("Selected Quarterly Features:")
    print(train_val_quarterly.columns)

    initial_train_size = 20
    val_size = 10
    full_train_val_size = 100

    lowest_validation_losses = []
    validation_predictions_per_split = []

    loss_plot_dir = f"results/loss_plot {start_of_train_run}"
    os.makedirs(loss_plot_dir, exist_ok=True)

    number_of_quarters = len(train_val_quarterly)
    print("n:", number_of_quarters)

    for train_proportion in range(initial_train_size, full_train_val_size, val_size):
        validation_predictions_current_split = []

        tf.keras.backend.clear_session()
        # Step 6: Split training and validation sets
        end_train_data_idx = int(number_of_quarters * train_proportion / full_train_val_size)
        end_val_data_idx = int(number_of_quarters * (train_proportion + val_size) / full_train_val_size)
        print(f"end_train_data_idx: {end_train_data_idx}, end_val_data_idx: {end_val_data_idx}, number of quarters in val: {end_val_data_idx - end_train_data_idx}")

        train_monthly = train_val_monthly.iloc[:end_train_data_idx * 3]
        val_monthly = train_val_monthly.iloc[end_train_data_idx * 3:end_val_data_idx * 3]

        train_quarterly = train_val_quarterly.iloc[:end_train_data_idx]
        val_quarterly = train_val_quarterly.iloc[end_train_data_idx:end_val_data_idx]

        month_start_training_data = train_monthly.index[0].strftime("%Y-%m")
        month_end_training_data = train_monthly.index[-1].strftime("%Y-%m")
        month_end_validation_data = val_monthly.index[-1].strftime("%Y-%m")

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

        # Confirm shapes
        print("Monthly Scaled:", train_monthly_scaled.shape, val_monthly_scaled.shape)
        print("Quarterly Scaled:", train_quarterly_scaled.shape, val_quarterly_scaled.shape)

        plot_scaled_data(train_monthly_scaled, val_monthly_scaled, train_quarterly_scaled, val_quarterly_scaled)

        # Step 1: Initialize empty lists for x and y

        x_train_monthly, x_train_quarterly, x_train_shift, y_train, _ = create_datapoints(train_monthly_scaled, train_quarterly_scaled, sequence_length)
        print("x_train_md shape:", x_train_monthly.shape)
        print("x_train_qd shape:", x_train_quarterly.shape)
        print("y_train shape:", y_train.shape)

        x_val_monthly, x_val_quarterly, x_val_shift, y_val, val_dates = create_datapoints(val_monthly_scaled, val_quarterly_scaled, sequence_length)
        print("x_val_md shape:", x_val_monthly.shape)
        print("x_val_qd shape:", x_val_quarterly.shape)
        print("y_val shape:", y_val.shape)

        training_repetition_count = 5
        val_losses = [] # we are gonna train it 5 times and take the average of the second and third best model for increased stability

        for training_repetition_idx in range(training_repetition_count):
            model = instantiate_model(
                sequence_length,
                len(monthly_features),
                len(quarterly_features),
                hidden_units,
                optimizer,
                learning_rate,
            )

            # Add EarlyStopping callback
            early_stopping = EarlyStopping(
                monitor="val_loss",         # Stop/restore based on the validation loss
                patience=5,                 # Stop training if no improvement for 5 epochs
                restore_best_weights=True   # Restore the best weights after stopping
            )

            initial_val_loss = model.evaluate([x_val_monthly, x_val_quarterly, x_val_shift], y_val, verbose=0)
            print(f"Initial validation Loss: {initial_val_loss}")
            initial_train_loss = model.evaluate([x_train_monthly, x_train_quarterly, x_train_shift], y_train, verbose=0)
            print(f"Initial train Loss: {initial_train_loss}")

            print("x_train_md type:", type(x_train_monthly), "shape:", x_train_monthly.shape)
            print("x_train_qd type:", type(x_train_quarterly), "shape:", x_train_quarterly.shape)
            print("y_train type:", type(y_train), "shape:", y_train.shape)
            print("x_val_md type:", type(x_val_monthly), "shape:", x_val_monthly.shape)
            print("x_val_qd type:", type(x_val_quarterly), "shape:", x_val_quarterly.shape)
            print("y_val type:", type(y_val), "shape:", y_val.shape)
            x_train_monthly = tf.convert_to_tensor(x_train_monthly, dtype=tf.float32)
            x_train_quarterly = tf.convert_to_tensor(x_train_quarterly, dtype=tf.float32)
            y_train = tf.convert_to_tensor(y_train, dtype=tf.float32)
            x_val_monthly = tf.convert_to_tensor(x_val_monthly, dtype=tf.float32)
            x_val_quarterly = tf.convert_to_tensor(x_val_quarterly, dtype=tf.float32)
            y_val = tf.convert_to_tensor(y_val, dtype=tf.float32)

            history = model.fit(
                [x_train_monthly, x_train_quarterly, x_train_shift],
                y_train,
                validation_data=([x_val_monthly, x_val_quarterly, x_val_shift], y_val),
                epochs=15,
                batch_size=batch_size,
                verbose=1,
                shuffle=True,
                callbacks=[early_stopping],
            )

            plot_loss(loss_plot_dir, month_start_training_data, month_end_training_data, month_end_validation_data, history)

            val_losses.append(min(history.history['val_loss']))

            # Step 3: Evaluate the model on validation data
            val_loss = model.evaluate([x_val_monthly, x_val_quarterly, x_val_shift], y_val, verbose=0)
            validation_predictions_current_split.append((
                val_loss,
                scaler_gdp.inverse_transform(model.predict([x_val_monthly, x_val_quarterly, x_val_shift], verbose=0)),
                scaler_gdp.inverse_transform(y_val),
                val_dates,
            ))
            print(f"Validation Loss: {val_loss}")
            train_loss = model.evaluate([x_train_monthly, x_train_quarterly, x_train_shift], y_train, verbose=0)
            print(f"Train Loss: {train_loss}")

            plot_y_train_val(y_train, y_val)

        val_losses.sort()
        lowest_validation_losses.append(sum(val_losses[1:-1]) / 3)
        validation_predictions_current_split.sort(key=lambda x: x[0])
        validation_predictions_per_split.append(validation_predictions_current_split[1]) # Select second best prediction

    validation_predictions_stationary = np.array([p for _, ps, _, _ in validation_predictions_per_split for p in ps])
    validation_ground_truth_stationary = np.array([y for _, _, ys, _ in validation_predictions_per_split for y in ys])
    flattened_val_dates = [d for _, _, _, dates in validation_predictions_per_split for d in dates]

    qd = pd.read_csv("FRED_QD.csv", index_col=0, parse_dates=True, date_format="%Y-%m-%d").iloc[2:]
    qd.index = pd.to_datetime(qd.index, format="%m/%d/%Y")
    gdp_qd = qd["GDPC1"].copy()
    gdp_qd: pd.Series = gdp_qd.loc[qd.index.repeat(3)]
    gdp_qd.index = pd.date_range(start=gdp_qd.index[0] - pd.DateOffset(months=5), periods=len(gdp_qd), freq="MS")
    print("gdp_qd length:", len(gdp_qd))
    gdp_qd: pd.Series = gdp_qd.loc[flattened_val_dates]
    print("gdp_qd shape:", gdp_qd.values.shape, "val_dates length:", len(flattened_val_dates))

    validation_predictions = validation_predictions_stationary + gdp_qd.values
    validation_ground_truth = validation_ground_truth_stationary + gdp_qd.values

    plt.figure(figsize=(12, 6))
    plt.plot(flattened_val_dates, validation_predictions, label="Predicted GDP", color="green")
    plt.plot(flattened_val_dates[::3], validation_ground_truth[::3], label="Actual GDP", color="orange")
    plt.title("Validation Predictions vs Actual GDP")
    plt.xlabel("Date")
    plt.ylabel("GDP")
    plt.legend()
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.savefig(f"{loss_plot_dir}/validation_predictions_full.png")
    plt.xlim(pd.Timestamp("2001-01-01"), pd.Timestamp("2013-01-01"))
    plt.ylim(14000, 18000)
    plt.savefig(f"{loss_plot_dir}/validation_predictions_2001_onward.png")
    plt.close()

    with open(f"{loss_plot_dir}/val_loss {start_of_train_run}.txt", "w") as f:
        f.write(f"Lowest validation losses per fold: {lowest_validation_losses}\n")
        f.write(f"Average: {sum(lowest_validation_losses) / len(lowest_validation_losses)}\n")
        f.write(f"Min: {min(lowest_validation_losses)}\n")
        f.write(f"Max: {max(lowest_validation_losses)}\n")
        f.write(f"Unpreprocessed MSE: {mean_squared_error(validation_ground_truth, validation_predictions)}\n")
        for k, v in param_dict.items():
            f.write(f"{k}: {v}\n")

    return lowest_validation_losses

def plot_loss(loss_plot_dir, month_start_training_data, month_end_training_data, month_end_validation_data, history):
    loss_plot_filename = f"{loss_plot_dir}/from_{month_start_training_data}_to_{month_end_training_data}_to_{month_end_validation_data}.png"
    plt.plot(history.history['loss'], label=f'Training Loss ({month_start_training_data} to {month_end_training_data})')
    plt.plot(history.history['val_loss'], label=f'Validation Loss ({month_end_training_data} to {month_end_validation_data})')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'Training and Validation Loss from {month_start_training_data} to {month_end_training_data} to {month_end_validation_data}')
    plt.legend()
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
    md_train_stationary, qd_train_stationary = load_train_data()


    LSTM_model_2(
        md_train_stationary,
        qd_train_stationary,
        feature_selection_method='rfe',
        n_features=15,
        dropout_rate=0.2977115755027825,
        optimizer='RMSprop',
        learning_rate=0.001,
        batch_size=16,
        hidden_units=32,
        sequence_length=13,
    )

# # Inputs
# monthly_input = layers.Input(shape=(timesteps, monthly_features))
# quarterly_input = layers.Input(shape=(timesteps // 3, quarterly_features))  # Quarterly is 3x slower

# # Quarterly LSTM
# quarterly_lstm = layers.LSTM(32, return_sequences=True)(quarterly_input)
# quarterly_upsampled = layers.UpSampling1D(size=3)(quarterly_lstm)  # Repeat each output 3 times to match monthly

# # Concatenate repeated quarterly output with monthly input
# monthly_augmented_input = layers.Concatenate()([monthly_input, quarterly_upsampled])

# # Monthly LSTM
# monthly_lstm = layers.LSTM(64, return_sequences=True)(monthly_augmented_input)

# # Output layer
# output = layers.TimeDistributed(layers.Dense(1))(monthly_lstm)

# model = tf.keras.Model(inputs=[monthly_input, quarterly_input], outputs=output)
# model.summary()



