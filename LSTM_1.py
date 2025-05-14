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
from stationarization import load_train_data
from feature_selection import combine_qd_and_md_as_qd, remove_highly_correlated_features, lasso_feature_selection, tree_based_feature_selection, rfe_feature_selection
from tensorflow.keras.callbacks import EarlyStopping
should_plot = False
# Set the random seed for reproducibility
def set_seed(seed):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)

# Set the start of the script for logging
start_of_script = datetime.datetime.now().strftime("%Y-%m-%d %Hh%Mm%Ss")

def LSTM_model_1(md_train_stationary, qd_train_stationary, feature_selection_method, n_features, dropout_rate, optimizer, batch_size, hidden_units, sequence_length, learning_rate=0.001):
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
    train_md = md_train_stationary[monthly_features]
    train_qd = qd_train_stationary[quarterly_features]

    # Print the separated features for verification
    #print("Selected Monthly Features:")
    #print(train_md.columns)

    #print("Selected Quarterly Features:")
    #print(train_qd.columns)
    
    # Step 1: Repeat each quarterly observation 3 times
    quarterly_repeated_values = train_qd.loc[train_qd.index.repeat(3)].reset_index(drop=True)

    # Step 2: Use the last 747 months (to match the repeated quarterly series)
    aligned_monthly = train_md # .iloc[-len(quarterly_repeated_values):].copy()

    if len(quarterly_repeated_values) != len(aligned_monthly):
        raise Exception("len(quarterly_repeated_values) != len(aligned_monthly)", len(quarterly_repeated_values), '!=', len(aligned_monthly))
    else:
        print("quarterly and monthly have same len:", len(quarterly_repeated_values), '==', len(aligned_monthly))

    # Step 3: Assign matching monthly index to repeated quarterly values
    quarterly_repeated_values.index = aligned_monthly.index

    # Final aligned series
    aligned_quarterly = quarterly_repeated_values.copy()

    # Confirm shapes
    print(f"{aligned_monthly.shape=}"), print(F"{aligned_quarterly.shape=}")

    X_train_val_monthly = aligned_monthly
    X_train_val_quarterly = aligned_quarterly

    initial_train_size = 20
    val_size = 10
    # initial_train_size = 80
    # val_size = 20
    full_train_val_size = 100

    lowest_validation_losses = []
    validation_predictions_per_split = []

    loss_plot_dir = f"results/loss_plot {start_of_script}"
    os.makedirs(loss_plot_dir, exist_ok=True)

    n = len(X_train_val_monthly)
    print("n:", n)

    for train_proportion in range(initial_train_size, full_train_val_size, val_size):
        validation_predictions_current_split = []

        tf.keras.backend.clear_session()
        # Step 6: Split training and validation sets
        end_train_data_idx = int(n * (train_proportion / 100))
        end_val_data_idx = int(n * ((train_proportion + val_size) / 100))
        print(f"end_train_data_idx: {end_train_data_idx}, end_val_data_idx: {end_val_data_idx}")

        X_train_monthly = X_train_val_monthly.iloc[:end_train_data_idx]
        X_val_monthly = X_train_val_monthly.iloc[end_train_data_idx:end_val_data_idx]

        X_train_quarterly = X_train_val_quarterly.iloc[:end_train_data_idx]
        X_val_quarterly = X_train_val_quarterly.iloc[end_train_data_idx:end_val_data_idx]

        month_start_training_data = X_train_monthly.index[0].strftime("%Y-%m")
        month_end_training_data = X_train_monthly.index[-1].strftime("%Y-%m")
        month_end_validation_data = X_val_monthly.index[-1].strftime("%Y-%m")

        print(f"Processing train-val split: start_train = {month_start_training_data}; end_train & start_val = {month_end_training_data}; val_end = {month_end_validation_data}")

        # Confirm shapes
        print(f"Train proportion: {train_proportion}%")
        print("Monthly:", X_train_monthly.shape, X_val_monthly.shape)
        print("Quarterly:", X_train_quarterly.shape, X_val_quarterly.shape)

        # Initialize the scaler
        scaler_md = StandardScaler()
        scaler_qd = StandardScaler()
        scaler_gdp = StandardScaler()

        # Fit the scaler on the training data and transform training, validation data
        X_train_monthly_scaled = pd.DataFrame(scaler_md.fit_transform(X_train_monthly), 
                                            index=X_train_monthly.index, 
                                            columns=X_train_monthly.columns)

        X_val_monthly_scaled = pd.DataFrame(scaler_md.transform(X_val_monthly), 
                                            index=X_val_monthly.index, 
                                            columns=X_val_monthly.columns)


        X_train_quarterly_scaled = pd.concat(
            [
                pd.DataFrame(scaler_qd.fit_transform(X_train_quarterly.drop(columns=["GDPC1"])), 
                            index=X_train_quarterly.index, 
                            columns=[c for c in X_train_quarterly.columns if c != "GDPC1"]),
                pd.DataFrame(scaler_gdp.fit_transform(X_train_quarterly[["GDPC1"]]),
                            index=X_train_quarterly.index, 
                            columns=["GDPC1"]),
            ],
            axis=1
        )

        X_val_quarterly_scaled = pd.concat(
            [
                pd.DataFrame(scaler_qd.transform(X_val_quarterly.drop(columns=["GDPC1"])), 
                            index=X_val_quarterly.index, 
                            columns=[c for c in X_val_quarterly.columns if c != "GDPC1"]),
                pd.DataFrame(scaler_gdp.transform(X_val_quarterly[["GDPC1"]]),
                            index=X_val_quarterly.index, 
                            columns=["GDPC1"]),
            ],
            axis=1
        )

        # Confirm shapes
        print("Monthly Scaled:", X_train_monthly_scaled.shape, X_val_monthly_scaled.shape)
        print("Quarterly Scaled:", X_train_quarterly_scaled.shape, X_val_quarterly_scaled.shape)

        plot_scaled_data(X_train_monthly_scaled, X_val_monthly_scaled, X_train_quarterly_scaled, X_val_quarterly_scaled)

        # Step 1: Initialize empty lists for validation x and y
        sequence_length = 12 # Number of months (so one year)

        # Step 1: Initialize empty lists for x and y
        x_data_monthly = []
        x_data_quarterly = []
        y_data = []

        for i in range(len(X_train_monthly) - sequence_length - 3):  # -3 to account for the next quarter
            x_seq_monthly = X_train_monthly_scaled.iloc[i:i + sequence_length].values
            x_seq_quarterly = X_train_quarterly_scaled.iloc[i:i + sequence_length].values
            y_seq = X_train_quarterly_scaled.iloc[i + sequence_length + 2][["GDPC1"]]

            x_data_monthly.append(x_seq_monthly)
            x_data_quarterly.append(x_seq_quarterly)
            y_data.append(y_seq)

        # Step 3: Convert lists to numpy arrays
        x_train_md = np.array(x_data_monthly)
        x_train_qd = np.array(x_data_quarterly)
        y_train = np.array(y_data)

        # Output shapes
        print("x_train_md shape:", x_train_md.shape)
        print("x_train_qd shape:", x_train_qd.shape)
        print("y_train shape:", y_train.shape)

        # Step 1: Initialize empty lists for x and y
        x_data_monthly_val = []
        x_data_quarterly_val = []
        y_data_val = []
        val_dates = []

        for i in range(len(X_val_monthly) - sequence_length - 3):  # -3 to account for the next quarter
            x_seq_monthly_val = X_val_monthly_scaled.iloc[i:i + sequence_length].values
            x_seq_quarterly_val = X_val_quarterly_scaled.iloc[i:i + sequence_length].values
            y_seq_val = X_val_quarterly_scaled.iloc[i + sequence_length + 2][["GDPC1"]]

            x_data_monthly_val.append(x_seq_monthly_val)
            x_data_quarterly_val.append(x_seq_quarterly_val)
            y_data_val.append(y_seq_val)
            val_dates.append(X_val_monthly_scaled.index[i + sequence_length + 2])

        # Step 3: Convert lists to numpy arrays
        x_val_md = np.array(x_data_monthly_val)
        x_val_qd = np.array(x_data_quarterly_val)
        y_val = np.array(y_data_val)

        # Output shapes
        print("x_val_md shape:", x_val_md.shape)
        print("x_val_qd shape:", x_val_qd.shape)
        print("y_val shape:", y_val.shape)

        # Create the model

        # Step 3: LSTM model
        val_losses = [] # we are gonna train it 5 times and take the average of the second and third best model for increased stability

        for i in range(5):
            monthly_input = Input(shape=(sequence_length, x_train_md.shape[2]))
            quarterly_input = Input(shape=(sequence_length, x_train_qd.shape[2]))

            combined_input = Concatenate(axis=-1)([monthly_input, quarterly_input])

            set_seed(base_seed + (hash(i) % 1_000_000_000))
            x = LSTM(hidden_units, return_sequences=False)(combined_input)
            # x = tf.keras.layers.BatchNormalization()(x)
            # x = tf.keras.layers.Relu()(x)
            x = tf.keras.layers.Dropout(dropout_rate)(x)
            output = Dense(1)(x)

            model = Model(inputs=[monthly_input, quarterly_input], outputs=output)
            
            if optimizer == 'adam':
                optimizer_instance = tf.keras.optimizers.Adam(learning_rate=learning_rate)
            elif optimizer == 'RMSprop':
                optimizer_instance = tf.keras.optimizers.SGD(learning_rate=learning_rate)
            model.compile(optimizer=optimizer_instance, loss="mse")

            # Summary
            model.summary()

            # Add EarlyStopping callback
            early_stopping = EarlyStopping(
                monitor="val_loss",  # Monitor validation loss
                patience=5,          # Stop training if no improvement for 5 epochs
                restore_best_weights=True  # Restore the best weights after stopping
            )

            # Step 1: Train the model
            print("start taining")

            # Step 3: Evaluate the model on validation data
            initial_val_loss = model.evaluate([x_val_md, x_val_qd], y_val, verbose=0)
            print(f"Initial validation Loss: {initial_val_loss}")
            initial_train_loss = model.evaluate([x_train_md, x_train_qd], y_train, verbose=0)
            print(f"Initial train Loss: {initial_train_loss}")

            print("Tensorflow executing eagerly:", tf.executing_eagerly())
            print("x_train_md type:", type(x_train_md), "shape:", x_train_md.shape)
            print("x_train_qd type:", type(x_train_qd), "shape:", x_train_qd.shape)
            print("y_train type:", type(y_train), "shape:", y_train.shape)
            print("x_val_md type:", type(x_val_md), "shape:", x_val_md.shape)
            print("x_val_qd type:", type(x_val_qd), "shape:", x_val_qd.shape)
            print("y_val type:", type(y_val), "shape:", y_val.shape)
            x_train_md = tf.convert_to_tensor(x_train_md, dtype=tf.float32)
            x_train_qd = tf.convert_to_tensor(x_train_qd, dtype=tf.float32)
            y_train = tf.convert_to_tensor(y_train, dtype=tf.float32)
            x_val_md = tf.convert_to_tensor(x_val_md, dtype=tf.float32)
            x_val_qd = tf.convert_to_tensor(x_val_qd, dtype=tf.float32)
            y_val = tf.convert_to_tensor(y_val, dtype=tf.float32)

            history = model.fit(
                [x_train_md, x_train_qd],
                y_train,
                validation_data=([x_val_md, x_val_qd], y_val),
                epochs=15,
                batch_size=batch_size,
                verbose=1,
                shuffle=True,
                callbacks=[early_stopping],
            )

            # Step 2: Plot training and validation loss
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

            val_losses.append(min(history.history['val_loss']))

            # Step 3: Evaluate the model on validation data
            val_loss = model.evaluate([x_val_md, x_val_qd], y_val, verbose=0)
            validation_predictions_current_split.append((
                val_loss,
                scaler_gdp.inverse_transform(model.predict([x_val_md, x_val_qd], verbose=0)),
                scaler_gdp.inverse_transform(y_val),
                val_dates,
            ))
            print(f"Validation Loss: {val_loss}")
            train_loss = model.evaluate([x_train_md, x_train_qd], y_train, verbose=0)
            print(f"Train Loss: {train_loss}")

            plot_y_train_val(y_train, y_val)
        
        val_losses.sort()
        lowest_validation_losses.append(sum(val_losses[1:4]) / 3)
        validation_predictions_current_split.sort(key=lambda x: x[0])
        validation_predictions_per_split.append(validation_predictions_current_split[1]) # Select second best prediction

    validation_predictions_stationary = np.array([p for _, p, _, _ in validation_predictions_per_split]).flatten()
    validation_ground_truth_stationary = np.array([y for _, _, y, _ in validation_predictions_per_split]).flatten()
    flattened_val_dates = [d for _, _, _, dates in validation_predictions_per_split for d in dates]

    qd = pd.read_csv("FRED_QD.csv", index_col=0, parse_dates=True, date_format="%Y-%m-%d").iloc[2:]
    qd.index = pd.to_datetime(qd.index, format="%m/%d/%Y")
    gdp_qd = qd["GDPC1"].copy()
    gdp_qd = gdp_qd.loc[qd.index.repeat(3)]
    gdp_qd.index = pd.date_range(start=gdp_qd.index[0] - pd.DateOffset(months=5), periods=len(gdp_qd), freq="MS")
    print("gdp_qd length:", len(gdp_qd))
    gdp_qd = gdp_qd.loc[flattened_val_dates].values
    # start_idx = gdp_qd.index.get_loc(md_train_stationary.index[2]) - 1
    # end_idx = gdp_qd.index.get_loc(md_train_stationary.index[-1])
    # gdp_qd = gdp_qd.values
    print("gdp_qd shape:", gdp_qd.shape, "val_dates length:", len(flattened_val_dates))
    # gdp_qd = gdp_qd[start_idx:end_idx]
    # print("gdp_qd shape:", gdp_qd.shape)
    # repeat the values 3 times to match the length of the validation predictions
    # gdp_qd = np.repeat(gdp_qd, 3)

    # gdp_qd = gdp_qd.iloc[start_idx:end_idx]
    # gdp_qd = gdp_qd.loc[gdp_qd.index.repeat(3)]
    # gdp_qd.index = md_train_stationary.index - pd.DateOffset(months=1)

    validation_predictions = validation_predictions_stationary + gdp_qd
    validation_ground_truth = validation_ground_truth_stationary + gdp_qd

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

    with open(f"{loss_plot_dir}/val_loss {start_of_script}.txt", "w") as f:
        f.write(f"Lowest validation losses per fold: {lowest_validation_losses}\n")
        f.write(f"Average: {sum(lowest_validation_losses) / len(lowest_validation_losses)}\n")
        f.write(f"Min: {min(lowest_validation_losses)}\n")
        f.write(f"Max: {max(lowest_validation_losses)}\n")
        f.write(f"Unpreprocessed MSE: {mean_squared_error(validation_ground_truth, validation_predictions)}\n")
        for k, v in param_dict.items():
            f.write(f"{k}: {v}\n")
    
    return lowest_validation_losses

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

def plot_scaled_data(X_train_monthly_scaled, X_val_monthly_scaled, X_train_quarterly_scaled, X_val_quarterly_scaled):
    if should_plot:
            # Plot standardized monthly data
        plt.figure(figsize=(12, 8))
        for column in X_train_monthly_scaled.columns:
            plt.plot(X_train_monthly_scaled.index, X_train_monthly_scaled[column], label=column)
        plt.title("Standardized Monthly Data (Training Set)")
        plt.xlabel("Date")
        plt.ylabel("Standardized Values")
        plt.legend(loc="upper right", bbox_to_anchor=(1.3, 1))
        plt.grid(True)
        plt.show()

            # Plot standardized quarterly data
        plt.figure(figsize=(12, 8))
        plt.plot(X_train_quarterly_scaled.index, X_train_quarterly_scaled["GDPC1"], label="Standardized GDP (Training Set)", color="blue")
        plt.title("Standardized Quarterly Data (Training Set)")
        plt.xlabel("Date")
        plt.ylabel("Standardized GDP")
        plt.legend()
        plt.grid(True)
        plt.show()

            # Plot standardized monthly validation data
        plt.figure(figsize=(12, 8))
        for column in X_val_monthly_scaled.columns:
            plt.plot(X_val_monthly_scaled.index, X_val_monthly_scaled[column], label=column)
        plt.title("Standardized Monthly Data (Validation Set)")
        plt.xlabel("Date")
        plt.ylabel("Standardized Values")
        plt.legend(loc="upper right", bbox_to_anchor=(1.3, 1))
        plt.grid(True)
        plt.show()

            # Plot standardized quarterly validation data
        plt.figure(figsize=(12, 8))
        plt.plot(X_val_quarterly_scaled.index, X_val_quarterly_scaled["GDPC1"], label="Standardized GDP (Validation Set)", color="orange")
        plt.title("Standardized Quarterly Data (Validation Set)")
        plt.xlabel("Date")
        plt.ylabel("Standardized GDP")
        plt.legend()
        plt.grid(True)
        plt.show()



if __name__ == "__main__":
    md_train_stationary, qd_train_stationary = load_train_data()


    LSTM_model_1(
        md_train_stationary, 
        qd_train_stationary, 
        feature_selection_method='rfe', 
        n_features=15, 
        dropout_rate=0.2977115755027825, 
        optimizer='RMSprop', 
        #learning_rate=0.001, 
        batch_size=16, 
        hidden_units=32, 
        sequence_length=12,
    )