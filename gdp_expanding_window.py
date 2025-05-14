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

import datetime

start_of_script = datetime.datetime.now().strftime("%Y-%m-%d %H_%M_%S")

should_plot = False

# Load datasets
fred_md = pd.read_csv("FRED_MD.csv", index_col=0, parse_dates=True, date_format="%m/%d/%Y")
fred_qd = pd.read_csv("FRED_QD.csv", index_col=0, parse_dates=True, date_format="%m/%d/%Y")

# Drop metadata rows (first row for MD, first two for QD)
fred_md = fred_md.iloc[1:].apply(pd.to_numeric, errors='coerce')
fred_qd = fred_qd.iloc[2:].apply(pd.to_numeric, errors='coerce')

# Convert index to datetime (if not already)
fred_md.index = pd.to_datetime(fred_md.index)
fred_qd.index = pd.to_datetime(fred_qd.index)

# Define selected features
# monthly_features = [
#     "VIXCLSx", "GS10", "TB3MS", "CPIAUCSL", "INDPRO", "M2SL",
#     "UNRATE", "PAYEMS", "HOUST", "RETAILx", "BUSINVx", "FEDFUNDS", "M1SL", "PERMIT"
# ]

monthly_features = [
    "VIXCLSx", "GS10", "TB3MS", "CPIAUCSL", "INDPRO",
    "UNRATE", "PAYEMS", "HOUST", "RETAILx", "BUSINVx", "FEDFUNDS", "PERMIT"
]
quarterly_features = ["GDPC1"]

# Subset features
monthly_selected = fred_md[monthly_features]
quarterly_selected = fred_qd[quarterly_features]

# Trim between 1962-07-01 and ...
start_date = pd.Timestamp("1962-07-01")
end_date = pd.Timestamp("2019-12-28")

monthly_trimmed = monthly_selected.loc[(monthly_selected.index >= start_date) & (monthly_selected.index <= end_date)]
quarterly_trimmed = quarterly_selected.loc[(quarterly_selected.index >= start_date) & (quarterly_selected.index <= end_date)]

# --- Final check ---
print("Date Range:", start_date.date(), "to", end_date.date())
print("Monthly shape:", monthly_trimmed.shape)
print("Quarterly shape:", quarterly_trimmed.shape)
print("Monthly missing values (null):\n", monthly_trimmed.isnull().sum())
print("Quarterly missing values (null):\n", quarterly_trimmed.isnull().sum())

print("Monthly missing values (na):\n", monthly_trimmed.isna().sum())
print("Quarterly missing values (na):\n", quarterly_trimmed.isna().sum())

monthly_missing_trimmed = monthly_trimmed.isna().sum().sort_values(ascending=False)
quarterly_missing_trimmed = quarterly_trimmed.isna().sum().sort_values(ascending=False)

# Filter to only show features with any missing values
monthly_missing_trimmed = monthly_missing_trimmed[monthly_missing_trimmed > 0]
quarterly_missing_trimmed = quarterly_missing_trimmed[quarterly_missing_trimmed > 0]

monthly_missing_trimmed, quarterly_missing_trimmed

from statsmodels.tsa.stattools import adfuller

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

stationarity_monthly = check_stationarity(monthly_trimmed)
stationarity_quarterly = check_stationarity(quarterly_trimmed)

# %%
# Apply differencing to non-stationary monthly features
non_stationary_monthly = stationarity_monthly[stationarity_monthly["Stationary"] == False].index.tolist()
monthly_stationary = monthly_trimmed.copy()
monthly_stationary[non_stationary_monthly] = monthly_stationary[non_stationary_monthly].diff()
monthly_stationary = monthly_stationary.iloc[3:]  # Drop the first three months (1 quarter) after differencing

# Apply differencing to non-stationary quarterly features
non_stationary_quarterly = stationarity_quarterly[stationarity_quarterly["Stationary"] == False].index.tolist()
quarterly_stationary = quarterly_trimmed.copy()
quarterly_stationary[non_stationary_quarterly] = quarterly_stationary[non_stationary_quarterly].diff()
quarterly_stationary = quarterly_stationary.iloc[1:]  # Drop the first quarter (three months) after differencing

# Re-check stationarity
stationarity_monthly_after = check_stationarity(monthly_stationary)
stationarity_quarterly_after = check_stationarity(quarterly_stationary)

assert stationarity_monthly_after["Stationary"].all(), "Not all monthly features are stationary after differencing." + str(stationarity_monthly_after[stationarity_monthly_after["Stationary"] == False])
assert stationarity_quarterly_after["Stationary"].all(), "Not all quarterly features are stationary after differencing." + str(stationarity_quarterly_after[stationarity_quarterly_after["Stationary"] == False])

# Step 1: Get list of stationary features (those that were not differenced)
stationary_monthly_cols = stationarity_monthly[stationarity_monthly["Stationary"] == True].index.tolist()
stationary_quarterly_cols = stationarity_quarterly[stationarity_quarterly["Stationary"] == True].index.tolist()

# Step 2: Create the final monthly dataset
# - Use differenced versions for non-stationary columns
# - Use original trimmed values for stationary columns
final_monthly_stationary = pd.concat([
    monthly_stationary[non_stationary_monthly],  # differenced
    monthly_trimmed[stationary_monthly_cols].loc[monthly_stationary.index]  # original
], axis=1).sort_index(axis=1)

# Step 3: Final quarterly dataset with only GDPC1 (already differenced if needed)
final_quarterly_stationary = quarterly_stationary[["GDPC1"]]

# Output dataset shapes and preview
final_monthly_shape = final_monthly_stationary.shape
final_quarterly_shape = final_quarterly_stationary.shape
final_monthly_stationary.head(), final_quarterly_stationary.head(), final_monthly_shape, final_quarterly_shape

if should_plot:
    # Plotting
    final_monthly_stationary.plot(subplots=True, figsize=(12, 20), title="Monthly Stationary Features")
    plt.tight_layout()
    plt.show()

    final_quarterly_stationary.plot(figsize=(10, 4), title="Quarterly Stationary GDP (GDPC1)")
    plt.tight_layout()
    plt.show()

# Model 1 preprocessing: repeating GDP values 

# Step 1: Repeat each quarterly observation 3 times
quarterly_repeated_values = final_quarterly_stationary.loc[final_quarterly_stationary.index.repeat(3)].reset_index(drop=True)

# Step 2: Use the last 747 months (to match the repeated quarterly series)
aligned_monthly = final_monthly_stationary# .iloc[-len(quarterly_repeated_values):].copy()

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

# Output sample
print("aligned_monthly:", aligned_monthly, sep="\n")
print("aligned_quarterly:", aligned_quarterly, sep="\n")

# Step 3: Split proportions
train_val_size = 0.85
test_size = 0.15

# Step 4: Split indices
n = len(aligned_monthly)
split_idx = int(n * train_val_size)

# Step 5: Create splits
X_train_val_monthly = aligned_monthly.iloc[:split_idx]
X_test_monthly = aligned_monthly.iloc[split_idx:]

X_train_val_quarterly = aligned_quarterly.iloc[:split_idx]
X_test_quarterly = aligned_quarterly.iloc[split_idx:]

# Confirm shapes
print("Monthly:", X_train_val_monthly.shape, X_test_monthly.shape)
print("Quarterly:", X_train_val_quarterly.shape, X_test_quarterly.shape)

initial_train_size = 15
val_size = 10
full_train_val_size = 85

lowest_validation_losses = []

loss_plot_dir = f"results/loss_plot {start_of_script}"
os.makedirs(loss_plot_dir, exist_ok=True)

for train_proportion in range(initial_train_size, full_train_val_size, val_size):
    # Step 6: Split training and validation sets
    end_train_data_idx = int(n * (train_proportion / 100))
    end_val_data_idx = int(n * ((train_proportion + val_size) / 100))

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
    print("Monthly:", X_train_monthly.shape, X_val_monthly.shape, X_test_monthly.shape)
    print("Quarterly:", X_train_quarterly.shape, X_val_quarterly.shape, X_test_quarterly.shape)

    # Initialize the scaler
    scaler = StandardScaler()

    # Fit the scaler on the training data and transform training, validation, and test sets
    X_train_monthly_scaled = pd.DataFrame(scaler.fit_transform(X_train_monthly), 
                                        index=X_train_monthly.index, 
                                        columns=X_train_monthly.columns)

    X_val_monthly_scaled = pd.DataFrame(scaler.transform(X_val_monthly), 
                                        index=X_val_monthly.index, 
                                        columns=X_val_monthly.columns)

    X_test_monthly_scaled = pd.DataFrame(scaler.transform(X_test_monthly), 
                                        index=X_test_monthly.index, 
                                        columns=X_test_monthly.columns)

    X_train_quarterly_scaled = pd.DataFrame(scaler.fit_transform(X_train_quarterly), 
                                            index=X_train_quarterly.index, 
                                            columns=X_train_quarterly.columns)

    X_val_quarterly_scaled = pd.DataFrame(scaler.transform(X_val_quarterly), 
                                        index=X_val_quarterly.index, 
                                        columns=X_val_quarterly.columns)

    X_test_quarterly_scaled = pd.DataFrame(scaler.transform(X_test_quarterly), 
                                        index=X_test_quarterly.index, 
                                        columns=X_test_quarterly.columns)

    # Confirm shapes
    print("Monthly Scaled:", X_train_monthly_scaled.shape, X_val_monthly_scaled.shape, X_test_monthly_scaled.shape)
    print("Quarterly Scaled:", X_train_quarterly_scaled.shape, X_val_quarterly_scaled.shape, X_test_quarterly_scaled.shape)

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

    for i in range(len(X_val_monthly) - sequence_length - 3):  # -3 to account for the next quarter
        x_seq_monthly_val = X_val_monthly_scaled.iloc[i:i + sequence_length].values
        x_seq_quarterly_val = X_val_quarterly_scaled.iloc[i:i + sequence_length].values
        y_seq_val = X_val_quarterly_scaled.iloc[i + sequence_length + 2][["GDPC1"]]

        x_data_monthly_val.append(x_seq_monthly_val)
        x_data_quarterly_val.append(x_seq_quarterly_val)
        y_data_val.append(y_seq_val)

    # Step 3: Convert lists to numpy arrays
    x_val_md = np.array(x_data_monthly_val)
    x_val_qd = np.array(x_data_quarterly_val)
    y_val = np.array(y_data_val)

    # Output shapes
    print("x_val_md shape:", x_val_md.shape)
    print("x_val_qd shape:", x_val_qd.shape)
    print("y_val shape:", y_val.shape)

    # Create the model

    # Step 1: Define inputs
    monthly_input = Input(shape=(sequence_length, x_train_md.shape[2]), name="monthly_input")
    quarterly_input = Input(shape=(sequence_length, x_train_qd.shape[2]), name="quarterly_input")

    # Step 2: Concatenate monthly + quarterly input along feature axis
    combined_input = Concatenate(axis=-1)([monthly_input, quarterly_input])

    # Step 3: LSTM model
    x = LSTM(32, return_sequences=False)(combined_input)[:, None]
    output = TimeDistributed(Dense(1))(x)[..., 0]

    # Step 4: Define model
    model = Model(inputs=[monthly_input, quarterly_input], outputs=output)
    model.compile(optimizer="adam", loss="mse")

    # Summary
    model.summary()

    # Step 1: Train the model
    print("start taining")

    # Step 3: Evaluate the model on validation data
    initial_val_loss = model.evaluate([x_val_md, x_val_qd], y_val, verbose=0)
    print(f"Initial validation Loss: {initial_val_loss}")
    initial_train_loss = model.evaluate([x_train_md, x_train_qd], y_train, verbose=0)
    print(f"Initial train Loss: {initial_train_loss}")

    history = model.fit(
        # [x_train_md[:400], x_train_qd[:400]],  # Inputs: monthly and quarterly sequences
        # y_train[:400],                  # Target: next quarter GDP
        # validation_data=([x_train_md[400:], x_train_qd[400:]], y_train[400:]),  # Validation data
        [x_train_md, x_train_qd],  # Inputs: monthly and quarterly sequences
        y_train,                  # Target: next quarter GDP
        validation_data=([x_val_md, x_val_qd], y_val),  # Validation data
        epochs=15,               # Number of epochs
        batch_size=32,           # Batch size
        verbose=1,                # Verbosity level'
        shuffle=True,           # Shuffle the data before each epoch
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

    lowest_validation_losses.append(min(history.history['val_loss']))

    # Step 3: Evaluate the model on validation data
    val_loss = model.evaluate([x_val_md, x_val_qd], y_val, verbose=0)
    print(f"Validation Loss: {val_loss}")
    train_loss = model.evaluate([x_train_md, x_train_qd], y_train, verbose=0)
    print(f"Train Loss: {train_loss}")

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

with open(f"{loss_plot_dir}/val_loss {start_of_script}.txt", "w") as f:
    f.write(f"Lowest validation losses per fold: {lowest_validation_losses}\n")
    f.write(f"Average: {sum(lowest_validation_losses) / len(lowest_validation_losses)}\n")
    f.write(f"Min: {min(lowest_validation_losses)}\n")
    f.write(f"Max: {max(lowest_validation_losses)}\n")

