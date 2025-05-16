# %%
import pandas as pd
import numpy as np
import tensorflow as tf
import random

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, root_mean_squared_error
from tensorflow import keras
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

import datetime
from check_stationarity import load_train_data

# %%

# Set the random seed for reproducibility
seed = 256
np.random.seed(seed)
tf.random.set_seed(seed)
random.seed(seed)

# train_md, train_qd = load_train_data()
_, train_data_univariate = load_train_data()
train_data_univariate = train_data_univariate["GDPC1"]

# Plot the original and transformed series
plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plt.plot(train_data_univariate, label="Original Series")
plt.title("Original Series")
plt.legend()



# %%
import matplotlib
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

ax1 = plt.subplot(211)
ax2 = plt.subplot(212)

# Plot acf and pacf
plot_acf(train_data_univariate)
plot_pacf(train_data_univariate, method="ywm")
ax1.tick_params(axis='both', labelsize=12)
ax2.tick_params(axis='both', labelsize=12)
# plt.show()


# %% [markdown]
# The blue regions the points are no longer statistically significant and from the plot we see the last lag that is statistically significant for autocorrelation plot ~2 and partial autocorrelation ~9

# %%
# Split train and val sets
initial_train_size = 20
val_size = 10
full_train_val_size = 100

lowest_validation_losses = []
validation_predictions_per_split = []

rmses = []

for train_proportion in range(initial_train_size, full_train_val_size, val_size):
    train_end = int(len(train_data_univariate) * (train_proportion / full_train_val_size))
    train = train_data_univariate.iloc[:train_end]
    val = train_data_univariate.iloc[int(len(train_data_univariate) * (train_proportion / full_train_val_size)):]

    # Build ARIMA model
    model = ARIMA(train, order=(2, 0, 2)).fit()

    forecasts = []

    # Copy the training data to use for recursive forecasting
    history = []  # Convert to a list for appending new values

    # Recursive forecasting
    for true_value in val:
        forecast = model.forecast(steps=1)[0]
        forecasts.append(forecast)

        model = model.append([true_value], refit=False)

        # Append the true value to the history for the next iteration
        history.append(true_value)
    
    rmse = root_mean_squared_error(forecasts, history)
    print("RMSE:", rmse)
    rmses.append(rmse)

    plt.plot(history, label="True Values", color="orange")
    plt.plot(forecasts, label="Forecasts", color="green")
    plt.title("ARIMA")
    plt.legend()
    plt.grid()

    # plt.show()

    print("history:", history)
    print("forecasts:", forecasts)

print("Mean RMSE:", np.mean(rmses))

# boxcox_diff_pred = model.forecast(len(val))
# # Inverse Box-Cox transformation
# forecast = inv_boxcox(boxcox_diff_pred, lambda_boxcox)




# %%
import itertools
import warnings

# Grid search for ARIMA model parameters
warnings.filterwarnings('ignore')

# Define parameter ranges
p = range(0, 5)
d = range(0, 1)
q = range(0, 5)
pdq = list(itertools.product(p, d, q))

# Initialize variables to track best model
best_aic = float("inf")
best_pdq = None
best_model = None

# Perform grid search
print("ARIMA Grid Search:")
print("=================")
for param in pdq:
    try:
        model = ARIMA(train, order=param)
        results = model.fit()
        aic = results.aic
        print(f"ARIMA{param} - AIC: {aic}")

        if aic < best_aic:
            best_aic = aic
            best_pdq = param
            best_model = results
    except Exception as e:
        continue

print("\nBest ARIMA model:")
print(f"ARIMA{best_pdq} - AIC: {best_aic}")

# Generate forecasts with the best model
forecasts = []
model = best_model

for true_value in val:
    forecast = model.forecast(steps=1)[0]
    forecasts.append(forecast)
    model = model.append([true_value], refit=False)

print("\nRMSE with best model:", root_mean_squared_error(val, forecasts))

# Plotting results
plt.figure(figsize=(12, 6))
plt.plot(val.index, val, label="True Values", color="orange")
plt.plot(val.index, forecasts, label="Forecasts", color="green")
plt.title(f"ARIMA{best_pdq} Forecasts")
plt.legend()
plt.grid()
# plt.show()

# %%
import plotly.graph_objects as go

def plot_forecasts(train: pd.DataFrame, val: pd.DataFrame, forecasts: list, title: str) -> None:
    """
    Function to plot the forecasts for GDP data.

    Parameters:
    - train: DataFrame containing the training data with a 'Date' column and 'GDP' column.
    - test: DataFrame containing the test data with a 'Date' column and 'GDP' column.
    - forecasts: List of forecasted values corresponding to the test data.
    - title: Title of the plot.
    """
    plt.figure(figsize=(12, 6))

    # Plot training data
    # plt.plot(train.index, inv_boxcox(train['GDPC1'], lambda_boxcox), label='Train', color='blue')

    # Plot validation data
    # plt.plot(val.index, inv_boxcox(val['GDPC1'], lambda_boxcox), label='Val', color='orange')

    # Plot forecast data
    plt.plot(val.index, forecasts, label='Forecast', color='green')

    # Add title and labels
    plt.title(title, fontsize=16)
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('GDP', fontsize=14)

    # Add legend
    plt.legend(fontsize=12)

    # Show grid
    plt.grid(True)

    # Display the plot
    # plt.show()

# Example usage
# Assuming `train`, `test`, and `forecasts` are already defined
# train: DataFrame with 'Date' and 'GDP' columns for training data
# test: DataFrame with 'Date' and 'GDP' columns for test data
# forecasts: List of forecasted GDP values
plot_forecasts(train, val, forecasts, 'ARIMA GDP Nowcasting')
mse = mean_squared_error(val, forecasts)
print("Mean Squared Error:", mse)



