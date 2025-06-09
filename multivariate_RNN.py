import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
import pandas as pd

def instantiate_multivariate_model(sequence_length, n_monthly_features, n_quarterly_features, n_hidden_units, optimizer_name, learning_rate, droupout_rate, Rnn=layers.LSTM):
    # Define model
    monthly_input = layers.Input(shape=(sequence_length // 3, n_monthly_features))
    quarterly_input = layers.Input(shape=(sequence_length // 3, n_quarterly_features))
    shift_input = layers.Input(shape=(sequence_length // 3, 1), dtype=tf.int32)

    full_input = layers.Concatenate(axis=-1)([monthly_input, quarterly_input])

    rnn = Rnn(n_hidden_units, return_sequences=False)(full_input)

    dropout = layers.Dropout(droupout_rate)(rnn)
    output = layers.Dense(1)(dropout)

    model = tf.keras.Model(inputs=[monthly_input, quarterly_input, shift_input], outputs=output)

    if optimizer_name == 'adam':
        optimizer_instance = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer_name == 'RMSprop':
        optimizer_instance = tf.keras.optimizers.SGD(learning_rate=learning_rate)

    model.compile(optimizer=optimizer_instance, loss="mse")

    return model

def create_datapoints_multivariate(monthly, quarterly, sequence_length):
    sequence_length_quarterly = sequence_length // 3

    x_monthly = []
    x_quarterly = []
    x_shift = []
    y = []
    y_dates = []

    for q in range(len(quarterly) - sequence_length_quarterly): # -1 to account for the next quarter
        m = q * 3
        m_end = m + sequence_length
        q_end = q + sequence_length_quarterly

        all_monthly = monthly.iloc[m:m_end]
        quarter = all_monthly.index.map(lambda d: d + pd.DateOffset(months=2 - ((d.month + 2) % 3)))

        summed_monthly = all_monthly.groupby(quarter).sum()
        print(f"{quarter=}")
        print(f"{summed_monthly.values.shape=}")

        x_monthly.append(
            summed_monthly.values
        )
        x_quarterly.append(
            quarterly.iloc[q:q_end].values
        )
        x_shift.append(
            np.arange(0, sequence_length // 3).reshape(-1, 1)
        )
        y.append(
            quarterly.iloc[q_end][["GDPC1"]]
        )
        y_dates.append(
            quarterly.index[q_end]
        )

    r = np.array(x_monthly), np.array(x_quarterly), np.array(x_shift), np.array(y), y_dates

    print(f"{r[0].shape=}")
    print(f"{r[1].shape=}")
    print(f"{r[2].shape=}")

    return r
