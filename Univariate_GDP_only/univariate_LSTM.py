from lstm_123 import train_and_evaluate_model
import numpy as np
import tensorflow as tf
from tensorflow import keras


def create_datapoints_univariate(monthly, quarterly, sequence_length):
    sequence_length_quarterly = sequence_length // 3

    x_monthly = []
    x_quarterly = []
    x_shift = []
    y = []
    y_dates = []

    for q in range(len(quarterly) - sequence_length_quarterly - 1): # -1 to account for the next quarter
        m = q * 3
        m_end = m + sequence_length
        q_end = q + sequence_length_quarterly

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
