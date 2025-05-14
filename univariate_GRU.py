from functools import partial

import tensorflow as tf
from keras import layers

from univariate_RNN import create_datapoints_univariate, instantiate_univariate_model
from lstm_123 import train_and_evaluate_model
from check_stationarity import load_train_data


if __name__ == "__main__":
    md_train_stationary, qd_train_stationary = load_train_data()

    train_and_evaluate_model(
        md_train_stationary,
        qd_train_stationary,
        feature_selection_method='pca',
        n_features=15,
        dropout_rate=0.2977115755027825,
        optimizer='RMSprop',
        learning_rate=0.005,
        batch_size=16,
        hidden_units=32,
        sequence_length=12,
        create_datapoints=create_datapoints_univariate,
        instantiate_model=partial(instantiate_univariate_model, Rnn=layers.GRU),
    )