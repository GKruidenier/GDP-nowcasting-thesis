import sys

from lstm_123 import train_and_evaluate_model, instantiate_model_duplicate_qd, instantiate_model_multilayer, instantiate_model_alternating_rnn, create_datapoints_lstm_1_2_and_3
from univariate_RNN import instantiate_univariate_model, create_datapoints_univariate
from multivariate_RNN import instantiate_multivariate_model, create_datapoints_multivariate
from check_stationarity import load_train_data, load_test_data

from keras import layers
from functools import partial


# param_dict = {
#     "feature_selection_method": feature_selection_method,
#     "n_features": n_features,
#     "dropout_rate": dropout_rate,
#     "optimizer": optimizer,
#     "batch_size": batch_size,
#     "hidden_units": hidden_units,
#     "sequence_length": sequence_length,
#     "learning_rate": learning_rate,
#     "add_recession_feature": add_recession_feature,
# }

# train_process = subprocess.run(["python", "train_with_hyper_parameters.py", "--model", model_id, "--rnn", rnn_name, "--feature-selection-method", feature_selection_method, "--n-features", str(n_features), "--dropout-rate", str(dropout_rate), "--optimizer", optimizer, "--batch-size", str(batch_size), "--hidden-units", str(hidden_units), "--learning-rate", str(learning_rate), "--add-recession-feature", str(add_recession_feature)], capture_output=True, text=True)

def main():
    try:
        print(" ".join(sys.argv))
        # Read hyperparameters from command line arguments
        model_id = read_str("--model")
        rnn_name = read_str("--rnn")
        feature_selection_method = read_str("--feature-selection-method")
        n_features = read_int("--n-features")
        dropout_rate = read_float("--dropout-rate")
        optimizer = read_str("--optimizer")
        batch_size = read_int("--batch-size")
        hidden_units = read_int("--hidden-units")
        learning_rate = read_float("--learning-rate")
        add_recession_feature = read_bool("--add-recession-feature")
    
        sequence_length = 12

        if model_id == '1':
            model = instantiate_model_duplicate_qd
            create_datapoints = create_datapoints_lstm_1_2_and_3
        elif model_id == '2':
            model = instantiate_model_multilayer
            create_datapoints = create_datapoints_lstm_1_2_and_3
        elif model_id == '3':
            model = instantiate_model_alternating_rnn
            create_datapoints = create_datapoints_lstm_1_2_and_3
        elif model_id == 'univariate':
            model = instantiate_univariate_model
            create_datapoints = create_datapoints_univariate
        elif model_id == 'multivariate':
            model = instantiate_multivariate_model
            create_datapoints = create_datapoints_multivariate

        if rnn_name.lower() == 'lstm':
            rnn = layers.LSTM
        elif rnn_name.lower() == 'gru':
            rnn = layers.GRU

        md_train, qd_train = load_train_data()
        md_test, qd_test = load_test_data()

        # Call the training function with the hyperparameters
        val_losses, train_losses = train_and_evaluate_model(
            md_train, qd_train,
            md_test_stationary=md_test, qd_test_stationary=qd_test,

            feature_selection_method=feature_selection_method,
            n_features=n_features,
            dropout_rate=dropout_rate,
            optimizer=optimizer,
            batch_size=batch_size,
            hidden_units=hidden_units,
            sequence_length=sequence_length,
            learning_rate=learning_rate,
            add_recession_feature=add_recession_feature,
            instantiate_model=partial(model, Rnn=rnn),
            create_datapoints=create_datapoints,
            verbose=False,
            model_description=f"{model_id}; {rnn_name}"
        )

        print(sum(train_losses) / len(train_losses))
        print(sum(val_losses) / len(val_losses))
    except Exception as e:
        print("inf")
        print("inf")
        print(e, file=sys.stderr)
        import traceback
        print(traceback.print_exc(), file=sys.stderr)

def read_str(name):
    idx = sys.argv.index(name)
    if idx == -1:
        return None
    return sys.argv[idx + 1]

def read_int(name):
    str = read_str(name)
    if str is None:
        return None
    return int(str)

def read_float(name):
    str = read_str(name)
    if str is None:
        return None
    return float(str)

def read_bool(name):
    str = read_str(name)
    if str is None:
        return None
    if str.lower() == "true":
        return True
    elif str.lower() == "false":
        return False
    else:
        raise ValueError(f"Invalid boolean value: {str}")

if __name__ == "__main__":
    main()
