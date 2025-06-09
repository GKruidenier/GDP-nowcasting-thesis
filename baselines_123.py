import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import root_mean_squared_error, mean_squared_error, mean_absolute_error

import os
import datetime
from functools import partial
import random
import math

import statsmodels.api as sm

# from stationarization import load_train_data
from check_stationarity import load_train_data, load_test_data
from feature_selection import remove_highly_correlated_features, lasso_feature_selection, tree_based_feature_selection, rfe_feature_selection, pca_feature_selection, combine_qd_with_summed_md

should_plot = False
fast_mode = False

# Set the random seed for reproducibility
def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)

def train_and_evaluate_model(md_train_stationary, qd_train_stationary, feature_selection_method, n_features, add_recession_feature=True, model_name='dfm', model_params={}, verbose=True, model_description=None, md_test_stationary=None, qd_test_stationary=None):
    if fast_mode:
        md_train_stationary = md_train_stationary.iloc[:300]
        qd_train_stationary = qd_train_stationary.iloc[:100]

    has_test_data = md_test_stationary is not None and qd_test_stationary is not None

    md_train_stationary.index = pd.DatetimeIndex(md_train_stationary.index, freq="MS")
    qd_train_stationary.index = pd.DatetimeIndex(qd_train_stationary.index, freq="QS-MAR")
    if has_test_data:
        md_test_stationary.index = pd.DatetimeIndex(md_test_stationary.index, freq="MS")
        qd_test_stationary.index = pd.DatetimeIndex(qd_test_stationary.index, freq="QS-MAR")

    start_of_train_run = datetime.datetime.now().strftime("%Y-%m-%d %Hh%Mm%Ss")

    base_seed = 2571267# + 12093871
    set_seed(base_seed)

    if model_description is None:
        model_description = model_name

    param_dict = {
        "feature_selection_method": feature_selection_method,
        "n_features": n_features,
        "add_recession_feature": add_recession_feature,
        "fast_mode": fast_mode,
        "model": model_description if model_description is not None else model_name,
        "base_seed": base_seed,
    }
    if verbose:
        print("Parameters:", param_dict)

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
    validation_predictions_per_split = []

    loss_plot_dir = f"results/loss_plot {start_of_train_run} {model_description}"
    if verbose:
        os.makedirs(loss_plot_dir, exist_ok=True)

    number_of_quarters = len(train_val_quarterly)
    if verbose:
        print("n:", number_of_quarters)

    best_epoch = {}
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

        if model_name in ["arimax", "arima"]:
            train_endog = train_quarterly_scaled[["GDPC1"]]
            train_exog = pd.concat([
                train_monthly_scaled.groupby(train_monthly_scaled.index.map(lambda date: date - pd.DateOffset(months=date.month % 3))).mean().iloc[1:],
                train_quarterly_scaled.drop(columns=["GDPC1"])
            ], axis=1)

            val_endog = val_quarterly_scaled[["GDPC1"]]
            val_exog = pd.concat([ 
                val_monthly_scaled.groupby(val_monthly_scaled.index.map(lambda date: date - pd.DateOffset(months=date.month % 3))).mean().iloc[1:],
                val_quarterly_scaled.drop(columns=["GDPC1"])
            ], axis=1)

            if should_eval_test_set:
                test_endog = test_quarterly_scaled[["GDPC1"]]
                test_exog = pd.concat([
                    test_monthly_scaled.groupby(test_monthly_scaled.index.map(lambda date: date - pd.DateOffset(months=date.month % 3))).mean().iloc[1:],
                    test_quarterly_scaled.drop(columns=["GDPC1"])
                ], axis=1)

            train_endog = train_endog.iloc[1:]
            train_exog = train_exog.iloc[:-1]
            train_exog.index = train_endog.index

            val_exog = pd.concat([
                train_exog.iloc[-1:],
                val_exog.iloc[:-1]
            ])
            val_exog.index = val_endog.index

            if should_eval_test_set:
                test_exog = pd.concat([
                    train_exog.iloc[-1:],
                    test_exog.iloc[:-1]
                ])
                test_exog.index = test_endog.index

        training_repetition_count = 1
        median_idx = 0
        val_losses = []
        val_forecasts = []
        test_dates = []

        for training_repetition_idx in range(training_repetition_count):
            set_seed(base_seed + (hash(training_repetition_idx) + hash(89520)) % 100_000_000)

            if model_name == "arimax":
                print(train_exog.shape)
                print(train_exog)
                model_untrained = sm.tsa.SARIMAX(train_endog, exog=train_exog, order=(model_params['endog_lags'], 0, model_params['error_lags']))
                model = model_untrained.fit()
            elif model_name == "arima":
                model_untrained = sm.tsa.ARIMA(train_endog, order=(model_params['endog_lags'], 0, model_params['error_lags']))
                model = model_untrained.fit()
            elif model_name == "dfm":
                model_untrained = sm.tsa.DynamicFactorMQ(train_monthly_scaled, endog_quarterly=train_quarterly_scaled, factors=model_params['factors'], factor_orders=model_params['factor_orders'])
                model = model_untrained.fit(tolerance=1e-5)

            ground_truth = []
            forecasts = []
            val_dates = []

            prediction_repetitions = 3 if model_name == "dfm" else 1

            if model_name in ["arimax", "arima"]: 
                for i in range(0, len(val_endog)):
                    updated_endog = val_endog.iloc[i:i+1]
                    updated_exog = val_exog.iloc[i:i+1]

                    if model_name == "arimax":
                        forecast = model.forecast(steps=1, exog=updated_exog)
                        model = model.append(updated_endog, exog=updated_exog, refit=False)
                    elif model_name == "arima":
                        forecast = model.forecast(steps=1)
                        model = model.append(updated_endog, refit=False)

                    ground_truth.append(updated_endog["GDPC1"].values[0])
                    forecasts.append(forecast.values[0])
                    val_dates.append(val_endog.index[i])
            else:
                for i in range(len(val_quarterly)):
                    updated_monthly = val_monthly_scaled.iloc[i*3:(i+1)*3]
                    updated_quarterly = val_quarterly_scaled.iloc[i:i+1]

                    forecast = model.forecast(steps=3)
                    model = model.append(updated_monthly, endog_quarterly=updated_quarterly, refit=False)

                    ground_truth.append(updated_quarterly["GDPC1"].values[0])
                    forecasts.append(forecast["GDPC1"].values)
                    val_dates.append(val_quarterly.index[i])

            ground_truth = np.array(ground_truth)
            forecasts = np.array(forecasts)

            val_loss = root_mean_squared_error(ground_truth.reshape(-1, 1).repeat(prediction_repetitions, axis=-1), forecasts)
            val_mae = mean_absolute_error(ground_truth.reshape(-1, 1).repeat(prediction_repetitions, axis=-1), forecasts)

            val_losses.append(val_loss)
            val_forecasts.append(forecasts)

            if should_eval_test_set:
                test_ground_truth = []
                test_forecasts = []

                if model_name in ["arimax", "arima"]:
                    for i in range(0, len(test_endog)):
                        updated_endog = test_endog.iloc[i:i+1]
                        updated_exog = test_exog.iloc[i:i+1]

                        if model_name == "arimax":
                            forecast = model.forecast(steps=1, exog=updated_exog)
                            model = model.append(updated_endog, exog=updated_exog, refit=False)
                        elif model_name == "arima":
                            forecast = model.forecast(steps=1)
                            model = model.append(updated_endog, refit=False)

                        test_ground_truth.append(updated_endog["GDPC1"].values[0])
                        test_forecasts.append(forecast.values[0])
                        test_dates.append(test_endog.index[i])
                else:
                    for i in range(len(test_quarterly)):
                        updated_monthly = test_monthly_scaled.iloc[i*3:(i+1)*3]
                        updated_quarterly = test_quarterly_scaled.iloc[i:i+1]

                        forecast = model.forecast(steps=3)
                        model = model.append(updated_monthly, endog_quarterly=updated_quarterly, refit=False)

                        test_ground_truth.append(updated_quarterly["GDPC1"].values[0])
                        test_forecasts.append(forecast["GDPC1"].values)
                        test_dates.append(test_quarterly.index[i])

                test_ground_truth = np.array(test_ground_truth)
                test_forecasts = np.array(test_forecasts)

                test_loss = root_mean_squared_error(test_ground_truth.reshape(-1, 1).repeat(prediction_repetitions, axis=-1), test_forecasts)
                test_mae = mean_absolute_error(test_ground_truth.reshape(-1, 1).repeat(prediction_repetitions, axis=-1), test_forecasts)

                print(f"Test ground truth shape: {test_ground_truth.shape}")
                print(f"Test prediction shape: {test_forecasts.shape}")

                test_predictions.append({
                    'predictions': scaler_gdp.inverse_transform(test_forecasts.reshape(-1, 1)),
                    'ground_truth': scaler_gdp.inverse_transform(test_ground_truth.reshape(-1, 1)),
                    'rmse': test_loss,
                    'mae': test_mae,
                    'dates': test_dates,
                })
            # test_predictions = scaler_gdp.inverse_transform(model.predict([x_val_monthly, x_val_quarterly, x_val_shift], verbose=0).reshape(-1, 1))
            # test_ground_truth = scaler_gdp.inverse_transform(y_val)

            if verbose:
                print(f"Validation Loss: {val_loss}")

            validation_predictions_current_split.append((
                val_loss,
                val_mae,
                scaler_gdp.inverse_transform(forecasts.reshape(-1, 1)),
                scaler_gdp.inverse_transform(ground_truth.reshape(-1, 1)),
                val_dates,
            ))

            best_epoch[train_proportion].append({
                'val_loss': val_loss,
                'epoch': -1,
            })

        print(f"{val_losses=}")
        val_losses.sort()
        # lowest_validation_losses.append(sum(val_losses[1:-1]) / (training_repetition_count - 2))
        # lowest_validation_losses.append(np.sum(val_losses[1:-1], axis=0) / (training_repetition_count - 2))
        lowest_validation_losses.append(val_losses[median_idx])

        best_validation_run = 0
        best_validation_loss = float("inf")

        for i, val_loss in enumerate(val_losses):
            if val_loss < best_validation_loss:
                best_validation_loss = val_loss
                best_validation_run = i

        validation_predictions_current_split.sort(key=lambda x: x[0])
        validation_predictions_per_split.append(validation_predictions_current_split[median_idx]) # Select second best prediction

    median_test_predictions = test_predictions[best_validation_run]

    test_predictions.sort(key=lambda x: x['rmse'])
    median_test_predictions = test_predictions[median_idx]

    # val_loss,
    # val_mae,
    # scaler_gdp.inverse_transform(forecasts.reshape(-1, 1)),
    # scaler_gdp.inverse_transform(ground_truth.reshape(-1, 1)),
    # val_dates,
    validation_predictions_diff_log = np.array([p for _, _, ps, _, _ in validation_predictions_per_split for p in ps]).flatten()
    validation_ground_truth_diff_log = np.array([y for _, _, _, ys, _ in validation_predictions_per_split for y in ys]).flatten()
    test_predictions_diff_log: np.ndarray = median_test_predictions['predictions'].flatten()
    test_ground_truth_diff_log: np.ndarray = median_test_predictions['ground_truth'].flatten()

    prediction_repetitions = validation_predictions_diff_log.shape[0] // validation_ground_truth_diff_log.shape[0]

    assert prediction_repetitions * validation_ground_truth_diff_log.shape[0] == validation_predictions_diff_log.shape[0], f"Length of predictions is not an integer multiple of ground truth. Predictions counts: {validation_predictions_diff_log.shape[0]}, ground truth count: {validation_ground_truth_diff_log.shape[0]}"
    assert prediction_repetitions * test_ground_truth_diff_log.shape[0] == test_predictions_diff_log.shape[0], f"Length of predictions is not {prediction_repetitions} times that of the ground truth. Predictions counts: {test_predictions_diff_log.shape[0]}, ground truth count: {test_ground_truth_diff_log.shape[0]}"

    val_dates = [d for _, _, _, _, dates in validation_predictions_per_split for d in dates]
    repeated_val_dates = [d + pd.DateOffset(months=month_offset) for _, _, _, _, dates in validation_predictions_per_split for d in dates for month_offset in range(prediction_repetitions)]
    test_dates = median_test_predictions['dates']
    repeated_test_dates = [d + pd.DateOffset(months=month_offset) for d in test_dates for month_offset in range(prediction_repetitions)]

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

        validation_mean_diff_log = np.mean(validation_ground_truth_diff_log).repeat(validation_ground_truth_diff_log.shape[0])
        validation_predictions_usd = np.exp(validation_predictions_diff_log + gdp_log_val.repeat(prediction_repetitions))
        validation_ground_truth_usd = np.exp(validation_ground_truth_diff_log + gdp_log_val)
        validation_mean_predictions_usd = np.exp(validation_mean_diff_log + gdp_log_val)

        test_mean_diff_log = np.mean(test_ground_truth_diff_log).repeat(test_ground_truth_diff_log.shape[0])
        test_predictions_usd = np.exp(test_predictions_diff_log + gdp_log_test.repeat(prediction_repetitions))
        test_ground_truth_usd = np.exp(test_ground_truth_diff_log + gdp_log_test)
        test_mean_predictions_usd = np.exp(test_mean_diff_log + gdp_log_test)

        print(f"{len(val_dates)=}")
        print(f"{len(repeated_val_dates)=}")
        print(f"{validation_ground_truth_diff_log.shape=}")

        val_df = pd.DataFrame({
            "predictions_usd": validation_predictions_usd,
            "ground_truth_usd": validation_ground_truth_usd.repeat(prediction_repetitions),
            "predictions_diff_log": validation_predictions_diff_log,
            "ground_truth_diff_log": validation_ground_truth_diff_log.repeat(prediction_repetitions),
        }, index=repeated_val_dates)

        val_df.to_csv(f"{loss_plot_dir}/val idx={0} {model_description}.csv")

        test_df = pd.DataFrame({
            "predictions_usd": test_predictions_usd,
            "ground_truth_usd": test_ground_truth_usd.repeat(prediction_repetitions),
            "predictions_diff_log": test_predictions_diff_log,
            "ground_truth_diff_log": test_ground_truth_diff_log.repeat(prediction_repetitions),
        }, index=repeated_test_dates)
        test_df.to_csv(f"{loss_plot_dir}/test idx={0} {model_description}.csv")

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
        plt.title(r"Validation Predictions vs Actual $\Delta \log(GDP)$ for " + model_description)
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
        plt.title(r"Validation Predictions VS Actual Percentage GDP Growth for " + model_description)
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
        plt.title("Test Predictions vs Actual GDP in billion USD")
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
        plt.title(r"Test Predictions VS Actual Percentage GDP Growth for " + model_description)
        plt.xlabel("Date")
        plt.ylabel(r"$\% \Delta GDP$")
        plt.legend()
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.savefig(f"{loss_plot_dir}/test_predictions_percentage_growth.png")
        plt.close()

        with open(f"{loss_plot_dir}/val_loss {start_of_train_run}.txt", "w") as f:
            f.write(f"Lowest validation losses per fold: {lowest_validation_losses}\n")
            lowest_validation_losses = np.array(lowest_validation_losses)
            print(lowest_validation_losses)
            f.write(f"Average: {np.mean(lowest_validation_losses)}\n")
            f.write(f"RMSE: {np.sqrt(np.mean(lowest_validation_losses))}\n")
            # f.write(f"Min: {min(lowest_validation_losses)}\n")
            # f.write(f"Max: {max(lowest_validation_losses)}\n")
            f.write(f"Unpreprocessed MAE: {mean_absolute_error(validation_ground_truth_usd.repeat(prediction_repetitions), validation_predictions_usd)}\n")
            f.write(f"Unpreprocessed RMSE: {root_mean_squared_error(validation_ground_truth_usd.repeat(prediction_repetitions), validation_predictions_usd)}\n")
            f.write(f"Mean-predict Unpreprocessed MAE: {mean_absolute_error(validation_ground_truth_usd, validation_mean_predictions_usd)}\n")
            f.write(f"Mean-predict Unpreprocessed RMSE: {root_mean_squared_error(validation_ground_truth_usd, validation_mean_predictions_usd)}\n")
            f.write(f"Transformed MAE: {mean_absolute_error(validation_ground_truth_diff_log.repeat(prediction_repetitions), validation_predictions_diff_log)}\n")
            f.write(f"Transformed RMSE: {root_mean_squared_error(validation_ground_truth_diff_log.repeat(prediction_repetitions), validation_predictions_diff_log)}\n")
            f.write(f"Mean-predict Transformed MAE: {mean_absolute_error(validation_ground_truth_diff_log, validation_mean_diff_log)}\n")
            f.write(f"Mean-predict Transformed RMSE: {root_mean_squared_error(validation_ground_truth_diff_log, validation_mean_diff_log)}\n")

            f.write("\n")
            f.write(f"Test RMSE: {median_test_predictions['rmse']}\n")
            f.write(f"Test MAE: {median_test_predictions['mae']}\n")
            f.write(f"Unpreprocessed Test MAE: {mean_absolute_error(test_ground_truth_usd.repeat(prediction_repetitions), test_predictions_usd)}\n")
            f.write(f"Unpreprocessed Test RMSE: {root_mean_squared_error(test_ground_truth_usd.repeat(prediction_repetitions), test_predictions_usd)}\n")
            f.write(f"Mean-predict Unpreprocessed Test MAE: {mean_absolute_error(test_ground_truth_usd, test_mean_predictions_usd)}\n")
            f.write(f"Mean-predict Unpreprocessed Test RMSE: {root_mean_squared_error(test_ground_truth_usd, test_mean_predictions_usd)}\n")
            f.write(f"Transformed Test MAE: {mean_absolute_error(test_ground_truth_diff_log.repeat(prediction_repetitions), test_predictions_diff_log)}\n")
            f.write(f"Transformed Test RMSE: {root_mean_squared_error(test_ground_truth_diff_log.repeat(prediction_repetitions), test_predictions_diff_log)}\n")
            f.write(f"Mean-predict Test MAE: {mean_absolute_error(test_ground_truth_diff_log, test_mean_diff_log)}\n")
            f.write(f"Mean-predict Test RMSE: {root_mean_squared_error(test_ground_truth_diff_log, test_mean_diff_log)}\n")


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

    return lowest_validation_losses, None

if __name__ == "__main__":
    md_train_stationary, qd_train_stationary = load_train_data()
    md_test_stationary, qd_test_stationary = load_test_data()
    # md_test_stationary = md_test_stationary.loc[:"2018-12-31"]
    # qd_test_stationary = qd_test_stationary.loc[:"2018-12-31"]

    import sys

    model_name = sys.argv[-1]
    if model_name == "arimax":
        # Best parameters:
        # feature_selection_method: lasso
        # n_features: 16
        # add_recession_feature: False
        # endog_lags: 3
        # error_lags: 3
        train_and_evaluate_model(
            md_train_stationary,
            qd_train_stationary,
            md_test_stationary=md_test_stationary,
            qd_test_stationary=qd_test_stationary,

            feature_selection_method='lasso',
            n_features=16,
            add_recession_feature=False,
            model_params={
                'endog_lags': 3,
                'error_lags': 3,
            },
            model_name="arimax",

            verbose=True,
        )
    elif model_name == "arima":
        # Best parameters:
        # feature_selection_method: lasso
        # n_features: 16
        # add_recession_feature: False
        # endog_lags: 3
        # error_lags: 3
        train_and_evaluate_model(
            md_train_stationary,
            qd_train_stationary,
            md_test_stationary=md_test_stationary,
            qd_test_stationary=qd_test_stationary,

            feature_selection_method='lasso',
            n_features=16,
            add_recession_feature=False,
            model_params={
                'endog_lags': 2,
                'error_lags': 7,
            },
            model_name="arima",

            verbose=True,
        )
    elif model_name == "dfm":
        # Best parameters:
        # feature_selection_method: lasso
        # n_features: 20
        # add_recession_feature: False
        # factors: 4
        # factor_orders: 2
        train_and_evaluate_model(
            md_train_stationary,
            qd_train_stationary,
            md_test_stationary=md_test_stationary,
            qd_test_stationary=qd_test_stationary,

            feature_selection_method='lasso',
            n_features=20,
            add_recession_feature=False,
            model_params={
                'factors': 4,
                'factor_orders': 2,
            },
            model_name="dfm",

            verbose=True,
        )
    else:
        train_and_evaluate_model(
            md_train_stationary,
            qd_train_stationary,
            md_test_stationary=md_test_stationary,
            qd_test_stationary=qd_test_stationary,

            feature_selection_method='lasso',
            n_features=15,
            # feature_selection_method='none',
            # n_features=14,
            add_recession_feature=True,

            verbose=True,
            model_name="dfm",

            # feature_selection_method='lasso',
            # dropout_rate=0.14959366179554065,
            # optimizer='adam',
            # batch_size=64,
            # hidden_units=32,
            # learning_rate=0.0038974266096597505,
            # add_recession_feature=True,
            # n_features=15,

            # feature_selection_method='lasso',
            # dropout_rate=0.24968606797287904,
            # optimizer='adam',
            # batch_size=16,
            # hidden_units=32,
            # learning_rate=0.007972544770044396,
            # add_recession_feature=False,
            # n_features=13,
        )