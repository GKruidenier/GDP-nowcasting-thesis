import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.statespace.dynamic_factor import DynamicFactor
from statsmodels.tsa.statespace.dynamic_factor_mq import DynamicFactorMQ

from check_stationarity import load_train_data

import time
now = time.time()

# Load the datasets
monthly_df, quarterly_df = load_train_data()

# Prepare GDP series aligned to monthly index
gdp = quarterly_df['GDPC1']
gdp_monthly = gdp.reindex(monthly_df.index)  # NaN in non-quarter-end months

# Select a subset of monthly indicators for modeling
# selected_indicators = monthly_df.columns[:40]
# X = monthly_df[selected_indicators].copy()
# X = pd.concat([monthly_df, quarterly_df.reindex(monthly_df.index)], axis=1)

# Standardize monthly data
monthly_scaler = StandardScaler()
quarterly_scaler = StandardScaler()
monthly_scaled = pd.DataFrame(monthly_scaler.fit_transform(monthly_df), index=monthly_df.index, columns=monthly_df.columns)
quarterly_scaled = pd.DataFrame(quarterly_scaler.fit_transform(quarterly_df), index=quarterly_df.index, columns=quarterly_df.columns)

# Split GDP into training and validation
train_val = pd.concat([monthly_scaled, quarterly_scaled], axis=1)

split_idx = int(len(train_val) * 0.8)
train = train_val.iloc[:split_idx]
val = train_val.iloc[split_idx:]

# Fit the Dynamic Factor Model to the training data
dfm = DynamicFactorMQ(train, k_endog_monthly=len(monthly_df.columns), factors=10, factor_orders=1)
dfm_result = dfm.fit(disp=10)

end = time.time()
diff = end - now
print("took", diff, "seconds")

f = dfm_result.forecast(steps=1)

print(f)
print(f["GDPC1"])
exit()

# Extract latent factor and regress GDP on it
factor_train = pd.Series(dfm_result.filtered_state[0], index=X_train.index)
common_idx = gdp_train.dropna().index.intersection(factor_train.index)
gdp_train_values = gdp_train.loc[common_idx]
gdp_factor_train = factor_train.loc[common_idx]
reg_coef = np.polyfit(gdp_factor_train, gdp_train_values, 1)

# Refit the model on the full dataset for prediction
dfm_full = DynamicFactor(X_scaled, k_factors=1, factor_order=1)
dfm_full_result = dfm_full.fit(disp=False)

# Get smoothed latent factor for the full period
factor_full = pd.Series(dfm_full_result.smoothed_state[0], index=X_scaled.index)

# Predict GDP in validation period
gdp_factor_val = factor_full.loc[gdp_val.index].dropna()
gdp_pred_val = reg_coef[0] * gdp_factor_val + reg_coef[1]
gdp_actual_val = gdp_val.loc[gdp_pred_val.index]

# Align and clean predictions
valid_idx = gdp_actual_val.dropna().index.intersection(gdp_pred_val.dropna().index)
gdp_actual_val_clean = gdp_actual_val.loc[valid_idx]
gdp_pred_val_clean = gdp_pred_val.loc[valid_idx]

# Evaluate
rmse = np.sqrt(mean_squared_error(gdp_actual_val_clean, gdp_pred_val_clean))
mae = mean_absolute_error(gdp_actual_val_clean, gdp_pred_val_clean)

print(rmse, mae)
end = time.time()
diff = end - now
print("took", diff, "seconds")