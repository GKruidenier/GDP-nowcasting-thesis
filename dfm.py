from statsmodels.tsa.statespace.varmax import VARMAX
from statsmodels.tsa.statespace.dynamic_factor import DynamicFactor
from check_stationarity import load_train_data
import pandas as pd
import numpy as np
import tools

# Load the data
md, qd = load_train_data()
# Fit a Dynamic Factor Model using state-space form
# We will use the original standardized data (X_scaled)
# Set k_factors to match number of latent factors (r)
dfm_model = DynamicFactor(X_scaled, k_factors=r, factor_order=1)
dfm_results = dfm_model.fit(disp=False)

# Extract smoothed states (estimated factors from Kalman filter)
kalman_factors = dfm_results.factors.smoothed

# Prepare dataframe for display
df_kalman_factors = pd.DataFrame(kalman_factors, index=X_df.index, columns=[f'Kalman_Factor{i+1}' for i in range(r)])

tools.display_dataframe_to_user(name="Kalman Smoothed Factors", dataframe=df_kalman_factors)

df_kalman_factors.head()
