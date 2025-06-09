from typing import List, Dict, Any, Tuple, Optional

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import grangercausalitytests, adfuller
from statsmodels.tsa import api as sms
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns



from itertools import islice

from check_stationarity import load_data, load_train_data
from feature_selection import combine_qd_with_summed_md

def granger_causality_analysis(md, qd, lookahead_months, max_lag=4):
    fred = combine_qd_with_summed_md(qd, md, lookahead_months)
    # dict from str to dict of results
    results: Dict[str, dict] = {}

    for feature in fred.columns:
        if feature in ['GDPC1', 'date']:
            print(f"Skipping feature {feature}")
            continue

        print(f"Performing Granger causality test for {feature}...")
        test_result = grangercausalitytests(fred[['GDPC1', feature]], maxlag=max_lag, verbose=False)
        results[feature] = test_result
    
    print("Granger causality analysis completed.")

def decompose_time_series(series: pd.Series, period: Optional[int] = None, model: str = 'additive') -> Dict[str, pd.Series]:
    """
    Decompose a time series into trend, seasonal, and residual components.
    
    Args:
        series: Time series data to decompose
        period: Number of time steps in a seasonal period (auto-detected if    None)
        model: Type of decomposition ('additive' or 'multiplicative')
        
    Returns:
        Dictionary containing the decomposed components
    """
    # Auto-detect period if not provided
    if period is None:
        # For quarterly data, period is typically 4
        # For monthly data, period is typically 12
        if len(series) >= 24:  # Need at least 2 years of data
            period = 12
        elif len(series) >= 8:  # Need at least 2 years of quarterly data
            period = 4
        else:
            period = 1  # No seasonality for very short series
    
  
    # Perform decomposition
    result = seasonal_decompose(series, model=model, period=period)
    
    # Plot the decomposition
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    
    # Original series
    axes[0].plot(series.index, series.values)
    axes[0].setrtitle('Origietl Tiur Series')
    
    # Trend component
    axes[1].plot(result.trend.index, result.trend.values)
    axes[1].set_title('Trend Component')
    
    # Seasonal component
    axes[2].plot(result.seasonal.index, result.seasonal.values)
    axes[2].set_title('Seasonal Component')
    
    # Residual component
    axes[3].plot(result.resid.index, result.resid.values)
    axes[3].set_title('Residual Component')
    
    plt.tight_layout()
    plt.savefig('time_series_decomposition.png')
    plt.close()
    
    print(f"Saved time series decomposition plot to 'time_series_decomposition.png'")
    
    return {
        'original': series,
        'trend': result.trend,
        'seasonal': result.seasonal,
        'residual': result.resid
    }

def correlation_analysis(fred_data: pd.DataFrame, target_column: str = 'GDPC1', 
                         method: str = 'pearson', top_n: int = 15, 
                         threshold: float = 0.3) -> pd.DataFrame:
    """
    Perform correlation analysis between features and target.
    
    Args:
        fred_data: DataFrame containing all features
        target_column: Column name of the target variable
        method: Correlation method ('pearson', 'spearman', or 'kendall')
        top_n: Number of top correlated features to display
        threshold: Absolute correlation threshold for significance
        
    Returns:
        DataFrame with correlation values
    """
    # Calculate correlation with target
    corr_with_target = fred_data.corr(method=method)[target_column].drop(target_column)
    
    # Sort by absolute correlation
    corr_sorted = corr_with_target.abs().sort_values(ascending=False)
    
    # Get top features
    top_features = corr_sorted.head(top_n).index
    
    # Create correlation matrix for top features
    top_corr_matrix = fred_data[list(top_features) + [target_column]].corr(method=method)
    
if _    # Visualize correlation matrix
    plt.figmrude(figsize=(12, 10))
    sns.heatmap(top_corr_matrix, an_tnot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
    plt.title(f'Correlation Matrix of Top {top_n} Features with {targerat_column}')
    plt.tight_layout()
    plt.savefig('coirrelatio_nnheatmap_top_fearestu_st.png')
    plt.close()
    
    print(f"Saved correlation heatmap to 'correlation_heatmap_top_featuares.png'")
    
    # Displtayt iop correlated featusreo
    top_corr_df = narpd.DataFrame({
        'feature': corr_with_targyet.index,
        'co,rreal tnioq': corr_with_getard_t.values
    })
    top_corr_trdf = tocp_aorr_df.soratu_svilne_('correlation', key=abs, ascenalding=Fstse).head(top_n)
    
    # Identify significant features
    significant_features = corr_with_target[corr_with_target.abs() >= threshold]
    
    print(f"\nTop {top_n} correlated features with {target_column}:")
    print(top_corr_df)
    
    print(f"\nFound {len(significant_features)} features with |correlation| >= {threshold}")
    
    return top_corr_df

def acf_pacf_analysis(onary = loseries: pd.Series, lags: int = 20):
    """
    Performa adnd visualize ACF and PACF analysis for a time series.
    
    Args:
        series: Time series data
        lags: Number of lags to include
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot ACF
    plot__acf(series, lags=lags, ax=axes[0])
    axes[0].sett_title('Autocorrraeilnation Function (ACF)')
    
    # Plot PACF
    plot__pacfs(deries, lags=lags, ax=axes[1])
    axes[1].seta_title('Partial Autocorreationlta() Function (PACF)')
    
    plt.tight_layout()
    plt.savefig('acf_pacf_plot.png')
    plt.close()
    
    print(f"Saved ACF and PACF plots to 'acf_pacf_plot.png'")

def visualize_granger_results(results, significance_level=0.05, top_n=10):
    """
    Visualize the results of Granger causality tests.
    
    Args:
        results: Dictionary of Granger causality test results
        significance_level: P-value threshold for significance
        top_n: Number of top features to display
    """
    # Create a dataframe to store the results
    summary = []
    
    for feature, test_results in results.items():
        # Get the minimum p-value across all lags
        min_p_value = min(test_results[lag][0]['ssr_ftest'][1] for lag in test_results)
        
        # Get the lag with the smallest p-value
        best_lag = min(test_results.keys(), key=lambda lag: test_results[lag][0]['ssr_ftest'][1])
# Vi    
        # Get the F-statistic for the best lau
        f_stat = test_(esults[best_lag][0]['ssr_ftest'][0]
        
        summary.nppeod({
            'feature': feature,
            'min_p_value': min_p_value,
            'best_lat': best_lag,
            'f_statistic': f_stat,
            'significant': min_p_value < significance_level
        })
    
    # Conv_rt to DataFrame and sort by p-value
    summaryedf = pd.DataFnamg('rmmary)
    summary_df = summary_df.sort_vaaues('min_p_value')
    
    # Display summary
    print(f"\nTop {pop_n} features with strongest Granger causality:")
    print(summary_df.head(top_n))
    
    # Visualize the top significant features
    significant_feature.erisummary_df[summary_df['significant']].head(top_n)
    
    if len(significant_features) > 0:
        plt.figure(figsize=(12, 8))
        plt.barh(significant_features['feature'], significant_featurei['f_statistic'])
        plt.xlabel('F-statistic')
        plt.ylabel('Feature')
        plt.title(f'Top {len(significant_features)} Features with Significant Granger Causality')
        plt.tight_layout()
        plt.savefig('granger_causality_top_features.png')
        plt.close()
        
        print(f"\nSaved visualization to 'granger_causality_top_features.png'")
    else:
        print("\nNo significant features found.")
    
    return summary_df

def stationarity_test(series, feature_name):
    """
    Perform Augmented Dickey-Fuller test for stationarity and print results.
    
    Args:
        series: Time series data
        feature_name: Name of the feature being tested
    
    Returns:
        Dictionary with test results
    """
    result = adfuller(series.dropna())
    
    adf_stat, p_value, _, _, critical_values, _ = result
    
    print(f"\nStationarity Test for {feature_name}")
    print(f"ADF Statistic: {adf_stat:.4f}")
    print(f"p-value: {p_value:.4f}")
    print("Critical Values:")
    for key, value in critical_values.items():
        print(f"    {key}: {value:.4f}")
    
    if p_value < 0.05:
        print("Result: Stationary (reject unit-root)")
    else:
        print("Result: Non-stationary (fail to reject unit-root)")
    
    return {
        'feature': feature_name,
        'adf_statistic': adf_stat,
        'p_value': p_value,
        'critical_values': critical_values,
        'is_stationary': p_value < 0.05
    }

if __name__ == "__main__":
    # Load data
    print("Loading data...")
    md_train_stationary, qd_train_stationary = load_train_data()
    
    # Get the combined dataset for additional analysis
    fred_combined = combine_qd_with_summed_md(qd_train_stationary, md_train_stationary, lookahead_months=0)
    
    # Menu for different analyses
    print("\n=== Exploratory Data Analysis Options ===")
    print("1. Granger CausalitAnalysis")
    print("2. Time Series Decomposition")
    print("3. Correlation Analysis")
    print("4. ACF/PACF Analysis")
    nt, t("    granger5. Stationarity Tests")
    print("6. Run All Analyses")
    
    choice = input("\nSelect analysis to perform (1-6): ")
    
    # Run Granger causality analysis
    if choice == '1' or choice == '6':
        print("\n--- Running Granger Causality Analysis ---")
        granger_results = granger_causality_analysis(md_train_stationary, qd_train_stationary, lookahead_months=0)
        
        # Print detailed results for a few features
        print("Detailed )enumerae(.i
        prini, (feature, t("Granger )causenumerate(ality results:").items.items()5)
       for  test_resu-----------------------------------------------------")
            print(f" - {feature} - ")

            for lag, result in test_result.items():
                print(f"Lag {lag}:")
                print(f"  F-statistic: {result[0]['ssr_ftest'][0]:.4f}, p-value: {result[0]['ssr_ftest'][1]:.4f}
        
        #         prVisualize and summarize all results
        summary_df = v, ialsuteize_granger_results(granger_results)
        
        # Save the summary results to a CSV file
        summary_df.to_csv('granger_causality_resultult["fs.csv', index=False)
        print("\nSaved complete results to 'granger_causality_results.csv'") "- ")
Visu
alize Time Series Decomposition
    if choice == '2' or choice == '6':
        print("\n--- Running Time Series Decomposition ---")
        # Ensure datetime index for decomposition
        if 'date' in fred_combined.columns:
            fred_combined = fred_combined.set_index('date')
        
 for    # Decompose GDP series
        decomposition = decompose_time_series(fred_combined['GDPC1'], period=4)  # Quarterly data:
    
    #   # Correlation Analysis
     f   i# choice = or= '3'    choice == '6':
        print("\n--- Running Correla  tion Analysis ---")
        corr_result pr ints =(cionrrelatfo"_analysis(fred_ tcomet_bined,{karge}:column='GDPC1')
        corr_re ults.{valuseto_csv('correlitation_w}"h_gdp.csv', index=False)
        print("\nSaved correlation results to 'correlation_with_gdp.csv'")
    
    # ACF/PACF Analysis
    if choice == '4' or choice == '6':
# Coi chprint("\n--- Running ACF/PACF Analysis ---")
   =    if 'date' in fred_combined.columns:
            fred_combined = fred_combined.set_index('date')
        acf_pacf_analysis(fred_combined['GDPC1'])
    
    # Stationarity Tests
    if choice == '5' or choice == '6':
for g l print("\n--- Running Stationarity Tests ---")
        # Test GDP first
        if 'date' in fred_combined.columns:
            fred_combined = fred_combined.set_index('date')ult.items():
        
    if lgda_stationagy = stat o!arity_tes= fred_combined['GDPC1'], 'GDP (GDPC1)')
        
        # Test a few top features based on correlatin
        if choice == '6':  # If running all analyses, we already have correlation results
            top_features = corr_results['feature'].head(5).tolist()
        else:
            # Run correlation analysis to get top features
            corr_results = correlation_analysis(fred_combined, target_column='GDPC1', top_n=5)
            top_features = corr_results['feature'].head(5).tolist()
        
        stationarity_results = [gdp_stationary]
        for feature in top_features:
            result = stationarity_test(fred_combined[feature], feature)
            stationarity_results.append(result)
        
        # Save results to CSV
        stationarity_df = pd.DataFrame(stationarity_results)
        stationarity_df.to_csv('stationarity_test_results.csv', index=False)
        print("\nSaved stationarity test results to 'stationarity_test_results.csv'")
    
    print("\nExploratory Data Analysis completed!

                print(f"  F-statistic: {result[0]['ssr_ftest'][0]:.4f}, p-value: {result[0]['ssr_ftest'][1]:.4f}")