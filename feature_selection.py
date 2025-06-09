import random
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
import math

# from check_stationarity import load_train_data
from sklearn.decomposition import PCA

from functools import partial
from typing import List, Tuple, Dict

def expert_selection(*ignored_args, **ignored_kwargs):
    return [
        "UMCSENTx",
        "NAPM",
        "PERMIT",
        "AMDMNOx",
        "S&P 500",
        "AWHMAN",
        "T10YFFM",
    ] + [
        "GPDIC1",
        "GDPC1",
    ]

from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.api import VAR
import numpy as np

def granger_causality_feature_selection(
    qd: pd.DataFrame, 
    md: pd.DataFrame, 
    target_column: str = "GDPC1", 
    n_features: int = 20, 
    max_lag: int = 4, 
    significance_level: float = 0.05,
    lookahead_months: int = 2,
    selection_criterion: str = "p_value",  # Options: "p_value", "f_stat", "coeff", "combined"
    visualize: bool = False
) -> Tuple[List[str], pd.DataFrame]:
    """
    Perform feature selection based on Granger causality tests.
    
    Parameters:
        qd (pd.DataFrame): Quarterly data DataFrame
        md (pd.DataFrame): Monthly data DataFrame
        target_column (str): Target column for Granger causality test (default: "GDPC1")
        n_features (int): Number of features to select
        max_lag (int): Maximum number of lags to test for Granger causality
        significance_level (float): P-value threshold for significance
        lookahead_months (int): Number of months to look ahead when combining monthly/quarterly data
        selection_criterion (str): Criterion for selecting features:
            - "p_value": Minimum p-value across lags (statistical significance)
            - "f_stat": Maximum F-statistic across lags (strength of causality)
            - "coeff": Coefficient magnitude from VAR model (effect size)
            - "combined": Combined score of F-statistic and coefficient magnitude
        visualize (bool): Whether to generate visualization of results
        
    Returns:
        selected_features (List[str]): List of selected feature names
        summary_df (pd.DataFrame): DataFrame with detailed Granger causality test results
    """
    # Combine quarterly and monthly data
    fred_combined = combine_qd_with_summed_md(qd, md, lookahead_months=lookahead_months)
    
    # Dictionary to store test results
    results = {}
    
    # Perform Granger causality tests for each feature
    for feature in fred_combined.columns:
        if feature == target_column or feature == 'date':
            continue
            
        # Test whether the feature Granger-causes the target
        try:
            # Use only complete rows (no NaN values)
            test_data = fred_combined[[target_column, feature]].dropna()
            
            # Skip features with insufficient data
            if len(test_data) < max_lag + 2:
                continue
                
            # Initialize feature results
            feature_results = {
                'p_values': [],
                'f_stats': [],
                'coeffs': [],
                'best_lag': None,
                'min_p_value': float('inf'),
                'max_f_stat': 0,
                'max_coeff': 0
            }
            
            # Run Granger causality test
            granger_test = grangercausalitytests(
                test_data, 
                maxlag=max_lag, 
                verbose=False
            )
            
            # Extract p-values and F-statistics for each lag
            for lag, lag_results in granger_test.items():
                p_value = lag_results[0]['ssr_ftest'][1]
                f_stat = lag_results[0]['ssr_ftest'][0]
                
                feature_results['p_values'].append((lag, p_value))
                feature_results['f_stats'].append((lag, f_stat))
                
                # Update min p-value and corresponding lag
                if p_value < feature_results['min_p_value']:
                    feature_results['min_p_value'] = p_value
                    feature_results['best_lag_p'] = lag
                
                # Update max F-statistic and corresponding lag
                if f_stat > feature_results['max_f_stat']:
                    feature_results['max_f_stat'] = f_stat
                    feature_results['best_lag_f'] = lag
            
            # If using coefficient magnitude, fit VAR model
            if selection_criterion in ['coeff', 'combined']:
                try:
                    # Fit VAR model for each lag and extract coefficients
                    var_model = VAR(test_data)
                    lag_coeffs = []
                    
                    for lag in range(1, max_lag + 1):
                        # Skip lags with insufficient data
                        if len(test_data) <= lag + 2:
                            lag_coeffs.append((lag, 0))
                            continue
                            
                        try:
                            model_fit = var_model.fit(lag)
                            
                            # Get coefficients of feature's impact on target
                            # Coefficients matrix shape: [n_vars, n_vars, lag]
                            # We want all lags of feature's impact on target
                            # Index 0: target equation, Index 1: feature
                            feature_coeffs = np.abs([model_fit.coefs[i, 0, 1] for i in range(lag)])
                            
                            # Use sum of absolute coefficients as measure of impact
                            coeff_magnitude = np.sum(feature_coeffs)
                            lag_coeffs.append((lag, coeff_magnitude))
                            
                            if coeff_magnitude > feature_results['max_coeff']:
                                feature_results['max_coeff'] = coeff_magnitude
                                feature_results['best_lag_c'] = lag
                                
                        except Exception as e:
                            lag_coeffs.append((lag, 0))
                            
                    feature_results['coeffs'] = lag_coeffs
                except Exception as e:
                    feature_results['coeffs'] = [(lag, 0) for lag in range(1, max_lag + 1)]
                    feature_results['max_coeff'] = 0
                    feature_results['best_lag_c'] = None
            
            results[feature] = feature_results
            
        except Exception as e:
            print(f"Error testing feature '{feature}': {str(e)}")
    
    # Create a summary DataFrame
    summary = []
    
    for feature, test_results in results.items():
        row = {
            'feature': feature,
            'min_p_value': test_results['min_p_value'],
            'best_lag_p': test_results.get('best_lag_p', None),
            'max_f_stat': test_results['max_f_stat'],
            'best_lag_f': test_results.get('best_lag_f', None),
            'significant': test_results['min_p_value'] < significance_level
        }
        
        if selection_criterion in ['coeff', 'combined']:
            row['max_coeff'] = test_results.get('max_coeff', 0)
            row['best_lag_c'] = test_results.get('best_lag_c', None)
            
        if selection_criterion == 'combined':
            # Normalize F-statistic and coefficient values to [0,1] range
            max_f_all = max(r['max_f_stat'] for r in results.values())
            max_c_all = max(r.get('max_coeff', 0) for r in results.values())
            
            if max_f_all > 0 and max_c_all > 0:
                norm_f = row['max_f_stat'] / max_f_all
                norm_c = row['max_coeff'] / max_c_all
                # Combined score: weighted average of normalized F-stat and coefficient
                row['combined_score'] = 0.5 * norm_f + 0.5 * norm_c
            else:
                row['combined_score'] = 0
        
        summary.append(row)
    
    # Convert to DataFrame and sort by the selected criterion
    summary_df = pd.DataFrame(summary)
    
    if selection_criterion == 'p_value':
        summary_df = summary_df.sort_values('min_p_value')
        ranking_col = 'min_p_value'
    elif selection_criterion == 'f_stat':
        summary_df = summary_df.sort_values('max_f_stat', ascending=False)
        ranking_col = 'max_f_stat'
    elif selection_criterion == 'coeff':
        summary_df = summary_df.sort_values('max_coeff', ascending=False)
        ranking_col = 'max_coeff'
    elif selection_criterion == 'combined':
        summary_df = summary_df.sort_values('combined_score', ascending=False)
        ranking_col = 'combined_score'
    
    # Select top n_features
    selected_features = summary_df['feature'].head(n_features).tolist()
    
    # Ensure the target column is included
    if target_column not in selected_features:
        selected_features.append(target_column)
    
    # Visualize results if requested
    if visualize:
        plt.figure(figsize=(12, 8))
        
        top_df = summary_df.head(min(n_features, 20))
        
        if selection_criterion == 'p_value':
            plt.barh(top_df['feature'], -np.log10(top_df['min_p_value']))
            plt.xlabel('-log10(p-value)')
            plt.title(f'Top Features by Granger Causality p-value')
        elif selection_criterion == 'f_stat':
            plt.barh(top_df['feature'], top_df['max_f_stat'])
            plt.xlabel('F-statistic')
            plt.title(f'Top Features by Granger Causality F-statistic')
        elif selection_criterion == 'coeff':
            plt.barh(top_df['feature'], top_df['max_coeff'])
            plt.xlabel('Coefficient Magnitude')
            plt.title(f'Top Features by VAR Coefficient Magnitude')
        elif selection_criterion == 'combined':
            plt.barh(top_df['feature'], top_df['combined_score'])
            plt.xlabel('Combined Score')
            plt.title(f'Top Features by Combined Granger Causality Score')
        
        plt.ylabel('Feature')
        plt.tight_layout()
        plt.savefig(f'granger_causality_{selection_criterion}.png')
        plt.close()
        
        print(f"Saved visualization to 'granger_causality_{selection_criterion}.png'")
    
    # Separate features into monthly and quarterly
    monthly_features = [f for f in selected_features if f in md.columns]
    quarterly_features = [f for f in selected_features if f in qd.columns]
    
    print(f"Granger causality selected {len(selected_features)} features ({selection_criterion} criterion):")
    print(f"- Monthly features: {len(monthly_features)}")
    print(f"- Quarterly features: {len(quarterly_features)}")
    
    return selected_features, summary_df

def correlation_analysis(fred_qd_and_md_as_qd: pd.DataFrame, correlation_threshold=0.9):
    """
    Perform correlation analysis and identify groups of highly correlated features.

    Parameters:
        fred_qd_and_md_as_qd (pd.DataFrame): The dataset containing features.
        correlation_threshold (float): The threshold above which features are considered highly correlated.

    Returns:
        correlation_matrix (pd.DataFrame): The full correlation matrix.
        top_20_features (pd.Series): Top 20 features most highly correlated with GDP.
        correlated_groups (list): List of groups of highly correlated features.
    """
    # Calculate the full correlation matrix
    correlation_matrix = fred_qd_and_md_as_qd.corr()

    # Identify the top 20 features most highly correlated with GDP
    top_20_features = correlation_matrix["GDPC1"].drop("GDPC1").sort_values(ascending=False).head(20)

    print("\nTop 20 Features Most Highly Correlated with GDP:")
    print(top_20_features)

    # Identify pairs of features with high correlation
    high_corr_pairs = correlation_matrix.where(
        np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
    ).stack().reset_index()
    high_corr_pairs.columns = ["Feature 1", "Feature 2", "Correlation"]
    high_corr_pairs = high_corr_pairs[high_corr_pairs["Correlation"].abs() > correlation_threshold]

    print("\nPairs of Features with High Correlation (|correlation| > {:.2f}):".format(correlation_threshold))
    print(high_corr_pairs)

    # Find groups of highly correlated features

    correlated_groups = set()

    for feature in correlation_matrix.columns:
        group = correlation_matrix.columns[
            correlation_matrix[feature].abs() > correlation_threshold
        ].tolist()

        group_reverse = group[::-1]

        for feature_in_group in group_reverse:
            for other_feature_in_group in group:
                if correlation_matrix[feature_in_group][other_feature_in_group] < correlation_threshold:
                    group.remove(other_feature_in_group)
                    break

        unique_members = set(group)
        correlated_groups.add(tuple(sorted(unique_members)))


    largest_supergroups = [
        group for group in correlated_groups if not any(
            group != other_group and set(other_group).issuperset(set(group)) for other_group in correlated_groups
        )
    ]

    # correlated_groups = []
    # visited = set()

    # for feature in correlation_matrix.columns:
    #     if feature not in visited:
    #         # Find all features highly correlated with the current feature
    #         # print(f"{correlation_matrix[feature]=}")
    #         group = set(
    #             correlation_matrix.columns[
    #                 correlation_matrix[feature].abs() > correlation_threshold
    #             ].tolist()
    #         )
    #         if group:
    #             correlated_groups.append(group)
    #             visited.update(group)

    print("\nGroups of Highly Correlated Features (|correlation| > {:.2f}):".format(correlation_threshold))
    for i, group in enumerate(largest_supergroups, 1):
        print(f"Group {i}: {group}")

    return correlation_matrix, top_20_features, largest_supergroups

def remove_highly_correlated_features(qd_and_md_as_qd, threshold=0.9):
    """
    From each largest supergroup, select one random feature and add it to a list of selected features.

    Parameters:
        largest_supergroups (list): List of largest groups of highly correlated features.

    Returns:
        selected_features (list): List of randomly selected features from each group.
    """
    correlation_matrix, _, largest_supergroups = correlation_analysis(qd_and_md_as_qd, threshold)
    selected_features = []

    for group in largest_supergroups:
        selected_feature = max(group, key=lambda feature: correlation_matrix["GDPC1"][feature])
        selected_features.append(selected_feature)

    return selected_features

def tree_based_feature_selection(fred_qd_and_md_as_qd, target_column="GDPC1", n_features=30):
    """
    Perform tree-based feature selection using a Random Forest Regressor.

    Parameters:
        fred_md (pd.DataFrame): The dataset containing features and the target column.
        target_column (str): The name of the target column (default is "GDP").
        n_features (int): The number of top features to select (default is 20).

    Returns:
        selected_features (list): List of the top selected feature names.
        feature_importances (pd.Series): Feature importances ranked by importance.
    """
    # Separate features and target
    X = fred_qd_and_md_as_qd.drop(columns=[target_column])[1:]
    y = fred_qd_and_md_as_qd[target_column][1:]

    # Ensure X and y are aligned
    X, y = X.align(y, join="inner", axis=0)

    # Train a Random Forest Regressor
    rf = RandomForestRegressor(random_state=42, n_estimators=100)
    rf.fit(X[:-1], y[1:])

    # Get feature importances
    feature_importances = pd.Series(rf.feature_importances_, index=X.columns)
    feature_importances = feature_importances.sort_values(ascending=False)

    # Select the top n_features
    selected_features = feature_importances.head(n_features).index.tolist()

    return selected_features, feature_importances

def rfe_feature_selection(fred_qd_and_md_as_qd, target_column="GDPC1", n_features=20):
    """
    Perform feature selection using Recursive Feature Elimination (RFE).

    Parameters:
        fred_md (pd.DataFrame): The dataset containing features and the target column.
        target_column (str): The name of the target column (default is "GDP").
        n_features (int): The number of top features to select (default is 20).

    Returns:
        selected_features (list): List of the top selected feature names.
        feature_rankings (pd.Series): Feature rankings from RFE.
    """
    # Separate features and target
    X = fred_qd_and_md_as_qd.drop(columns=[target_column]).iloc[1:]
    y = fred_qd_and_md_as_qd[target_column].iloc[1:]

    # Ensure X and y are aligned
    X, y = X.align(y, join="inner", axis=0)

    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    y_scaler = StandardScaler()
    y_scaled = y_scaler.fit_transform(y.values.reshape(-1, 1)).flatten()

    # Use a linear regression model for RFE
    model = LinearRegression()
    rfe = RFE(estimator=model, n_features_to_select=n_features)
    rfe.fit(X_scaled[:-1], y_scaled[1:])

    # Get feature rankings
    feature_rankings = pd.Series(rfe.ranking_, index=X.columns)
    feature_rankings = feature_rankings.sort_values(ascending=False)  # Sort in descending order
    selected_features = feature_rankings[feature_rankings == 1].index.tolist()

    return selected_features, feature_rankings

def lasso_feature_selection(fred_qd_and_md_as_qd, target_column="GDPC1", n_features=30, seed=42, alpha=None):
    """
    Perform feature selection using Lasso Regression (L1 regularization).

    Parameters:
        fred_md (pd.DataFrame): The dataset containing features and the target column.
        target_column (str): The name of the target column (default is "GDP").
        alpha (float): Regularization strength for Lasso (default is 0.01).

    Returns:
        selected_features (list): List of the top selected feature names.
        feature_coefficients (pd.Series): Coefficients of the features from Lasso, sorted in descending order.
    """
    if alpha is None and n_features is not None:
        n_features_to_alpha = {
            109: 0.008655228552735476, 108: 0.008612081936071872, 107: 0.0087421713639949, 106: 0.00880355045542366, 105: 0.008927604498213393, 104: 0.00909876428960483, 103: 0.009116970916948328, 102: 0.009450991197298489, 101: 0.009536391148832547, 100: 0.010105563835009473,
            99: 0.010217282513963031, 98: 0.010623421760393156, 97: 0.010676645209700065, 96: 0.010946789025418137, 95: 0.01176360211133073, 94: 0.012049199251217661, 93: 0.0119891335720703, 92: 0.012097468391626879, 91: 0.01218240514103954, 90: 0.012354071858496391,
            89: 0.012565779664111716, 88: 0.01269200443137325, 87: 0.014110500805381008, 86: 0.014266494724680103, 85: 0.014540011574743515, 84: 0.014656739602924806, 83: 0.01610054789475816, 82: 0.016392834328144677, 81: 0.018500242654177778, 80: 0.018760936326061933,
            79: 0.018892657517297, 78: 0.01992045061406252, 77: 0.020444908568774955, 76: 0.02094127076780477, 75: 0.02125759821967845, 74: 0.022414048162414033, 73: 0.022866613455772462, 72: 0.023004156593384322, 71: 0.023870811427318037, 70: 0.02440152037245366,
            69: 0.024646636577907853, 68: 0.02484450115829357, 67: 0.025144280387377493, 66: 0.026222280881320916, 65: 0.026248503162202234, 64: 0.029387003026069423, 63: 0.02941639002909549, 62: 0.029504727477769246, 61: 0.029920489361972634, 60: 0.030769671644183663,
            59: 0.03114094528682549, 58: 0.033902238691009244, 57: 0.034829598583226394, 56: 0.03500409522059851, 55: 0.03560394928261017, 54: 0.03690837826777261, 53: 0.038955173848942294, 52: 0.040503650650865156, 51: 0.04370026363374009, 50: 0.04413923803516067,
            49: 0.044360376059329916, 48: 0.044627204609199944, 47: 0.04530131890125031, 46: 0.0453919668403717, 45: 0.045437358807212065, 44: 0.04571066542986643, 43: 0.046169834559000006, 42: 0.04663361609723256, 41: 0.047718070574990064, 40: 0.04951579336312736,
            39: 0.04976386798328217, 38: 0.0512786347175573, 37: 0.05374500346199966, 36: 0.05740995531461513, 35: 0.05763993982530268, 34: 0.0577552773448931, 33: 0.05927583605197224, 32: 0.06011111349363877, 31: 0.060654282572082344, 30: 0.06163206368419866,
            29: 0.06256304413305051, 28: 0.06603254803124443, 27: 0.0679745439977937, 26: 0.06907033183679377, 25: 0.07025396817579024, 24: 0.0721035876032693, 23: 0.07422413088796924, 22: 0.07444702602724992, 21: 0.07841865376613541, 20: 0.0792063785699423,
            19: 0.08112935377484048, 18: 0.08537210072848589, 17: 0.09055794098377935, 16: 0.09174228370040183, 15: 0.10457640218714266, 14: 0.10993540239385131, 13: 0.110596666039608, 12: 0.13520464388439185, 11: 0.15046583476899697, 10: 0.19106556215879342,
            9: 0.20348354618935818, 8: 0.23996730921644358, 7: 0.25556357983705286, 6: 0.3522393197980002, 5: 0.3714017452517151, 4: 0.3736357341830121, 3: 0.5794399283572936, 2: 0.6901952679163122, 1: 0.6908854631842285,
        }
        alpha = n_features_to_alpha.get(n_features, None)
        if alpha is None:
            raise ValueError(f"Can't map n_features '{n_features}' to alpha value. Please provide an n_features value between 9 and 106 or update the n_features_to_alpha dictionary.")
    # Separate features and target
    X = fred_qd_and_md_as_qd.drop(columns=[target_column])[1:]
    y = fred_qd_and_md_as_qd[target_column][1:]

    # Ensure X and y are aligned
    X, y = X.align(y, join="inner", axis=0)

    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    y_scaler = StandardScaler()
    y_scaled = y_scaler.fit_transform(y.values.reshape(-1, 1)).flatten()

    # Apply Lasso
    lasso = Lasso(alpha=alpha, random_state=seed)
    lasso.fit(X_scaled[:-1], y_scaled[1:])

    # Get feature coefficients
    coefficients = lasso.coef_
    feature_coefficients = pd.Series(coefficients, index=X.columns)

    # Select features with non-zero coefficients
    selected_features = feature_coefficients[feature_coefficients != 0].index.tolist()

    return selected_features, feature_coefficients

def pca_feature_selection(fred_qd, fred_md, target_column="GDPC1", n_features=30):
    """
    Perform PCA on fred_qd and fred_md separately, and sum the top principal components of both datasets.

    Parameters:
        fred_qd (pd.DataFrame): Quarterly data features.
        fred_md (pd.DataFrame): Monthly data features.
        target_column (str): The name of the target column (default is "GDPC1").
        n_features (int): The number of top principal components to sum (default is 20).

    Returns:
        combined_pca_components (pd.Series): Summed top principal components of fred_qd and fred_md.
    """
    n_quarterly_features = n_features // 2
    n_monthly_features = n_features - n_quarterly_features

    qd_scaler = StandardScaler()
    md_scaler = StandardScaler()

    qd_scaled = qd_scaler.fit_transform(fred_qd.drop(columns=[target_column]))
    md_scaled = md_scaler.fit_transform(fred_md)

    # Perform PCA on fred_qd
    pca_qd = PCA(n_components=n_quarterly_features)
    pca_qd_components = pca_qd.fit_transform(qd_scaled)
    pca_qd_df = pd.DataFrame(pca_qd_components, index=fred_qd.index, columns=[f"PCA_QD_{i+1}" for i in range(n_quarterly_features)])
    pca_qd_df[target_column] = fred_qd[target_column]

    # Perform PCA on fred_md
    pca_md = PCA(n_components=n_monthly_features)
    pca_md_components = pca_md.fit_transform(md_scaled)
    pca_md_df = pd.DataFrame(pca_md_components, index=fred_md.index, columns=[f"PCA_MD_{i+1}" for i in range(n_monthly_features)])

    return pca_qd_df, pca_md_df


def plot_comparison(feature_importances, feature_rankings, feature_coefficients, output_file_prefix="feature_selection_comparison"):
    """
    Plot and compare selected features and their coefficients/importances for Tree-Based, RFE, and Lasso methods.

    Parameters:
        feature_importances (pd.Series): Feature importances from the tree-based model.
        feature_rankings (pd.Series): Feature rankings from RFE.
        feature_coefficients (pd.Series): Coefficients from Lasso Regression.
        output_file_prefix (str): The prefix for the file paths to save the plots.
    """
    # Tree-Based Feature Importance Plot
    plt.figure(figsize=(10, 8))
    feature_importances_sorted = feature_importances.sort_values(ascending=False).head(30)
    sns.barplot(x=feature_importances_sorted, y=feature_importances_sorted.index, color="blue", alpha=0.7)
    plt.title("Top Features Selected by Tree-Based Model", fontsize=16)
    plt.xlabel("Gini Importance", fontsize=14)
    plt.ylabel("Features", fontsize=14)
    plt.tight_layout()
    tree_output_file = f"{output_file_prefix}_tree_based.png"
    plt.savefig(tree_output_file)
    plt.close()
    print(f"Tree-based feature importance plot saved to {tree_output_file}")

    # RFE Feature Rankings Plot
    plt.figure(figsize=(10, 8))
    #rfe_selected_features = feature_rankings[feature_rankings == 1].index
    rfe_selected_rankings = feature_rankings.sort_values(ascending = True).head(30)
    sns.barplot(x=rfe_selected_rankings, y=rfe_selected_rankings.index, color="green", alpha=0.7)
    plt.title("Top Features Selected by RFE", fontsize=16)
    plt.xlabel("Ranking", fontsize=14)
    plt.ylabel("Features", fontsize=14)
    plt.tight_layout()
    rfe_output_file = f"{output_file_prefix}_rfe.png"
    plt.savefig(rfe_output_file)
    plt.close()
    print(f"RFE feature rankings plot saved to {rfe_output_file}")

    # Lasso Feature Coefficients Plot
    plt.figure(figsize=(10, 8))
    lasso_selected_features = feature_coefficients.abs().sort_values(ascending=False).head(30)
    lasso_selected_features = lasso_selected_features[lasso_selected_features != 0]
    print(lasso_selected_features)
    sns.barplot(x=lasso_selected_features, y=lasso_selected_features.index, color="red", alpha=0.7)
    plt.title("Top Features Selected by Lasso Regression", fontsize=16)
    plt.xlabel("Coefficients", fontsize=14)
    plt.ylabel("Features", fontsize=14)
    plt.tight_layout()
    lasso_output_file = f"{output_file_prefix}_lasso.png"
    plt.savefig(lasso_output_file)
    plt.close()
    print(f"Lasso feature coefficients plot saved to {lasso_output_file}")



def plot_correlation_heatmap(df, output_file="correlation_heatmap.png"):
    """
    Plot a heatmap of the correlation matrix and save it to a file.

    Parameters:
        df (pd.DataFrame): The dataset containing features.
        output_file (str): The file path to save the heatmap.
    """
    # Calculate the correlation matrix
    corr_matrix = df.corr()

    # Create the heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=False, cmap="coolwarm", fmt=".2f", cbar=True)
    plt.title("Feature Correlation Heatmap", fontsize=16)
    plt.tight_layout()

    # Save the heatmap to a file
    plt.savefig(output_file)
    plt.close()
    print(f"Correlation heatmap saved to {output_file}")

def map_month_to_quarter(monthly_date, lookahead_months=2):
    result = monthly_date - pd.DateOffset(
        months=(monthly_date.month - lookahead_months + 2) % 3 + lookahead_months - 2
    )

    assert (monthly_date.month, lookahead_months, result.month,) in ({
        (1, 0, 3),
        (2, 0, 3),
        (3, 0, 3),
        (4, 0, 6),
        (5, 0, 6),
        (6, 0, 6),
        (7, 0, 9),
        (8, 0, 9),
        (9, 0, 9),
        (10, 0, 12),
        (11, 0, 12),
        (12, 0, 12),

        (1, 1, 12),
        (2, 1, 3),
        (3, 1, 3),
        (4, 1, 3),
        (5, 1, 6),
        (6, 1, 6),
        (7, 1, 6),
        (8, 1, 9),
        (9, 1, 9),
        (10, 1, 9),
        (11, 1, 12),
        (12, 1, 12),

        (1, 2, 12),
        (2, 2, 12),
        (3, 2, 3),
        (4, 2, 3),
        (5, 2, 3),
        (6, 2, 6),
        (7, 2, 6),
        (8, 2, 6),
        (9, 2, 9),
        (10, 2, 9),
        (11, 2, 9),
        (12, 2, 12),
    }), f"Unexpected mapping for {monthly_date} with lookahead {lookahead_months}: {result.month}"

    return result

def combine_qd_with_summed_md(qd, md, lookahead_months=2):
    quarter = md.index.map(partial(map_month_to_quarter, lookahead_months=lookahead_months))
    md_summed = md.groupby(quarter).mean()

    return pd.concat([qd, md_summed], axis=1)

def combine_qd_and_md_as_qd(fred_qd, fred_md):
    fred_md_as_qd = fred_md.loc[fred_qd.index].copy()

    # Concatenate fred_qd with fred_md_as_qd
    fred_qd_and_md_as_qd = pd.concat([fred_qd, fred_md_as_qd], axis=1)
    return fred_qd_and_md_as_qd

def fit_lasso_alpha():
    from check_stationarity import load_train_data
    fred_md, fred_qd = load_train_data()
    fred_qd_and_md_as_qd = combine_qd_with_summed_md(fred_qd, fred_md)

    max_n_features = 100
    min_n_features = 2
    n_features_to_alpha = {}
    alpha = 0.005
    while True:
        f, c = lasso_feature_selection(fred_qd_and_md_as_qd, alpha=alpha)
        found_n_features = len(f)

        if found_n_features not in n_features_to_alpha:
            print(f"Found n_features={found_n_features} for alpha={alpha:.6f}")
        n_features_to_alpha[found_n_features] = alpha

        if found_n_features < min_n_features:
            break
    
        alpha *= 1.001

    print("{")
    for n, a in n_features_to_alpha.items():
        print(f"{n}: {a}, ", end="")
        if n % 10 == 0:
            print("\n\t", end="")
    print("\n}")

    for i in range(max_n_features, min_n_features - 1, -1):
        if i not in n_features_to_alpha:
            print(f"No alpha found for n_features={i}.")

def verify_lasso_alpha():
    from check_stationarity import load_train_data
    fred_md, fred_qd = load_train_data()
    fred_qd_and_md_as_qd = combine_qd_with_summed_md(fred_qd, fred_md)
    print(fred_qd_and_md_as_qd)

    for target_n_features in range(10, 101):
        f, c = lasso_feature_selection(fred_qd_and_md_as_qd, n_features=target_n_features)
        found_n_features = len(f)
        if found_n_features != target_n_features:
            print(f"mismatch: target={target_n_features}, found={found_n_features}")

def plots():
    from check_stationarity import load_train_data
    fred_md, fred_qd = load_train_data()

    print(f"{fred_md.index=}")
    print(f"{fred_qd.index=}")
    print(f"{fred_md.shape=}")
    print(f"{fred_qd.shape=}")

    fred_md_as_qd = fred_md.loc[fred_qd.index].copy()

    # Concatenate fred_qd with fred_md_as_qd
    # fred_qd_and_md_as_qd = pd.concat([fred_qd, fred_md_as_qd], axis=1)
    fred_qd_and_md_as_qd = combine_qd_with_summed_md(fred_qd, fred_md)
    print(len(fred_qd_and_md_as_qd.columns))

    selected_features_lasso, feature_coefficients = lasso_feature_selection(fred_qd_and_md_as_qd)

    # Perform correlation analysis
    correlation_matrix, top_20_features, correlated_groups = correlation_analysis(fred_qd_and_md_as_qd)

    # Ensure the GDP column is retained in fred_md
    # fred_md["GDP"] = fred_qd["GDPC1"].reindex(fred_md.index, method="ffill")

    # Plot and save the correlation heatmap
    print("\nGenerating correlation heatmap...")
    plot_correlation_heatmap(fred_qd_and_md_as_qd, output_file="results/data_exploration/correlation_heatmap.png")

    # Remove highly correlated features
    #print("\nRemoving highly correlated features...")
    #fred_no_corr = remove_highly_correlated_features(fred_qd_and_md_as_qd.drop(columns=["GDP"]), threshold=0.9)

    # Add GDP back to the dataset after removing correlated features
    #fred_no_corr["GDP"] = fred_qd["GDPC1"].reindex(fred_md.index, method="ffill")

    # Perform tree-based feature selection
    selected_features_tree, feature_importances = tree_based_feature_selection(fred_qd_and_md_as_qd)

    # Perform RFE-based feature selection
    selected_features_rfe, feature_rankings = rfe_feature_selection(fred_qd_and_md_as_qd)

    # Perform Lasso-based feature selection
    selected_features_lasso, feature_coefficients = lasso_feature_selection(fred_qd_and_md_as_qd)

    # Check overlap between top 20 features from all methods
    top_20_tree = set(feature_importances.head(20).index)
    top_20_rfe = set(feature_rankings[feature_rankings == 1].index)
    top_20_lasso = set(selected_features_lasso)
    overlap = top_20_tree.intersection(top_20_rfe).intersection(top_20_lasso)

    print("\nOverlap Between Top Features (Tree-Based, RFE, and Lasso):")
    print(overlap)

    # Save the selected features and their importances/rankings
    feature_importances.to_csv("feature_importances_tree.csv", header=True)
    feature_rankings.to_csv("feature_rankings_rfe.csv", header=True)
    feature_coefficients.to_csv("feature_coefficients_lasso.csv", header=True)
    with open("selected_features_tree.txt", "w") as f:
        f.write("\n".join(selected_features_tree))
    with open("selected_features_rfe.txt", "w") as f:
        f.write("\n".join(selected_features_rfe))
    with open("selected_features_lasso.txt", "w") as f:
        f.write("\n".join(selected_features_lasso))

    # Plot and save the comparison of feature selection methods
    plot_comparison(feature_importances, feature_rankings, feature_coefficients, output_file_prefix="feature_selection_comparison")

if __name__ == "__main__":
    verify_lasso_alpha()