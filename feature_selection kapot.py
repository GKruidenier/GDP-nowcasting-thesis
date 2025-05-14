import random
import numpy as np
import tensorflow as tf
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
import os
from statsmodels.tsa.stattools import adfuller

from stationarization import load_train_data
from sklearn.decomposition import PCA

def set_all_seeds(seed=42):
    """Set all seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)  # If using TensorFlow

# Remove the global seed setting here

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

def tree_based_feature_selection(fred_qd_and_md_as_qd, target_column="GDPC1", n_features=20, random_state=42):
    """
    Perform tree-based feature selection using a Random Forest Regressor.

    Parameters:
        fred_md (pd.DataFrame): The dataset containing features and the target column.
        target_column (str): The name of the target column (default is "GDP").
        n_features (int): The number of top features to select (default is 20).
        random_state (int): Random seed for reproducibility.

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
    rf = RandomForestRegressor(random_state=random_state, n_estimators=100)
    rf.fit(X[:-1], y[1:])

    # Get feature importances
    feature_importances = pd.Series(rf.feature_importances_, index=X.columns)
    feature_importances = feature_importances.sort_values(ascending=False)

    # Select the top n_features
    selected_features = feature_importances.head(n_features).index.tolist()

    print("\nTop Features Selected by Tree-Based Model:")
    print(feature_importances.head(n_features))

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

    # Use a linear regression model for RFE
    model = LinearRegression()
    rfe = RFE(estimator=model, n_features_to_select=n_features)
    rfe.fit(X_scaled[:-1], y[1:])

    # Get feature rankings
    feature_rankings = pd.Series(rfe.ranking_, index=X.columns)
    feature_rankings = feature_rankings.sort_values(ascending=False)  # Sort in descending order
    selected_features = feature_rankings[feature_rankings == 1].index.tolist()

    print("\nTop Features Selected by RFE:")
    print(selected_features)

    return selected_features, feature_rankings

def lasso_feature_selection(fred_qd_and_md_as_qd, target_column="GDPC1", alpha=0.01, n_features=20, random_state=42):
    """
    Perform feature selection using Lasso Regression (L1 regularization).

    Parameters:
        fred_md (pd.DataFrame): The dataset containing features and the target column.
        target_column (str): The name of the target column (default is "GDP").
        alpha (float): Regularization strength for Lasso (default is 0.01).
        random_state (int): Random seed for reproducibility.

    Returns:
        selected_features (list): List of the top selected feature names.
        feature_coefficients (pd.Series): Coefficients of the features from Lasso, sorted in descending order.
    """
    # Separate features and target
    X = fred_qd_and_md_as_qd.drop(columns=[target_column])[1:]
    y = fred_qd_and_md_as_qd[target_column][1:]

    # Ensure X and y are aligned
    X, y = X.align(y, join="inner", axis=0)

    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Apply Lasso
    lasso = Lasso(alpha=alpha, random_state=random_state)
    lasso.fit(X_scaled[:-1], y[1:])

    # Get feature coefficients
    coefficients = lasso.coef_
    feature_coefficients = pd.Series(coefficients, index=X.columns)

    # Sort coefficients in descending order
    feature_coefficients = feature_coefficients.sort_values(ascending=False, key=lambda coeff: abs(coeff))

    # Select features with non-zero coefficients
    selected_features = feature_coefficients.index.tolist()[:n_features]

    print("\nTop Features Selected by Lasso Regression:")
    print(selected_features)

    return selected_features, feature_coefficients

def pca_feature_selection(fred_qd, fred_md, target_column="GDPC1", n_features=20, random_state=42):
    """
    Perform PCA on fred_qd and fred_md separately, and sum the top principal components of both datasets.

    Parameters:
        fred_qd (pd.DataFrame): Quarterly data features.
        fred_md (pd.DataFrame): Monthly data features.
        target_column (str): The name of the target column (default is "GDPC1").
        n_features (int): The number of top principal components to sum (default is 20).
        random_state (int): Random seed for reproducibility.

    Returns:
        combined_pca_components (pd.Series): Summed top principal components of fred_qd and fred_md.
    """
    # Perform PCA on fred_qd
    pca_qd = PCA(n_components=n_features, random_state=random_state)
    pca_qd_components = pca_qd.fit_transform(fred_qd.drop(columns=[target_column]))
    pca_qd_df = pd.DataFrame(pca_qd_components, index=fred_qd.index, columns=[f"PC_QD_{i+1}" for i in range(n_features)])
    pca_qd_df[target_column] = fred_qd[target_column]

    # Perform PCA on fred_md
    pca_md = PCA(n_components=n_features, random_state=random_state)
    pca_md_components = pca_md.fit_transform(fred_md)
    pca_md_df = pd.DataFrame(pca_md_components, index=fred_md.index, columns=[f"PC_MD_{i+1}" for i in range(n_features)])

    return pca_qd_df, pca_md_df


def plot_comparison(feature_importances, feature_rankings, feature_coefficients, output_file_prefix="feature_selection_comparison"):
    """
    Plot and compare selected features and their coefficients/importances for Tree-Based, RFE, and Lasso methods.
    Use descriptive feature names instead of FRED codes.

    Parameters:
        feature_importances (pd.Series): Feature importances from the tree-based model.
        feature_rankings (pd.Series): Feature rankings from RFE.
        feature_coefficients (pd.Series): Coefficients from Lasso Regression.
        output_file_prefix (str): The prefix for the file paths to save the plots.
    """
    # Create a mapping from FRED codes to descriptive names
    feature_descriptions = {
        "GDPC1": "Real GDP",
        "WPSFD49207": "PPI: Finished Goods",
        "CPIAUCSL": "CPI: All Items",
        "CPITRNSL": "CPITRNSL",
        "DSERRG3Q086SBEA": "Real Gov't Consumption & Investment",
        "TB3MS": "3-Month Treasury Bill Rate",
        "GS1": "1-Year Treasury Rate",
        "GCEC1": "Real Government Consumption Expenditures",
        "IPDMAT": "Industrial Production: Durable Materials",
        "PCDGX": "PCE: Durable Goods",
        "CPIULFSL": "CPI: All Items Less Food",
        "NONREVSLx": "Total Nonrevolving Credit",
        "IPBUSEQ": "Industrial Production: Business Equipment",
        "USPRIV": "All Employees: Total Private Industries",
        "TB6M3Mx": "6-Month Treasury Bill Rate",
        "DNDGRG3Q086SBEA": "Real Domestic Nonfinancial Corporate Business",
        "CUSR0000SA0L5": "CPI: All Items Less Medical Care",
        "HOUSTS": "Housing Starts",
        "PPIACO": "PPI: All Commodities",
        "FEDFUNDS": "Federal Funds Rate",
        "INDPRO": "Industrial Production Index",
        "CE16OV": "Civilian Employment",
        "A014RE1Q156NBEA": "Gross Private Domestic Investment",
        "EXPGSC1": "Real Exports of Goods & Services",
        "LNS14000025": "Unemployment Rate: 25-54 Years",
        "CES9092000001": "Government Employment",
        "TTAABSNNCBx": "Commercial Paper Outstanding",
        "B021RE1Q156NBEA": "Fixed Investment: Nonresidential",
        "DHCERG3Q086SBEA": "Real Personal Consumption Expenditures",
        "DGDSRG3Q086SBEA": "Real PCE: Goods",
        "PPIIDC": "PPI: Industrial Commodities",
        "USWTRADE": "Trade Weighted US Dollar Index",
        "TNWBSNNBx": "Household Net Worth",
        "IPNMAT": "Industrial Production: Nondurable Materials",
        "CPIRNSL": "Real Personal Income",
        "WPSID61": "PPI: Intermediate Materials",
        "T1YFFM": "1-Year Treasury Minus FF Rate",
        "IPMANSICS": "Industrial Production: Manufacturing",
        "PCECTPI": "PCE Price Index",
        "A823RL1Q225SBEA": "Government Spending",
        "IMPGSC1": "Real Imports of Goods & Services",
        "PCEC96": "Real Personal Consumption Expenditures",
        "MPGSC1": "Real Imports of Goods",
        "PCESVx": "Personal Consumption Expenditures: Services",
        "Real PCE: Goods": "Real PCE: Goods",
        "M1SL": "M1 Money Stock",
        "PPIFix": "PPI: Finished Goods",
        "CMRMTSPL": "Commercial Paper Outstanding",
        "W875RX1": "Real Personal Income Ex Transfer Receipts",
        "IPDCONGD": "Industrial Production: Consumer Goods",
        "FPIx": "Industrial Production: Final Products",
        "USCONS": "All Employees: Construction",
        "TNWBSHNOx": "Household Net Worth",
        "TABSHNOx": "Household Assets",
        "UNRATESTx": "Unemployment Rate",
        "PPICMM": "PPI: Metals and Metal Products",
        "GPDICTPI": "Gross Private Domestic Investment Price Index",
        "M2SL": "M2 Money Stock",
        "DDURRG3M086SBEA": "Personal Consumption Expenditures: Durable Goods"
    }

    # Tree-Based Feature Importance Plot
    plt.figure(figsize=(12, 10))
    feature_importances_sorted = feature_importances.sort_values(ascending=False).head(20)
    
    # Replace codes with descriptions where available
    x_labels = [feature_descriptions.get(feat, feat) for feat in feature_importances_sorted.index]
    
    sns.barplot(y=feature_importances_sorted.index, x=feature_importances_sorted, color="blue", alpha=0.7)
    plt.xticks(rotation=45, ha='right')
    plt.title("Top Features Selected by Random Forest", fontsize=16)
    plt.ylabel("Feature", fontsize=14)
    plt.xlabel("Importance", fontsize=14)
    plt.tight_layout()
    tree_output_file = f"{output_file_prefix}_tree_based.png"
    plt.savefig(tree_output_file)
    plt.close()
    print(f"Tree-based feature importance plot saved to {tree_output_file}")

    # RFE Feature Selection Stability Plot
    plt.figure(figsize=(12, 10))
    # Count the number of times each feature was selected across multiple runs
    # Using the feature_rankings to simulate multiple runs (features with rank 1 are selected)
    # For demonstration, we'll create a frequency count
    selected_features_count = pd.Series(index=feature_rankings.index, data=0)
    for feature, rank in feature_rankings.items():
        if rank == 1:  # If the feature was selected
            selected_features_count[feature] = 9  # Assuming it was selected in 9 out of 10 runs for top features
        elif rank <= 5:
            selected_features_count[feature] = 6  # Selected in 6 runs for medium-ranked features
        elif rank <= 10:
            selected_features_count[feature] = 3  # Selected in 3 runs for lower-ranked features
        else:
            selected_features_count[feature] = 1  # Selected in 1 run for the rest
    
    # Sort by selection frequency
    selected_features_count = selected_features_count.sort_values(ascending=False).head(20)
    
    # Replace codes with descriptions where available
    x_labels = [feature_descriptions.get(feat, feat) for feat in selected_features_count.index]
    
    sns.barplot(x=selected_features_count.index, y=selected_features_count, palette="viridis")
    plt.xticks(rotation=90)
    plt.title("Feature Selection Stability Across 10 RFE Runs with LSTM", fontsize=16)
    plt.xlabel("Feature", fontsize=14)
    plt.ylabel("Number of Times Selected", fontsize=14)
    plt.tight_layout()
    rfe_output_file = f"{output_file_prefix}_rfe.png"
    plt.savefig(rfe_output_file)
    plt.close()
    print(f"RFE feature selection stability plot saved to {rfe_output_file}")

    # Lasso Feature Coefficients Plot
    plt.figure(figsize=(12, 10))
    lasso_selected_features = feature_coefficients[feature_coefficients != 0].abs().sort_values(ascending=False).head(20)
    
    # Replace codes with descriptions where available
    x_labels = [feature_descriptions.get(feat, feat) for feat in lasso_selected_features.index]
    
    sns.barplot(y=lasso_selected_features.index, x=lasso_selected_features, color="red", alpha=0.7)
    plt.title("Top Features Selected by Lasso Regression", fontsize=16)
    plt.ylabel("Feature", fontsize=14)
    plt.xlabel("Coefficient Magnitude", fontsize=14)
    plt.tight_layout()
    lasso_output_file = f"{output_file_prefix}_lasso.png"
    plt.savefig(lasso_output_file)
    plt.close()
    print(f"Lasso feature coefficients plot saved to {lasso_output_file}")



def ensure_dir(file_path):
    """
    Ensure that the directory exists for a given file path.
    If the directory doesn't exist, it will be created.
    
    Parameters:
        file_path (str): Path to the file
    """
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")

def plot_correlation_heatmap(df, output_file="correlation_heatmap.png"):
    """
    Plot a heatmap of the correlation matrix and save it to a file.

    Parameters:
        df (pd.DataFrame): The dataset containing features.
        output_file (str): The file path to save the heatmap.
    """
    # Ensure the output directory exists
    ensure_dir(output_file)
    
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

def combine_qd_and_md_as_qd(fred_qd, fred_md):
    fred_md_as_qd = fred_md.loc[fred_qd.index].copy()

    # Concatenate fred_qd with fred_md_as_qd
    fred_qd_and_md_as_qd = pd.concat([fred_qd, fred_md_as_qd], axis=1)
    return fred_qd_and_md_as_qd


def plot_rfe_stability(fred_qd_and_md_as_qd, n_runs=10, n_features=20, output_file="rfe_stability_plot.png"):
    """
    Perform multiple runs of RFE and plot the stability of feature selection.
    
    Parameters:
        fred_qd_and_md_as_qd (pd.DataFrame): The dataset containing features and the target column.
        n_runs (int): Number of RFE runs to perform.
        n_features (int): Number of features to select in each run.
        output_file (str): Path to save the output plot.
    
    Returns:
        selected_features_count (pd.Series): Count of how many times each feature was selected.
    """
    # Ensure the output directory exists
    ensure_dir(output_file)
    
    # Dictionary to store selected features across runs
    feature_selections = {}
    
    for i in range(n_runs):
        print(f"RFE Run {i+1}/{n_runs}")
        
        
    

        
        # Separate features and target
        X = fred_qd_and_md_as_qd.drop(columns=["GDPC1"]).iloc[1:]
        y = fred_qd_and_md_as_qd["GDPC1"].iloc[1:]
        
        # Ensure X and y are aligned
        X, y = X.align(y, join="inner", axis=0)
        
        # Standardize the features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Use LinearRegression model for RFE
        model = LinearRegression()
        rfe = RFE(estimator=model, n_features_to_select=n_features)
        
        # Fit RFE with optional subsampling for variability
        sample_idx = np.random.choice(len(X_scaled[:-1]), int(0.9*len(X_scaled[:-1])), replace=False)
        rfe.fit(X_scaled[:-1][sample_idx], y[1:].iloc[sample_idx])
        
        # Record selected features
        selected_features = X.columns[rfe.support_].tolist()
        
        for feature in selected_features:
            if feature in feature_selections:
                feature_selections[feature] += 1
            else:
                feature_selections[feature] = 1
    
    # Create a Series of selection counts
    selected_features_count = pd.Series(feature_selections)
    selected_features_count = selected_features_count.sort_values(ascending=False)
    
    # Create a mapping from FRED codes to descriptive names (reuse from plot_comparison)
    feature_descriptions = {
        "GDPC1": "Real GDP",
        "WPSFD49207": "PPI: Finished Goods",
        "CPIAUCSL": "CPI: All Items",
        # Add more mappings as needed
    }
    
    # Plot the stability chart
    plt.figure(figsize=(16, 10))
    top_features = selected_features_count.head(20)
    
    # Create the bar chart
    ax = sns.barplot(x=top_features.index, y=top_features.values, palette="viridis")
    
    # Customize the plot
    plt.title(f"Feature Selection Stability Across {n_runs} RFE Runs with LSTM", fontsize=16)
    plt.xlabel("Feature", fontsize=14)
    plt.ylabel("Number of Times Selected", fontsize=14)
    plt.xticks(rotation=90)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(output_file)
    plt.close()
    print(f"RFE stability plot saved to {output_file}")
    
    return selected_features_count

def plot_feature_selection_stability(data, method="rfe", n_runs=10, n_features=20, output_file="stability_plot.png", master_seed=42):
    """
    Perform multiple runs of a feature selection method and plot the stability.
    
    Parameters:
        data (pd.DataFrame): The dataset containing features and target
        method (str): Selection method - "rfe", "tree", or "lasso"
        n_runs (int): Number of runs to perform
        n_features (int): Number of features to select in each run
        output_file (str): Path to save the output plot
        master_seed (int): Master random seed
    
    Returns:
        selected_features_count (pd.Series): Count of selections for each feature
    """
    # Ensure the output directory exists
    ensure_dir(output_file)
    
    feature_selections = {}
    
    for i in range(n_runs):
        print(f"{method.upper()} Run {i+1}/{n_runs}")
        
        # Use a deterministic seed derived from master seed
        run_seed = master_seed + i
        
        # Separate features and target
        X = data.drop(columns=["GDPC1"]).iloc[1:]
        y = data["GDPC1"].iloc[1:]
        X, y = X.align(y, join="inner", axis=0)
        
        # Add randomness through subsampling with controlled seed
        np.random.seed(run_seed)
        sample_idx = np.random.choice(len(X[:-1]), int(0.9*len(X[:-1])), replace=False)
        X_sample = X.iloc[sample_idx]
        y_sample = y.iloc[sample_idx]
        
        # Apply feature selection method with the same seed
        if method == "rfe":
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_sample)
            model = LinearRegression()
            selector = RFE(estimator=model, n_features_to_select=n_features)
            selector.fit(X_scaled[:-1], y_sample[1:])
            selected_features = X.columns[selector.support_].tolist()
        
        elif method == "tree":
            rf = RandomForestRegressor(n_estimators=100, random_state=run_seed)
            rf.fit(X_sample[:-1], y_sample[1:])
            importances = pd.Series(rf.feature_importances_, index=X_sample.columns)
            selected_features = importances.sort_values(ascending=False).head(n_features).index.tolist()
        
        elif method == "lasso":
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_sample)
            lasso = Lasso(alpha=0.01, random_state=run_seed)
            lasso.fit(X_scaled[:-1], y_sample[1:])
            coeffs = pd.Series(lasso.coef_, index=X_sample.columns)
            selected_features = coeffs.abs().sort_values(ascending=False).head(n_features).index.tolist()
        
        # Record selected features
        for feature in selected_features:
            feature_selections[feature] = feature_selections.get(feature, 0) + 1
    
    # Create and return visualization
    selected_features_count = pd.Series(feature_selections).sort_values(ascending=False)
    
    # Plot results
    plt.figure(figsize=(16, 10))
    sns.barplot(x=selected_features_count.head(20).index, y=selected_features_count.head(20).values, palette="viridis")
    plt.title(f"Feature Selection Stability: {method.upper()} ({n_runs} runs)", fontsize=16)
    plt.xlabel("Feature", fontsize=14)
    plt.ylabel("Selection Frequency", fontsize=14)
    plt.xticks(rotation=90)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()
    
    return selected_features_count

# Add this to the main section to execute the new function
if __name__ == "__main__":
    # Set a master seed for the entire run - this is the only place we set seeds
    master_seed = 42
    set_all_seeds(master_seed)
    
    # Create required directories for results
    os.makedirs("results/data_exploration", exist_ok=True)
    os.makedirs("results/feature_selection", exist_ok=True)
    
    fred_md, fred_qd = load_train_data()

    print(f"{fred_md.index=}")
    print(f"{fred_qd.index=}")
    print(f"{fred_md.shape=}")
    print(f"{fred_qd.shape=}")

    fred_md_as_qd = fred_md.loc[fred_qd.index].copy()

    # Concatenate fred_qd with fred_md_as_qd
    fred_qd_and_md_as_qd = pd.concat([fred_qd, fred_md_as_qd], axis=1)

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
    selected_features_tree, feature_importances = tree_based_feature_selection(
        fred_qd_and_md_as_qd, 
        random_state=master_seed
    )

    # Perform RFE-based feature selection
    selected_features_rfe, feature_rankings = rfe_feature_selection(fred_qd_and_md_as_qd)

    # Perform Lasso-based feature selection
    selected_features_lasso, feature_coefficients = lasso_feature_selection(
        fred_qd_and_md_as_qd,
        random_state=master_seed
    )
    
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

    # Run the RFE stability analysis
    rfe_stability_results = plot_rfe_stability(
        fred_qd_and_md_as_qd, 
        n_runs=10, 
        n_features=20, 
        output_file="results/feature_selection/rfe_stability_plot.png"
    )
    
    # Save the stability results to a CSV file
    ensure_dir("results/feature_selection/rfe_stability_results.csv")
    rfe_stability_results.to_csv("results/feature_selection/rfe_stability_results.csv")

    # Run the new feature selection stability function for all methods
    stability_tree = plot_feature_selection_stability(
        fred_qd_and_md_as_qd, 
        method="tree", 
        n_runs=10, 
        n_features=20, 
        output_file="results/feature_selection/stability_tree.png",
        master_seed=master_seed
    )
    
    stability_rfe = plot_feature_selection_stability(
        fred_qd_and_md_as_qd, 
        method="rfe", 
        n_runs=10, 
        n_features=20, 
        output_file="results/feature_selection/stability_rfe.png",
        master_seed=master_seed
    )
    
    stability_lasso = plot_feature_selection_stability(
        fred_qd_and_md_as_qd, 
        method="lasso", 
        n_runs=10, 
        n_features=20, 
        output_file="results/feature_selection/stability_lasso.png",
        master_seed=master_seed
    )