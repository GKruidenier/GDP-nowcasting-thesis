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

from stationarization import load_train_data
# from check_stationarity import load_train_data
from sklearn.decomposition import PCA

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

    print("\nTop Features Selected by RFE:")
    print(selected_features)

    return selected_features, feature_rankings

def lasso_feature_selection(fred_qd_and_md_as_qd, target_column="GDPC1", n_features=30):
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

    n_features_to_alpha = {
        106: 0.005007504501350202, 105: 0.005025561291896462, 104: 0.005080123274404351, 103: 0.005247389258821229, 102: 0.00531232091864954, 101: 0.0053571256102915585, 100: 0.005379669465406822,
        99: 0.005405550060690964, 98: 0.005669602793357783, 97: 0.005719136065390105, 96: 0.006555453414172951, 95: 0.006614709860967925, 94: 0.006917048990181719, 93: 0.007172711730915726, 92: 0.007313930839993622, 91: 0.007448987516963053, 90: 0.0076919421381369145,
        89: 0.00797146229936312, 88: 0.007973169779851401, 87: 0.007978638767942596, 86: 0.008121096337424112, 85: 0.008348325858824343, 84: 0.008507585594264977, 83: 0.009181097589979917, 82: 0.009460661073706578, 81: 0.009722455085408581, 80: 0.009786825823649875,
        79: 0.010721204402549684, 78: 0.010727638090099611, 77: 0.011184368754784593, 76: 0.011302405530854666, 75: 0.011397731246747547, 74: 0.011569959868318548, 73: 0.011692066067788432, 72: 0.012493400625140377, 71: 0.012598771323022069, 70: 0.01282372076531823,
        69: 0.013253897918235932, 68: 0.013401813411409226, 67: 0.014001765913286751, 66: 0.014179277513345585, 65: 0.01489870274677974, 64: 0.017900821590309347, 63: 0.01975154771601703, 62: 0.02039576007132232, 61: 0.02061101135792602, 60: 0.021475640679347457,
        59: 0.021494974554961695, 58: 0.022796575970395333, 57: 0.024315175822996405, 56: 0.02504054386294114, 55: 0.02683711648508762, 54: 0.0269419698433289, 53: 0.027144763726200026, 52: 0.027895909606727887, 51: 0.029620629785987184, 50: 0.029852539184698955,
        49: 0.03012238404334267, 48: 0.030231003731907704, 47: 0.031631745405912126, 46: 0.031822077756061704, 45: 0.032341715174365765, 44: 0.032791056772973025, 43: 0.03392155777595872, 42: 0.03511209159036118, 41: 0.035270428243523826, 40: 0.035910947258667884,
        39: 0.038165541494728186, 38: 0.04306949276605957, 37: 0.043276692133704615, 36: 0.04338066531754857, 35: 0.04408904825541718, 34: 0.04549974350448505, 33: 0.0465628598471005, 32: 0.047026051701414925, 31: 0.04763652529211293, 30: 0.0486035605542735,
        29: 0.04985869808892756, 28: 0.04986381107739353, 27: 0.051346078084459325, 26: 0.05210632553156745, 25: 0.05380581456299029, 24: 0.05404844985507107, 23: 0.057735426687331004, 22: 0.05782208180483692, 21: 0.05843232027886644, 20: 0.06466718974628234,
        19: 0.06695687398410809, 18: 0.0693276295338234, 17: 0.0899455483698591, 16: 0.09809059267574015, 15: 0.09969235930860011, 14: 0.10761688556959333, 13: 0.1102305371791774, 12: 0.11845855675512107, 11: 0.12376153369654992, 10: 0.12540576739037576,
        9: 0.12544338912059286,
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
    lasso = Lasso(alpha=alpha, random_state=42)
    lasso.fit(X_scaled[:-1], y_scaled[1:])

    # Get feature coefficients
    coefficients = lasso.coef_
    feature_coefficients = pd.Series(coefficients, index=X.columns)

    # Select features with non-zero coefficients
    selected_features = feature_coefficients[feature_coefficients != 0].index.tolist()

    print("\nTop Features Selected by Lasso Regression:")
    print(selected_features)

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
    # Perform PCA on fred_qd
    pca_qd = PCA(n_components=n_features)
    pca_qd_components = pca_qd.fit_transform(fred_qd.drop(columns=[target_column]))
    pca_qd_df = pd.DataFrame(pca_qd_components, index=fred_qd.index, columns=[f"PC_QD_{i+1}" for i in range(n_features)])
    pca_qd_df[target_column] = fred_qd[target_column]

    # Perform PCA on fred_md
    pca_md = PCA(n_components=n_features)
    pca_md_components = pca_md.fit_transform(fred_md)
    pca_md_df = pd.DataFrame(pca_md_components, index=fred_md.index, columns=[f"PC_MD_{i+1}" for i in range(n_features)])

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
    plt.xlabel("Importance", fontsize=14)
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

def combine_qd_with_summed_md(qd, md):
    quarter = md.index.map(lambda d: d - pd.DateOffset(months=d.month % 3))
    md_summed = md.groupby(quarter).sum()

    return pd.concat([qd, md_summed], axis=1)

def combine_qd_and_md_as_qd(fred_qd, fred_md):
    fred_md_as_qd = fred_md.loc[fred_qd.index].copy()

    # Concatenate fred_qd with fred_md_as_qd
    fred_qd_and_md_as_qd = pd.concat([fred_qd, fred_md_as_qd], axis=1)
    return fred_qd_and_md_as_qd


if __name__ == "__main__":
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