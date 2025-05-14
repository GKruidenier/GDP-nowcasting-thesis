import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import random

from stationarization import load_train_data

def _error_correlation_analysis(fred_qd_and_md_as_qd: pd.DataFrame, correlation_threshold=0.9):
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
        if len(unique_members) > 1:
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

def _error_remove_highly_correlated_features(largest_supergroups):
    """
    From each largest supergroup, select one random feature and add it to a list of selected features.

    Parameters:
        largest_supergroups (list): List of largest groups of highly correlated features.

    Returns:
        selected_features (list): List of randomly selected features from each group.
    """
    selected_features = []

    for group in largest_supergroups:
        selected_feature = random.choice(group)
        selected_features.append(selected_feature)

    return selected_features

def _error_plot_correlation_heatmaps(correlation_matrix: pd.DataFrame, correlated_groups: list, num_groups=3):
    """
    Plot correlation heatmaps for a few random groups of highly correlated features.

    Parameters:
        correlation_matrix (pd.DataFrame): The full correlation matrix.
        correlated_groups (list): List of groups of highly correlated features.
        num_groups (int): Number of random groups to plot heatmaps for.
    """
    # Select random groups
    random_groups = random.sample(correlated_groups, min(num_groups, len(correlated_groups)))

    for i, group in enumerate(random_groups, 1):
        group_matrix = correlation_matrix.loc[group, group]
        plt.figure(figsize=(8, 6))
        sns.heatmap(group_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
        plt.title(f"Correlation Heatmap for Group {i}")
        plt.show()

# Example usage in the main block
if __name__ == "__main__":
    fred_md, fred_qd = load_train_data()
       
    print(f"{fred_md.index=}")
    print(f"{fred_qd.index=}")
    print(f"{fred_md.shape=}")
    print(f"{fred_qd.shape=}")

    fred_md_as_qd = fred_md.loc[fred_qd.index].copy()

    # Concatenate fred_qd with fred_md_as_qd
    fred_qd_and_md_as_qd = pd.concat([fred_qd, fred_md_as_qd], axis=1)

    # Perform correlation analysis
    correlation_matrix, top_20_features, correlated_groups = _error_correlation_analysis(fred_qd_and_md_as_qd)

    # Plot heatmaps for a few random groups
    _error_plot_correlation_heatmaps(correlation_matrix, correlated_groups, num_groups=3)

