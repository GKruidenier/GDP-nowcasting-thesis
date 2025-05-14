# Remove outliers using the rm_outliers function from fbi package

fred_md_transformed_rm_outliers <- rm_outliers.fredmd(Fred_md_transformed)
fred_qd_transformed_rm_outliers <- rm_outliers.fredmd(Fred_qd_transformed)



# Plot features before and after removing outliers
# Select a random subset of features again
random_features_outliers <- sample(colnames(fred_md_transformed_rm_outliers)[-1], 5)  # Exclude the date column

# Plot features before and after removing outliers
par(mfrow = c(2, 5))  # Set up plotting area

for (feature in random_features_outliers) {
    # Plot before removing outliers
    plot(Fred_md_transformed[[feature]], type = "l", main = paste("Before Outliers Removal:", feature), xlab = "Index", ylab = "Value")
    
    # Plot after removing outliers
    plot(fred_md_transformed_rm_outliers[[feature]], type = "l", main = paste("After Outliers Removal:", feature), xlab = "Index", ylab = "Value")
}

# Select a random subset of features again for FRED_QD
random_features_outliers_qd <- sample(colnames(Fred_qd_transformed_rm_outliers)[-1], 5)  # Exclude the date column

# Plot features before and after removing outliers for FRED_QD
par(mfrow = c(2, 5))  # Set up plotting area

for (feature in random_features_outliers_qd) {
    # Plot before removing outliers
    plot(Fred_qd_transformed[[feature]], type = "l", main = paste("Before Outliers Removal:", feature), xlab = "Index", ylab = "Value")
    
    # Plot after removing outliers
    plot(Fred_qd_transformed_rm_outliers[[feature]], type = "l", main = paste("After Outliers Removal:", feature), xlab = "Index", ylab = "Value")
}