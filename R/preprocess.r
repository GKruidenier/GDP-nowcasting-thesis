# Load necessary library
library(ggplot2)
library(fbi)


Fred_md_raw <- fredmd("FRED_MD.csv", date_start = NULL, date_end= NULL, transform = FALSE)
fred_qd_raw <- fredqd("FRED_QD.csv", date_start = NULL, date_end= NULL, transform = FALSE)

Fred_md_transformed <- fredmd("FRED_MD.csv", date_start = NULL, date_end= NULL, transform = TRUE)
Fred_qd_transformed <- fredqd("FRED_QD.csv", date_start = NULL, date_end= NULL, transform = TRUE)


# Select a random subset of features
set.seed(123)  # For reproducibility
random_features <- sample(colnames(Fred_md_raw)[-1], 5)  # Exclude the date column

# Plot features before and after transformation
par(mfrow = c(2, 5))  # Set up plotting area

for (feature in random_features) {
    # Plot before transformation
    plot(Fred_md_raw[[feature]], type = "l", main = paste("Before Transformation:", feature), xlab = "Index", ylab = "Value")
    
    # Plot after transformation
    plot(Fred_md_transformed[[feature]], type = "l", main = paste("After Transformation:", feature), xlab = "Index", ylab = "Value")

}

# Select a random subset of features for FRED_QD
random_features_qd <- sample(colnames(fred_qd_raw)[-1], 5)  # Exclude the date column

# Plot features before and after transformation for FRED_QD
par(mfrow = c(2, 5))  # Set up plotting area

for (feature in random_features_qd) {
    # Plot before transformation
    plot(fred_qd_raw[[feature]], type = "l", main = paste("Before Transformation:", feature), xlab = "Index", ylab = "Value")
    
    # Plot after transformation
    plot(Fred_qd_transformed[[feature]], type = "l", main = paste("After Transformation:", feature), xlab = "Index", ylab = "Value")
}

# Save transformed data to CSV files
write.csv(Fred_md_transformed, file = "c:/Users/giada/Documents/GDP nowcasting thesis/Fred_md_transformed.csv", row.names = FALSE)
write.csv(Fred_qd_transformed, file = "c:/Users/giada/Documents/GDP nowcasting thesis/Fred_qd_transformed.csv", row.names = FALSE)

