library(fbi)
library(ggplot2)

rm_outliers.fredqd <- function(object) {
  # Error checking
  if (!inherits(object, "fredqd"))
    stop("Object must be of class 'fredqd'")

  data <- object
  N <- ncol(data)
  X <- data[, 2:N]

  # Calculate median of each series
  median_X <- apply(X, 2, stats::median, na.rm = TRUE)

  # Repeat median of each series over all data points in the series
  median_X_mat <- matrix(rep(median_X, nrow(X)), nrow = nrow(X),
                         ncol = ncol(X), byrow = TRUE)

  # Calculate quartiles
  Q <- apply(X, 2, stats::quantile, probs = c(0.25, 0.75), na.rm = TRUE)

  # Calculate interquartile range (IQR) of each series
  IQR <- Q[2, ] - Q[1, ]

  # Repeat IQR of each series over all data points in the series
  IQR_mat <- matrix(rep(IQR, nrow(X)), nrow = nrow(X),
                    ncol = ncol(X), byrow = TRUE)

  # Determine outliers
  Z <- abs(X - median_X_mat)
  outlier <- (Z > (10 * IQR_mat))

  # Replace outliers with NaN
  Y <- X
  Y[outlier] <- NA

  # Cleaned data
  outdata <- data
  outdata[, 2:N] <- Y
  class(outdata) <- c("data.frame", "fredqd")
  return(outdata)

  # Print the number of outliers
  print("Number of outliers:", quote = FALSE)
  print(sum(outlier, na.rm = TRUE), quote = FALSE)
}

## Step 1: Load Data
# Update these paths to match your local file paths
fred_md_path <- "FRED_MD.csv"
fred_qd_path <- "FRED_QD.csv"

# Load data with transformation
md <- fredmd(fred_md_path, date_start = NULL, date_end = NULL, transform = TRUE)
qd <- fredqd(fred_qd_path, date_start = NULL, date_end = NULL, transform = TRUE)

## Step 2: Prepare data - handle missing values and outliers
# Remove outliers first
md_clean <- rm_outliers.fredmd(md)
qd_clean <- rm_outliers.fredqd(qd)

# Check for columns with too many missing values and remove them
col_na_prop <- apply(is.na(md_clean), 2, mean)
md_select <- md_clean[, (col_na_prop < 0.05)]
col_na_prop_qd <- apply(is.na(qd_clean), 2, mean)
qd_select <- qd_clean[, (col_na_prop_qd < 0.05)]

# Create a balanced dataset by removing rows with any missing values
md_bal <- na.omit(md_select)
X_bal <- md_bal[,2:ncol(md_bal)]
rownames(X_bal) <- md_bal[,1]
qd_bal <- na.omit(qd_select)
X_bal_qd <- qd_bal[,2:ncol(qd_bal)]
rownames(X_bal_qd) <- qd_bal[,1]

# View the balanced dataset
head(X_bal)
head(X_bal_qd)

## Step 3: Explore Dataset
# View basic descriptions
describe_md("RPI")
describe_qd("GDPC1")

# Plot an example series
plot(md[["UMCSENTx"]], type = "l", main = "Consumer Sentiment (UMCSENTx)", ylab = "Index")

## Step 4: Estimate Common Factors using rpca
kmax <- 8
factors_md <- rpca(X_bal, kmax = kmax, standardize = FALSE, tau = 0)
factors_qd <- rpca(X_bal_qd, kmax = kmax, standardize = FALSE, tau = 0)

# View the estimated factors
head(factors_md$Fhat)
head(factors_qd$Fhat)

# Plot the first factor
plot(factors_md$Fhat[,1], type = "l", main = "Factor 1 md", ylab = "Factor Value")
plot(factors_qd$Fhat[,1], type = "l", main = "Factor 1 qd", ylab = "Factor Value")

xpt <- list(c(1,2,3,4), c(2,3,4,5), c(3,4,5,6), c(4,5,6,7))
se.rpca(factors_md, xpoints = xpt, qq = 50)

# Step 4b: Impute missing values using tw_apc
md_imputed <- tw_apc(md_select, 8)
qd_imputed <- tw_apc(qd_select, 8)

## Step 5: Forecasting Example (Industrial Production)
# Find the Industrial Production column in the balanced data
y <- X_bal[,"INDPRO"]  # Target variable
X <- factors_md$Fhat    # Estimated factors

# Linear model with first 3 factors
model <- lm(y ~ X[,1:3])
summary(model)

## Step 6: Construct Factor-Based Diffusion Index
fdi <- cumsum(factors_md$Fhat[,1])
plot(fdi, type = "l", main = "Factor-Based Diffusion Index", ylab = "Cumulative Factor 1")

## Conclusion
cat("This notebook performed PCA-based factor analysis on FRED-MD data and used it for forecasting and index construction.")
