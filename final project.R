# ========================= 1 ==================================================
library(tidyverse)
library(alr4)
library(caret)

data("Downer")
View(Downer)
sum(is.na(Downer))
dim(Downer)

Downer_df = Downer[, c("calving", "daysrec", "ck", "ast", "urea", "pcv","outcome")]
Downer_df = na.omit(Downer_df)
dim(Downer_df)


train_index = sample(1:nrow(Downer_df), size = 0.8*nrow(Downer_df), replace = FALSE)
train_index
train_data = Downer_df[train_index, ]
test_data = Downer_df[-train_index, ]

train_data
logistic_model = glm(outcome ~ calving  +  daysrec + ck + ast + urea + pcv,
                     data = train_data, family = "binomial")

predicted = predict(logistic_model,test_data,type = "response",cutoff = 0.5)
predicted = ifelse(predicted > 0.5,1,0)
confusion_matrix = table(test_data$outcome, predicted)

accuracy = sum(diag(confusion_matrix)) / sum(confusion_matrix)
accuracy

# =================================== 2 ========================================

# a : load the dataset
library(glmnet)
library(MASS)
data("Boston")
View(Boston)

glimpse(Boston)
# b: Lasso Regression Modeling 
x = model.matrix(medv ~ ., Boston,alpha = 1 )[,-1]
y = Boston$medv

set.seed(12345)
train_lasso_index = sample(1:nrow(Boston),size = 0.8 * nrow(Boston))
train_lasso_data = Boston[train_lasso_index,]
lasso_model = cv.glmnet(x = as.matrix(train_data[,-14]), y= train_lasso_data$medv, alpha = 1)

# ============================ 2
# Install and load necessary packages
install.packages("MASS")
install.packages("glmnet")
library(MASS)
library(glmnet)

# (a) Loading the Dataset
data("Boston")
head(Boston)
boston_data  =  Boston
# (b) Lasso Regression Modeling
X  =  as.matrix(boston_data[, -14])  # Features
y  =  boston_data$medv  # Target variable

# Use cross-validation to find optimal lambda
cv_fit  =  cv.glmnet(X, y, alpha=1)  # alpha=1 for Lasso regression
cv_fit$cvm
# Get the optimal lambda
optimal_lambda  =  cv_fit$lambda.min

# Build the Lasso model with the optimal lambda
lasso_model  =  glmnet(X, y, alpha=1, lambda=optimal_lambda)


# (c) Variable Selection:
# Extract coefficients
lasso_coefficients  =  coef(lasso_model)

# Identify features with non-zero coefficients
selected_features  =  which(lasso_coefficients[-1, ] != 0)
selected_features

# (d) evaluation
lasso_cv_metrics  =  cv_fit$cvm  # Mean cross-validated error

feature_importance  =  lasso_coefficients[selected_features + 1]


# (e) Interpretation
cat("Optimal Lambda:", optimal_lambda, "\n")
cat("Selected Features:", colnames(X)[selected_features], "\n")
cat("Coefficients of Selected Features:", feature_importance, "\n")
cat("Mean Squared Error (CV):", min(lasso_cv_metrics), "\n")

# ======================== 3 ====================================
# (a) load the faithful dataset in r
# Load necessary libraries
# Install and load necessary packages
install.packages("boot")
library(boot)

# Load the Faithful dataset
data(faithful)

# Set seed for reproducibility
set.seed(123)

# Create a data frame with predictors (waiting) and response variable (eruptions)
faithful_data  =  data.frame(waiting = faithful$waiting, eruptions = faithful$eruptions)

# Function to perform polynomial regression and calculate R^2
poly_regression  =  function(data, degree) {
  model  =  lm(eruptions ~ poly(waiting, degree, raw = TRUE), data = data)
  return(summary(model)$r.squared)
}
degree = 1:4
# Perform 10-fold cross-validation for degrees 1 to 4
cv_results  =  sapply(degree, function(degree) {
  cv_model  =  cv.glm(faithful_data, poly_regression, degree = degree,glmfit = TRUE)
  return(max(cv_model$delta))
})

# Identify the degree with the highest average cross-validated R^2
best_degree  =  which.max(cv_results)
best_r_squared  =  cv_results[best_degree]

# Display the selected degree and its corresponding average R^2 value
cat("Selected Degree:", best_degree, "\n")
cat("Corresponding Average R^2 Value:", best_r_squared, "\n")


# bard

# (a) Load the faithful dataset
library(datasets)
data(faithful)

# (b) Define functions for polynomial regression and cross-validation


cv_r2  =  function(model, data, folds) {
  library(caret)
  control  =  trainControl(method = "cv", number = folds)
  r2_score  =  function(pred, obs) {
    1 - mean((pred - obs)^2) / var(obs)
  }
  fit  =  train(model, data, metric = r2_score, trControl = control)
  fit$results$r2
}

# Compute cross-validated R2 for each degree
degrees  =  1:4
r2_scores  =  sapply(degrees, function(degree) {
  model  =  fit_poly_model(faithful, degree)
  cv_r2(model, faithful, 10)
})

# (d) Identify the degree with the highest average R2
best_degree  =  degrees[which.max(mean(r2_scores))]
best_r2  =  mean(r2_scores[degrees == best_degree])

cat("Best degree:", best_degree, "\n")
cat("Average R2:", best_r2)



# ==================== ==========================================================
# (a) Load the faithful dataset in R.
data(faithful)

# (b) Implement polynomial regression models with degrees from 1 to 4.
degrees  =  1:4
cv_r2_values  =  numeric(length(degrees))

for (degree in degrees) {
  # Create polynomial features
  poly_features  =  poly(faithful$waiting, degree, raw = TRUE)
  
  # Fit linear regression model
  lm_model  =  lm(faithful$eruptions ~ poly_features)
  
  # (c) Use 10-fold cross-validation to compute R2 values for each model.
  cv_r2  =  summary(lm_model)$r.squared
  cv_r2_values[degree]  =  cv_r2
}

# (d) Identify the degree that corresponds to the highest average cross-validated R2 value.
best_degree  =  which.max(cv_r2_values)
best_avg_r2  =  mean(cv_r2_values)

# Provide the selected degree and its corresponding average R2 value as the solution.
cat("Selected Degree:", best_degree, "\n")
cat("Average Cross-validated R2 Value:", best_avg_r2, "\n")


















# 4th question refer chapter 3.1 from book and for 5th question chapter 9 204 page
