#Loading necessary libraries 

library(readr)
library(dplyr)
library(ggplot2)
library(tidyr)
library(caret)

if (require("plumber")) {
  require("plumber")
} else {
  install.packages("plumber", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

## caret ----
if (require("caret")) {
  require("caret")
} else {
  install.packages("caret", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")}

#Importing the dataset

Crop_recommendation <- read_csv("data/Crop_recommendation.csv")
View(Crop_recommendation)

# Measures of Frequency
frequency_table <- table(Crop_recommendation$label)
print("Measures of Frequency:")
print(frequency_table)

# Measures of Central Tendency
central_tendency <- summary(Crop_recommendation[, c("N", "P", "K", "temperature", "humidity", "ph", "rainfall")])
print("Measures of Central Tendency:")
print(central_tendency)

# Measures of Distribution
distribution <- sapply(Crop_recommendation[, c("N", "P", "K", "temperature", "humidity", "ph", "rainfall")], sd)
print("Measures of Distribution:")
print(distribution)

# Measures of Relationship
correlation_matrix <- cor(Crop_recommendation[, c("N", "P", "K", "temperature", "humidity", "ph", "rainfall")])
print("Measures of Relationship (Correlation Matrix):")
print(correlation_matrix)
#STEP 3. Issue 2: Inferential Statistics - ANOVA
# Example: One-way ANOVA to test if there are any significant differences in the 'N' values among different crops
anova_result <- aov(N ~ label, data = Crop_recommendation)
print("ANOVA Results:")
print(summary(anova_result))


# Load necessary libraries
library(ggplot2)

# Univariate Plot for 'N' values
ggplot(Crop_recommendation, aes(x = label, y = N)) +
  geom_boxplot(fill = "lightblue") +
  labs(title = "Boxplot of N values for each crop",
       x = "Crop",
       y = "N Value")

# The above code can be repeated for other variables (P, K, temperature, humidity, ph, rainfall) to create univariate plots for each.

# Multivariate Plot for 'N' and 'P' values
ggplot(Crop_recommendation, aes(x = N, y = P, color = label)) +
  geom_point() +
  labs(title = "Scatter plot of N vs P for each crop",
       x = "N Value",
       y = "P Value")

# The above code can be repeated for other combinations of variables to create multivariate plots.



# Note: This can be repeated for other variables (P, K, temperature...


# Confirmation of the presence of missing values
missing_values <- sum(is.na(Crop_recommendation))
print(paste("Number of Missing Values:", missing_values))

# Data imputation (not applicable but did for practic)
# For simplicity, let's impute missing values with the mean of each column
Crop_recommendation_imputed <- Crop_recommendation %>%
  mutate_all(~ifelse(is.na(.), mean(., na.rm = TRUE), .))

# Confirm that missing values are imputed
missing_values_after_imputation <- sum(is.na(Crop_recommendation_imputed))
print(paste("Number of Missing Values After Imputation:", missing_values_after_imputation))

# Data transformation (not applicable but did for practice)
# For example, you can log-transform the 'rainfall' variable
Crop_recommendation_transformed <- Crop_recommendation_imputed %>%
  mutate(rainfall_log = log1p(rainfall))

# Display the head of the transformed dataset
print("Head of the Transformed Dataset:")
print(head(Crop_recommendation_transformed))

# Load necessary libraries ----
library(caret)

# Data Splitting ----
set.seed(123)  # For reproducibility
index <- createDataPartition(Crop_recommendation$label, p = 0.8, list = FALSE)
train_data <- Crop_recommendation[index, ]
test_data <- Crop_recommendation[-index, ]

# Linear Algorithm ----
## Linear Discriminant Analysis ----
### Linear Discriminant Analysis with caret ----
# We train the following models, all of which are using 5-fold cross validation
#   LDA
#   CART

train_control <- trainControl(method = "cv", number = 5)
# We also apply a standardize data transform to make the mean = 0 and
# standard deviation = 1
recommendation_caret_model_lda <- train(label ~ .,
                                 data = train_data,
                                 method = "lda", metric = "Accuracy",
                                 preProcess = c("center", "scale"),
                                 trControl = train_control)

#### Display the model's details ----
print(recommendation_caret_model_lda)

#### Make predictions ----
predictions <- predict(recommendation_caret_model_lda,
                       test_data[, 1:7])


# Non-Linear Algorithm ----
##Classification and Regression Trees: Decision tree for a classification problem with caret ----

#### Train the model ----
set.seed(7)
# We apply the 5-fold cross validation resampling method
train_control <- trainControl(method = "cv", number = 5)
recommendation_caret_model_rpart <- train(label ~ ., data = train_data,
                                   method = "rpart", metric = "Accuracy",
                                   trControl = train_control)

#### Display the model's details ----
print(recommendation_caret_model_rpart)

#### Make predictions ----
predictions <- predict(recommendation_caret_model_rpart,
                       test_data[, 1:7])

#### Display the model's evaluation metrics ----
table(predictions, test_data$label)



###Model Performance Comparison ----

## Call the `resamples` Function ----
# We then create a list of the model results and pass the list as an argument
# to the `resamples` function.

results <- resamples(list(LDA = recommendation_caret_model_lda, CART = recommendation_caret_model_rpart))

# Display the Results ----
## 1. Table Summary ----
# This is the simplest comparison. It creates a table with one model per row
# and its corresponding evaluation metrics displayed per column.

summary(results)

## 2. Box and Whisker Plot ----
# This is useful for visually observing the spread of the estimated accuracies
# for different algorithms and how they relate.

scales <- list(x = list(relation = "free"), y = list(relation = "free"))
bwplot(results, scales = scales)

## 3. Dot Plots ----
# They show both the mean estimated accuracy as well as the 95% confidence
# interval (e.g. the range in which 95% of observed scores fell).

scales <- list(x = list(relation = "free"), y = list(relation = "free"))
dotplot(results, scales = scales)

## 4. Scatter Plot Matrix ----
# This is useful when considering whether the predictions from two
# different algorithms are correlated. If weakly correlated, then they are good
# candidates for being combined in an ensemble prediction.

splom(results)

## 5. Pairwise xyPlots ----
# You can zoom in on one pairwise comparison of the accuracy of trial-folds for
# two models using an xyplot.

# xyplot plots to compare models
xyplot(results, models = c("LDA", "CART"))


## 6. Statistical Significance Tests ----
# This is used to calculate the significance of the differences between the
# metric distributions of the various models.

### Upper Diagonal ----
# The upper diagonal of the table shows the estimated difference between the
# distributions. If we think that LDA is the most accurate model from looking
# at the previous graphs, we can get an estimate of how much better it is than
# specific other models in terms of absolute accuracy.

### Lower Diagonal ----
# The lower diagonal contains p-values of the null hypothesis.
# The null hypothesis is a claim that "the models are the same".
# A lower p-value is better (more significant).

diffs <- diff(results)

summary(diffs)

# Data Splitting
set.seed(123)
index <- createDataPartition(Crop_recommendation$label, p = 0.8, list = FALSE)
train_data <- Crop_recommendation[index, ]
test_data <- Crop_recommendation[-index, ]

# Hyperparameter Tuning using Grid Search (Random Forest)
tune_grid <- expand.grid(
  mtry = seq(1, 7, by = 1)
)

tune_control <- trainControl(method = "cv", number = 5)
tuned_model <- train(
  label ~ .,
  data = train_data,
  method = "rf",
  trControl = tune_control,
  tuneGrid = tune_grid
)

# Print the best tuning parameters
print("Best Tuning Parameters:")
print(tuned_model$bestTune)

# Ensemble (Random Forest)
ensemble_models <- lapply(1:5, function(seed) {
  set.seed(seed)
  train(label ~ ., data = train_data, method = "rf", trControl = tune_control, tuneGrid = tune_grid)
})

# Manually combine predictions for classification (majority vote)
ensemble_predict <- function(models, newdata) {
  predictions <- lapply(models, function(model) predict(model, newdata))
  ensemble_prediction <- do.call(cbind, predictions)
  
  # Use majority vote for classification
  majority_vote <- apply(ensemble_prediction, 1, function(row) {
    names(which.max(table(row)))
  })
  
  # Convert to factor and set levels explicitly to include all possible levels
  levels_to_include <- unique(c(levels(factor(test_data$label)), levels(factor(majority_vote))))
  majority_vote <- factor(majority_vote, levels = levels_to_include)
  
  return(majority_vote)
}

# Apply ensemble_predict function to your models
ensemble_predictions <- ensemble_predict(ensemble_models, test_data)

# Evaluate the ensemble predictions (for confusion matrix)
confusion_matrix <- confusionMatrix(ensemble_predictions, factor(test_data$label, levels = levels(ensemble_predictions)))
print("Confusion Matrix:")
print(confusion_matrix)

# We can create an API to access the model from outside R using a package
# called Plumber.


# Saving and Load your Model ----
# Saving a model into a file allows you to load it later and use it to make
# predictions. Saved models can be loaded by calling the `readRDS()` function

saveRDS(recommendation_caret_model_lda, "./data/saved_recommendation_caret_model_lda.rds")

setwd("../BIProject-Crop-Prediction-Analytics")

# Process a Plumber API ----
# This allows us to process a plumber API
api <- plumber::plumb("API.R")

# Run the API on a specific port ----
# Specify a constant localhost port to use
api$run(host = "127.0.0.1", port = 5022)


# We test the API using the following values:
# for the arguments:
# N, P, K, temperature, humidity, ph, rainfall
# 45, 32, 24, 78, 22, 9, and 10 respectively should output "mango"