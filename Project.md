Business Intelligence Project Markdown
================
\<Virginia Wang’ang’a\>
\<26/11/2023\>

- [Student Details](#student-details)
- [Setup Chunk](#setup-chunk)
- [Loading the Crop Recommendation
  Dataset](#loading-the-crop-recommendation-dataset)
  - [Description of the Dataset](#description-of-the-dataset)
- [Inferential Statistics :ANOVA](#inferential-statistics-anova)
- [**Basic Visualization - Univariate and Multivariate
  Plots**](#basic-visualization---univariate-and-multivariate-plots)
- [**Preprocessing and Data
  Transformation**](#preprocessing-and-data-transformation)
- [**Training the Model**](#training-the-model)
- [**Hyper-Parameter Tuning and
  Ensembled**](#hyper-parameter-tuning-and-ensembled)
- [**Consolidation**](#consolidation)
- [](#section)

# Student Details

<table>
<colgroup>
<col style="width: 53%" />
<col style="width: 46%" />
</colgroup>
<tbody>
<tr class="odd">
<td><strong>Student ID Numbers and Names of Group Members</strong></td>
<td><ol type="1">
<li>126761 - B- Virginia Wang’ang’a</li>
</ol></td>
</tr>
<tr class="even">
<td><strong>GitHub Classroom Group Name</strong></td>
<td>B</td>
</tr>
<tr class="odd">
<td><strong>Course Code</strong></td>
<td>BBT4206</td>
</tr>
<tr class="even">
<td><strong>Course Name</strong></td>
<td>Business Intelligence II</td>
</tr>
<tr class="odd">
<td><strong>Program</strong></td>
<td>Bachelor of Business Information Technology</td>
</tr>
<tr class="even">
<td><strong>Semester Duration</strong></td>
<td>21<sup>st</sup> August 2023 to 28<sup>th</sup> November 2023</td>
</tr>
</tbody>
</table>

# Setup Chunk

We start by installing all the required packages

``` r
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
```

------------------------------------------------------------------------

**Note:** the following “*KnitR*” options have been set as the defaults
in this markdown:  
`knitr::opts_chunk$set(echo = TRUE, warning = FALSE, eval = TRUE, collapse = FALSE, tidy.opts = list(width.cutoff = 80), tidy = TRUE)`.

More KnitR options are documented here
<https://bookdown.org/yihui/rmarkdown-cookbook/chunk-options.html> and
here <https://yihui.org/knitr/options/>.

``` r
knitr::opts_chunk$set(
    eval = TRUE,
    echo = TRUE,
    warning = FALSE,
    collapse = FALSE,
    tidy = TRUE
)
```

------------------------------------------------------------------------

**Note:** the following “*R Markdown*” options have been set as the
defaults in this markdown:

> output:  
>   
> github_document:  
> toc: yes  
> toc_depth: 4  
> fig_width: 6  
> fig_height: 4  
> df_print: default  
>   
> editor_options:  
> chunk_output_type: console

# Loading the Crop Recommendation Dataset

The 20230412-20230719-BI1-BBIT4-1-Crop recommendation dataset is then
loaded. The dataset and its metadata are available here:
<https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset>

``` r
Crop_recommendation <- read_csv("data/Crop_recommendation.csv")
```

    ## Rows: 2200 Columns: 8
    ## ── Column specification ────────────────────────────────────────────────────────
    ## Delimiter: ","
    ## chr (1): label
    ## dbl (7): N, P, K, temperature, humidity, ph, rainfall
    ## 
    ## ℹ Use `spec()` to retrieve the full column specification for this data.
    ## ℹ Specify the column types or set `show_col_types = FALSE` to quiet this message.

``` r
View(Crop_recommendation)
```

## Description of the Dataset

We then display the number of observations and number of variables. We
have 101 observations and 100 variables .

``` r
# Load necessary libraries
library(dplyr)

# Measures of Frequency
frequency_table <- table(Crop_recommendation$label)
print("Measures of Frequency:")
```

    ## [1] "Measures of Frequency:"

``` r
print(frequency_table)
```

    ## 
    ##       apple      banana   blackgram    chickpea     coconut      coffee 
    ##         100         100         100         100         100         100 
    ##      cotton      grapes        jute kidneybeans      lentil       maize 
    ##         100         100         100         100         100         100 
    ##       mango   mothbeans    mungbean   muskmelon      orange      papaya 
    ##         100         100         100         100         100         100 
    ##  pigeonpeas pomegranate        rice  watermelon 
    ##         100         100         100         100

``` r
# Measures of Central Tendency
central_tendency <- summary(Crop_recommendation[, c("N", "P", "K", "temperature", "humidity", "ph", "rainfall")])
print("Measures of Central Tendency:")
```

    ## [1] "Measures of Central Tendency:"

``` r
print(central_tendency)
```

    ##        N                P                K           temperature    
    ##  Min.   :  0.00   Min.   :  5.00   Min.   :  5.00   Min.   : 8.826  
    ##  1st Qu.: 21.00   1st Qu.: 28.00   1st Qu.: 20.00   1st Qu.:22.769  
    ##  Median : 37.00   Median : 51.00   Median : 32.00   Median :25.599  
    ##  Mean   : 50.55   Mean   : 53.36   Mean   : 48.15   Mean   :25.616  
    ##  3rd Qu.: 84.25   3rd Qu.: 68.00   3rd Qu.: 49.00   3rd Qu.:28.562  
    ##  Max.   :140.00   Max.   :145.00   Max.   :205.00   Max.   :43.675  
    ##     humidity           ph           rainfall     
    ##  Min.   :14.26   Min.   :3.505   Min.   : 20.21  
    ##  1st Qu.:60.26   1st Qu.:5.972   1st Qu.: 64.55  
    ##  Median :80.47   Median :6.425   Median : 94.87  
    ##  Mean   :71.48   Mean   :6.469   Mean   :103.46  
    ##  3rd Qu.:89.95   3rd Qu.:6.924   3rd Qu.:124.27  
    ##  Max.   :99.98   Max.   :9.935   Max.   :298.56

``` r
# Measures of Distribution
distribution <- sapply(Crop_recommendation[, c("N", "P", "K", "temperature", "humidity", "ph", "rainfall")], sd)
print("Measures of Distribution:")
```

    ## [1] "Measures of Distribution:"

``` r
print(distribution)
```

    ##           N           P           K temperature    humidity          ph 
    ##  36.9173338  32.9858827  50.6479305   5.0637486  22.2638116   0.7739377 
    ##    rainfall 
    ##  54.9583885

``` r
# Measures of Relationship
correlation_matrix <- cor(Crop_recommendation[, c("N", "P", "K", "temperature", "humidity", "ph", "rainfall")])
print("Measures of Relationship (Correlation Matrix):")
```

    ## [1] "Measures of Relationship (Correlation Matrix):"

``` r
print(correlation_matrix)
```

    ##                       N           P           K temperature     humidity
    ## N            1.00000000 -0.23145958 -0.14051184  0.02650380  0.190688379
    ## P           -0.23145958  1.00000000  0.73623222 -0.12754113 -0.118734116
    ## K           -0.14051184  0.73623222  1.00000000 -0.16038713  0.190858861
    ## temperature  0.02650380 -0.12754113 -0.16038713  1.00000000  0.205319677
    ## humidity     0.19068838 -0.11873412  0.19085886  0.20531968  1.000000000
    ## ph           0.09668285 -0.13801889 -0.16950310 -0.01779502 -0.008482539
    ## rainfall     0.05902022 -0.06383905 -0.05346135 -0.03008378  0.094423053
    ##                       ph    rainfall
    ## N            0.096682846  0.05902022
    ## P           -0.138018893 -0.06383905
    ## K           -0.169503098 -0.05346135
    ## temperature -0.017795017 -0.03008378
    ## humidity    -0.008482539  0.09442305
    ## ph           1.000000000 -0.10906948
    ## rainfall    -0.109069484  1.00000000

# Inferential Statistics :ANOVA

``` r
anova_result <- aov(N ~ label, data = Crop_recommendation)
print("ANOVA Results:")
```

    ## [1] "ANOVA Results:"

``` r
print(summary(anova_result))
```

    ##               Df  Sum Sq Mean Sq F value Pr(>F)    
    ## label         21 2686561  127931   897.6 <2e-16 ***
    ## Residuals   2178  310433     143                   
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

# **Basic Visualization - Univariate and Multivariate Plots**

``` r
ggplot(Crop_recommendation, aes(x = label, y = N)) +
  geom_boxplot(fill = "red") +
  labs(title = "Boxplot of N values for each crop",
       x = "Crop",
       y = "N Value")
```

![](Project_files/figure-gfm/Your%20Seventh%20Code%20Chunk-1.png)<!-- -->

``` r
# The above code can be repeated for other variables (P, K, temperature, humidity, ph, rainfall) to create univariate plots for each.

# Multivariate Plot for 'N' and 'P' values
ggplot(Crop_recommendation, aes(x = N, y = P, color = label)) +
  geom_point() +
  labs(title = "Scatter plot of N vs P for each crop",
       x = "N Value",
       y = "P Value")
```

![](Project_files/figure-gfm/Your%20Seventh%20Code%20Chunk-2.png)<!-- -->

``` r
# The above code can be repeated for other combinations of variables to create multivariate plots.
```

# **Preprocessing and Data Transformation**

``` r
# Confirmation of the presence of missing values
missing_values <- sum(is.na(Crop_recommendation))
print(paste("Number of Missing Values:", missing_values))
```

    ## [1] "Number of Missing Values: 0"

``` r
# Data imputation (not applicable but did for practic)
# For simplicity, let's impute missing values with the mean of each column
Crop_recommendation_imputed <- Crop_recommendation %>%
  mutate_all(~ifelse(is.na(.), mean(., na.rm = TRUE), .))

# Confirm that missing values are imputed
missing_values_after_imputation <- sum(is.na(Crop_recommendation_imputed))
print(paste("Number of Missing Values After Imputation:", missing_values_after_imputation))
```

    ## [1] "Number of Missing Values After Imputation: 0"

``` r
# Data transformation (not applicable but did for practice)
# For example, you can log-transform the 'rainfall' variable
Crop_recommendation_transformed <- Crop_recommendation_imputed %>%
  mutate(rainfall_log = log1p(rainfall))

# Display the head of the transformed dataset
print("Head of the Transformed Dataset:")
```

    ## [1] "Head of the Transformed Dataset:"

``` r
print(head(Crop_recommendation_transformed))
```

    ## # A tibble: 6 × 9
    ##       N     P     K temperature humidity    ph rainfall label rainfall_log
    ##   <dbl> <dbl> <dbl>       <dbl>    <dbl> <dbl>    <dbl> <chr>        <dbl>
    ## 1    90    42    43        20.9     82.0  6.50     203. rice          5.32
    ## 2    85    58    41        21.8     80.3  7.04     227. rice          5.43
    ## 3    60    55    44        23.0     82.3  7.84     264. rice          5.58
    ## 4    74    35    40        26.5     80.2  6.98     243. rice          5.50
    ## 5    78    42    42        20.1     81.6  7.63     263. rice          5.57
    ## 6    69    37    42        23.1     83.4  7.07     251. rice          5.53

# **Training the Model**

``` r
# Data Splitting ----
set.seed(123)  # For reproducibility
index <- createDataPartition(Crop_recommendation$label, p = 0.8, list = FALSE)
train_data <- Crop_recommendation[index, ]
test_data <- Crop_recommendation[-index, ]


## Linear Discriminant Analysis ----
### Linear Discriminant Analysis with caret ----
# We train the following models, all of which are using 5-fold cross validation
#   LDA
#   CART

train_control <- trainControl(method = "cv", number = 5)
# We also apply a standardize data transform to make the mean = 0 and
# standard deviation = 1
farming_caret_model_lda <- train(label ~ .,
                                  data = train_data,
                                  method = "lda", metric = "Accuracy",
                                  preProcess = c("center", "scale"),
                                  trControl = train_control)

#### Display the model's details ----
print(farming_caret_model_lda)
```

    ## Linear Discriminant Analysis 
    ## 
    ## 1760 samples
    ##    7 predictor
    ##   22 classes: 'apple', 'banana', 'blackgram', 'chickpea', 'coconut', 'coffee', 'cotton', 'grapes', 'jute', 'kidneybeans', 'lentil', 'maize', 'mango', 'mothbeans', 'mungbean', 'muskmelon', 'orange', 'papaya', 'pigeonpeas', 'pomegranate', 'rice', 'watermelon' 
    ## 
    ## Pre-processing: centered (7), scaled (7) 
    ## Resampling: Cross-Validated (5 fold) 
    ## Summary of sample sizes: 1408, 1408, 1408, 1408, 1408 
    ## Resampling results:
    ## 
    ##   Accuracy   Kappa    
    ##   0.9630682  0.9613095

``` r
#### Make predictions ----
predictions <- predict(farming_caret_model_lda,
                       test_data[, 1:7])


# Non-Linear Algorithm ----
##Classification and Regression Trees: Decision tree for a classification problem with caret ----

#### Train the model ----
set.seed(7)
# We apply the 5-fold cross validation resampling method
train_control <- trainControl(method = "cv", number = 5)
farming_caret_model_rpart <- train(label ~ ., data = train_data,
                                    method = "rpart", metric = "Accuracy",
                                    trControl = train_control)

#### Display the model's details ----
print(farming_caret_model_rpart)
```

    ## CART 
    ## 
    ## 1760 samples
    ##    7 predictor
    ##   22 classes: 'apple', 'banana', 'blackgram', 'chickpea', 'coconut', 'coffee', 'cotton', 'grapes', 'jute', 'kidneybeans', 'lentil', 'maize', 'mango', 'mothbeans', 'mungbean', 'muskmelon', 'orange', 'papaya', 'pigeonpeas', 'pomegranate', 'rice', 'watermelon' 
    ## 
    ## No pre-processing
    ## Resampling: Cross-Validated (5 fold) 
    ## Summary of sample sizes: 1408, 1408, 1408, 1408, 1408 
    ## Resampling results across tuning parameters:
    ## 
    ##   cp          Accuracy    Kappa    
    ##   0.04315476  0.86136364  0.8547619
    ##   0.04702381  0.81306818  0.8041667
    ##   0.04761905  0.04545455  0.0000000
    ## 
    ## Accuracy was used to select the optimal model using the largest value.
    ## The final value used for the model was cp = 0.04315476.

``` r
#### Make predictions ----
predictions <- predict(farming_caret_model_rpart,
                       test_data[, 1:7])

#### Display the model's evaluation metrics ----
table(predictions, test_data$label)
```

    ##              
    ## predictions   apple banana blackgram chickpea coconut coffee cotton grapes jute
    ##   apple          20      0         0        0       0      0      0      0    0
    ##   banana          0     20         0        0       0      0      0      0    0
    ##   blackgram       0      0        19        0       0      0      0      0    0
    ##   chickpea        0      0         0       20       0      0      0      0    0
    ##   coconut         0      0         0        0      20      0      0      0    0
    ##   coffee          0      0         0        0       0     20      0      0    4
    ##   cotton          0      0         0        0       0      0     20      0    0
    ##   grapes          0      0         0        0       0      0      0     20    0
    ##   jute            0      0         0        0       0      0      0      0    0
    ##   kidneybeans     0      0         0        0       0      0      0      0    0
    ##   lentil          0      0         0        0       0      0      0      0    0
    ##   maize           0      0         1        0       0      0      0      0    0
    ##   mango           0      0         0        0       0      0      0      0    0
    ##   mothbeans       0      0         0        0       0      0      0      0    0
    ##   mungbean        0      0         0        0       0      0      0      0    0
    ##   muskmelon       0      0         0        0       0      0      0      0    0
    ##   orange          0      0         0        0       0      0      0      0    0
    ##   papaya          0      0         0        0       0      0      0      0    0
    ##   pigeonpeas      0      0         0        0       0      0      0      0    0
    ##   pomegranate     0      0         0        0       0      0      0      0    0
    ##   rice            0      0         0        0       0      0      0      0   16
    ##   watermelon      0      0         0        0       0      0      0      0    0
    ##              
    ## predictions   kidneybeans lentil maize mango mothbeans mungbean muskmelon
    ##   apple                 0      0     0     0         0        0         0
    ##   banana                0      0     0     0         0        0         0
    ##   blackgram             0     20     0     0        20        0         0
    ##   chickpea              0      0     0     0         0        0         0
    ##   coconut               0      0     0     0         0        0         0
    ##   coffee                0      0     0     0         0        0         0
    ##   cotton                0      0     2     0         0        0         0
    ##   grapes                0      0     0     0         0        0         0
    ##   jute                  0      0     0     0         0        0         0
    ##   kidneybeans          20      0     0     0         0        0         0
    ##   lentil                0      0     0     0         0        0         0
    ##   maize                 0      0    18     0         0        0         0
    ##   mango                 0      0     0    20         0        0         0
    ##   mothbeans             0      0     0     0         0        0         0
    ##   mungbean              0      0     0     0         0       20         0
    ##   muskmelon             0      0     0     0         0        0        20
    ##   orange                0      0     0     0         0        0         0
    ##   papaya                0      0     0     0         0        0         0
    ##   pigeonpeas            0      0     0     0         0        0         0
    ##   pomegranate           0      0     0     0         0        0         0
    ##   rice                  0      0     0     0         0        0         0
    ##   watermelon            0      0     0     0         0        0         0
    ##              
    ## predictions   orange papaya pigeonpeas pomegranate rice watermelon
    ##   apple            0      0          0           0    0          0
    ##   banana           0      0          0           0    0          0
    ##   blackgram        0      0          0           0    0          0
    ##   chickpea         0      0          0           0    0          0
    ##   coconut          0      0          0           0    0          0
    ##   coffee           0      0          0           0    0          0
    ##   cotton           0      0          0           0    0          0
    ##   grapes           0      0          0           0    0          0
    ##   jute             0      0          0           0    0          0
    ##   kidneybeans      0      0          0           0    0          0
    ##   lentil           0      0          0           0    0          0
    ##   maize            0      0          0           0    0          0
    ##   mango            0      0          0           0    0          0
    ##   mothbeans        0      0          0           0    0          0
    ##   mungbean         0      0          0           0    0          0
    ##   muskmelon        0      0          0           0    0          0
    ##   orange          20      0          0           0    0          0
    ##   papaya           0     20          0           0    0          0
    ##   pigeonpeas       0      0         20           0    0          0
    ##   pomegranate      0      0          0          20    0          0
    ##   rice             0      0          0           0   20          0
    ##   watermelon       0      0          0           0    0         20

``` r
###Model Performance Comparison ----

## Call the `resamples` Function ----
# We then create a list of the model results and pass the list as an argument
# to the `resamples` function.

results <- resamples(list(LDA = farming_caret_model_lda, CART = farming_caret_model_rpart))

# Display the Results ----
## 1. Table Summary ----
# This is the simplest comparison. It creates a table with one model per row
# and its corresponding evaluation metrics displayed per column.

summary(results)
```

    ## 
    ## Call:
    ## summary.resamples(object = results)
    ## 
    ## Models: LDA, CART 
    ## Number of resamples: 5 
    ## 
    ## Accuracy 
    ##           Min.   1st Qu.    Median      Mean   3rd Qu.      Max. NA's
    ## LDA  0.9573864 0.9573864 0.9573864 0.9630682 0.9659091 0.9772727    0
    ## CART 0.8181818 0.8352273 0.8579545 0.8613636 0.8607955 0.9346591    0
    ## 
    ## Kappa 
    ##           Min.   1st Qu.    Median      Mean   3rd Qu.      Max. NA's
    ## LDA  0.9553571 0.9553571 0.9553571 0.9613095 0.9642857 0.9761905    0
    ## CART 0.8095238 0.8273810 0.8511905 0.8547619 0.8541667 0.9315476    0

``` r
## 2. Box and Whisker Plot ----
# This is useful for visually observing the spread of the estimated accuracies
# for different algorithms and how they relate.

scales <- list(x = list(relation = "free"), y = list(relation = "free"))
bwplot(results, scales = scales)
```

![](Project_files/figure-gfm/Your%20nine%20Code%20Chunk%7D%20#%20Confirmation%20of%20the%20presence%20of%20missing-1.png)<!-- -->

``` r
## 3. Dot Plots ----
# They show both the mean estimated accuracy as well as the 95% confidence
# interval (e.g. the range in which 95% of observed scores fell).

scales <- list(x = list(relation = "free"), y = list(relation = "free"))
dotplot(results, scales = scales)
```

![](Project_files/figure-gfm/Your%20nine%20Code%20Chunk%7D%20#%20Confirmation%20of%20the%20presence%20of%20missing-2.png)<!-- -->

``` r
## 4. Scatter Plot Matrix ----
# This is useful when considering whether the predictions from two
# different algorithms are correlated. If weakly correlated, then they are good
# candidates for being combined in an ensemble prediction.

splom(results)
```

![](Project_files/figure-gfm/Your%20nine%20Code%20Chunk%7D%20#%20Confirmation%20of%20the%20presence%20of%20missing-3.png)<!-- -->

``` r
## 5. Pairwise xyPlots ----
# You can zoom in on one pairwise comparison of the accuracy of trial-folds for
# two models using an xyplot.

# xyplot plots to compare models
xyplot(results, models = c("LDA", "CART"))
```

![](Project_files/figure-gfm/Your%20nine%20Code%20Chunk%7D%20#%20Confirmation%20of%20the%20presence%20of%20missing-4.png)<!-- -->

``` r
## 6. Statistical Significance Tests ----
# This is used to calculate the significance of the differences between the
# metric distributions of the various models.
diffs <- diff(results)

summary(diffs)
```

    ## 
    ## Call:
    ## summary.diff.resamples(object = diffs)
    ## 
    ## p-value adjustment: bonferroni 
    ## Upper diagonal: estimates of the difference
    ## Lower diagonal: p-value for H0: difference = 0
    ## 
    ## Accuracy 
    ##      LDA      CART  
    ## LDA           0.1017
    ## CART 0.008679       
    ## 
    ## Kappa 
    ##      LDA      CART  
    ## LDA           0.1065
    ## CART 0.008679

# **Hyper-Parameter Tuning and Ensembled**

``` r
# Data Splitting
set.seed(70)
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
```

    ## [1] "Best Tuning Parameters:"

``` r
print(tuned_model$bestTune)
```

    ##   mtry
    ## 2    2

``` r
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
```

    ## [1] "Confusion Matrix:"

``` r
print(confusion_matrix)
```

    ## Confusion Matrix and Statistics
    ## 
    ##              Reference
    ## Prediction    apple banana blackgram chickpea coconut coffee cotton grapes jute
    ##   apple           0      0         0        0       0      0      0      0    0
    ##   banana          0      0         0        0       0      0      0      0    0
    ##   blackgram       0      0         0        0       0      0      0      0    0
    ##   chickpea        0      0         0        0       0      0      0      0    0
    ##   coconut         0      0         0        0       0      0      0      0    0
    ##   coffee          0      0         0        0       0      0      0      0    0
    ##   cotton          0      0         0        0       0      0      0      0    0
    ##   grapes          0      0         0        0       0      0      0      0    0
    ##   jute            0      0         0        0       0      0      0      0    0
    ##   kidneybeans     0      0         0        0       0      0      0      0    0
    ##   lentil          0      0         0        0       0      0      0      0    0
    ##   maize           0      0         0        0       0      0      0      0    0
    ##   mango           0      0         0        0       0      0      0      0    0
    ##   mothbeans       0      0         0        0       0      0      0      0    0
    ##   mungbean        0      0         0        0       0      0      0      0    0
    ##   muskmelon       0      0         0        0       0      0      0      0    0
    ##   orange          0      0         0        0       0      0      0      0    0
    ##   papaya          0      0         0        0       0      0      0      0    0
    ##   pigeonpeas      0      0         0        0       0      0      0      0    0
    ##   pomegranate     0      0         0        0       0      0      0      0    0
    ##   rice            0      0         0        0       0      0      0      0    0
    ##   watermelon      0      0         0        0       0      0      0      0    0
    ##   1              20      0         0        0       0      0      0      0    0
    ##   10              0      0         0        0       0      0      0      0    0
    ##   11              0      0         0        0       0      0      0      0    0
    ##   12              0      0         0        0       0      0      0      0    0
    ##   13              0      0         0        0       0      0      0      0    0
    ##   14              0      0         0        0       0      0      0      0    0
    ##   15              0      0         0        0       0      0      0      0    0
    ##   16              0      0         0        0       0      0      0      0    0
    ##   17              0      0         0        0       0      0      0      0    0
    ##   18              0      0         0        0       0      0      0      0    0
    ##   19              0      0         0        0       0      0      0      0    0
    ##   2               0     20         0        0       0      0      0      0    0
    ##   20              0      0         0        0       0      0      0      0    0
    ##   21              0      0         0        0       0      0      0      0    0
    ##   22              0      0         0        0       0      0      0      0    0
    ##   3               0      0        20        0       0      0      0      0    0
    ##   4               0      0         0       20       0      0      0      0    0
    ##   5               0      0         0        0      20      0      0      0    0
    ##   6               0      0         0        0       0     20      0      0    0
    ##   7               0      0         0        0       0      0     20      0    0
    ##   8               0      0         0        0       0      0      0     20    0
    ##   9               0      0         0        0       0      0      0      0   20
    ##              Reference
    ## Prediction    kidneybeans lentil maize mango mothbeans mungbean muskmelon
    ##   apple                 0      0     0     0         0        0         0
    ##   banana                0      0     0     0         0        0         0
    ##   blackgram             0      0     0     0         0        0         0
    ##   chickpea              0      0     0     0         0        0         0
    ##   coconut               0      0     0     0         0        0         0
    ##   coffee                0      0     0     0         0        0         0
    ##   cotton                0      0     0     0         0        0         0
    ##   grapes                0      0     0     0         0        0         0
    ##   jute                  0      0     0     0         0        0         0
    ##   kidneybeans           0      0     0     0         0        0         0
    ##   lentil                0      0     0     0         0        0         0
    ##   maize                 0      0     0     0         0        0         0
    ##   mango                 0      0     0     0         0        0         0
    ##   mothbeans             0      0     0     0         0        0         0
    ##   mungbean              0      0     0     0         0        0         0
    ##   muskmelon             0      0     0     0         0        0         0
    ##   orange                0      0     0     0         0        0         0
    ##   papaya                0      0     0     0         0        0         0
    ##   pigeonpeas            0      0     0     0         0        0         0
    ##   pomegranate           0      0     0     0         0        0         0
    ##   rice                  0      0     0     0         0        0         0
    ##   watermelon            0      0     0     0         0        0         0
    ##   1                     0      0     0     0         0        0         0
    ##   10                   20      0     0     0         0        0         0
    ##   11                    0     20     0     0         0        0         0
    ##   12                    0      0    20     0         0        0         0
    ##   13                    0      0     0    20         0        0         0
    ##   14                    0      0     0     0        20        0         0
    ##   15                    0      0     0     0         0       20         0
    ##   16                    0      0     0     0         0        0        20
    ##   17                    0      0     0     0         0        0         0
    ##   18                    0      0     0     0         0        0         0
    ##   19                    0      0     0     0         0        0         0
    ##   2                     0      0     0     0         0        0         0
    ##   20                    0      0     0     0         0        0         0
    ##   21                    0      0     0     0         0        0         0
    ##   22                    0      0     0     0         0        0         0
    ##   3                     0      0     0     0         0        0         0
    ##   4                     0      0     0     0         0        0         0
    ##   5                     0      0     0     0         0        0         0
    ##   6                     0      0     0     0         0        0         0
    ##   7                     0      0     0     0         0        0         0
    ##   8                     0      0     0     0         0        0         0
    ##   9                     0      0     0     0         0        0         0
    ##              Reference
    ## Prediction    orange papaya pigeonpeas pomegranate rice watermelon  1 10 11 12
    ##   apple            0      0          0           0    0          0  0  0  0  0
    ##   banana           0      0          0           0    0          0  0  0  0  0
    ##   blackgram        0      0          0           0    0          0  0  0  0  0
    ##   chickpea         0      0          0           0    0          0  0  0  0  0
    ##   coconut          0      0          0           0    0          0  0  0  0  0
    ##   coffee           0      0          0           0    0          0  0  0  0  0
    ##   cotton           0      0          0           0    0          0  0  0  0  0
    ##   grapes           0      0          0           0    0          0  0  0  0  0
    ##   jute             0      0          0           0    0          0  0  0  0  0
    ##   kidneybeans      0      0          0           0    0          0  0  0  0  0
    ##   lentil           0      0          0           0    0          0  0  0  0  0
    ##   maize            0      0          0           0    0          0  0  0  0  0
    ##   mango            0      0          0           0    0          0  0  0  0  0
    ##   mothbeans        0      0          0           0    0          0  0  0  0  0
    ##   mungbean         0      0          0           0    0          0  0  0  0  0
    ##   muskmelon        0      0          0           0    0          0  0  0  0  0
    ##   orange           0      0          0           0    0          0  0  0  0  0
    ##   papaya           0      0          0           0    0          0  0  0  0  0
    ##   pigeonpeas       0      0          0           0    0          0  0  0  0  0
    ##   pomegranate      0      0          0           0    0          0  0  0  0  0
    ##   rice             0      0          0           0    0          0  0  0  0  0
    ##   watermelon       0      0          0           0    0          0  0  0  0  0
    ##   1                0      0          0           0    0          0  0  0  0  0
    ##   10               0      0          0           0    0          0  0  0  0  0
    ##   11               0      0          0           0    0          0  0  0  0  0
    ##   12               0      0          0           0    0          0  0  0  0  0
    ##   13               0      0          0           0    0          0  0  0  0  0
    ##   14               0      0          0           0    0          0  0  0  0  0
    ##   15               0      0          0           0    0          0  0  0  0  0
    ##   16               0      0          0           0    0          0  0  0  0  0
    ##   17              20      0          0           0    0          0  0  0  0  0
    ##   18               0     20          0           0    0          0  0  0  0  0
    ##   19               0      0         20           0    0          0  0  0  0  0
    ##   2                0      0          0           0    0          0  0  0  0  0
    ##   20               0      0          0          20    0          0  0  0  0  0
    ##   21               0      0          0           0   17          0  0  0  0  0
    ##   22               0      0          0           0    0         20  0  0  0  0
    ##   3                0      0          0           0    0          0  0  0  0  0
    ##   4                0      0          0           0    0          0  0  0  0  0
    ##   5                0      0          0           0    0          0  0  0  0  0
    ##   6                0      0          0           0    0          0  0  0  0  0
    ##   7                0      0          0           0    0          0  0  0  0  0
    ##   8                0      0          0           0    0          0  0  0  0  0
    ##   9                0      0          0           0    3          0  0  0  0  0
    ##              Reference
    ## Prediction    13 14 15 16 17 18 19  2 20 21 22  3  4  5  6  7  8  9
    ##   apple        0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
    ##   banana       0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
    ##   blackgram    0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
    ##   chickpea     0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
    ##   coconut      0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
    ##   coffee       0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
    ##   cotton       0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
    ##   grapes       0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
    ##   jute         0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
    ##   kidneybeans  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
    ##   lentil       0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
    ##   maize        0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
    ##   mango        0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
    ##   mothbeans    0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
    ##   mungbean     0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
    ##   muskmelon    0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
    ##   orange       0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
    ##   papaya       0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
    ##   pigeonpeas   0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
    ##   pomegranate  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
    ##   rice         0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
    ##   watermelon   0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
    ##   1            0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
    ##   10           0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
    ##   11           0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
    ##   12           0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
    ##   13           0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
    ##   14           0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
    ##   15           0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
    ##   16           0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
    ##   17           0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
    ##   18           0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
    ##   19           0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
    ##   2            0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
    ##   20           0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
    ##   21           0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
    ##   22           0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
    ##   3            0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
    ##   4            0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
    ##   5            0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
    ##   6            0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
    ##   7            0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
    ##   8            0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
    ##   9            0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
    ## 
    ## Overall Statistics
    ##                                      
    ##                Accuracy : 0          
    ##                  95% CI : (0, 0.0083)
    ##     No Information Rate : 0.0455     
    ##     P-Value [Acc > NIR] : 1          
    ##                                      
    ##                   Kappa : 0          
    ##                                      
    ##  Mcnemar's Test P-Value : NA         
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: apple Class: banana Class: blackgram
    ## Sensitivity               0.00000       0.00000          0.00000
    ## Specificity               1.00000       1.00000          1.00000
    ## Pos Pred Value                NaN           NaN              NaN
    ## Neg Pred Value            0.95455       0.95455          0.95455
    ## Prevalence                0.04545       0.04545          0.04545
    ## Detection Rate            0.00000       0.00000          0.00000
    ## Detection Prevalence      0.00000       0.00000          0.00000
    ## Balanced Accuracy         0.50000       0.50000          0.50000
    ##                      Class: chickpea Class: coconut Class: coffee Class: cotton
    ## Sensitivity                  0.00000        0.00000       0.00000       0.00000
    ## Specificity                  1.00000        1.00000       1.00000       1.00000
    ## Pos Pred Value                   NaN            NaN           NaN           NaN
    ## Neg Pred Value               0.95455        0.95455       0.95455       0.95455
    ## Prevalence                   0.04545        0.04545       0.04545       0.04545
    ## Detection Rate               0.00000        0.00000       0.00000       0.00000
    ## Detection Prevalence         0.00000        0.00000       0.00000       0.00000
    ## Balanced Accuracy            0.50000        0.50000       0.50000       0.50000
    ##                      Class: grapes Class: jute Class: kidneybeans Class: lentil
    ## Sensitivity                0.00000     0.00000            0.00000       0.00000
    ## Specificity                1.00000     1.00000            1.00000       1.00000
    ## Pos Pred Value                 NaN         NaN                NaN           NaN
    ## Neg Pred Value             0.95455     0.95455            0.95455       0.95455
    ## Prevalence                 0.04545     0.04545            0.04545       0.04545
    ## Detection Rate             0.00000     0.00000            0.00000       0.00000
    ## Detection Prevalence       0.00000     0.00000            0.00000       0.00000
    ## Balanced Accuracy          0.50000     0.50000            0.50000       0.50000
    ##                      Class: maize Class: mango Class: mothbeans Class: mungbean
    ## Sensitivity               0.00000      0.00000          0.00000         0.00000
    ## Specificity               1.00000      1.00000          1.00000         1.00000
    ## Pos Pred Value                NaN          NaN              NaN             NaN
    ## Neg Pred Value            0.95455      0.95455          0.95455         0.95455
    ## Prevalence                0.04545      0.04545          0.04545         0.04545
    ## Detection Rate            0.00000      0.00000          0.00000         0.00000
    ## Detection Prevalence      0.00000      0.00000          0.00000         0.00000
    ## Balanced Accuracy         0.50000      0.50000          0.50000         0.50000
    ##                      Class: muskmelon Class: orange Class: papaya
    ## Sensitivity                   0.00000       0.00000       0.00000
    ## Specificity                   1.00000       1.00000       1.00000
    ## Pos Pred Value                    NaN           NaN           NaN
    ## Neg Pred Value                0.95455       0.95455       0.95455
    ## Prevalence                    0.04545       0.04545       0.04545
    ## Detection Rate                0.00000       0.00000       0.00000
    ## Detection Prevalence          0.00000       0.00000       0.00000
    ## Balanced Accuracy             0.50000       0.50000       0.50000
    ##                      Class: pigeonpeas Class: pomegranate Class: rice
    ## Sensitivity                    0.00000            0.00000     0.00000
    ## Specificity                    1.00000            1.00000     1.00000
    ## Pos Pred Value                     NaN                NaN         NaN
    ## Neg Pred Value                 0.95455            0.95455     0.95455
    ## Prevalence                     0.04545            0.04545     0.04545
    ## Detection Rate                 0.00000            0.00000     0.00000
    ## Detection Prevalence           0.00000            0.00000     0.00000
    ## Balanced Accuracy              0.50000            0.50000     0.50000
    ##                      Class: watermelon Class: 1 Class: 10 Class: 11 Class: 12
    ## Sensitivity                    0.00000       NA        NA        NA        NA
    ## Specificity                    1.00000  0.95455   0.95455   0.95455   0.95455
    ## Pos Pred Value                     NaN       NA        NA        NA        NA
    ## Neg Pred Value                 0.95455       NA        NA        NA        NA
    ## Prevalence                     0.04545  0.00000   0.00000   0.00000   0.00000
    ## Detection Rate                 0.00000  0.00000   0.00000   0.00000   0.00000
    ## Detection Prevalence           0.00000  0.04545   0.04545   0.04545   0.04545
    ## Balanced Accuracy              0.50000       NA        NA        NA        NA
    ##                      Class: 13 Class: 14 Class: 15 Class: 16 Class: 17
    ## Sensitivity                 NA        NA        NA        NA        NA
    ## Specificity            0.95455   0.95455   0.95455   0.95455   0.95455
    ## Pos Pred Value              NA        NA        NA        NA        NA
    ## Neg Pred Value              NA        NA        NA        NA        NA
    ## Prevalence             0.00000   0.00000   0.00000   0.00000   0.00000
    ## Detection Rate         0.00000   0.00000   0.00000   0.00000   0.00000
    ## Detection Prevalence   0.04545   0.04545   0.04545   0.04545   0.04545
    ## Balanced Accuracy           NA        NA        NA        NA        NA
    ##                      Class: 18 Class: 19 Class: 2 Class: 20 Class: 21 Class: 22
    ## Sensitivity                 NA        NA       NA        NA        NA        NA
    ## Specificity            0.95455   0.95455  0.95455   0.95455   0.96136   0.95455
    ## Pos Pred Value              NA        NA       NA        NA        NA        NA
    ## Neg Pred Value              NA        NA       NA        NA        NA        NA
    ## Prevalence             0.00000   0.00000  0.00000   0.00000   0.00000   0.00000
    ## Detection Rate         0.00000   0.00000  0.00000   0.00000   0.00000   0.00000
    ## Detection Prevalence   0.04545   0.04545  0.04545   0.04545   0.03864   0.04545
    ## Balanced Accuracy           NA        NA       NA        NA        NA        NA
    ##                      Class: 3 Class: 4 Class: 5 Class: 6 Class: 7 Class: 8
    ## Sensitivity                NA       NA       NA       NA       NA       NA
    ## Specificity           0.95455  0.95455  0.95455  0.95455  0.95455  0.95455
    ## Pos Pred Value             NA       NA       NA       NA       NA       NA
    ## Neg Pred Value             NA       NA       NA       NA       NA       NA
    ## Prevalence            0.00000  0.00000  0.00000  0.00000  0.00000  0.00000
    ## Detection Rate        0.00000  0.00000  0.00000  0.00000  0.00000  0.00000
    ## Detection Prevalence  0.04545  0.04545  0.04545  0.04545  0.04545  0.04545
    ## Balanced Accuracy          NA       NA       NA       NA       NA       NA
    ##                      Class: 9
    ## Sensitivity                NA
    ## Specificity           0.94773
    ## Pos Pred Value             NA
    ## Neg Pred Value             NA
    ## Prevalence            0.00000
    ## Detection Rate        0.00000
    ## Detection Prevalence  0.05227
    ## Balanced Accuracy          NA

# **Consolidation**

``` r
# Saving the model
saveRDS(farming_caret_model_lda, "./data/saved_farming_caret_model_lda.rds")



# Process a Plumber API ----
#api <- plumber::plumb("API.R")

# Run the API on a specific port ----
# Specify a constant localhost port to use
#api$run(host = "127.0.0.1", port = 5022)


# We test the API using the following values:
# for the arguments:
# N, P, K, temperature, humidity, ph, rainfall
# 90, 42, 43, 20, 82, 6.5, and 202.93 respectively should output "rice"
```

# 
