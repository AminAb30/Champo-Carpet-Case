# Champo Carpets R Code

---
title: "Champo Carpets_ Improving Business-to-Business Sales Using Machine Learning
  Algorithms"
author: "Amin Abbasi"
date: "2024-11-27"
output:
  html_document:
    df_print: paged
  word_document: default
  pdf_document: default
---
#Install libraries
```{#r}
install.packages(c("caret", "ROCR", "randomForest", "e1071", "rpart", "pROC", "kknn"))
install.packages("ROSE")

```

#Necessary Libraries
```{r}
library(caret)# For model training and evaluation
library(ROCR)
library(randomForest)
library(e1071)
library(rpart)
library(pROC)
library(kknn)
library(readxl)  # For reading Excel files
library(ggplot2)
library(ROSE)

```

##Importing the Dataset

```{r}


datasample <- read_excel("Champo_Carpets_Improving_Business_to_Business_Sales_Using_Machine.xlsx", sheet = "Data on Sample ONLY")


datacluster <- read_excel("Champo_Carpets_Improving_Business_to_Business_Sales_Using_Machine.xlsx", sheet = "Data for Clustering")


datasample$"Order Conversion" <- as.factor(datasample$"Order Conversion")



datasample_org <- read_excel("Champo_Carpets_Improving_Business_to_Business_Sales_Using_Machine.xlsx", sheet = "Data on Sample ONLY")

datacluster_org <- read_excel("Champo_Carpets_Improving_Business_to_Business_Sales_Using_Machine.xlsx", sheet = "Data for Clustering")

```
#Exploring the dataset

```{r}


# View the structure of the dataset of "Data on Sample ONLY" sheet
str(datasample)

# Get a summary of the dataset of "Data on Sample ONLY" sheet
summary(datasample)

```


Remove Space in the Variable Names
```{r}
# Rename variables: Replace spaces with underscores
names(datasample) <- gsub(" ", "_", names(datasample))
names(datacluster) <- gsub(" ", "_", names(datacluster))
```



##Data Cleaning

```{r}

datasample <- datasample[, !names(datasample) %in% c("ITEM_NAME", "CountryName", "ShapeName")]

```

#Handling Missing Values
Check Missing Values in Each Column for both dataset
```{r}
colSums(is.na(datasample))

```

Because only 39 rows of 5821 rows have missing value, I decided to remove them in the handeling missing value part
```{r}

datasample <- na.omit(datasample)
```
The SampleCluster dataset did not have any missing values, so there is no need to address that issue.
>>The missing values have been successfully removed.


#Handling Outliers

For clustering tasks such as K-means, which is sensitive to outliers, it is necessary to manage outliers for the data cluster dataset.

#Cap Outliers 
cap the values at the IQR thresholds for the Datacluster

```{r}
handle_outliers_iqr_cap <- function(datacluster) {
  numeric_columns <- sapply(datacluster, is.numeric)  # Identify numeric columns
  
  datacluster[numeric_columns] <- lapply(datacluster[numeric_columns], function(column) {
    if (is.numeric(column)) {
      Q1 <- quantile(column, 0.25, na.rm = TRUE)
      Q3 <- quantile(column, 0.75, na.rm = TRUE)
      IQR <- Q3 - Q1
      lower_bound <- Q1 - 1.5 * IQR
      upper_bound <- Q3 + 1.5 * IQR
      
      # Cap the values at lower and upper bounds
      column[column < lower_bound] <- lower_bound
      column[column > upper_bound] <- upper_bound
    }
    return(column)
  })
  
  return(datacluster)
}

# Apply the capping method
data_capped <- handle_outliers_iqr_cap(datacluster)

```

cap the values at the IQR thresholds for the DataSample

```{r}
handle_outliers_iqr_cap <- function(datasample) {
  numeric_columns <- sapply(datasample, is.numeric)  # Identify numeric columns
  
  datasample[numeric_columns] <- lapply(datasample[numeric_columns], function(column) {
    if (is.numeric(column)) {
      Q1 <- quantile(column, 0.25, na.rm = TRUE)
      Q3 <- quantile(column, 0.75, na.rm = TRUE)
      IQR <- Q3 - Q1
      lower_bound <- Q1 - 1.5 * IQR
      upper_bound <- Q3 + 1.5 * IQR
      
      # Cap the values at lower and upper bounds
      column[column < lower_bound] <- lower_bound
      column[column > upper_bound] <- upper_bound
    }
    return(column)
  })
  
  return(datasample)
}

# Apply the capping method
data_capped <- handle_outliers_iqr_cap(datacluster)

library(ggplot2)

numeric_columns <- names(data_capped)[sapply(data_capped, is.numeric)]

for (column_name in numeric_columns) {
  print(
    ggplot(data_capped, aes_string(y = paste0("`", column_name, "`"))) +
      geom_boxplot(outlier.colour = "blue", outlier.shape = 16, outlier.size = 2) +
      labs(title = paste("Boxplot of", column_name, "After Capping"), y = column_name) +
      theme_minimal()
  )
}


```
>>The outliers have been successfully removed.

#Check for duplicates
```{r}
# Check for duplicate rows for DataCluster
duplicate_CCode <- datacluster$"Row Labels"[duplicated(datacluster$"Row Labels")]


print(duplicate_CCode)
```
#Normalize
High variability in scale between features can introduce noise in certain algorithms. Therefore, I decided to normalize the data using the min/max method.I implemented normalization specifically for the Datacluster and DataSample.
Normalize All Numerical Variables:
```{r}
# Normalize numerical columns (min-max scaling)
normalize <- function(x) {
  return((x - min(x)) / (max(x) - min(x)))
}

# Apply normalization only to numeric columns

numeric_columns <- sapply(datacluster, is.numeric)
datacluster[numeric_columns] <- lapply(datacluster[numeric_columns], normalize)

str(datacluster)



# Apply normalization only to numeric columns

numeric_columns <- sapply(datasample, is.numeric)
datasample[numeric_columns] <- lapply(datasample[numeric_columns], normalize)

```




#Correlation Matrix
dentify highly correlated numerical features
```{r}

#  Filter out numeric columns
numeric_datasample <- datasample[sapply(datasample, is.numeric)]

#  Calculate the correlation matrix for numeric columns
correlation_matrix <- cor(numeric_datasample)

# View the correlation matrix
print("Correlation Matrix:")
print(correlation_matrix)

# Identify highly correlated features
threshold <- 0.6  # Define the correlation threshold
highly_correlated <- which(abs(correlation_matrix) > threshold & abs(correlation_matrix) < 1, arr.ind = TRUE)

#Create a data frame of highly correlated pairs
highly_correlated_pairs <- data.frame(
  Feature1 = rownames(correlation_matrix)[highly_correlated[, 1]],
  Feature2 = colnames(correlation_matrix)[highly_correlated[, 2]],
  Correlation = correlation_matrix[highly_correlated]
)

print("Highly Correlated Feature Pairs:")
print(highly_correlated_pairs)


```




##EDA part

```{r}
ggplot(datasample_org, aes(x = ShapeName, y = AreaFt)) +
  geom_bar(stat = "identity", fill = "blue") +
  labs(title = "AreaFt vs ShapeName", x = "Shape or carpets", y = "Area(Feet)") +
  theme_minimal()

```
The bar chart illustrates the relationship between ShapeName (categories: REC, ROUND, SQUARE) and AreaFt, showing that the rectangular shape (REC) has a significantly larger total area compared to ROUND and SQUARE, which have much smaller and similar areas. This indicates that rectangular shapes carpets in terms of total area, possibly due to higher usage. In contrast, ROUND and SQUARE shapes carpets occupy considerably less area, suggesting they are less prevalent or required in the dataset's context. This distribution highlights the importance of rectangular shapes in the given scenario.


Bar Chart: AreaFt (X) vs ITEM_NAME (Y)
```{r}
ggplot(datasample_org, aes(x = AreaFt, y = ITEM_NAME)) +
  geom_bar(stat = "identity", fill = "green") +
  labs(title = "AreaFt vs Carpet Names ", x = "Area(Feet)", y = "Carpet Type") +
  theme_minimal()

```
The bar chart depicts the total `AreaFt` (in square feet) for various carpet types, with `Carpet Type` on the Y-axis and `AreaFt` on the X-axis. Among the carpet types, `HAND TUFTED` dominates, accounting for the largest area, significantly outpacing all others. `DURRY` and `HANDWOVEN` also contribute notably but are far behind `HAND TUFTED`. In contrast, carpet types like `TABLE TUFTED`, `POWER LOOM JACQUARD`, and `GUN TUFTED` occupy minimal areas, indicating their lower prevalence or usage. This distribution suggests that `HAND TUFTED` carpets are a primary focus or in highest demand within the dataset, while other types play a much smaller role in terms of area covered.


Bar Chart: AreaFt (X) vs CountryName (Y)
```{r}
ggplot(datasample_org, aes(x = AreaFt, y = CountryName)) +
  geom_bar(stat = "identity", fill = "red") +
  labs(title = "AreaFt vs CountryName", x = "Area(Feet)", y = "Country Name") +
  theme_minimal()

```
The bar chart illustrates the distribution of total Area in feet across various countries, with Country Name on the Y-axis and AreaFt on the X-axis. India overwhelmingly dominates with the largest AreaFt, significantly surpassing all other countries. The USA ranks second but lags far behind India in terms of total area. Other countries such as Belgium, Israel, and Italy contribute minimally, while countries like the UK, UAE, and China show negligible contributions. This distribution highlights India as the primary contributor to the dataset in terms of area, with a stark disparity compared to other nations.


Bar Chart: AreaFt (X) vs CustomerCode (Y)
```{r}
ggplot(datasample_org, aes(x = AreaFt, y = CustomerCode)) +
  geom_bar(stat = "identity", fill = "purple") +
  labs(title = "Area(Ft) vs CustomerCode", x = "Area(Feet)", y = "Customer Code") +
  theme_minimal()

```
The bar chart depicts the distribution of total Area(Ft) across different CustomerCode values, with CustomerCode on the Y-axis and Area(Ft) on the X-axis. Among the customers, CTS overwhelmingly accounts for the largest Area(Ft), far surpassing all others. Other customer codes, such as H-2, M-4, and C-5, contribute moderately but are significantly smaller in comparison. The remaining customer codes show minimal or negligible contributions. This chart highlights CTS as the dominant customer in terms of total area, with a notable disparity compared to all other customers in the dataset.



"Order Conversion" distriburion
```{r}
# Count the values in the "Order Conversion" column
order_conversion_counts <- table(datasample_org$`Order Conversion`)

# Convert the table to a data frame for visualization
order_conversion_df <- as.data.frame(order_conversion_counts)
colnames(order_conversion_df) <- c("OrderConversion", "Count")

# Load ggplot2 for visualization
library(ggplot2)

# Create the pie chart
ggplot(order_conversion_df, aes(x = "", y = Count, fill = factor(OrderConversion))) +
  geom_bar(stat = "identity", width = 1) +
  coord_polar(theta = "y") +  # Convert to a pie chart
  scale_fill_manual(values = c("red", "green"), labels = c("0: Did Not Order", "1: Ordered")) +
  labs(
    title = "Order Conversion Distribution",
    fill = "Order Conversion"
  ) +
  theme_minimal() +
  theme(axis.text.x = element_blank())  # Remove x-axis labels for cleaner pie chart

```
The pie chart represents the distribution of `Order Conversion`, highlighting the proportion of customers who placed an order after receiving a sample (`1: Ordered`) versus those who did not (`0: Did Not Order`). The red segment, representing customers who did not order, occupies the majority of the chart, indicating that most customers did not convert to placing an order. The green segment, representing customers who placed an order, is comparatively smaller, showing that a smaller proportion of customers successfully converted. This visualization emphasizes the disparity between the two groups, suggesting a potential area for improvement in conversion strategies.



#EDA od the datacluster dataset
"Sum of QtyRequired", "Sum of TotalArea" and "Sum of Amount" in "Row Labels" that shows the behiviar of different group of costumers in the terms ofNumber of units ordered , Carpet area and carpet (in USD)
```{r}


# Bar chart for "Sum of QtyRequired"
ggplot(datacluster_org, aes(x = `Row Labels`, y = `Sum of QtyRequired`)) +
  geom_bar(stat = "identity", fill = "blue") +
  labs(
    title = "Sum of Number of units ordered vs Coustomer Groups",
    x = "Row Labels",
    y = "Sum of QtyRequired"
  ) +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))  # Rotate X-axis labels if needed

# Bar chart for "Sum of TotalArea"
ggplot(datacluster_org, aes(x = `Row Labels`, y = `Sum of TotalArea`)) +
  geom_bar(stat = "identity", fill = "green") +
  labs(
    title = "Sum of TotalArea vs Coustomer Groups",
    x = "Row Labels",
    y = "Sum of TotalArea"
  ) +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# Bar chart for "Sum of Amount"
ggplot(datacluster_org, aes(x = `Row Labels`, y = `Sum of Amount`)) +
  geom_bar(stat = "identity", fill = "red") +
  labs(
    title = "Sum of Amount vs Coustomer Groups",
    x = "Row Labels",
    y = "Sum of Amount"
  ) +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

```
##Model Imputation

#predivtive model on Sample Data

```{r}
datasample <- datasample[, !(names(datasample) %in% c("India", "OtherCountries", "Other","CustomerCode"))]

set.seed(123)  # For reproducibility
train_index <- sample(1:nrow(datasample), 0.8 * nrow(datasample))  # Randomly select 80% of the rows
train_data <- datasample[train_index, ]  # Training set (80% of the data)
test_data <- datasample[-train_index, ]  # Testing set (remaining 20% of the data)

```

##Train a Predictive Model:



#Train Logistic Regression
```{r}
logistic_model <- glm(Order_Conversion ~ ., data = train_data, family = binomial)

# Predictions
logistic_probs <- predict(logistic_model, test_data, type = "response")
logistic_classes <- ifelse(logistic_probs > 0.5, 1, 0)

```

#Train Decision Tree
```{r}
decision_tree_model <- rpart(Order_Conversion ~ ., data = train_data, method = "class")

# Predictions
decision_tree_probs <- predict(decision_tree_model, test_data, type = "prob")[, 2]
decision_tree_classes <- predict(decision_tree_model, test_data, type = "class")

```

#Train Random Forest
```{r}
random_forest_model <- randomForest(Order_Conversion ~ ., data = train_data, ntree = 100)

# Predictions
random_forest_probs <- predict(random_forest_model, test_data, type = "prob")[, 2]
random_forest_classes <- predict(random_forest_model, test_data)


```


# Train K-Nearest Neighbors (KNN)

```{r}
knn_model <- train(Order_Conversion ~ ., data = train_data, method = "kknn", tuneLength = 5)

# Predictions
knn_probs <- predict(knn_model, test_data, type = "prob")[, 2]
knn_classes <- predict(knn_model, test_data)


```

#Evaluation Metrics
```{r}
evaluate_model <- function(actual, predicted, probs) {
  cm <- confusionMatrix(as.factor(predicted), as.factor(actual), positive = "1")
  
  # Calculate metrics
  accuracy <- cm$overall["Accuracy"]
  precision <- cm$byClass["Pos Pred Value"]
  recall <- cm$byClass["Sensitivity"]
  f1 <- 2 * (precision * recall) / (precision + recall)
  false_alarm <- cm$byClass["Specificity"]
  
  return(c(Accuracy = accuracy, Precision = precision, Recall = recall, F1 = f1, False_Alarm = 1 - false_alarm))
}


```

#Confusion Matrices for Each Model
```{r}
# Logistic Regression
logistic_cm <- confusionMatrix(as.factor(logistic_classes), as.factor(test_data$Order_Conversion), positive = "1")
print("Confusion Matrix - Logistic Regression")
print(logistic_cm)

# Decision Tree
decision_tree_cm <- confusionMatrix(as.factor(decision_tree_classes), as.factor(test_data$Order_Conversion), positive = "1")
print("Confusion Matrix - Decision Tree")
print(decision_tree_cm)

# Random Forest
random_forest_cm <- confusionMatrix(as.factor(random_forest_classes), as.factor(test_data$Order_Conversion), positive = "1")
print("Confusion Matrix - Random Forest")
print(random_forest_cm)

# KNN
knn_cm <- confusionMatrix(as.factor(knn_classes), as.factor(test_data$Order_Conversion), positive = "1")
print("Confusion Matrix - KNN")
print(knn_cm)

```


#Evaluation Metrics for Each Model
```{r}
logistic_metrics <- evaluate_model(test_data$Order_Conversion, logistic_classes, logistic_probs)
decision_tree_metrics <- evaluate_model(test_data$Order_Conversion, decision_tree_classes, decision_tree_probs)
random_forest_metrics <- evaluate_model(test_data$Order_Conversion, random_forest_classes, random_forest_probs)
knn_metrics <- evaluate_model(test_data$Order_Conversion, knn_classes, knn_probs)

# Combine metrics into a single table
metrics_table <- rbind(Logistic = logistic_metrics,
                       DecisionTree = decision_tree_metrics,
                       RandomForest = random_forest_metrics,
                       KNN = knn_metrics)

print(metrics_table)


```

#Charts for Each Model
```{r}
plot_model_performance <- function(probs, actual, model_name) {
  pred <- prediction(probs, as.numeric(as.character(actual)))
  
  # ROC Curve
  perf_roc <- performance(pred, "tpr", "fpr")
  plot(perf_roc, col = "blue", lwd = 2, main = paste("ROC Curve -", model_name))
  abline(a = 0, b = 1, lty = 2, col = "red")
  
  # Gain Chart
  perf_gain <- performance(pred, "tpr", "rpp")
  plot(perf_gain, col = "green", lwd = 2, main = paste("Gain Chart -", model_name))
  abline(a = 0, b = 1, lty = 2, col = "red")
  
  # Response Chart
  perf_response <- performance(pred, "lift", "rpp")
  plot(perf_response, col = "orange", lwd = 2, main = paste("Response Chart -", model_name))
  
  # Lift Chart
  perf_lift <- performance(pred, "lift", "rpp")
  plot(perf_lift, col = "purple", lwd = 2, main = paste("Lift Chart -", model_name))
}

# Logistic Regression
plot_model_performance(logistic_probs, test_data$Order_Conversion, "Logistic Regression")

# Decision Tree
plot_model_performance(decision_tree_probs, test_data$Order_Conversion, "Decision Tree")

# Random Forest
plot_model_performance(random_forest_probs, test_data$Order_Conversion, "Random Forest")

# KNN
plot_model_performance(knn_probs, test_data$Order_Conversion, "KNN")


```


```{r}
# Logistic Regression
logistic_pred <- prediction(logistic_probs, as.numeric(as.character(test_data$Order_Conversion)))

# Decision Tree
decision_tree_pred <- prediction(decision_tree_probs, as.numeric(as.character(test_data$Order_Conversion)))

# Random Forest
random_forest_pred <- prediction(random_forest_probs, as.numeric(as.character(test_data$Order_Conversion)))

# KNN
knn_pred <- prediction(knn_probs, as.numeric(as.character(test_data$Order_Conversion)))


logistic_roc <- performance(logistic_pred, "tpr", "fpr")
decision_tree_roc <- performance(decision_tree_pred, "tpr", "fpr")
random_forest_roc <- performance(random_forest_pred, "tpr", "fpr")
knn_roc <- performance(knn_pred, "tpr", "fpr")


# Base plot for Logistic Regression
plot(logistic_roc, col = "blue", lwd = 2, main = "ROC Curves for All Models", xlab = "False Positive Rate", ylab = "True Positive Rate")

# Add Decision Tree ROC
plot(decision_tree_roc, col = "green", lwd = 2, add = TRUE)

# Add Random Forest ROC
plot(random_forest_roc, col = "red", lwd = 2, add = TRUE)

# Add KNN ROC
plot(knn_roc, col = "purple", lwd = 2, add = TRUE)

# Add a diagonal reference line
abline(a = 0, b = 1, lty = 2, col = "black")

# Add legend
legend("bottomright", legend = c("Logistic Regression", "Decision Tree", "Random Forest", "KNN"),
       col = c("blue", "green", "red", "purple"), lwd = 2)


```

#Print AUC for Each Model
```{r}
# Logistic Regression AUC
logistic_auc <- performance(logistic_pred, "auc")@y.values[[1]]

# Decision Tree AUC
decision_tree_auc <- performance(decision_tree_pred, "auc")@y.values[[1]]

# Random Forest AUC
random_forest_auc <- performance(random_forest_pred, "auc")@y.values[[1]]

# KNN AUC
knn_auc <- performance(knn_pred, "auc")@y.values[[1]]

# Print AUC values
print(paste("Logistic Regression AUC:", logistic_auc))
print(paste("Decision Tree AUC:", decision_tree_auc))
print(paste("Random Forest AUC:", random_forest_auc))
print(paste("KNN AUC:", knn_auc))


```
#Run All Models on Balanced and Imbalanced Data

#Difference Between Balanced and Imbalanced Results


```{#r}
library(ROSE)

balanced_data <- ovun.sample(Order_Conversion ~ ., data = train_data, method = "over", seed = 123)$data


# Train Logistic Regression on Balanced Data
logistic_model_balanced <- glm(Order_Conversion ~ ., data = balanced_data, family = binomial)

# Generate predictions
logistic_probs_balanced <- predict(logistic_model_balanced, test_data, type = "response")
logistic_classes_balanced <- ifelse(logistic_probs_balanced > 0.5, 1, 0)

decision_tree_model_balanced <- rpart(Order_Conversion ~ ., data = balanced_data, method = "class")

# Predictions
decision_tree_probs_balanced <- predict(decision_tree_model_balanced, test_data, type = "prob")[, 2]
decision_tree_classes_balanced <- predict(decision_tree_model_balanced, test_data, type = "class")

random_forest_model_balanced <- randomForest(Order_Conversion ~ ., data = balanced_data, ntree = 100)

# Predictions
random_forest_probs_balanced <- predict(random_forest_model_balanced, test_data, type = "prob")[, 2]
random_forest_classes_balanced <- predict(random_forest_model_balanced, test_data)

knn_model_balanced <- train(Order_Conversion ~ ., data = balanced_data, method = "kknn", tuneLength = 5)

# Predictions
knn_probs_balanced <- predict(knn_model_balanced, test_data, type = "prob")[, 2]
knn_classes_balanced <- predict(knn_model_balanced, test_data)


logistic_metrics_balanced <- evaluate_model(test_data$Order_Conversion, logistic_classes_balanced, logistic_probs_balanced)
decision_tree_metrics_balanced <- evaluate_model(test_data$Order_Conversion, decision_tree_classes_balanced, decision_tree_probs_balanced)
random_forest_metrics_balanced <- evaluate_model(test_data$Order_Conversion, random_forest_classes_balanced, random_forest_probs_balanced)
knn_metrics_balanced <- evaluate_model(test_data$Order_Conversion, knn_classes_balanced, knn_probs_balanced)

# Combine results into a table
balanced_metrics <- rbind(
  Logistic = logistic_metrics_balanced,
  DecisionTree = decision_tree_metrics_balanced,
  RandomForest = random_forest_metrics_balanced,
  KNN = knn_metrics_balanced
)

print("Metrics for Balanced Data:")
print(balanced_metrics)



# Combine metrics into a single table for comparison
final_comparison <- list(
  Imbalanced = imbalanced_metrics,
  Balanced = balanced_metrics
)

print("Final Comparison of Metrics:")
print(final_comparison)



```


```{r}

library(ROSE)

balanced_data <- ovun.sample(Order_Conversion ~ ., data = train_data, method = "over", seed = 123)$data


# Metrics for Imbalanced Data
imbalanced_metrics <- rbind(
  Logistic = evaluate_model(test_data$Order_Conversion, logistic_classes, logistic_probs),
  DecisionTree = evaluate_model(test_data$Order_Conversion, decision_tree_classes, decision_tree_probs),
  RandomForest = evaluate_model(test_data$Order_Conversion, random_forest_classes, random_forest_probs),
  KNN = evaluate_model(test_data$Order_Conversion, knn_classes, knn_probs)
)

# Metrics for Balanced Data (repeat model training on balanced_data and then evaluate)
balanced_metrics <- rbind(
  Logistic = evaluate_model(test_data$Order_Conversion, logistic_classes_balanced, logistic_probs_balanced),
  DecisionTree = evaluate_model(test_data$Order_Conversion, decision_tree_classes_balanced, decision_tree_probs_balanced),
  RandomForest = evaluate_model(test_data$Order_Conversion, random_forest_classes_balanced, random_forest_probs_balanced),
  KNN = evaluate_model(test_data$Order_Conversion, knn_classes_balanced, knn_probs_balanced)
)

# Combine and compare
print("Metrics for Imbalanced Data:")
print(imbalanced_metrics)
print("Metrics for Balanced Data:")
print(balanced_metrics)

```










---
title: "HW5-f-kmeans"
author: "Mahtab and Amin"
date: "2024-12-04"
output:
  pdf_document: default
  html_document: default
---

```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = TRUE)
```


I import the original data to R studio and use it. I did the following:
# Specify the file and the sheet name or index
# Read the specific sheet
# View the data


```{r}
#install.packages("readxl") 
library(readxl)
file_path <- "Champo_Carpets_Improving_Business_to_Business_Sales_Using_Machine.xlsx"
sheet_name <- "Data for Clustering"  # Replace with your sheet name or use sheet index, e.g., 1

data <- read_excel(file_path, sheet = sheet_name)


head(data)

str(data)
```
#missing values
I checked for missing values but there were no missing values to delet.

```{r}
 sum(is.na(data))
```
there is no missing value


#normalize
for k-means it is neccessary to normalize the data so that none of the variables dominant the distane similarity.
I used min-max for that.

```{r}
library(dplyr)
myscale <- function(x) {
(x - min(x)) / (max(x) - min(x))
}
Data <- data %>% mutate_if(is.numeric, myscale)

head(Data)

```
#outlier
at first, I was trying to remove the outliers from the data but for some coloumns that are related to dummy variables it will delet the only 1 in the whole colomn as the outlier. So I decided not to delet the outliers regarding the dataset nature and commented all codes for it.

```{r}
#out_lier <- function(x) {
#  if (is.numeric(x)) {
#    q1 <- quantile(x, 0.25, na.rm = TRUE)
#    q3 <- quantile(x, 0.75, na.rm = TRUE)
#    IQR_val <- q3 - q1
#    lower <- q1 - 1.5*IQR_val
#    upper <- q3 + 1.5*IQR_val
#    return(x < lower | x > upper) # Directly return the logical vector
#  } else {
#    return(rep(FALSE, length(x))) # No outliers for non-numeric data
#  }
#}
#
#
#data_no_out <- Data %>% 
#  mutate(across(where(is.numeric), ~ ifelse(out_lier(.), median(., na.rm = TRUE), .)))
#
#sum_outliers <- sum(sapply(data_no_out %>% select(where(is.numeric)), function(col) sum(out_lier(col))))
#
#sum_outliers

```

```{r}

```
#making it a data frame and also deleting the labels:
I converted the data set to a data frame and also deleted first colomn which are the labels for each instance and are not numerical and we can just delet this colomn.

```{r}

df <- as.data.frame(Data)
dfi <- df[-1]
any(is.na(dfi))    # Checks for NA
#any(is.nan(dfi))   # Checks for NaN
#any(is.infinite(dfi)) # Checks for Inf or -Inf
str(dfi)

```



#kmeans models
at first I just tried to implement the kmeans with 2 clusters but in the following I will find the best number of clusters.

```{r}
kModel1 <- kmeans(dfi , centers = 2, nstart = 100)

kModel1

```

```{r}
library(factoextra)
fviz_cluster(kModel1, dfi)

```


#now I try different cluster numbers for visualization.

```{r}
kModel2 <- kmeans(dfi , centers = 2, nstart = 100)
kModel3 <- kmeans(dfi , centers = 3, nstart = 100)
kModel4 <- kmeans(dfi , centers = 4, nstart = 100)
kModel5 <- kmeans(dfi , centers = 5, nstart = 100)


fp2 <- fviz_cluster(kModel2, dfi, geom = "point")
fp3 <- fviz_cluster(kModel3, dfi, geom = "point")
fp4 <- fviz_cluster(kModel4, dfi, geom = "point")
fp5 <- fviz_cluster(kModel5, dfi, geom = "point")

library(gridExtra)
grid.arrange(fp2, fp3, fp4, fp5, nrow = 2)

```
#in the following code we can use scree plot and elbow method to find the best K. with WSS
```{r}
set.seed(123)
fviz_nbclust(dfi , kmeans, method = "wss")
```
#in the following code we can use Silloutee to find the best K. 

```{r}
set.seed(123)
fviz_nbclust(dfi , kmeans, method = "silhouette")
```

```{r}
library(cluster)
avg_sil <- function(k){
  kModel1 <- kmeans(dfi, centers = k , nstart = 100)
  ss <- silhouette(kModel1$cluster , dist(dfi))
  mean(ss[,3])
}


avg_sil(2)
avg_sil(3)
avg_sil(4)
avg_sil(6)


```
#As we can see 2 clusters are the best regarding silhouette

```{r}
cluster_summary <- df %>%
  mutate(Cluster = kModel2$cluster) %>%
  group_by(Cluster) %>%
  summarise(across(everything(), mean))
print(cluster_summary)

```

```{r}
library(GGally)
ggpairs(dfi, aes(color = factor(kModel2$cluster)))

```
#the cluster characteristics summarized by the mean values of the variables for each cluster.

Business Insights:
Cluster 0 (High Volume and Area Customers):

Significantly larger values for total quantities (Sum of QtyRequired) and total area (Sum of TotalArea), indicating that these customers are high-volume buyers.
A dominant presence in categories like DURRY, HAND TUFTED, and KNOTTED, suggesting a preference for these product types.
Higher total monetary value (Sum of Amount), making this cluster the key revenue driver.
Cluster 1 (Low Volume and Area Customers):

Lower values across all metrics, indicating smaller-scale customers.
Focus on categories such as HANDLOOM and JACQUARD but with significantly lower quantities and areas compared to Cluster 0.
These customers may need targeted engagement strategies to increase their purchase volume.
#Recommendations:
Cluster 0:

Prioritize these customers for premium offerings and bulk discounts.
Tailor marketing efforts toward their preferred product types (DURRY, HAND TUFTED, KNOTTED).

Cluster 1:

Develop loyalty programs or promotional campaigns to encourage repeat purchases and higher order volumes.
Offer educational or promotional materials highlighting the benefits of other product categories to expand their buying preferences.




```{r}
```


#part g

#recommender system:

I used cosine similarity to find n nearest neighbors for a target costumer.
to create a customer-product purchase matrix, I deleted coloumns 2 to 4 and only keep row labels whcih is first column and also the numbers for each product type which are in columns 5 to end.
 Parameters:
   purchase_matrix: Customer-product matrix (rows: customers, cols: products, values: purchase counts or binary)
   target_customer: ID of the target customer (row name in the matrix)
   top_n_neighbors: Number of nearest neighbors to consider



```{r}

library(proxy)

fil_recommender <- function(purchase_matrix, target_customer, top_n_neighbors = 3) {
  
  rownames(purchase_matrix) <- purchase_matrix$Customer
  
#Cosine Similarity
  similarity_matrix <- as.matrix(simil(purchase_matrix[-1], method = "cosine"))
  
  #Exclude Self-Similarity
  diag(similarity_matrix) <- NA
  
  neighbors <- sort(similarity_matrix[target_customer, ], decreasing = TRUE, na.last = TRUE)
  nearest_neighbors <- head(names(neighbors), top_n_neighbors)
  
#Aggregate Product Purchases of Nearest Neighbors
  neighbors_purchases <- purchase_matrix[nearest_neighbors, -1, drop = FALSE]  # Exclude Customer column
  aggregated_purchases <- colSums(neighbors_purchases, na.rm = TRUE)
  
# Filter Out Products Already Purchased by the target Customer
  target_customer_purchases <- purchase_matrix[target_customer, -1] 
  recommendations <- aggregated_purchases[target_customer_purchases == 0]
  
  recommendations <- sort(recommendations, decreasing = TRUE)
  
  return(recommendations)
}


#to create a customer-product purchase matrix. I deleted coloumns 2 to 4 and only keep row labels whcih is first column and also the numbers for each product type which are in columns 5 to end.

data_rec <- df[, -c(2:4)]
purchase_matrix <- data.frame(data_rec)
colnames(purchase_matrix)[1] <- "Customer"  # Rename the first column to "Customer"

recommendations <- fil_recommender(purchase_matrix, target_customer = "A-11", top_n_neighbors = 2)


print(recommendations)

```

```{r}


```

```{r}

```





