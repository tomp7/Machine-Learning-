
```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = TRUE)
knitr::include_graphics
```

# Practical Machine Learning Project

The goal of this project is to predict the manner in which they did the exercise. This is the "classe" variable in the training set. Will also use our prediction model to predict 20 different test cases.

*** 

## Background 
Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement - a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: http://groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset).

Read more: http://groupware.les.inf.puc-rio.br/har#ixzz3xsbS5bVX

***

## Preparing Data 

Below we begin by loading in the data 

```{r library}
library(knitr)
library(caret)
library(rpart)
library(rpart.plot)
library(rattle)
library(randomForest)
library(gbm)
library(AppliedPredictiveModeling)
library(corrplot)
set.seed(12345)
```

```{r data_load}
# URL for the data download
train_data <- "http://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
test_data  <- "http://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"

# training and test data sets downloadd 
training <- read.csv(url(train_data))
testing  <- read.csv(url(test_data))

# breaking up the training and test data sets by taking 70% of the data for training 
inTrain  <- createDataPartition(training$classe, p=0.7, list=FALSE)
train_set <- training[inTrain, ]
test_set  <- training[-inTrain, ]
dim(train_set)
dim(test_set)
```
Both created datasets have 160 variables. Those variables have plenty of NA, that can be removed with the cleaning procedures below. The Near Zero variance (NZV) variables are also removed and the ID variables as well.

The training and test data are loaded above and a training set has been created to assist in our creation of a model for the test data set. We see that both the training and test sets hav 160 variables, but differing number of values from the data partition. The basic set up has been completed and next we will look to clean the data by removing NA's and the variables with near zero variance as well. 

### Cleaning the Data 

Removal of near zero variance variables 
```{r Cleaning_NZV}
# Variables with nearly zero variance removed in both training and test sts
NZV <- nearZeroVar(train_set)
train_set <- train_set[, -NZV]
test_set  <- test_set[, -NZV]
dim(train_set)
dim(test_set)
```

Removal of variables that mostly contain NA's as they would have not have significant impact 
```{r Cleaning_NA}
# variables that mostly contain NA are removed 
NA_Var    <- sapply(train_set, function(x) mean(is.na(x))) > 0.95
train_set <- train_set[, NA_Var==FALSE]
test_set  <- test_set[, NA_Var==FALSE]
dim(train_set)
dim(test_set)
```

Removal of variables that only act as identification
``` {r Cleaning_ID}
# Columns 1 to 5 removed as they are only identification variables 
train_set <- train_set[, -(1:5)]
test_set  <- test_set[, -(1:5)]
dim(train_set)
dim(test_set)
```

Our two step clean up of the data sets has reduced our variable counts down to only 54 variables, but the number of observations remains the same. Now that our data has been cleaned we can proceed to analysis. 

*** 

## Correlation 

Prior to the start of the modeling process we take a look at the correlation plot below to see which variables to see which variables have the strongest correlations. Darker colors indicate stronger correlations 

```{r Correlate}
cor_mat <- cor(train_set[, -54])
corrplot(cor_mat, order = "FPC", method = "color", type = "lower", tl.cex = 0.8, tl.col = rgb(0, 0, 0))
```

In order to see the highest variable correlations we run the code below and create a cutoff of 0.75

```{r Corr_Check}
high_corr = findCorrelation(cor_mat, cutoff = 0.75)
names(train_set)[high_corr]
```

*** 

## Modeling 

For the predictive model building we will incorporate the use of three different methods applied on the training data set to help us determine the best one to apply. The preferred method will be chosen based on which results in the highest accuracy on the test data set. The three methods include: Random Forests, Decision Tree, and Generalized Boosted Modeling.

### Method 1: Random Forest 

Below is the code for the first attempted method, Random Forest. The method is selected, a model is fit on the training data and then applied to the test set. A confusion matrix shows the prediction and a plot of the matrix helps us visualize and see the accuracy. 

```{r model_1}
# Model 1
set.seed(12345)
control_RF <- trainControl(method="cv", number=3, verboseIter=FALSE)
modFit_RF <- train(classe ~ ., data=train_set, method="rf", trControl=control_RF)
modFit_RF$finalModel
```

```{r RF_1}
# prediction on test dataset with confusion matrix 
pred_RF <- predict(modFit_RF, newdata=test_set)
Matrix_RF <- confusionMatrix(pred_RF, test_set$classe)
Matrix_RF
```

```{r RF_plot}
# visualization of confusion matrix 
plot(Matrix_RF$table, col = Matrix_RF$byClass, 
     main = paste("Random Forest - Accuracy =", round(Matrix_RF$overall['Accuracy'], 4)))

```

As a result of our Random Forest method we see that the resulting Accuracy is at 0.9983. This is a strong level of accuracy, but we continue on with the next two methods to see if they are even better. 

***

### Method 2: Decision Tree

The second method used is the decision tree. 

``` {r model_2}
set.seed(12345)
modFit_DT <- rpart(classe ~ ., data=train_set, method="class")
fancyRpartPlot(modFit_DT)
```

```{r DT_1}
# prediction on test dataset with confusion matrix 
pred_DT <- predict(modFit_DT, newdata=test_set, type="class")
Matrix_DT <- confusionMatrix(pred_DT, test_set$classe)
Matrix_DT
```

```{r DT_plot}
# visualization of confusion matrix 
plot(Matrix_DT$table, col = Matrix_DT$byClass, 
     main = paste("Decision Tree - Accuracy =", round(Matrix_DT$overall['Accuracy'], 4)))
```

As a result of our Decision Tree method we see that the resulting Accuracy is at 0.7587 which is significantly lower than the accuracy of the Random Forest. Lower accuracy leads to more error and the opposite of what we want. We move on to the final model to test its accuracy and then will decide on the best. 

***

### Method 3: Gradient Boosting Machine (GBM)

The final method being used in our search for the most accurate predictor. 

``` {r model_3}
set.seed(12345)
control_GBM <- trainControl(method = "repeatedcv", number = 5, repeats = 1)
modFit_GBM  <- train(classe ~ ., data=train_set, method = "gbm",
                    trControl = control_GBM, verbose = FALSE)
modFit_GBM$finalModel
```

```{r GBM_1}
# prediction on test dataset with confusion matrix 
pred_GBM <- predict(modFit_GBM, newdata=test_set)
Matrix_GBM <- confusionMatrix(pred_GBM, test_set$classe)
Matrix_GBM
```

```{r GBM_plot}
# visualization of confusion matrix 
plot(Matrix_GBM$table, col = Matrix_GBM$byClass, 
     main = paste("GBM - Accuracy =", round(Matrix_GBM$overall['Accuracy'], 4)))
```

After the GBM was run it tells us that:

A gradient boosted model with multinomial loss function.  
150 iterations were performed.  
There were 53 predictors of which 53 had non-zero influence.  

We also see that the accuracy (0.9878) is much higher than that of the Decision Tree, but still not strong as the Random Forest. 

*** 

## Conlusion 

After having run our three different methods on our data, we can directly compare all three accuracies. 

1) Random Forest: 0.9983
2) Decision Tree: 0.7587
3) Gradient Boosting Machine (GBM): 0.9878

By looking at this data we see that the Decision Tree provided us with the least accuarate model and while close, the Random Forest is slightly more accurate than the GBM model. This means we can come to the conclusion that the most accurate and therefore best model results from the Random Forest. Now we apply the Random Forest to the validation data. 

``` {r final_validation}
Quiz_Results <- predict(modFit_RF, newdata=testing)
Quiz_Results
```

The resulting quiz answers are above

***
