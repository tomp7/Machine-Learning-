---
title: "C8_W4_Project"
author: "Tom"
date: "April 16, 2020"
output:
  html_document:
    keep_md: true
---



# Practical Machine Learning Project

The goal of this project is to predict the manner in which they did the exercise. This is the "classe" variable in the training set. Will also use our prediction model to predict 20 different test cases.

*** 

## Background 
Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement - a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: http://groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset).

Read more: http://groupware.les.inf.puc-rio.br/har#ixzz3xsbS5bVX

***

## Preparing Data 

Below we begin by loading in the data 


```r
library(knitr)
```

```
## Warning: package 'knitr' was built under R version 3.5.3
```

```r
library(caret)
```

```
## Warning: package 'caret' was built under R version 3.5.3
```

```
## Loading required package: lattice
```

```
## Warning: package 'lattice' was built under R version 3.5.3
```

```
## Loading required package: ggplot2
```

```
## Warning: package 'ggplot2' was built under R version 3.5.3
```

```r
library(rpart)
library(rpart.plot)
```

```
## Warning: package 'rpart.plot' was built under R version 3.5.3
```

```r
library(rattle)
```

```
## Warning: package 'rattle' was built under R version 3.5.3
```

```
## Rattle: A free graphical interface for data science with R.
## Version 5.3.0 Copyright (c) 2006-2018 Togaware Pty Ltd.
## Type 'rattle()' to shake, rattle, and roll your data.
```

```r
library(randomForest)
```

```
## Warning: package 'randomForest' was built under R version 3.5.3
```

```
## randomForest 4.6-14
```

```
## Type rfNews() to see new features/changes/bug fixes.
```

```
## 
## Attaching package: 'randomForest'
```

```
## The following object is masked from 'package:rattle':
## 
##     importance
```

```
## The following object is masked from 'package:ggplot2':
## 
##     margin
```

```r
library(gbm)
```

```
## Warning: package 'gbm' was built under R version 3.5.3
```

```
## Loaded gbm 2.1.5
```

```r
library(AppliedPredictiveModeling)
```

```
## Warning: package 'AppliedPredictiveModeling' was built under R version 3.5.3
```

```r
library(corrplot)
```

```
## Warning: package 'corrplot' was built under R version 3.5.3
```

```
## corrplot 0.84 loaded
```

```r
set.seed(12345)
```


```r
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
```

```
## [1] 13737   160
```

```r
dim(test_set)
```

```
## [1] 5885  160
```
Both created datasets have 160 variables. Those variables have plenty of NA, that can be removed with the cleaning procedures below. The Near Zero variance (NZV) variables are also removed and the ID variables as well.

The training and test data are loaded above and a training set has been created to assist in our creation of a model for the test data set. We see that both the training and test sets hav 160 variables, but differing number of values from the data partition. The basic set up has been completed and next we will look to clean the data by removing NA's and the variables with near zero variance as well. 

### Cleaning the Data 

Removal of near zero variance variables 

```r
# Variables with nearly zero variance removed in both training and test sts
NZV <- nearZeroVar(train_set)
train_set <- train_set[, -NZV]
test_set  <- test_set[, -NZV]
dim(train_set)
```

```
## [1] 13737   106
```

```r
dim(test_set)
```

```
## [1] 5885  106
```

Removal of variables that mostly contain NA's as they would have not have significant impact 

```r
# variables that mostly contain NA are removed 
NA_Var    <- sapply(train_set, function(x) mean(is.na(x))) > 0.95
train_set <- train_set[, NA_Var==FALSE]
test_set  <- test_set[, NA_Var==FALSE]
dim(train_set)
```

```
## [1] 13737    59
```

```r
dim(test_set)
```

```
## [1] 5885   59
```

Removal of variables that only act as identification

```r
# Columns 1 to 5 removed as they are only identification variables 
train_set <- train_set[, -(1:5)]
test_set  <- test_set[, -(1:5)]
dim(train_set)
```

```
## [1] 13737    54
```

```r
dim(test_set)
```

```
## [1] 5885   54
```

Our two step clean up of the data sets has reduced our variable counts down to only 54 variables, but the number of observations remains the same. Now that our data has been cleaned we can proceed to analysis. 

*** 

## Correlation 

Prior to the start of the modeling process we take a look at the correlation plot below to see which variables to see which variables have the strongest correlations. Darker colors indicate stronger correlations 


```r
cor_mat <- cor(train_set[, -54])
corrplot(cor_mat, order = "FPC", method = "color", type = "lower", tl.cex = 0.8, tl.col = rgb(0, 0, 0))
```

![](C8_W4_Course_Project_files/figure-html/Correlate-1.png)<!-- -->

In order to see the highest variable correlations we run the code below and create a cutoff of 0.75


```r
high_corr = findCorrelation(cor_mat, cutoff = 0.75)
names(train_set)[high_corr]
```

```
##  [1] "accel_belt_z"      "roll_belt"         "accel_belt_y"     
##  [4] "total_accel_belt"  "accel_dumbbell_z"  "accel_belt_x"     
##  [7] "pitch_belt"        "magnet_dumbbell_x" "accel_dumbbell_y" 
## [10] "magnet_dumbbell_y" "accel_arm_x"       "accel_dumbbell_x" 
## [13] "accel_arm_z"       "magnet_arm_y"      "magnet_belt_z"    
## [16] "accel_forearm_y"   "gyros_forearm_y"   "gyros_dumbbell_x" 
## [19] "gyros_dumbbell_z"  "gyros_arm_x"
```

*** 

## Modeling 

For the predictive model building we will incorporate the use of three different methods applied on the training data set to help us determine the best one to apply. The preferred method will be chosen based on which results in the highest accuracy on the test data set. The three methods include: Random Forests, Decision Tree, and Generalized Boosted Modeling.

### Method 1: Random Forest 

Below is the code for the first attempted method, Random Forest. The method is selected, a model is fit on the training data and then applied to the test set. A confusion matrix shows the prediction and a plot of the matrix helps us visualize and see the accuracy. 


```r
# Model 1
set.seed(12345)
control_RF <- trainControl(method="cv", number=3, verboseIter=FALSE)
modFit_RF <- train(classe ~ ., data=train_set, method="rf", trControl=control_RF)
modFit_RF$finalModel
```

```
## 
## Call:
##  randomForest(x = x, y = y, mtry = param$mtry) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 27
## 
##         OOB estimate of  error rate: 0.2%
## Confusion matrix:
##      A    B    C    D    E  class.error
## A 3904    1    0    0    1 0.0005120328
## B    6 2649    2    1    0 0.0033860045
## C    0    4 2391    1    0 0.0020868114
## D    0    0    7 2245    0 0.0031083481
## E    0    0    0    5 2520 0.0019801980
```


```r
# prediction on test dataset with confusion matrix 
pred_RF <- predict(modFit_RF, newdata=test_set)
Matrix_RF <- confusionMatrix(pred_RF, test_set$classe)
Matrix_RF
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1674    5    0    0    0
##          B    0 1133    4    0    0
##          C    0    1 1022    7    0
##          D    0    0    0  957    4
##          E    0    0    0    0 1078
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9964          
##                  95% CI : (0.9946, 0.9978)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9955          
##                                           
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   0.9947   0.9961   0.9927   0.9963
## Specificity            0.9988   0.9992   0.9984   0.9992   1.0000
## Pos Pred Value         0.9970   0.9965   0.9922   0.9958   1.0000
## Neg Pred Value         1.0000   0.9987   0.9992   0.9986   0.9992
## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
## Detection Rate         0.2845   0.1925   0.1737   0.1626   0.1832
## Detection Prevalence   0.2853   0.1932   0.1750   0.1633   0.1832
## Balanced Accuracy      0.9994   0.9969   0.9972   0.9960   0.9982
```


```r
knitr::opts_chunk$set(fig.path = "README_figs/README-")
# visualization of confusion matrix 
plot(Matrix_RF$table, col = Matrix_RF$byClass, 
     main = paste("Random Forest - Accuracy =", round(Matrix_RF$overall['Accuracy'], 4)))
```

![](C8_W4_Course_Project_files/figure-html/RF_plot-1.png)<!-- -->

As a result of our Random Forest method we see that the resulting Accuracy is at 0.9983. This is a strong level of accuracy, but we continue on with the next two methods to see if they are even better. 

***

### Method 2: Decision Tree

The second method used is the decision tree. 


```r
set.seed(12345)
modFit_DT <- rpart(classe ~ ., data=train_set, method="class")
fancyRpartPlot(modFit_DT)
```

![](README_figs/README-model_2-1.png)<!-- -->


```r
# prediction on test dataset with confusion matrix 
pred_DT <- predict(modFit_DT, newdata=test_set, type="class")
Matrix_DT <- confusionMatrix(pred_DT, test_set$classe)
Matrix_DT
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1530  269   51   79   16
##          B   35  575   31   25   68
##          C   17   73  743   68   84
##          D   39  146  130  702  128
##          E   53   76   71   90  786
## 
## Overall Statistics
##                                          
##                Accuracy : 0.7368         
##                  95% CI : (0.7253, 0.748)
##     No Information Rate : 0.2845         
##     P-Value [Acc > NIR] : < 2.2e-16      
##                                          
##                   Kappa : 0.6656         
##                                          
##  Mcnemar's Test P-Value : < 2.2e-16      
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9140  0.50483   0.7242   0.7282   0.7264
## Specificity            0.9014  0.96650   0.9502   0.9100   0.9396
## Pos Pred Value         0.7866  0.78338   0.7543   0.6131   0.7305
## Neg Pred Value         0.9635  0.89051   0.9422   0.9447   0.9384
## Prevalence             0.2845  0.19354   0.1743   0.1638   0.1839
## Detection Rate         0.2600  0.09771   0.1263   0.1193   0.1336
## Detection Prevalence   0.3305  0.12472   0.1674   0.1946   0.1828
## Balanced Accuracy      0.9077  0.73566   0.8372   0.8191   0.8330
```


```r
# visualization of confusion matrix 
plot(Matrix_DT$table, col = Matrix_DT$byClass, 
     main = paste("Decision Tree - Accuracy =", round(Matrix_DT$overall['Accuracy'], 4)))
```

![](README_figs/README-DT_plot-1.png)<!-- -->

As a result of our Decision Tree method we see that the resulting Accuracy is at 0.7587 which is significantly lower than the accuracy of the Random Forest. Lower accuracy leads to more error and the opposite of what we want. We move on to the final model to test its accuracy and then will decide on the best. 

***

### Method 3: Gradient Boosting Machine (GBM)

The final method being used in our search for the most accurate predictor. 


```r
set.seed(12345)
control_GBM <- trainControl(method = "repeatedcv", number = 5, repeats = 1)
modFit_GBM  <- train(classe ~ ., data=train_set, method = "gbm",
                    trControl = control_GBM, verbose = FALSE)
modFit_GBM$finalModel
```

```
## A gradient boosted model with multinomial loss function.
## 150 iterations were performed.
## There were 53 predictors of which 53 had non-zero influence.
```


```r
# prediction on test dataset with confusion matrix 
pred_GBM <- predict(modFit_GBM, newdata=test_set)
Matrix_GBM <- confusionMatrix(pred_GBM, test_set$classe)
Matrix_GBM
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1670   11    0    2    0
##          B    4 1115   16    5    2
##          C    0   12 1006   16    1
##          D    0    1    4  941   10
##          E    0    0    0    0 1069
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9857          
##                  95% CI : (0.9824, 0.9886)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9819          
##                                           
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9976   0.9789   0.9805   0.9761   0.9880
## Specificity            0.9969   0.9943   0.9940   0.9970   1.0000
## Pos Pred Value         0.9923   0.9764   0.9720   0.9843   1.0000
## Neg Pred Value         0.9990   0.9949   0.9959   0.9953   0.9973
## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
## Detection Rate         0.2838   0.1895   0.1709   0.1599   0.1816
## Detection Prevalence   0.2860   0.1941   0.1759   0.1624   0.1816
## Balanced Accuracy      0.9973   0.9866   0.9873   0.9865   0.9940
```


```r
# visualization of confusion matrix 
plot(Matrix_GBM$table, col = Matrix_GBM$byClass, 
     main = paste("GBM - Accuracy =", round(Matrix_GBM$overall['Accuracy'], 4)))
```

![](README_figs/README-GBM_plot-1.png)<!-- -->

After the GBM was run it tells us that:

A gradient boosted model with multinomial loss function.  
150 iterations were performed.  
There were 53 predictors of which 53 had non-zero influence.  

We also see that the accuracy (0.9878) is much higher than that of the Decision Tree, but still not strong as the Random Forest. 

*** 

## Conclusion 

After having run our three different methods on our data, we can directly compare all three accuracies. 

1) Random Forest: 0.9983
2) Decision Tree: 0.7587
3) Gradient Boosting Machine (GBM): 0.9878

By looking at this data we see that the Decision Tree provided us with the least accuarate model and while close, the Random Forest is slightly more accurate than the GBM model. This means we can come to the conclusion that the most accurate and therefore best model results from the Random Forest. Now we apply the Random Forest to the validation data. 


```r
Quiz_Results <- predict(modFit_RF, newdata=testing)
Quiz_Results
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```

The resulting quiz answers are above

***
