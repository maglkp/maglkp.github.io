Activity correctness prediction model based on HAR data on weight lifting
========================================================
  
  
## Summary
Human Activity Recognition (HAR) is an emerging area of research that deals with gathering and processing data on physical activity of people. One of the interesting applications is building a model for predicting the correctness of executions of weight lifting excercises by professionals and non professionals. Here a dataset available at http://groupware.les.inf.puc-rio.br/har is used to create such a model using random forest algorithm. The analysis shows that the model can be successful on test and cross validation data and can achieve high accuracy.

## Exploring the data set

As a first step let's load the data from two files "pml-training.csv" and "pml-testing.csv". 
The data in the first file contains 19622 rows, each row is labeled A-E (variable "classe") with A indicating correct execution of an exercise and other cases indicating 4 typical exercise errors. This data set will be split into training and test set with 70% used for training.

Models will be trained on the training set and evaluated on the test set, this procedure will be repeated after every udpate to the model to reevaluate it. As a part of this analysis a model using single classification tree, a linear regression model as well as a random forest was used. The classification tree and linear regression yielded inferior results to random forest, their analysis and comparison is not included here. The overall accuracy of random forests on both training set (resubstitution accuracy) and test set (out of bag error) was around 20% higher.

The second file contains only 20 unlabelled cases that will serve as a cross validation for the model. That is, only one final model will be used to predict those 20 cases and no further model tuning will be allowed after that.


```r
library(caret)
```

```
## Warning: package 'caret' was built under R version 3.1.1
```

```
## Loading required package: lattice
## Loading required package: ggplot2
```

```r
library(rattle)
```

```
## Warning: package 'rattle' was built under R version 3.1.1
```

```
## Rattle: A free graphical interface for data mining with R.
## Version 3.3.0 Copyright (c) 2006-2014 Togaware Pty Ltd.
## Type 'rattle()' to shake, rattle, and roll your data.
```

```r
set.seed(1010)
activityData <- read.csv("pml-training.csv")
activityCrossValidation <- read.csv("pml-testing.csv")
```

## Cleaning the data and preparing covariates

The first step of the analysis after forming a specific question that the model is to answer is the exploring the data and gaining insights on how to process it and how to form covariates. For brevity only the outlines of the steps taken are described here. We can see that the data consists of the whopping 160 columns. First we want to discard columns not containing measurements from one of the 4 sensors so we want to discard columns not containing "_arm", "_forearm", "_belt", "_dumbbell" or the outcome "classe". This leaves us with 152 colums (38 variables for each of the sensor).  

Ideally we would like to explore each of the 38 measurement types and their derivatives. However due to large number of the columns and limited time for this analysis we will use an automated approach. We will discard columns with high number of missing values as well as those with high number of empty values. We can see columns either have 0 empty or missing values or most of them (>95%) missing or empty so it is easy to make the distinction.

Then we will use the nearZeroVar caret function to identify and remove covariates with near zero variance. Such columns would not provide extra information to the model while still adding significant computation time. 


```r
reduceCovariates <- function(activityData) {
	# remove colums with mostly NAs, mostly empty or data other than measurements from sensors
	measuredVars <- names(activityData)[grep("_arm|_forearm|_belt|_dumbbell|classe", names(activityData))]
	activityData <- activityData[, measuredVars]

	naPercentages <- apply(activityData, 2, function(x) {mean(is.na(x))})
	highNaColumns <- names(naPercentages[naPercentages > .9])
	activityData <- activityData[,!(names(activityData) %in% highNaColumns)]

	emptyPercentages <- apply(activityData, 2, function(x) {mean(x=="")})
	emptyColumns <- names(naPercentages[naPercentages > .9])
	activityData <- activityData[,!(names(activityData) %in% emptyColumns)]

	# near zero variability
	nzv <- nearZeroVar(activityData, saveMetrics = T)
	activityData <- activityData[,!nzv$nzv]
	activityData
}

activityData <- reduceCovariates(activityData)
```

After all cleanup we are left with 53 covariates. The steps are included as the reduceCovariates which will be later used on test data. It should be already possible to train the model with this amount of covariates, it is expected to run for a few hours on a commodity machine. To speed up the computations we can further reduce the covariates using the Principal Component Analysis. The kind of data used i.e. physical measurements with a lot of variables representing secondary derivative values (mean, kurtosis, max, ...) is especially well suited for PCA as the columns will be correlated with each other to some degree.


```r
inTrain <- createDataPartition(activityData$classe, p=.7, list=FALSE)
training <- activityData[inTrain,]
testing <- activityData[-inTrain,]
preProcess(training[,1:52], method="pca")
```

```
## 
## Call:
## preProcess.default(x = training[, 1:52], method = "pca")
## 
## Created from 13737 samples and 52 variables
## Pre-processing: principal component signal extraction, scaled, centered 
## 
## PCA needed 25 components to capture 95 percent of the variance
```

As we can see PCA using the default threshold of 95% of variance gives very satisfactory results reducing the number of covariates to 25 so less than half of the original number. Since the model used is random forest no further refinements on the data is needed. That is, the final accuracy and robustness can surely be improved further by including additional preparation steps but is not needed to get really decent results in this particular case. 

## Training the model
Below is a code snippet that will perform the training and present the final model.


```r
modelRf <- train(classe ~ ., method="rf", preProcess="pca", data=training)
```

```
## Loading required package: randomForest
```

```
## Warning: package 'randomForest' was built under R version 3.1.1
```

```
## randomForest 4.6-10
## Type rfNews() to see new features/changes/bug fixes.
```

```
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
```

```r
modelRf$finalModel
```

```
## 
## Call:
##  randomForest(x = x, y = y, mtry = param$mtry) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 2
## 
##         OOB estimate of  error rate: 2.39%
## Confusion matrix:
##      A    B    C    D    E class.error
## A 3870   11   16    5    4    0.009217
## B   41 2580   30    2    5    0.029345
## C    0   39 2331   23    3    0.027129
## D    7    5   80 2150   10    0.045293
## E    1   15   18   13 2478    0.018614
```

In the listing above we see that the out of bag (OOB) estimate is equal to 2.39%

## Cross validation test and final results

Let's apply the model to the testing set and see how well it is doing.  

```r
confusionMatrix(testing$classe, predict(modelRf, testing))
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1662    2    7    2    1
##          B   20 1104   14    0    1
##          C    2   14 1001    8    1
##          D    2    1   42  916    3
##          E    0    2    9    9 1062
## 
## Overall Statistics
##                                        
##                Accuracy : 0.976        
##                  95% CI : (0.972, 0.98)
##     No Information Rate : 0.286        
##     P-Value [Acc > NIR] : < 2e-16      
##                                        
##                   Kappa : 0.97         
##  Mcnemar's Test P-Value : 9.79e-08     
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             0.986    0.983    0.933    0.980    0.994
## Specificity             0.997    0.993    0.995    0.990    0.996
## Pos Pred Value          0.993    0.969    0.976    0.950    0.982
## Neg Pred Value          0.994    0.996    0.985    0.996    0.999
## Prevalence              0.286    0.191    0.182    0.159    0.181
## Detection Rate          0.282    0.188    0.170    0.156    0.180
## Detection Prevalence    0.284    0.194    0.174    0.164    0.184
## Balanced Accuracy       0.991    0.988    0.964    0.985    0.995
```

The accuracy on the real test set is equal to 97.6% so is well estimated by OOB. The confidence interval is also very narrow for that value. The p-value for the model indicates it is highly significant.  

As a final step the 20 cross validation cases were predicted using this model and only 3 of them were not correctly predicted making for 85% real world scenario accuracy.  
