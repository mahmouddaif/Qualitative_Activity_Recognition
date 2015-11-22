# Qualitative Activity Recognition
Mahmoud Daif  
November 22, 2015  

# Overview
The approach we propose for the Weight Lifting Exercises dataset is to investigate "how (well)" an activity was performed.
Six young health participants were asked to perform one set of 10 repetitions of the Unilateral Dumbbell Biceps Curl in five different fashions: exactly according to the specification (Class A), throwing the elbows to the front (Class B), lifting the dumbbell only halfway (Class C), lowering the dumbbell only halfway (Class D) and throwing the hips to the front (Class E)
Our goal is to be able to predict the class given measurements.

# Reading data



```r
trainData <- read.csv("data/training/pml-training.csv",na.strings=c("NA","NaN", " ",""))
testData <- read.csv("Data/test/pml-testing.csv",na.strings=c("NA","NaN", " ",""))
```

#Removing Columns With Mjority Missing Values
1- Setting the max number of missing values threshold to be more than half of number of rows

```r
numberOfRows <- nrow(trainData)

naThreshold <- numberOfRows/2
```

2- Removing rows with majority of missing values

```r
rowsToMentain <- names(trainData[colSums(is.na(trainData)) < naThreshold])
rowsToMentainTest <- rowsToMentain[rowsToMentain != "classe"]

trainData <- trainData[,rowsToMentain]

testData <- testData[,rowsToMentainTest]
```

3- Setting seed for reproducible results

```r
set.seed(32323)
```

4- Removing unnecessary variables like id, name and timestamp related variables


```r
colsR <- c("X","user_name","raw_timestamp_part_1","raw_timestamp_part_2", "cvtd_timestamp")
inR <- match(colsR, colnames(trainData))
trainData <- trainData[,-inR]
```

5- removing variables with near zero variance

```r
nzv <- nearZeroVar(trainData)

trainData <- trainData[,-nzv]
```

6- Shuffling training data

```r
trainData <- trainData[sample(nrow(trainData)),]
```

7- Splitting training data to 80 % and 20% to calculate confusion matrix and be able to test our model

```r
inTrain <- createDataPartition(y=trainData$classe, p = 0.8, list = FALSE)

training <- trainData[inTrain,]
testing <- trainData[-inTrain,]
```

8- training model using random forest and repeated random sub sampling validation


```r
control <- trainControl(method="repeatedcv", number=10, repeats=3, allowParallel = TRUE)
cl <- makeCluster(detectCores())
registerDoParallel(cl)

modFit <- train(classe ~ ., method="rf", data = training, trControl = control )
stopCluster(cl)
```

9- Summary of final model

```r
print(modFit$finalModel)
```

```
## 
## Call:
##  randomForest(x = x, y = y, mtry = param$mtry) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 27
## 
##         OOB estimate of  error rate: 0.18%
## Confusion matrix:
##      A    B    C    D    E  class.error
## A 4462    1    0    0    1 0.0004480287
## B    5 3030    2    1    0 0.0026333114
## C    0    3 2734    1    0 0.0014609204
## D    0    0    9 2564    0 0.0034978624
## E    0    0    0    5 2881 0.0017325017
```

#In-sample and Out-Sample Errors
1- In-Sample and Out-Sample error
The we will use accuracy from the confusion matrix to express in sample and out of sample errors.
The in sample accuracy is expected to be higher than the out sample accuracy.
This is because the model is always a little fine tuned (overfitted) to the training data

In sample

```r
confusionMatrix(training$classe, predict(modFit,training))
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 4464    0    0    0    0
##          B    0 3038    0    0    0
##          C    0    0 2738    0    0
##          D    0    0    0 2573    0
##          E    0    0    0    0 2886
## 
## Overall Statistics
##                                      
##                Accuracy : 1          
##                  95% CI : (0.9998, 1)
##     No Information Rate : 0.2843     
##     P-Value [Acc > NIR] : < 2.2e-16  
##                                      
##                   Kappa : 1          
##  Mcnemar's Test P-Value : NA         
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   1.0000   1.0000   1.0000   1.0000
## Specificity            1.0000   1.0000   1.0000   1.0000   1.0000
## Pos Pred Value         1.0000   1.0000   1.0000   1.0000   1.0000
## Neg Pred Value         1.0000   1.0000   1.0000   1.0000   1.0000
## Prevalence             0.2843   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2843   0.1935   0.1744   0.1639   0.1838
## Detection Prevalence   0.2843   0.1935   0.1744   0.1639   0.1838
## Balanced Accuracy      1.0000   1.0000   1.0000   1.0000   1.0000
```

Outsample

```r
confusionMatrix(testing$classe, predict(modFit,testing))
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1116    0    0    0    0
##          B    0  757    2    0    0
##          C    0    4  680    0    0
##          D    0    0    1  642    0
##          E    0    2    0    1  718
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9975          
##                  95% CI : (0.9953, 0.9988)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9968          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   0.9921   0.9956   0.9984   1.0000
## Specificity            1.0000   0.9994   0.9988   0.9997   0.9991
## Pos Pred Value         1.0000   0.9974   0.9942   0.9984   0.9958
## Neg Pred Value         1.0000   0.9981   0.9991   0.9997   1.0000
## Prevalence             0.2845   0.1945   0.1741   0.1639   0.1830
## Detection Rate         0.2845   0.1930   0.1733   0.1637   0.1830
## Detection Prevalence   0.2845   0.1935   0.1744   0.1639   0.1838
## Balanced Accuracy      1.0000   0.9958   0.9972   0.9991   0.9995
```

#Test Data Prediction


```r
testClasses <- predict(modFit,testData)
testClasses
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```
