library(fields)
library(caret)
library(randomForest)
library(doParallel)
########################## Functions ############################################
pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}


setwd("E:/Courses/Practical Machine Learning/Project")

#reading empty values as na 

trainData <- read.csv("data/training/pml-training.csv",na.strings=c("NA","NaN", " ",""))
testData <- read.csv("Data/test/pml-testing.csv",na.strings=c("NA","NaN", " ",""))

numberOfRows <- nrow(trainData)

naThreshold <- numberOfRows/2

rowsToMentain <- names(trainData[colSums(is.na(trainData)) < naThreshold])
rowsToMentainTest <- rowsToMentain[rowsToMentain != "classe"]

trainData <- trainData[,rowsToMentain]

testData <- testData[,rowsToMentainTest]

set.seed(32323)

#removing unnecessary variables

colsR <- c("X","user_name","raw_timestamp_part_1","raw_timestamp_part_2", "cvtd_timestamp")
inR <- match(colsR, colnames(trainData))
trainData <- trainData[,-inR]

nzv <- nearZeroVar(trainData)

trainData <- trainData[,-nzv]


#shuffle training data rowwise
trainData <- trainData[sample(nrow(trainData)),]

inTrain <- createDataPartition(y=trainData$classe, p = 0.8, list = FALSE)

training <- trainData[inTrain,]
testing <- trainData[-inTrain,]

control <- trainControl(method="repeatedcv", number=10, repeats=3, allowParallel = TRUE)
cl <- makeCluster(detectCores())
registerDoParallel(cl)

modFit <- train(classe ~ ., method="rf", data = training, trControl = control )
stopCluster(cl)
print(modFit$finalModel)
confusionMatrix(testing$classe, predict(modFit,testing))
confusionMatrix(training$classe, predict(modFit,training))
testClasses <- predict(modFit,testData)
pml_write_files(testClasses)
