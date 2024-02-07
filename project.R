library(ggplot2);
library(caret);
library(rpart);
#library(rpart.plot);
#library(randomForest);
#library(RColorBrewer);
#library(rattle)

trData <- read.csv("./pml-training.csv", na.strings=c("#DIV/0!","NA",""))
trData <- trData[,colSums(is.na(trData)) == 0]
trData <- trData[,-c(1:7)]
trData$classe <- factor(trData$classe)

trPartition <- createDataPartition(y=trData$classe, p=0.75, list=FALSE)
training <- trData[trPartition, ] 
testing <- trData[-trPartition, ]

tstData <- read.csv("./pml-testing.csv", na.strings=c("#DIV/0!","NA",""))
tstData <- tstData[,-c(1:7)]
tstData <- tstData[,colSums(is.na(testing)) == 0]

ggplot(training, aes(x=classe)) + geom_bar()

set.seed(23674)

# Recursive Partition Decision Tree
ptm <- proc.time() # Start the clock!
trainControl_cv <- trainControl(method="cv", number=10)
modFit_tr_rpart <- train(classe ~ ., method="rpart", data=training, trControl=trainControl_cv)
pred_tr_rpart <- predict(modFit_tr_rpart, testing)
confusionMatrix(pred_tr_rpart, testing$classe)
proc.time() - ptm # Stop the clock

# Random Forest - PCA
ptm <- proc.time() # Start the clock!
trainControl_oob <- trainControl(method="oob", number=10, preProcOptions = list(thresh=0.8))
modFit_tr_rf_pca <- train(classe ~ ., method="rf", data=training, preProcess="pca", trControl=trainControl_oob)
pred_tr_rf_pca <- predict(modFit_tr_rf_pca, testing)
confusionMatrix(pred_tr_rf_pca, testing$classe)
pred_tstData_rf_pca <- predict(modFit_tr_rf_pca, tstData)
proc.time() - ptm # Stop the clock

# Random Forest
ptm <- proc.time() # Start the clock!
trainControl_oob <- trainControl(method="oob", number=10)
modFit_tr_rf <- train(classe ~ ., method="rf", data=training, trControl=trainControl_oob)
pred_tr_rf <- predict(modFit_tr_rf, testing)
confusionMatrix(pred_tr_rf, testing$classe)
pred_tstData_rf <- predict(modFit_tr_rf, tstData)
proc.time() - ptm # Stop the clock

# Generalize Boosted Regression - PCA
ptm <- proc.time() # Start the clock!
modFit_tr_gbm_pca <- train(classe ~ ., method="gbm", data=training,
                           preProcess="pca", trControl=trainControl(preProcOptions = list(thresh=0.8)),
                           trace=FALSE)
pred_tr_gbm_pca <- predict(modFit_tr_gbm_pca, testing)
confusionMatrix(pred_tr_gbm_pca, testing$classe)
proc.time() - ptm # Stop the clock

# Generalize Boosted Regression
ptm <- proc.time() # Start the clock!
modFit_tr_gbm <- train(classe ~ ., method="gbm", data=training, trace=FALSE)
pred_tr_gbm <- predict(modFit_tr_gbm, testing)
confusionMatrix(pred_tr_gbm, testing$classe)
proc.time() - ptm # Stop the clock

#modFit_rpart <- rpart(classe ~ ., data=training, method="class")
#pred_rpart <- predict(modFit_rpart, testing, type = "class")
#confusionMatrix(pred_rpart, testing$classe)
#rpart.plot(modFit_rpart)

pred_tstData <- predict(modFit_tr_rf, tstData)
