##############################Wavelets with Backward stepwise regression######################
install.packages("ISLR")
install.packages("tree")
install.packages("ModelMetrics")
install.packages("ranger")
install.packages("xgboost")
install.packages("FactoMineR")
install.packages("factoextra")
install.packages("caret")
install.packages("pca3d")
install.packages("mlbench")
install.packages("leaps")
install.packages("GA")
install.packages("doParallel")
install.packages("randomForest")
install.packages("caretEnsemble")
####LIBRARY#######
library(rpart)
library(dplyr)
library(class)
library(ISLR)
library(tree)
library(ModelMetrics)
library(ranger)
library(vtreat)
library(magrittr)
library(xgboost)
library(FactoMineR)
library(factoextra)
library(ggplot2)
library(caret)
library(rpart.plot)
library(pROC)
library(pca3d)
library(mlbench)
library(leaps)
library(MASS)
library(GA)
library(doParallel)
library(randomForest)
library(caretEnsemble)
library(knitr)
########PREPARING DATA##########
data <- read.csv2("02062020waveletsf.csv")
rownames(data) <- data$id
diagnosis1 <- data$diagnosis
data_scale <- scale(data[ ,-c(1,2)])
data_scale <- as.data.frame(data_scale)
data_scale <- mutate(data_scale, diagnosis = diagnosis1)
data_pca <- data[ ,-c(1,2)]
# #######################EXPLORE DATA#############################
pca_obj <- PCA(data_pca, ncp = 20)
fviz_pca_biplot(pca_obj, col.ind = diagnosis1)
fviz_screeplot(pca_obj)
# 
# pca_1 <- prcomp(data_pca, scale. = T)
# new_data <- pca_1$x
# new_data <- as.data.frame(new_data)
# new_data <- mutate(new_data, diagnosis = ifelse(diagnosis1 == "HEALTH",0,1))
# pca3d(pca_1, group = diagnosis1,  legend="topleft")



#######################CLASSIFICATION#########################

##################DATA PARTITION#############
set.seed(527)
data.pca <- as.data.frame(pca_obj$ind$coord)
data.pca <- mutate(data.pca, diagnosis = diagnosis1)
indxTrain <- createDataPartition(y = data.pca$diagnosis,p = 0.65,list = FALSE)
training_pca <- data.pca[indxTrain,]
testing_pca <- data.pca[-indxTrain,]
myFolds_pca <- createFolds(training_pca$diagnosis, k = 5)

#############################KNN#####################
set.seed(527)
training_pca$diagnosis <- as.factor(training_pca$diagnosis)
trctrl_pca <- trainControl(method = "cv", number = 5, 
                          verboseIter = FALSE,
                          savePredictions = TRUE,
                          classProbs=TRUE,
                          returnResamp = "final",
                          index = myFolds_pca)

knn_fit_pca <- train(diagnosis ~., data = training_pca, method = "knn",
                    preProcess= c( "center", "scale"),
                    trControl=trctrl_pca)

test_pred_pca <- predict(knn_fit_pca, newdata = testing_pca)
metrics_knn <- confusionMatrix(as.factor(test_pred_pca),as.factor(testing_pca$diagnosis))

##################RANDOM FOREST#######################
set.seed(527)
modelo_rf <- train(diagnosis ~ ., data = training_pca,
                   method = "ranger", preProcess= c("center", "scale"),
                   trControl = trctrl_pca)
test_pred_rf <- predict(modelo_rf, newdata = testing_pca)
metrics_rf <- confusionMatrix(as.factor(test_pred_rf),as.factor(testing_pca$diagnosis))

#Accuracy 58,33%
################ADABOOST########################
set.seed(527)
model1 <- train(diagnosis ~ ., data = training_pca, method = "ada",
                verbose = T,preProcess= c( "center", "scale"), trControl = trctrl_pca)
test_pred_ada <- predict(model1, testing_pca)
metrics_ada <- confusionMatrix(as.factor(test_pred_ada),as.factor(testing_pca$diagnosis))

#Accuracy 58.33%

##################XGBOOSTREE#########################
set.seed(527)
model_xgbt <- train(diagnosis ~ ., data = training_pca, method = "xgbTree",
                    verbose = T, preProcess= c( "center", "scale"), 
                    trControl = trctrl_pca)
test_pred_xg <- predict(model_xgbt, testing_pca)
metrics_xgb <- confusionMatrix(as.factor(test_pred_xg),as.factor(testing_pca$diagnosis))
#Accuracy 50%
################## XGBOOST + RANGER #######################
# set.seed(527)
# y <- training_pca$diagnosis
# x <- training_pca[ ,-17]
# models <- caretList(x, y, trControl = trctrl_pca,
#                     methodList=c("xgbTree","ranger"))
# ens <- caretEnsemble(models)
# test_pred_ens <- predict(ens, testing_pca)
# metrics_ens <-confusionMatrix(as.factor(test_pred_ens),as.factor(testing_pca$diagnosis))
# summary(ens)

#######################GLMNET###################
set.seed(527)

modelglmnet2 <- train(diagnosis~., training_pca, method = "glmnet", preProcess= c("center", "scale"), 
                       trControl = trctrl_pca)

test_pred_glm <- predict(modelglmnet2, testing_pca)
metrics_glmnet <-confusionMatrix(as.factor(test_pred_glm),as.factor(testing_pca$diagnosis))

##################ROTATION FOREST###############
set.seed(527)
model_rotf <- train(diagnosis ~ ., data = training_pca, method = "rotationForest",
                    verbose = T, preProcess= c( "center", "scale"), 
                    trControl = trctrl_pca)
test_pred_rotf <- predict(model_rotf, testing_pca)
metrics_rotf <- confusionMatrix(as.factor(test_pred_rotf),as.factor(testing_pca$diagnosis))

#####################Näive Bayes##############
set.seed(527)
model_nb <- train(diagnosis ~ ., data = training_pca, method = "naive_bayes",
                  verbose = T, preProcess= c( "center", "scale"), 
                  trControl = trctrl_pca)
test_pred_nb <- predict(model_nb, testing_pca)
metrics_nb <- confusionMatrix(as.factor(test_pred_nb),as.factor(testing_pca$diagnosis))

##################COMPARE MODELS################
set.seed(527)
model_list <- list(
  KNN = knn_fit_pca,
  ADABOOST = model1,
  XGBTREE = model_xgbt,
  RANDOM_FOREST = modelo_rf,
  GLMNET = modelglmnet2,
  ROT_FOREST = model_rotf,
  NB = model_nb)
#Collect resamples from the CV folds
resamps <- resamples(model_list)
summary(resamps)
bwplot(resamps, models = resamps$models, metric = resamps$metrics[1])

# ##############tRAIN - tesT
# Sensitivity <- c(metrics_ada$byClass[1],metrics_knn$byClass[1],metrics_glmnet$byClass[1],
#                  metrics_rf$byClass[1],metrics_rotf$byClass[1],metrics_xgb$byClass[1],
#                  metrics_nb$byClass[1], metrics_ens$byClass[1])
# Sensitivity <- (round(Sensitivity,4))*100
# Specificity <- c(metrics_ada$byClass[2],metrics_knn$byClass[2],metrics_glmnet$byClass[2],
#                  metrics_rf$byClass[2],metrics_rotf$byClass[2],metrics_xgb$byClass[2],
#                  metrics_nb$byClass[2] , metrics_ens$byClass[2] )
# Specificity <- (round(Specificity,4))*100
# Accuracy <- c(metrics_ada$overall[1],metrics_knn$overall[1],metrics_glmnet$overall[1],
#               metrics_rf$overall[1],metrics_rotf$overall[1],metrics_xgb$overall[1],
#               metrics_nb$overall[1], metrics_ens$overall[1])
# Accuracy <- (round(Accuracy,4))*100
# Models_list <- c("ADABOOST", "KNN", "GLMNET","RANDOM FOREST","ROTATION FOREST","XGBTREE",
#                  "NÄIVE BAYES","ENSEMBLE MODEL")
# metrics_final <- data.frame(Models_list,Sensitivity,Specificity,Accuracy)
# kable(metrics_final, caption = "Performance Metrics for Train-Test models")
# capture.output(metrics_final, file="metrics.wavePCAf.xls")
# 
# ############Cross_Validation#########
# cv <- summary(resamps)
# capture.output(cv, file="cvwave_PCA.xls")
