##############################Wavelets with RF######################
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

########PREPARING DATA##########
data <- read.csv2("02062020waveletsf.csv")
rownames(data) <- data$id
diagnosis1 <- data$diagnosis
data_scale <- scale(data[ ,-c(1,2)])
data_scale <- as.data.frame(data_scale)
data_scale <- mutate(data_scale, diagnosis = diagnosis1)
data_pca <- data[ ,-c(1,2)]
#######################EXPLORE DATA#############################
# pca_obj <- PCA(data_scale)
# fviz_pca_biplot(pca_obj, col.ind = diagnosis1)
# fviz_screeplot(pca_obj)

# pca_1 <- prcomp(data_pca, scale. = T)
# new_data <- pca_1$x
# new_data <- as.data.frame(new_data)
# new_data <- mutate(new_data, diagnosis = ifelse(diagnosis1 == "HEALTH",0,1))
# pca3d(pca_1, group = diagnosis1,  legend="topleft")

#####################FEATURE SELECTION########################
#Random forest selection function##
set.seed(527)
data_y <- as.factor(data_scale$diagnosis)
data_x <- as.data.frame(data_scale[ , -21])
control <- rfeControl(functions=rfFuncs, method="cv", number=10)
results <- rfe(data_x, data_y, sizes=c(1:20), rfeControl=control)
print(results)
predictors(results)
plot(results, type=c("g", "o"))
#Variables : cordh2 cora entrodh1 enedh1 cordh1

#####################CLASSIFICATION###########################

##################DATA PARTITION#############
set.seed(527)
data.knn_rf <- data_scale[ , c(1,2,3,4,5,6,7,8,9,10,12,13,14,18,19,21)]
indxTrain <- createDataPartition(y = data.knn_rf$diagnosis,p = 0.65,list = FALSE)
training_rf <- data.knn_rf[indxTrain,]
testing_rf <- data.knn_rf[-indxTrain,]
myFolds_rf <- createFolds(training_rf$diagnosis, k = 5)

##################KNN#########################
set.seed(527)
training_rf$diagnosis <- as.factor(training_rf$diagnosis)
trctrl_rf <- trainControl(method = "cv", number = 5, 
                          verboseIter = FALSE,
                          savePredictions = TRUE,
                          classProbs=TRUE,
                          index = myFolds_rf,
                          returnResamp = "final")
knn_fit_rf <- train(diagnosis ~., data = training_rf, method = "knn",
                    trControl=trctrl_rf)

test_pred_rf <- predict(knn_fit_rf, newdata = testing_rf)
metrics_knn <- confusionMatrix(as.factor(test_pred_rf),as.factor(testing_rf$diagnosis))

##################RANDOM FOREST#######################
set.seed(527)
modelo_rf <- train(diagnosis ~ ., data = training_rf,
                   method = "ranger",
                   trControl = trctrl_rf)
test_pred_rf <- predict(modelo_rf, newdata = testing_rf)
metrics_rf <- confusionMatrix(as.factor(test_pred_rf),as.factor(testing_rf$diagnosis))

################ADABOOST########################
set.seed(527)
model1 <- train(diagnosis ~ ., data = training_rf, method = "ada",
                verbose = T, trControl = trctrl_rf)
test_pred_ada <- predict(model1$finalModel, testing_rf)
metrics_ada <- confusionMatrix(as.factor(test_pred_ada),as.factor(testing_rf$diagnosis))
#################XGBTREE##########################
set.seed(527)
model_xgbt <- train(diagnosis ~ ., data = training_rf, method = "xgbTree",
                    verbose = T, trControl = trctrl_rf)
test_pred_xgb <- predict(model_xgbt, testing_rf)
metrics_xgb <- confusionMatrix(as.factor(test_pred_xgb),as.factor(testing_rf$diagnosis))
##################XGBTREE + RANGER#######################
# y <- training_rf$diagnosis
# x <- training_rf[ ,-17 ]
# models <- caretList(x, y, trControl = trctrl_rf,
#                     methodList=c("xgbTree","ranger"))
# ens <- caretEnsemble(models)
# test_pred_ens <- predict(ens, testing_rf)
# metrics_ens <- confusionMatrix(as.factor(test_pred_ens),as.factor(testing_rf$diagnosis))
# summary(ens)

#######################GLMNET###################
set.seed(527)
modelglmnet2 <- train(diagnosis~., training_rf, method = "glmnet", preProcess= c("center", "scale"), 
                     trControl = trctrl_rf)

test_pred_glm <- predict(modelglmnet2, testing_rf)
metrics_glmnet <- confusionMatrix(as.factor(test_pred_glm),as.factor(testing_rf$diagnosis))

##################ROTATION FOREST###############
set.seed(527)
model_rotf <- train(diagnosis ~ ., data = training_rf, method = "rotationForest",
                    verbose = T, preProcess= c( "center", "scale"), 
                    trControl = trctrl_rf)
test_pred_rotf <- predict(model_rotf, testing_rf)
metrics_rotf <- confusionMatrix(as.factor(test_pred_rotf),as.factor(testing_rf$diagnosis))

#####################Näive Bayes##############
set.seed(527)
model_nb <- train(diagnosis ~ ., data = training_rf, method = "naive_bayes",
                  verbose = T, preProcess= c( "center", "scale"), 
                  trControl = trctrl_rf)
test_pred_nb <- predict(model_nb, testing_rf)
metrics_nb <- confusionMatrix(as.factor(test_pred_nb),as.factor(testing_rf$diagnosis))

#################COMPARE MODELS#####################
set.seed(527)
model_list <- list(
  knn = knn_fit_rf,
  ADABOOST = model1,
  xgb = model_xgbt,
  RF = modelo_rf,
  glmnet = modelglmnet2,
  Rot_Forest = model_rotf,
  NB = model_nb)
#Collect resamples from the CV folds
resamps <- resamples(model_list)
summary(resamps)
bwplot(resamps, models = resamps$models, metric = resamps$metrics[1])


