install.packages("xgboost")
install.packages("dplyr")
install.packages("caret")
install.packages("pROC")
install.packages("e1071")      
install.packages("randomForest")
install.packages("rpart")      
install.packages("ggplot2")    



library(dplyr)
library(caret)
library(pROC)
library(xgboost)
library(e1071)      
library(randomForest)
library(rpart)      
library(ggplot2)   


set.seed(42)

# Crear dataset sintético
# Instalar y cargar librerías necesarias (Ejecutar solo si aún no están instaladas)
if (!require("dplyr")) install.packages("dplyr", dependencies = TRUE)
if (!require("caret")) install.packages("caret", dependencies = TRUE)
if (!require("pROC")) install.packages("pROC", dependencies = TRUE)
if (!require("xgboost")) install.packages("xgboost", dependencies = TRUE)
if (!require("e1071")) install.packages("e1071", dependencies = TRUE)
if (!require("randomForest")) install.packages("randomForest", dependencies = TRUE)
if (!require("rpart")) install.packages("rpart", dependencies = TRUE)
if (!require("ggplot2")) install.packages("ggplot2", dependencies = TRUE)

# Cargar librerías
library(dplyr)
library(caret)
library(pROC)
library(xgboost)
library(e1071)     
library(randomForest)
library(rpart)      
library(ggplot2)    


# Configuración inicial
# Librerías necesarias
library(xgboost)
library(e1071)
library(randomForest)
library(rpart)
library(pROC)
library(caret)
library(DMwR)  # Para SMOTE (balanceo de clases)

# Configuración inicial
set.seed(42)


n <- 10000
data <- data.frame(
  age = sample(18:70, n, replace = TRUE),
  income = sample(2000:10000, n, replace = TRUE),
  gender = sample(c("Male", "Female"), n, replace = TRUE),
  web_visits = sample(1:50, n, replace = TRUE),
  purchased = sample(c(0, 1), n, replace = TRUE)
)


data$gender <- as.factor(data$gender)
data$purchased <- as.factor(data$purchased)


data$gender <- as.numeric(data$gender) - 1


data$age <- as.numeric(data$age)
data$income <- as.numeric(data$income)
data$web_visits <- as.numeric(data$web_visits)


set.seed(123)
trainIndex <- createDataPartition(data$purchased, p = .8, list = FALSE, times = 1)
train_data <- data[trainIndex,]
test_data <- data[-trainIndex,]


train_matrix <- xgb.DMatrix(data = as.matrix(train_data[, -5]), label = as.numeric(train_data$purchased) - 1)
test_matrix <- xgb.DMatrix(data = as.matrix(test_data[, -5]), label = as.numeric(test_data$purchased) - 1)
xgb_model <- xgboost(data = train_matrix, max.depth = 6, eta = 0.3, nrounds = 100, objective = "binary:logistic", verbose = 0)


svm_model <- svm(purchased ~ ., data = train_data, kernel = "linear", probability = TRUE)


logistic_model <- glm(purchased ~ ., data = train_data, family = "binomial")


rf_model <- randomForest(purchased ~ ., data = train_data)


tree_model <- rpart(purchased ~ ., data = train_data, method = "class")


calc_roc <- function(pred, actual) {
  roc_obj <- roc(actual, pred)
  auc(roc_obj)
}


xgb_pred <- predict(xgb_model, test_matrix)
xgb_auc <- calc_roc(xgb_pred, as.numeric(test_data$purchased) - 1)


svm_pred <- predict(svm_model, test_data, probability = TRUE)
svm_pred <- attr(svm_pred, "probabilities")[,2]
svm_auc <- calc_roc(svm_pred, as.numeric(test_data$purchased) - 1)


logistic_pred <- predict(logistic_model, test_data, type = "response")
logistic_auc <- calc_roc(logistic_pred, as.numeric(test_data$purchased) - 1)


rf_pred <- predict(rf_model, test_data, type = "prob")[,2]
rf_auc <- calc_roc(rf_pred, as.numeric(test_data$purchased) - 1)


tree_pred <- predict(tree_model, test_data, type = "prob")[,2]
tree_auc <- calc_roc(tree_pred, as.numeric(test_data$purchased) - 1)


cat("AUC para cada modelo (en formato decimal y porcentaje):\n")
cat("XGBoost: ", xgb_auc, " / ", round(xgb_auc * 100, 2), "%\n")
cat("SVM: ", svm_auc, " / ", round(svm_auc * 100, 2), "%\n")
cat("Regresión Logística: ", logistic_auc, " / ", round(logistic_auc * 100, 2), "%\n")
cat("Random Forest: ", rf_auc, " / ", round(rf_auc * 100, 2), "%\n")
cat("Árbol de Decisión: ", tree_auc, " / ", round(tree_auc * 100, 2), "%\n")