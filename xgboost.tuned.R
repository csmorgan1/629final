require(xgboost)
require(tidyverse)
require(doParallel)
set.seed(1234)
training_data <- readRDS("sparsetrain.RDS")
training_label <- readRDS("sparseoutput.train.RDS")
training_label.1 <- training_label - 1
cl <- makePSOCKcluster(20)
registerDoParallel(cl)

final_model <-
  xgboost(
    data = training_data,
    label = training_label.1,
    booster = "gbtree",
    objective = "binary:logistic",
    nrounds = 50,
    max_depth = 2,
    eta = 0.2,
    gamma = 0,
    colsample_bytree = 1,
    min_child_weight = 1,
    subsample = 1,
    eval_metric = "error",
    eval_metric = "auc",
    nthread=20
  )

stopCluster(cl)

xgb.save(final_model,"xgb.final")