require(caret)
require(xgboost)
require(Matrix)
library(doParallel)
set.seed(1234)
cl <- makePSOCKcluster(20)
registerDoParallel(cl)

train_data <- readRDS("sparsetrain.RDS")
train_label <- readRDS("sparseoutput.train.RDS")
train_data1 <- as.matrix(train_data)
#train_label1 <- as.numeric(train_label)
grid_tune <- expand.grid(
  nrounds = seq(50,200,50),
  max_depth = seq(2,6,1),
  eta = seq(0.2,1,0.2),
  gamma = 0,
  min_child_weight = 1,
  subsample = 1
)
ctrl <-
  trainControl(
    method = "cv",
    number = 5,
    savePredictions = "final",
    classProbs = TRUE,
    allowParallel = TRUE
  )
model <-
  train(
    x = train_data1,
    y = train_label,
    trControl = ctrl,
    tuneGrid = grid_tune,
    method = "xgbTree"
  )
stopCluster(cl)
saveRDS(model,"xgbTree.caret.RDS")

