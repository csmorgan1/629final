library(caret)
library(randomForest)
library(doParallel)
library(tidyverse)
set.seed(1234)
dat <- readRDS('finaldata.train.RDS') %>% na.omit()
cl <- makePSOCKcluster(20)
registerDoParallel(cl)

trControl = trainControl(method = "repeatedcv",
                         number = 10,
                         repeats = 3,
                         savePredictions = "final",
                         classProbs = TRUE,
                         allowParallel = TRUE,
                         summaryFunction = twoClassSummary)

rf_default <-
  train(
    AnxietyICD ~ .,
    data = dat,
    method = "rf",
    metric = "ROC",
    trControl = trControl,
    tuneGrid = expand.grid(.mtry=c(1:40))
  )

stopCluster(cl)

saveRDS(rf_default,"rf_default")

