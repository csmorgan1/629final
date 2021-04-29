library(glmnet)
library(caret)
library(doParallel)
library(tidyverse)
set.seed(1234)
cl <- makePSOCKcluster(20)
registerDoParallel(cl)
dat <- readRDS("finaldata.train.RDS") %>% na.omit()

model <- train(
  AnxietyICD ~ .,
  data = dat,
  method = "glmnet",
  trControl = trainControl(
    "repeatedcv",
    number = 10,
    repeats = 3,
    allowParallel = TRUE,
    classProbs = TRUE
  ),
  tuneLength = 100,
)

x <- model.matrix(AnxietyICD~.,data=dat)[,-1]
y <- as.numeric(dat$AnxietyICD)

par <- model$bestTune
elnet.train <- glmnet::glmnet(x,
                              y,
                              family="binomial",
                              alpha = par[1],
                              lambda = par[2])
stopCluster(cl)
saveRDS(model,"tune.enet.caret.RDS")
saveRDS(elnet.train,"glmnet.model.RDS")