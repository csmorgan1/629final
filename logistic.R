require(dplyr)
require(doParallel)
require(pROC)
require(kableExtra)
require(cvms)
require(caret)
set.seed(1234)
training <- readRDS("finaldata.train.RDS")
cl <- makePSOCKcluster(20)
registerDoParallel(cl)
trainging <- training %>% select(-1) %>% na.omit()
testing <- readRDS("finaldata.test.RDS")
testing <- testing %>% select(-1) %>% na.omit()
glm.fits <- glm(AnxietyICD~.,data = training,family=binomial)
probs <- predict(glm.fits,newdata=testing,type="response")
preds <- ifelse(probs>0.5,1,0)
Truth_f <- testing$AnxietyICD
Truth_n <- as.numeric(testing$AnxietyICD)-1
brier.glm <- mean((probs-Truth_n)^2)
roc.glm <- roc(Truth_f~as.vector(probs))
AUC <- auc(roc.glm)
auc.1 <- roc.glm$auc
out <- data.frame(Accuracy <- mean(Truth_n==preds),
                  Brier <- brier.glm,
                  AUC <- auc.1) %>%
  rename(Accuracy = Accuracy....mean.Truth_n....preds.,
         BrierScore =  Brier....brier.glm)
out.kable <- out %>% round(digits=3) %>% kable(caption = "GLM Accuracy, Brier Score, AUC") %>% kable_classic(full_width=F)
test.tab.glm <- table(predicted=preds,actual=Truth_n)
test.conmat.glm <- confusionMatrix(test.tab.glm,positive="1")
glm.measures <- test.conmat.glm$ByClass
kable.glm.measures <- glm.measures %>% as.data.frame() %>% round(digits=3) %>% kable(caption = "GLM Performance Measures") %>% kable_classic(full_width=F)
stopCluster(cl)
saveRDS(glm.fits,"glm.fits.RDS")
saveRDS(out,"glm.out.RDS")
saveRDS(out.kable,"glm.kable.out.RDS")
saveRDS(roc.glm,"glm.roc.RDS")
saveRDS(glm.measures,"glm.measures.RDS")
saveRDS(kable.glm.measures,"glm.kable.measures.RDS")