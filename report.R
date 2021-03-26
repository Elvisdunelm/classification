
# Load the packages
library("tidyverse")
library("skimr")
library("ggplot2")
library("dplyr")
library("mlr3")
library("mlr3learners")
library("mlr3proba")
library("data.table")
library("mlr3verse")
library("rsample")
library("recipes")
library("keras")

install.packages("gridExtra")
library(gridExtra)


# Read the data
patient <- readr::read_csv("https://raw.githubusercontent.com/Elvisdunelm/loan/main/heart_failure.csv")
skim(patient)

# prepare the data
patients <- patient %>%
  mutate(anaemia = as.factor(anaemia),diabetes = as.factor(diabetes),high_blood_pressure = as.factor(high_blood_pressure),
          sex = as.factor(sex),smoking = as.factor(smoking),fatal_mi = as.factor(fatal_mi),
         kidney_function = case_when(serum_creatinine >= 0.6  & serum_creatinine <= 1.35 ~ 'good',
                                     serum_creatinine < 0.6  | serum_creatinine > 1.35 ~ 'poor'))

# simple visualisations of the data
p1 <- ggplot(patients,aes(anaemia)) + geom_bar(aes(fill = fatal_mi))
p2 <- ggplot(patients,aes(diabetes)) + geom_bar(aes(fill = fatal_mi))
p3 <- ggplot(patients,aes(high_blood_pressure)) + geom_bar(aes(fill = fatal_mi))
p4 <- ggplot(patients,aes(sex)) + geom_bar(aes(fill = fatal_mi))
p5 <- ggplot(patients,aes(smoking)) + geom_bar(aes(fill = fatal_mi))
p6 <- ggplot(patients,aes(kidney_function)) + geom_bar(aes(fill = fatal_mi))
grid.arrange(p1, p2,p3, p4, p5, p6, nrow = 2)

patients <- patients %>%
  select(-kidney_function)

DataExplorer::plot_bar(patients, ncol = 3)
DataExplorer::plot_histogram(patients, ncol = 3)

# Fitting a logistic regresion model
fit.heart.lr <- glm(fatal_mi ~.,binomial,patients)
summary(fit.heart.lr)

pred.heart.lr <- predict(fit.heart.lr,type = "response")
ggplot(data.frame(x = pred.heart.lr), aes(x = x)) + geom_histogram()

conf.mat <- table(`true fatal MI` = patients$fatal_mi, `predict fatal MI` = pred.heart.lr > 0.5)
conf.mat
conf.mat/rowSums(conf.mat)*100

levels(patients$fatal_mi)
y_hat <- factor(ifelse(pred.heart.lr > 0.5, "1", "0"))
mean(I(y_hat == patients$fatal_mi))
table(truth = patients$fatal_mi, prediction = y_hat)

# Linear Discriminant Analysis
heart_lda <- MASS::lda(fatal_mi~.,patients)
heart_lda

heart_pred <-predict(heart_lda,patients)
heart_pred
mean(I(heart_pred$class == patients$fatal_mi))
table(truth = patients$fatal_mi, prediction = heart_pred$class)

# MLR3 log_reg
set.seed(212)
task_heart <- TaskClassif$new(id = "heart",
                               backend = patients,
                               target = "fatal_mi")

learner_lr <- lrn("classif.log_reg",predict_type = "prob")
learner_lr$train(task_heart)

pred_lr <- learner_lr$predict(task_heart)
pred_lr$score(list(msr("classif.ce"),
                   msr("classif.acc"),
                   msr("classif.auc"),
                   msr("classif.fpr"),
                   msr("classif.fnr"))
              )
pred_lr$confusion

# MLR3 LDA

set.seed(212)
task_heart <- TaskClassif$new(id = "heart",
                              backend = patients,
                              target = "fatal_mi")

learner_lda <- lrn("classif.lda",predict_type = "prob")
learner_lda$train(task_heart)

pred_lda <- learner_lda$predict(task_heart)
pred_lda$score(list(msr("classif.ce"),
                    msr("classif.acc"),
                    msr("classif.auc"),
                    msr("classif.fpr"),
                    msr("classif.fnr")))
pred_lda$confusion

# dive deeper in to MLR3 and run a best-practise analysis 
#cross validation

set.seed(212) # set seed for reproducibility
heart_task <- TaskClassif$new(id = "heart",
                               backend = patients, 
                               target = "fatal_mi",
                               positive = "1")


cv5 <- rsmp("cv", folds = 10)
cv5$instantiate(heart_task)

learner_baseline <- lrn("classif.featureless", predict_type = "prob")
learner_cart <- lrn("classif.rpart", predict_type = "prob")
learner_lr <- lrn("classif.log_reg",predict_type = "prob")

res_baseline <- resample(heart_task, learner_baseline, cv5, store_models = TRUE)
res_cart <- resample(heart_task, learner_cart, cv5, store_models = TRUE)

res <- benchmark(data.table(
  task       = list(heart_task),
  learner    = list(learner_baseline,
                    learner_cart,
                    learner_lr),
  resampling = list(cv5)
), store_models = TRUE)

res$aggregate()

res$aggregate(list(msr("classif.ce"),
                   msr("classif.acc"),
                   msr("classif.auc"),
                   msr("classif.fpr"),
                   msr("classif.fnr")))

# use bootstrap

set.seed(212) # set seed for reproducibility
heart_task <- TaskClassif$new(id = "heart",
                              backend = patients, 
                              target = "fatal_mi",
                              positive = "1")

rb <- rsmp("bootstrap", repeats = 10 ,ratio = 1)
rb$instantiate(heart_task)

learner_baseline <- lrn("classif.featureless", predict_type = "prob")
learner_cart <- lrn("classif.rpart", predict_type = "prob")

res_baseline <- resample(heart_task, learner_baseline, rb, store_models = TRUE)

res_cart <- resample(heart_task, learner_cart, rb, store_models = TRUE)

res <- benchmark(data.table(
  task       = list(heart_task),
  learner    = list(learner_baseline,
                    learner_cart),
  resampling = list(rb)
), store_models = TRUE)

res$aggregate()

res$aggregate(list(msr("classif.ce"),
                   msr("classif.acc"),
                   msr("classif.auc"),
                   msr("classif.fpr"),
                   msr("classif.fnr")))

# Classification trees using cv

set.seed(212) 
heart_task <- TaskClassif$new(id = "heart",
                              backend = patients, 
                              target = "fatal_mi",
                              positive = "1")


cv5 <- rsmp("cv", folds = 10)
cv5$instantiate(heart_task)

learner_baseline <- lrn("classif.featureless", predict_type = "prob")
learner_cart <- lrn("classif.rpart", predict_type = "prob")

res_baseline <- resample(heart_task, learner_baseline, cv5, store_models = TRUE)
res_cart <- resample(heart_task, learner_cart, cv5, store_models = TRUE)

res <- benchmark(data.table(
  task       = list(heart_task),
  learner    = list(learner_baseline,
                    learner_cart),
  resampling = list(cv5)
), store_models = TRUE)

trees <- res$resample_result(2)
tree1 <- trees$learners[[5]]
tree1_rpart <- tree1$model
plot(tree1_rpart,compress = TRUE,margin = 0.1)
text(tree1_rpart,use.n = TRUE,cex = 0.8)

plot(res$resample_result(2)$learners[[2]]$model, compress = TRUE, margin = 0.1)
text(res$resample_result(2)$learners[[2]]$model, use.n = TRUE, cex = 0.8)

learner_cart_cv <- lrn("classif.rpart", predict_type = "prob", xval = 10)
res_cart_cv <- resample(heart_task, learner_cart_cv, cv5, store_models = TRUE)
rpart::plotcp(res_cart_cv$learners[[3]]$model)

learner_cart_cp <- lrn("classif.rpart", predict_type = "prob", cp = 0.1,id = "cartcp")

res <- benchmark(data.table(
  task       = list(heart_task),
  learner    = list(learner_baseline,
                    learner_cart,
                    learner_cart_cp),
  resampling = list(cv5)
), store_models = TRUE)

res$aggregate(list(msr("classif.ce"),
                   msr("classif.acc"),
                   msr("classif.auc"),
                   msr("classif.fpr"),
                   msr("classif.fnr")))

learner_xgboost <- lrn("classif.xgboost", predict_type = "prob")
pl_xgb <- po("encode") %>>%
  po(learner_xgboost)

learner_ranger   <- lrn("classif.ranger", predict_type = "prob")
learner_log_reg  <- lrn("classif.log_reg", predict_type = "prob")


# super learning
set.seed(212)

heart_task <- TaskClassif$new(id = "heart",
                              backend = patients, 
                              target = "fatal_mi",
                              positive = "1")


cv5 <- rsmp("cv", folds = 5)
cv5$instantiate(heart_task)

learner_baseline <- lrn("classif.featureless", predict_type = "prob")
learner_cart <- lrn("classif.rpart", predict_type = "prob")
learner_log_reg  <- lrn("classif.log_reg", predict_type = "prob")
learner_cart_cp <- lrn("classif.rpart", predict_type = "prob", cp = 0.1,id = "cartcp")
learner_xgboost <- lrn("classif.xgboost", predict_type = "prob")
learner_ranger   <- lrn("classif.ranger", predict_type = "prob")

learnersp_log_reg <- lrn("classif.log_reg", predict_type = "prob", id = "super")

pl_factor <- po("encode")

spr_lrn <- gunion(list(
  gunion(list(
    po("learner_cv", learner_baseline),
    po("learner_cv", learner_cart),
    po("learner_cv", learner_cart_cp),
    po("learner_cv", learner_ranger),
    po("learner_cv", learner_log_reg)
  )),
  pl_factor %>>%
    po("learner_cv",learner_xgboost)
)) %>>%
  po("featureunion") %>>%
  po(learnersp_log_reg)

spr_lrn$plot()

res_spr <- resample(heart_task, spr_lrn, cv5, store_models = TRUE)
res_spr$aggregate(list(msr("classif.ce"),
                       msr("classif.acc"),
                       msr("classif.fpr"),
                       msr("classif.fnr"),
                       msr("classif.auc")))

# Deep learning preparation
set.seed(212) 
heart_split <- initial_split(patients)
heart_train <- training(heart_split)

heart_split2 <- initial_split(testing(heart_split), 0.5)
heart_validate <- training(heart_split2)
heart_test <- testing(heart_split2)

cake <- recipe(fatal_mi ~ ., data = patients) %>%
  step_meanimpute(all_numeric()) %>% # impute missings on numeric values with the mean
  step_center(all_numeric()) %>% # center by subtracting the mean from all numeric features
  step_scale(all_numeric()) %>% # scale by dividing by the standard deviation on all numeric features
  step_unknown(all_nominal(), -all_outcomes()) %>% # create a new factor level called "unknown" to account for NAs in factors, except for the outcome (response can't be NA)
  step_dummy(all_nominal(), one_hot = TRUE) %>% # turn all factors into a one-hot coding
  prep(training = heart_train) # learn all the parameters of preprocessing on the training data

heart_train_final <- bake(cake, new_data = heart_train) # apply preprocessing to training data
heart_validate_final <- bake(cake, new_data = heart_validate) # apply preprocessing to validation data
heart_test_final <- bake(cake, new_data = heart_test) # apply preprocessing to testing data

# Keras
heart_train_x <- heart_train_final %>%
  select(-starts_with("fatal_")) %>%
  as.matrix()
heart_train_y <- heart_train_final %>%
  select(fatal_mi_X0) %>%
  as.matrix()

heart_validate_x <- heart_validate_final %>%
  select(-starts_with("fatal_")) %>%
  as.matrix()
heart_validate_y <- heart_validate_final %>%
  select(fatal_mi_X0) %>%
  as.matrix()

heart_test_x <- heart_test_final %>%
  select(-starts_with("fatal_")) %>%
  as.matrix()
heart_test_y <- heart_test_final %>%
  select(fatal_mi_X0) %>%
  as.matrix()

# first
deep.net <- keras_model_sequential() %>%
  layer_dense(units = 32, activation = "relu",
              input_shape = c(ncol(heart_train_x))) %>%
  layer_dense(units = 32, activation = "relu") %>%
  layer_dense(units = 1, activation = "sigmoid")
# Have a look at it
deep.net

# This must then be "compiled".  See lectures on the optimiser.
deep.net %>% compile(
  loss = "binary_crossentropy",
  optimizer = optimizer_rmsprop(),
  metrics = c("accuracy")
)

# Finally, fit the neural network!  We provide the training data, and
# also a list of validation data.  We can use this to monitor for
# overfitting. See lectures regarding mini batches
deep.net %>% fit(
  heart_train_x, heart_train_y,
  epochs = 50, batch_size = 32,
  validation_data = list(heart_validate_x, heart_validate_y),
)

# To get the probability predictions on the test set:
pred_test_prob <- deep.net %>% predict_proba(heart_test_x)

# To get the raw classes (assuming 0.5 cutoff):
pred_test_res <- deep.net %>% predict_classes(heart_test_x)

# Confusion matrix/accuracy/AUC metrics
# (recall, in Lab03 we got accuracy ~0.80 and AUC ~0.84 from the super learner,
# and around accuracy ~0.76 and AUC ~0.74 from best other models)
table(pred_test_res, heart_test_y)
yardstick::accuracy_vec(as.factor(heart_test_y),
                        as.factor(pred_test_res))
yardstick::roc_auc_vec(factor(heart_test_y, levels = c("1","0")),
                       c(pred_test_prob))

set.seed(212)
deep.net <- keras_model_sequential() %>%
  layer_dense(units = 32, activation = "relu",
              input_shape = c(ncol(heart_train_x))) %>%
  layer_batch_normalization() %>%
  layer_dropout(rate = 0.4) %>%
  layer_dense(units = 32, activation = "relu") %>%
  layer_batch_normalization() %>%
  layer_dropout(rate = 0.4) %>%
  layer_dense(units = 32, activation = "relu") %>%
  layer_batch_normalization() %>%
  layer_dropout(rate = 0.4) %>%
  layer_dense(units = 32, activation = "relu") %>%
  layer_batch_normalization() %>%
  layer_dropout(rate = 0.4) %>%
  layer_dense(units = 1, activation = "sigmoid")

deep.net

deep.net %>% compile(
  loss = "binary_crossentropy",
  optimizer = optimizer_rmsprop(),
  metrics = c("accuracy")
)

deep.net %>% fit(
  heart_train_x, heart_train_y,
  epochs = 50, batch_size = 32,
  validation_data = list(heart_validate_x, heart_validate_y),
)

pred_test_prob <- deep.net %>% predict_proba(heart_test_x)

pred_test_res <- deep.net %>% predict_classes(heart_test_x)

table(pred_test_res, heart_test_y)
yardstick::accuracy_vec(as.factor(heart_test_y),
                        as.factor(pred_test_res))
yardstick::roc_auc_vec(factor(heart_test_y, levels = c("1","0")),
                       c(pred_test_prob))

yardstick::accuracy(heart_test,as.factor(heart_test_y),
                    as.factor(pred_test_res))


# MLR3 log_reg performance report
set.seed(212)
task_heart <- TaskClassif$new(id = "heart",
                              backend = patients,
                              target = "fatal_mi")

learner_lr <- lrn("classif.log_reg",predict_type = "prob")
learner_lr$train(task_heart)

pred_lr <- learner_lr$predict(task_heart)
pred_lr$score(list(msr("classif.ce"),
                   msr("classif.acc"),
                   msr("classif.auc"),
                   msr("classif.fpr"),
                   msr("classif.fnr"))
)
pred_lr$confusion

autoplot(pred_lr)
