dm = data.matrix(df)
dm_train = data.matrix(df.train)
dm_test = data.matrix(df.test)

train.array <- t(dm_train[,colFeat,with=FALSE])
dim(train.array) <- c(28, 28, 1, nrow(dm_train))
test.array <- t(dm_test)
dim(test.array) <- c(28, 28, 1, nrow(dm_test))

data <- mx.symbol.Variable("data")

conv1 <- mx.symbol.Convolution(data=data, kernel=c(2,2), num_filter=10)
tanh1 <- mx.symbol.Activation(data=conv1, act_type="tanh")
pool1 <- mx.symbol.Pooling(data=tanh1, pool_type="max",
                           kernel=c(2,2), stride=c(2,2))

flatten <- mx.symbol.Flatten(data=pool1)
fc1 <- mx.symbol.FullyConnected(data=flatten, num_hidden=40)
tanh3 <- mx.symbol.Activation(data=fc1, act_type="tanh")

fc2 <- mx.symbol.FullyConnected(data=tanh3, num_hidden=5)

lenet <- mx.symbol.SoftmaxOutput(data=fc2)

devices = lapply(1:4, function(i) {
  mx.cpu(i)
})

mx.set.seed(0)
system.time({
  model <- mx.model.FeedForward.create(
    lenet, 
    X=train.array, 
    y = dm_train[,colTarget],
    array.layout = c(28,28),
    eval.metric = mx.metric.accuracy,
    optimizer = "sgd",
    learning.rate = 0.01, 
    momentum = 0.9,
    initializer = mx.init.Xavier(),
    num.round = 40,
    ctx = devices)
})

pred_train_prob = predict(model, dm_train[,colFeat])
pred_train_guess = max.col(t(pred_train_prob))-1

pred_test_prob = predict(model, dm_test[,colFeat])
pred_test_guess = max.col(t(pred_test_prob))-1

confMat_train = caret::confusionMatrix(table(pred_train_guess, dm_train[,colTarget]))
confMat_test = caret::confusionMatrix(table(pred_test_guess, dm_test[,colTarget]))

acc_train = sum(diag(confMat_train$table)) / sum(confMat_train$table)
acc_test = sum(diag(confMat_test$table)) / sum(confMat_test$table)
