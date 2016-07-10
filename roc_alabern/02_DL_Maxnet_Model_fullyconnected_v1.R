dm = data.matrix(df)
dm_train = data.matrix(df.train)
dm_test = data.matrix(df.test)

# data <- mx.symbol.Variable("data")
# fc1 <- mx.symbol.FullyConnected(data, name="fc1", num_hidden=16)
# softmax <- mx.symbol.SoftmaxOutput(fc1, name="sm")

# data <- mx.symbol.Variable("data")
# fc1 <- mx.symbol.FullyConnected(data, name="fc1", num_hidden=32)
# act1 <- mx.symbol.Activation(fc1, name="relu1", act_type="relu")
# fc2 <- mx.symbol.FullyConnected(act1, name="fc2", num_hidden=16)
# softmax <- mx.symbol.SoftmaxOutput(fc2, name="sm")

data <- mx.symbol.Variable("data")
fc1 <- mx.symbol.FullyConnected(data, name="fc1", num_hidden=64)
act1 <- mx.symbol.Activation(fc1, name="relu1", act_type="relu")
fc2 <- mx.symbol.FullyConnected(act1, name="fc2", num_hidden=32)
act2 <- mx.symbol.Activation(fc2, name="relu2", act_type="relu")
fc3 <- mx.symbol.FullyConnected(act2, name="fc3", num_hidden=10)
softmax <- mx.symbol.SoftmaxOutput(fc3, name="sm")

devices = lapply(1:4, function(i) {
  mx.cpu(i)
})

mx.set.seed(0)
system.time({
model <- mx.model.FeedForward.create(
  softmax, 
  X = t(dm_train[,colFeat]), 
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
