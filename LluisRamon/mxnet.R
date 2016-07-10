mnist_mini <- read.csv("data/mnist_mini.csv", header=FALSE)

mnist_mini$V1 <- NULL
mnist_mini$y <- mnist_mini$V2
mnist_mini$V2 <- NULL

library("caret")

set.seed(123456)
ind <- createDataPartition(mnist_mini$y, p = 0.7, list = FALSE)

train <- mnist_mini[ind, ]
test <- mnist_mini[-ind, ]

library("mxnet")

# Adapted from
# http://dmlc.ml/rstats/2015/11/03/training-deep-net-with-R.html

train.x <- data.matrix(train[, -785])
train.y <- train[, 785]

train.x <- t(train.x/255)

test.x <- data.matrix(test[, -785])
test.x <- t(test.x/255)

data <- mx.symbol.Variable("data")
fc1 <- mx.symbol.FullyConnected(data, name="fc1", num_hidden=128)
act1 <- mx.symbol.Activation(fc1, name="relu1", act_type="relu")
fc2 <- mx.symbol.FullyConnected(act1, name="fc2", num_hidden=64)
act2 <- mx.symbol.Activation(fc2, name="relu2", act_type="relu")
fc3 <- mx.symbol.FullyConnected(act2, name="fc3", num_hidden=10)
softmax <- mx.symbol.SoftmaxOutput(fc3, name="sm")

mx.set.seed(0)
devices <- mx.cpu()
model <- mx.model.FeedForward.create(softmax, 
                                     X = train.x, y = train.y,
                                     ctx = devices, num.round = 30, 
                                     array.batch.size = 300,
                                     learning.rate = 0.07, 
                                     momentum = 0.9,  
                                     eval.metric = mx.metric.accuracy,
                                     initializer = mx.init.uniform(0.07),
                                     epoch.end.callback = mx.callback.log.train.metric(100))



preds <- predict(model, test.x)
preds_class <- max.col(t(preds)) - 1

table(preds_class, test[, 785])

confusionMatrix(preds_class, test[, 785])

# Accuracy : 0.9396

mx.model.save(model, "prefix", 10)
graph.viz("prefix-symbol.json")
