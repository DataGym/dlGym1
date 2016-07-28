mnist_mini <- read.csv("~/Downloads/dlGym1-master/data/mnist_mini.csv", 
                       header=FALSE)

mnist_mini$V1 <- NULL
mnist_mini$y <- mnist_mini$V2
mnist_mini$V2 <- NULL

library("caret")

set.seed(123456)
ind <- createDataPartition(mnist_mini$y, p = 0.7, list = FALSE)

train <- mnist_mini[ind, ]
test <- mnist_mini[-ind, ]

library("mxnet")

train.x <- data.matrix(train[, -785])
train.y <- train[, 785]

train.x <- t(train.x/255)

test.x <- data.matrix(test[, -785])
test.x <- t(test.x/255)


data <- mx.symbol.Variable("data")
fc1 <- mx.symbol.FullyConnected(data, name="fc1", num_hidden=128)
act1 <- mx.symbol.Activation(fc1, name="relu1", act_type="relu")
bn1 <- mx.symbol.BatchNorm(act1)
fc2 <- mx.symbol.FullyConnected(bn1, name="fc2", num_hidden=64)
act2 <- mx.symbol.Activation(fc2, name="relu2", act_type="relu")
bn2 <- mx.symbol.BatchNorm(act2)
fc3 <- mx.symbol.FullyConnected(bn2, name="fc3", num_hidden=10)
softmax <- mx.symbol.SoftmaxOutput(fc3, name="sm")

mx.set.seed(0)
model <- mx.model.FeedForward.create(softmax, 
                                      X = train.x,
                                      y = train.y,
                                      num.round = 50, 
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

# Accuracy : 0.9493