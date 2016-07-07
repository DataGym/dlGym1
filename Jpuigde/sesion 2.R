require(mxnet)
require(xgboost)
require(data.table)
list.files("data")

data <- fread("data/mnist_mini.csv")[-1,-1,with=F]
ind <- sample(1:nrow(data),round((nrow(data)*0.7)))
train.x <- t(data.matrix(data[ind,-1,with=F]/255))
train.y <- data[ind,V2]
test.x <- t(data.matrix(data[-ind,-1,with=F]/255))
test.y <- data[-ind,V2]


model <- mx.mlp(train.x, train.y, hidden_node=10, out_node=10, out_activation="softmax",
                num.round=20, array.batch.size=15, learning.rate=0.07, momentum=0.9, 
                eval.metric=mx.metric.accuracy)

resp <- predict(model,test.x)

pred.label = max.col(t(resp))-1

resultat = data.table(resp=pred.label,test.y)
resultat[,.N,by=.(resp,test.y)][resp==test.y,sum(N)]/length(test.y)



