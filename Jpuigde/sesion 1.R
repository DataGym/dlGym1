require(data.table)
require(h2o)
# install.packages("h2o")

list.files("data/",full.names = T)
data <- fread(list.files("data/",full.names = T))[,-1,with=F]

ind <- sample(1:nrow(data),6000)

train=as.h2o(data[ind])
test=as.h2o(data[-ind])

h2o.init(nthreads = 7)
model <- h2o.deeplearning(names(train)[-1],names(train)[1],train,hidden=c(784,500,200,10),hidden_dropout_ratios = rep(0.8,4))

response <- predict(model,test)

resp_ <- round(as.data.frame(response),0)
resp_[resp_>9]<-9 
resp_[resp_<0]<-0 
resp_[is.na(resp_)]<-0

resultado <- data.table(resp_,as.data.frame(test[1]))[,.N,by=.(predict,V2)]

resultado[predict==V2][,sum(N)]/nrow(test)



mx.mlp










