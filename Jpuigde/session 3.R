require(FNN)
require(data.table)

data <- fread("data/mnist_mini.csv")[-1,-1,with=F]

ind     <- sample(1:nrow(data),round((nrow(data)*0.7)))
train.x <- data.matrix(data[ind,-1,with=F])
train.y <- data[ind,V2]
test.x  <- data.matrix(data[-ind,-1,with=F])
test.y  <- data[-ind,V2]

model <- get.knnx(train.x,test.x,20)

resp = matrix(train.y[model$nn.index],ncol = 20)
respuesta <- unlist(apply(resp,1,function(x){w=table(x);as.numeric(names(w[max(w)==w][1]))} ))
resultat = data.table(resp=respuesta,test.y)
resultat[,.N,by=.(resp,test.y)][resp==test.y,sum(N)]/length(test.y)


result2 <- knn(train.x,test.x,train.y,k=20)
resultat2 = data.table(resp=as.numeric(as.vector(result2)),test.y)
resultat2[,.N,by=.(resp,test.y)][resp==test.y,sum(N)]/length(test.y)


