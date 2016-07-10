library(data.table)
library(mxnet)

strFile = "data/mnist_mini.csv"
strModels = "roc_alabern/models/"
  
# readLines(strFile, n = 2)

df = fread("data/mnist_mini.csv", header=FALSE)

df = df[,-1,with=FALSE]

colTarget = 1
colFeat = 2:785

for (i in colFeat) {
  df[[i]] = df[[i]]/255 - 0.2
}

s = runif(nrow(df))
ind = s<=0.65
df.train = df[ind, ]
df.test = df[!ind, ]


