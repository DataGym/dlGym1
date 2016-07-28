mnist_mini <- read.csv("data/mnist_mini.csv", header=FALSE)
mnist_mini$V1 <- NULL
mnist_mini$y <- factor(mnist_mini$V2)
mnist_mini$V2 <- NULL

head(mnist_mini)
dim(mnist_mini)

library("caret")

set.seed(123456)
ind <- createDataPartition(mnist_mini$y, p = 0.7, list = FALSE)
train <- mnist_mini[ind, ]
test <- mnist_mini[-ind, ]

library("ranger")

mod_ranger <- ranger(y ~ ., data = train, write.forest = TRUE, probability = FALSE)
pred <- predict(mod_ranger, test)

confusionMatrix(test$y, pred$predictions)

# Accuracy : 0.9486