library("mxnet")

#loading tou dataset
train <- read.csv('data/train.csv', header=TRUE)
test <- read.csv('data/test.csv', header=TRUE)
train <- data.matrix(train)
test <- data.matrix(test)






train.x <- train[,-1] #afairesi tou label - unsupervised ?
train.y <- train[,1]  # to anagnoristiko

train.x <- t(train.x/255) #kanonikipoiisi se range [0-1]
test <- t(test/255)


#hidden layers kai arxitektoniki tou diktioy

data <- mx.symbol.Variable("data")
fc1 <- mx.symbol.FullyConnected(data, name="fc1", num_hidden=128) #proto layer
act1 <- mx.symbol.Activation(fc1, name="relu1", act_type="relu")
fc2 <- mx.symbol.FullyConnected(act1, name="fc2", num_hidden=64) #deytero pairnei times apo to proto
act2 <- mx.symbol.Activation(fc2, name="relu2", act_type="relu")
fc3 <- mx.symbol.FullyConnected(act2, name="fc3", num_hidden=10) #output layer o.p.
softmax <- mx.symbol.SoftmaxOutput(fc3, name="sm")

devices <- mx.cpu()



#train

mx.set.seed(0)
model <- mx.model.FeedForward.create(softmax, X=train.x, y=train.y,
                                     ctx=devices, num.round=10, array.batch.size=100,
                                     learning.rate=0.07, momentum=0.9,  eval.metric=mx.metric.accuracy,
                                     initializer=mx.init.uniform(0.07),
                                     epoch.end.callback=mx.callback.log.train.metric(100))

#predictions
preds <- predict(model, test)
dim(preds)

pred.label <- max.col(t(preds)) - 1
table(pred.label)



#save as csv

submission <- data.frame(ImageId=1:ncol(test), Label=pred.label)
write.csv(submission, file='submission.csv', row.names=FALSE, quote=FALSE)