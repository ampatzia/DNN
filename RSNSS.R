library(caret)
library(RSNNS)


mnist.set <- read.csv('data/train.csv', header=TRUE)
mnist.targets<- decodeClassLabels(mnist.set[,1])

mnist.set<-splitForTrainingAndTest(mnist.set,mnist.targets, ratio=0.2) #split train and test
mnist.set<- normTrainingAndTestSet(mnist.set, type="0_1")


ptm <- proc.time()
mnist.model <- mlp(mnist.set$inputsTrain, mnist.set$targetsTrain,size = 50,
                   learnFuncParams = c(0.1),maxit = 100,
                   inputsTest = mnist.set$inputsTest,
                   targetsTest = mnist.set$targetsTest)
proc.time() - ptm
#runtime 839.75
#94.3

#train set classify
#Prepei na fortothei prota i caret kai meta i RSNNS
train.con <- confusionMatrix(mnist.set$targetsTrain, mnist.model$fitted.values)
dimnames(train.con)$targets <-c(0:9)
dimnames(train.con)$predictions <- c(0:9)
train.con

caret::confusionMatrix(train.con)



test.con <- confusionMatrix(mnist.set$targetsTest,mnist.model$fittedTestValues)
dimnames(test.con)$targets<-c(0:9)
dimnames(test.con)$predictions<-c(0:9)
test.con
caret::confusionMatrix(test.con)




#Deep learning
# 
# install.packages("drat", repos="https://cran.rstudio.com")
# drat:::addRepo("dmlc")
# install.packages("mxnet")


library("mxnet")

mnist.set2 <- read.csv('data/train.csv', header=TRUE)
ind=sample(2,nrow(mnist.set2),replace=T,prob=c(0.8,0.2))
train <- mnist.set2[ind==1,]
test <- mnist.set2[ind==2,]

train.x <- train[,-1] #afairesi tou label - unsupervised
train.y <- train[,1]  # to anagnoristiko


test.x <- test[,-1] #afairesi tou label - unsupervised
test.y <- test[,1]  # to anagnoristiko
train.x <- t(train.x/255) #kanonikipoiisi se range [0-1]
test.x <- t(test.x/255)

#hidden layers kai arxitektoniki tou diktioy

data <- mx.symbol.Variable("data")
fc1 <- mx.symbol.FullyConnected(data, name="fc1", num_hidden=128) #proto layer
act1 <- mx.symbol.Activation(fc1, name="relu1", act_type="relu")
fc2 <- mx.symbol.FullyConnected(act1, name="fc2", num_hidden=64) #deytero pairnei times apo to proto
act2 <- mx.symbol.Activation(fc2, name="relu2", act_type="relu")
fc3 <- mx.symbol.FullyConnected(act2, name="fc3", num_hidden=10) #output layer o.p.
softmax <- mx.symbol.SoftmaxOutput(fc3, name="sm")

devices <- mx.cpu()



mx.set.seed(0)
ptm <- proc.time()
mx.set.seed(0)
model <- mx.model.FeedForward.create(softmax, X=train.x, y=train.y,
                                     ctx=devices, num.round=10, array.batch.size=100,
                                     learning.rate=0.07, momentum=0.9,  eval.metric=mx.metric.accuracy,
                                     initializer=mx.init.uniform(0.07),
                                     epoch.end.callback=mx.callback.log.train.metric(100))
 proc.time()-ptm
 
 
 
 #5.02 deyteroplepta!
 
#predictions
 
 preds <- predict(model, test.x)
 dim(preds)
 
 pred.label <- max.col(t(preds)) - 1
 table(pred.label)

 caret::confusionMatrix(pred.label,test.y)
 
 #6 seconds
 #96.92
 
 
 
 
 #laser regrassion
 
 data(snnsData)
 laser <- snnsData$laser_1000.pat
 inputs <- laser[, inputColumns(laser)]
 targets <- laser[, outputColumns(laser)]
 patterns <- splitForTrainingAndTest(inputs, targets, ratio = 0.3)
 
 model <- elman(patterns$inputsTrain, patterns$targetsTrain,
                size = c(8, 8), learnFuncParams = c(0.1), maxit = 500,
                inputsTest = patterns$inputsTest, targetsTest = patterns$targetsTest,
                linOut = FALSE)
 
 test.con <- confusionMatrix(patterns$targetsTest,model$fittedTestValues)
