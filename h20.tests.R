

 mfile= 'C:\\Users\\USEFIL\\Documents\\Test\\Data\\prac_train.csv'
 mfile2= 'C:\\Users\\USEFIL\\Documents\\Test\\Data\\prac_test.csv'
 
 
 trData<-h2o.importFile(path = mfile,sep=',')
 tsData<-h2o.importFile(path = mfile2,sep=',')
 trData[,1]<-as.factor(trData[,1])
 tsData[,1]<-as.factor(tsData[,1])

ptm<-proc.time()
res.dl<-h2o.deeplearning(x = 2:785,
                        y=1,
                        training_frame = trData,
                        validation_frame = tsData,
                        activation = "RectifierWithDropout",
                        distribution ="multinomial",
                        hidden=c(200,200,200),
                        input_dropout_ratio = 0.2,
                        l1 = 1e-5,
                        epochs = 10)



pred.dl<-h2o.predict(object=res.dl,newdata=tsData[,-1])
proc.time()-ptm
sum(diag(table(prac_test$label,pred.dl.df[,1])))
#95.49% xoris multinomial
#123.76

#96.74 me
#123.89


ptm<-proc.time()
res.dl <- h2o.deeplearning(x = 2:785, y = 1, training_frame= trData, activation = "Tanh",hidden=c(500,500,1000),
                           	epochs = 20,rate=0.01,rate_annealing = 0.001)
proc.time()-ptm

pred.dl<-h2o.predict(object=res.dl,newdata=tsData[,-1])
 pred.dl.df<-as.data.frame(pred.dl)
sum(diag(table(prac_test$label,pred.dl.df[,1])))

#3226.28
#96.74

ptm<-proc.time()

res.dl <- h2o.deeplearning(x = 2:785, y = 1, training_frame= trData, activation = "RectifierWithDropout",
                           	hidden=c(1024,1024,2048),epochs = 200, adaptive_rate = FALSE, rate=0.01, rate_annealing = 1.0e-6,
                           	rate_decay = 1.0, momentum_start = 0.5,momentum_ramp = 32000*12, momentum_stable = 0.99, input_dropout_ratio = 0.2,
                           	l1 = 1.0e-5,l2 = 0.0,max_w2 = 15.0, initial_weight_distribution = "Normal",initial_weight_scale = 0.01,
                           	nesterov_accelerated_gradient = T, loss = "CrossEntropy", fast_mode = T, diagnostics = T, ignore_const_cols = T,
                           	force_load_balance = T)
proc.time()-ptm

# 4962.97 
#98.14

 pred.dl<-h2o.predict(object=res.dl,newdata=tsData[,-1])
 pred.dl.df<-as.data.frame(pred.dl)
 table(prac_test$label,pred.dl.df[,1])
 
 #Anomaly Detection
 
 library(h2o)
 h2oServer <- h2o.init(nthreads=-1)
 
 train_ecg<-h2o.importFile(path ="http://h2o-public-test-data.s3.amazonaws.com/smalldata/anomaly/ecg_discord_train.csv",header=FALSE,sep=",")
 
 test_ecg<-h2o.importFile(path ="http://h2o-public-test-data.s3.amazonaws.com/smalldata/anomaly/ecg_discord_test.csv",header=FALSE,sep=",")
 a<-names(train_ecg)
 ptm<-proc.time()
 anomaly_model<-h2o.deeplearning(training_frame=train_ecg,a,activation="Tanh",autoencoder = TRUE,hidden=c(50,20,50),l1=1e-4,epochs = 100)
 proc.time()-ptm
 
 #elapse 1.45
 
 recon_error<-h2o.anomaly(anomaly_model, test_ecg)
 recon_error <- as.data.frame(recon_error)

# plot error
 library(ggplot2)
 recon_error$sub<-seq(1:23)
 
 
 ggplot(data=recon_error, aes(x=sub, y=Reconstruction.MSE)) +
   geom_line(color="firebrick") +
   geom_point(color="black")+
   xlab("Subject") + ylab("Error") +
   ggtitle("Error per Subject")

 