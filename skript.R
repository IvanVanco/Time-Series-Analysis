#################################################
#Montly sales data for 3 years and 3 months######
#We will use it for forecasting##################
#Created by Ivan Mitrovic 15.04.2020#############
#################################################


####################################################################################################################################
###############################################ONE DIMENSIONAL TIME SERIES ANALISYS#################################################
####################################################################################################################################

#clear all enviroment variables
rm(list = ls())

################################################
##############Loading libraries#################
################################################

#install.packages("fpp2")
library(fpp2)
#install.packages("uroot")
library(uroot)
#install.packages("RODBC")
library(RODBC)
#install.packages("tsfknn")
library(tsfknn)
#install.packages("nnfor")
library(nnfor)

###########################################################################################################
##################1. Import data from MS SQL Server to data frame using ODBC connection - RODBC############
###########################################################################################################

connection <- odbcDriverConnect(connection = "Driver={SQL Server Native Client 11.0};server=sqlrep.algrosso.com;database=AlGrosso Analitycs;uid=ReadOnly;pwd=BBRO.124578;")

data <- sqlFetch(connection, "dbo.Test_ProdajaPoMesecima", colnames = FALSE, rownames = TRUE)

######Only for MAC users######
path <- "data.csv"
data <- read.csv(path, stringsAsFactors = FALSE, header = TRUE)

###########################################################################################################
#######################################2. DATA Transformation phase########################################
###########################################################################################################

#removing last month from data frame, which is incomplete
data <- head(data, -1)

#dividing revenue by 1000 for tracking convinience in plots
data[,3] <- round(data[,3]/1000, digits = 0)
  
tdata <- ts(data[,3], start = c(2017,1), frequency = 12)

###########################################################################################################
#########################################3. Exploratory analisis###########################################
###########################################################################################################

summary(tdata)
autoplot(tdata) + 
         ggtitle("Monthly sales revenue")+
         ylab("Revenue in thousands")

####Trend#######

#Data has a trend
plot(tdata)
abline(reg = lm(tdata~time(tdata)))

####Season#######

#We have highest sales revenue in month of december in every year
boxplot(tdata~cycle(tdata))

#We also have same line motions around mid mar
trend <-  predict(lm(tdata~time(tdata)))
ggseasonplot(tdata/trend)+
  ggtitle("Seasonal Plot: Monthly sales revenue")+
  ylab("Revenue in thousands")

###########################################################################################################
#########################################4. Splitting data#################################################
###########################################################################################################
ttest <- tail(tdata, 6)
ttrain <- head(tdata, 33)

##########################################################################################################
##########################################5. Simple benchmark models######################################
##########################################################################################################

#################################################
################5.1 Mean method##################
#################################################
meanf <- meanf(ttrain, h=6)

#forcasting values as vector
fcvektormeanf <- c(rep(meanf$mean[1], times = 6))

#Sd residuals, Test Apsolute mean error, Test Root Mean Square Error
meanfsummary <- c(res_sd = sd(meanf$residuals, na.rm = TRUE), 
                  MAE = mean(abs(ttest-fcvektormeanf)), 
                  RMSE = sqrt(mean((ttest-fcvektormeanf)^2)))
meanfsummary

#Real differences in values between test and trained data. 
#We can used them to investigate if our model estimated values below tested, 
#or it exceeded it in some months(Can be used as subjective criterion i nsome cases) 
as.vector(ttest - abs(fcvektormeanf))

autoplot(tdata)+
  autolayer(meanf, series = "Mean Method", PI = FALSE)

#################################################
#########5.2 Naive method - Random walk##########
#################################################
rwf <- rwf(ttrain, h=6) #or naive(ttrain, h=6)

fcvektorrw <- c(rep(rwf$mean[1], times = 6))

rwfsummary <- c(res_sd = sd(rwf$residuals, na.rm = TRUE), 
                MAE = mean(abs(ttest-fcvektorrw)), 
                RMSE = sqrt(mean((ttest-fcvektorrw)^2)))
rwfsummary

as.vector(ttest - abs(fcvektorrw))

autoplot(tdata)+
  autolayer(rwf, series = "Random walk", PI = FALSE)

##############################################################
######################5.3 Drift method########################
##############################################################
rwd <- rwf(ttrain, h=6, drift=TRUE)

fcvektorrwd <- as.numeric(rwd$mean)

rwdsummary <- c(res_sd = sd(rwd$residuals, na.rm = TRUE), 
                MAE = mean(abs(ttest-fcvektorrwd)), 
                RMSE = sqrt(mean((ttest-fcvektorrwd)^2)))
rwdsummary

as.vector(ttest - abs(fcvektorrwd))

autoplot(tdata)+
  autolayer(rwd, series = "Drift method", PI = FALSE)

#################################################
###########5.4 Seasonal naive method#############
#################################################
snaive <- snaive(ttrain, h=6)

fcvektorsn <- as.numeric(snaive$mean)

snaivesummary <- c(res_sd = sd(snaive$residuals, na.rm = TRUE), 
                   MAE = mean(abs(ttest-fcvektorsn)), 
                   RSE = sqrt(mean((ttest-fcvektorsn)^2)))
snaivesummary

as.vector(ttest - abs(fcvektorsn))

autoplot(tdata)+
  autolayer(snaive, series = "S Naive", PI = FALSE)

##########################################################################################################
#####################################5. Exponential smoothing models- ETS#################################
##########################################################################################################

#####################################
#####5.1 Simple ETS method###########
#####################################
sets <- ses(ttrain, h=6)

fcvektorsets <- c(sets$mean)

setssummary <- c(res_sd = sd(sets$residuals, na.rm = TRUE), 
                 MAE = mean(abs(ttest-fcvektorsets)), 
                 RSE = sqrt(mean((ttest-fcvektorsets)^2)))
setssummary

as.vector(ttest - abs(fcvektorsets))

autoplot(tdata)+
  autolayer(sets, series = "Simple ETS", PI = FALSE)

#############################
#5.2 Holt-Winters-HW#########
#############################
hw <- hw(ttrain, h = 6) #same as seasonal = "additive"

fcvektorhw <- as.numeric(hw$mean)

hwsummary <- c(res_sd = sd(hw$residuals, na.rm = TRUE), 
               MAE = mean(abs(ttest-fcvektorhw)), 
               RSE = sqrt(mean((ttest-fcvektorhw)^2)))
hwsummary

as.vector(ttest - abs(fcvektorhw))

autoplot(tdata)+
  autolayer(hw, series = "Holt-Winter", PI = FALSE)

#############################
#5.3 Automated ETS###########
#############################
aets <- ets(ttrain)
aets <- predict(aets, h=6)
#it is suggesting ETS(M,N,N), which is not as good as HW

fcvektoraets <- as.numeric(aets$mean)

aetssummary <- round(c(res_sd = sd(aets$residuals, na.rm = TRUE), 
                       MAE = mean(abs(ttest-fcvektoraets)), 
                       RSE = sqrt(mean((ttest-fcvektoraets)^2))),4)
aetssummary

as.vector(ttest - abs(fcvektoraets))

autoplot(tdata)+
  autolayer(aets, series = "Automated ETS", PI = FALSE)

##########################################################################################################
########################################6. ARIMA family models############################################
##########################################################################################################

#########################################
#6.1 Moving average model - MA###########
#########################################
ma <- arima(ttrain, order = c(0L, 0L, 1L))

checkresiduals(ma)

yma <- predict(ma, 6)

fcvektorma <- as.numeric(yma$pred)

masummary <- c(res_sd = sqrt(ma$sigma2), 
               MAE = mean(abs(ttest-fcvektorma)), 
               RSE = sqrt(mean((ttest-fcvektorma)^2)))
masummary

as.vector(ttest - abs(fcvektorma))

autoplot(tdata)+
  autolayer(ts(fcvektorma, start = c(2019,10),frequency = 12), series = "MA(1)", PI = FALSE)

#########################################
#6.2 Autoregression model - AR###########
#########################################
ar <- arima(ttrain, order = c(1L, 0L, 0L))

checkresiduals(ar)

yar <- predict(ar, 6)

fcvektorar <- as.numeric(yar$pred)

arsummary <- c(res_sd = sqrt(ar$sigma2), 
               MAE = mean(abs(ttest-fcvektorar)), 
               RSE = sqrt(mean((ttest-fcvektorar)^2)))
arsummary

as.vector(ttest - abs(fcvektorar))

autoplot(tdata)+
  autolayer(ts(fcvektorar, start = c(2019,10),frequency = 12), series = "AR(1)", PI = FALSE)

###############################################################
#6.3 Autoregression model with moving average - ARMA###########
###############################################################
arma <- arima(ttrain, order = c(1L, 0L, 1L))

checkresiduals(arma)

yarma <- predict(arma, 6)

fcvektorarma <- as.numeric(yarma$pred)

armasummary <- c(res_sd = sqrt(arma$sigma2), 
                 MAE = mean(abs(ttest-fcvektorarma)), 
                 RSE = sqrt(mean((ttest-fcvektorarma)^2)))
armasummary

as.vector(ttest - abs(fcvektorarma))

autoplot(tdata)+
  autolayer(ts(fcvektorarma, start = c(2019,10),frequency = 12), series = "ARMA(1,1)", PI = FALSE)

###############################################################
#6.4 Auto Regressive Integrated Moving Average - ARIMA#########
###############################################################
#########################################
#6.4.1 ARIMA(1,0,0)(0,1,0)[12]###########
#########################################
arima1 <- auto.arima(ttrain, 
                    stepwise = FALSE,
                    approximation = FALSE,
                    trace = TRUE)

#we also need to do it manualy, because of error encountered 
arima1 <- arima(ttrain, order = c(1L, 0L, 0L),
                seasonal = list(order = c(0L, 1L, 0L), period = 12))

checkresiduals(arima1)

yarima1 <- predict(arima1, 6)

fcvektorarima1 <- as.numeric(yarima1$pred)

arima1summary <- c(res_sd = sqrt(arima1$sigma2), 
                   MAE = mean(abs(ttest-fcvektorarima1)), 
                   RSE = sqrt(mean((ttest-fcvektorarima1)^2)))
arima1summary

as.vector(ttest - abs(fcvektorarima1))

autoplot(tdata)+
  autolayer(ts(fcvektorarima1, start = c(2019,10),frequency = 12), series = "ARIMA(1,0,0)(0,1,0)[12]", PI = FALSE)

################################
#6.4.2 ARIMA(0,1,3)#############
################################
nsdiffs(ttrain, test = "ch")

arima2 <- auto.arima(ttrain, 
                     seasonal = TRUE,
                     seasonal.test = "ch", 
                     stepwise = FALSE,
                     approximation = FALSE,
                     trace = TRUE)

yarima2 <- predict(arima2, 6)

fcvektorarima2 <- as.numeric(yarima2$pred)

arima2summary <- c(res_sd = sqrt(arima2$sigma2), 
                   MAE = mean(abs(ttest-fcvektorarima2)), 
                   RSE = sqrt(mean((ttest-fcvektorarima2)^2)))
arima2summary

as.vector(ttest - abs(fcvektorarima2))

autoplot(tdata)+
  autolayer(ts(fcvektorarima2, start = c(2019,10),frequency = 12), series = "ARIMA(0,1,3)", PI = FALSE)


##########################################################################################################
######################################7. Machine learning models##########################################
##########################################################################################################

#########################
#7.1 KNN model###########
#########################
knn <- knn_forecasting(ttrain, h = 6, lags = 1:12, k = c(9), cf = c("mean"))  #best model
summary(knn)

#I used this different combination models, and model with k=9 is best fitted
knn2 <- knn_forecasting(ttrain, h = 6, lags = 1:12, k = c(2), cf = c("mean"))
knn3 <- knn_forecasting(ttrain, h = 6, lags = 1:12, k = c(3), cf = c("mean"))
knn4 <- knn_forecasting(ttrain, h = 6, lags = 1:12, k = c(4), cf = c("mean"))
knn5 <- knn_forecasting(ttrain, h = 6, lags = 1:12, k = c(5), cf = c("mean"))
knn6 <- knn_forecasting(ttrain, h = 6, lags = 1:12, k = c(6), cf = c("mean"))
knn7 <- knn_forecasting(ttrain, h = 6, lags = 1:12, k = c(7), cf = c("mean"))
knn8 <- knn_forecasting(ttrain, h = 6, lags = 1:12, k = c(8), cf = c("mean"))
knn23 <- knn_forecasting(ttrain, h = 6, lags = 1:12, k = c(2,3), cf = c("mean")) 
knn34 <- knn_forecasting(ttrain, h = 6, lags = 1:12, k = c(3,4), cf = c("mean")) 
knn45 <- knn_forecasting(ttrain, h = 6, lags = 1:12, k = c(4,5), cf = c("mean")) 
knn56 <- knn_forecasting(ttrain, h = 6, lags = 1:12, k = c(5,6), cf = c("mean"))
knn67 <- knn_forecasting(ttrain, h = 6, lags = 1:12, k = c(6,7), cf = c("mean"))
knn78 <- knn_forecasting(ttrain, h = 6, lags = 1:12, k = c(7,8), cf = c("mean"))
knn234 <- knn_forecasting(ttrain, h = 6, lags = 1:12, k = c(2,3,4), cf = c("mean"))
knn345 <- knn_forecasting(ttrain, h = 6, lags = 1:12, k = c(3,4,5), cf = c("mean")) 
knn456 <- knn_forecasting(ttrain, h = 6, lags = 1:12, k = c(4,5,6), cf = c("mean")) 
knn567 <- knn_forecasting(ttrain, h = 6, lags = 1:12, k = c(5,6,7), cf = c("mean")) 
knn678 <- knn_forecasting(ttrain, h = 6, lags = 1:12, k = c(6,7,8), cf = c("mean"))

fcvektorknn <- as.numeric(knn$prediction)
fcvektorknn2 <- as.numeric(knn2$prediction)
fcvektorknn3 <- as.numeric(knn3$prediction)
fcvektorknn4 <- as.numeric(knn4$prediction)
fcvektorknn5 <- as.numeric(knn5$prediction)
fcvektorknn6 <- as.numeric(knn6$prediction)
fcvektorknn7 <- as.numeric(knn7$prediction)
fcvektorknn8 <- as.numeric(knn8$prediction)
fcvektorknn23 <- as.numeric(knn23$prediction)
fcvektorknn34 <- as.numeric(knn34$prediction)
fcvektorknn45 <- as.numeric(knn45$prediction)
fcvektorknn56 <- as.numeric(knn56$prediction)
fcvektorknn67 <- as.numeric(knn67$prediction)
fcvektorknn78 <- as.numeric(knn78$prediction)
fcvektorknn234 <- as.numeric(knn234$prediction)
fcvektorknn345 <- as.numeric(knn345$prediction)
fcvektorknn456 <- as.numeric(knn456$prediction)
fcvektorknn567 <- as.numeric(knn567$prediction)
fcvektorknn678 <- as.numeric(knn678$prediction)

#Standard error for KNN is not calculated because there are no residuals or evaluated values for training data,
#and there are only predicted values for 6 months ahead. We will compare more important other two parametres
knnsummary <- c(res_sd = 0, 
                MAE = mean(abs(ttest-fcvektorknn)), 
                RSE = sqrt(mean((ttest-fcvektorknn)^2)))
knn2summary <- c(0,mean(abs(ttest-fcvektorknn2)), sqrt(mean((ttest-fcvektorknn2)^2)))
knn3summary <- c(0,mean(abs(ttest-fcvektorknn3)), sqrt(mean((ttest-fcvektorknn3)^2)))
knn4summary <- c(0,mean(abs(ttest-fcvektorknn4)), sqrt(mean((ttest-fcvektorknn4)^2)))
knn5summary <- c(0,mean(abs(ttest-fcvektorknn5)), sqrt(mean((ttest-fcvektorknn5)^2)))
knn6summary <- c(0,mean(abs(ttest-fcvektorknn6)), sqrt(mean((ttest-fcvektorknn6)^2)))
knn7summary <- c(0,mean(abs(ttest-fcvektorknn7)), sqrt(mean((ttest-fcvektorknn7)^2)))
knn8summary <- c(0,mean(abs(ttest-fcvektorknn8)), sqrt(mean((ttest-fcvektorknn8)^2)))
knn23summary <- c(0,mean(abs(ttest-fcvektorknn23)), sqrt(mean((ttest-fcvektorknn23)^2)))
knn34summary <- c(0,mean(abs(ttest-fcvektorknn34)), sqrt(mean((ttest-fcvektorknn34)^2)))
knn45summary <- c(0,mean(abs(ttest-fcvektorknn45)), sqrt(mean((ttest-fcvektorknn45)^2)))
knn56summary <- c(0,mean(abs(ttest-fcvektorknn56)), sqrt(mean((ttest-fcvektorknn56)^2)))
knn67summary <- c(0,mean(abs(ttest-fcvektorknn67)), sqrt(mean((ttest-fcvektorknn67)^2)))
knn78summary <- c(0,mean(abs(ttest-fcvektorknn78)), sqrt(mean((ttest-fcvektorknn78)^2)))
knn234summary <- c(0,mean(abs(ttest-fcvektorknn234)), sqrt(mean((ttest-fcvektorknn234)^2)))
knn345summary <- c(0,mean(abs(ttest-fcvektorknn345)), sqrt(mean((ttest-fcvektorknn345)^2)))
knn456summary <- c(0,mean(abs(ttest-fcvektorknn456)), sqrt(mean((ttest-fcvektorknn456)^2)))
knn567summary <- c(0,mean(abs(ttest-fcvektorknn567)), sqrt(mean((ttest-fcvektorknn567)^2)))
knn678summary <- c(0,mean(abs(ttest-fcvektorknn678)), sqrt(mean((ttest-fcvektorknn678)^2)))

comparings[order(comparings$MAE),]<- data.frame(rbind(knnsummary,knn2summary,knn3summary,knn4summary,knn5summary,knn6summary,
                                                      knn7summary, knn8summary, knn23summary, knn34summary, knn45summary, 
                                                      knn56summary,knn67summary, knn78summary, knn234summary, knn345summary, 
                                                      knn456summary, knn567summary, knn678summary),
                                                row.names = c("KNN9", "KNN2", "KNN3","KNN4", "KNN5", "KNN6", "KNN7", "KNN8",
                                                              "KNN23", "KNN34", "KNN45", "KNN56", "KNN67", "KNN78",
                                                              "KNN234", "KNN345", "KNN456", "KNN567", "KNN678"))
colnames(comparings) <- c("SD Residuals", "MAE", "RMSE")
comparings

as.vector(ttest - abs(fcvektorknn))

autoplot(tdata)+
  autolayer(ts(fcvektorknn, start = c(2019,10),frequency = 12), series = "KNN k=9", PI = FALSE)

#####################################
#7.2 Neural Networks models##########
#####################################
mlp <- mlp(ttrain, lags=1:12, reps = 10000)

plot(mlp)

ymlp <- forecast(mlp, h=6)
fcvektormlp <- as.numeric(ymlp$mean)

#Standard error for MLP is not calculated because there are no residuals or evaluated values for training data,
#and there are only predicted values for 6 months ahead. We will compare more important other two parametres
mlpsummary <- c(res_sd = 0, 
                MAE = round(mean(abs(ttest-fcvektormlp)),2), 
                RSE = round(sqrt(mean((ttest-fcvektormlp)^2)),2))
mlpsummary

as.vector(ttest - abs(fcvektormlp))

autoplot(tdata)+
  autolayer(ts(fcvektormlp, start = c(2019,10),frequency = 12), series = "Neural network-MLP", PI = FALSE)

#####################################################
#7. Choosing best fitting model for forecasting######
#####################################################
sumarno<- data.frame(rbind(meanfsummary, rwfsummary, rwdsummary, snaivesummary, setssummary, hwsummary, 
                           aetssummary, arsummary, masummary, armasummary, arima1summary, arima2summary, knnsummary, mlpsummary),
                     row.names = c("Mean", "Random walk-Naive", "Drift","S Naive", "Simple ETS", "HW", "Automated ETS", "AR(1)",
                                   "MA(1)", "ARMA(1,1)", "ARIMA(1,0,0)(0,1,0)[12]", "ARIMA(0,1,3)","KNN k=9", "Neural networks-MLP"))
colnames(sumarno) <- c("SD Residuals", "MAE", "RMSE")
sumarno

#Plots for Benchmark models
autoplot(tdata)+
  autolayer(meanf, series = "Mean Method", PI = FALSE)+
  autolayer(rwf, series = "Random walk", PI = FALSE)+
  autolayer(rwd, series = "Drift method", PI = FALSE)+
  autolayer(snaive, series = "S Naive", PI = FALSE)

#Plots for ETS models
autoplot(tdata)+
  autolayer(sets, series = "Simple ETS", PI = FALSE)+
  autolayer(hw, series = "Holt-Winter", PI = FALSE)+
  autolayer(aets, series = "Automated ETS", PI = FALSE)

#Plots for ARIMA models
autoplot(tdata)+
  autolayer(ts(fcvektorma, start = c(2019,10),frequency = 12), series = "MA(1)", PI = FALSE)+
  autolayer(ts(fcvektorar, start = c(2019,10),frequency = 12), series = "AR(1)", PI = FALSE)+
  autolayer(ts(fcvektorarma, start = c(2019,10),frequency = 12), series = "ARMA(1,1)", PI = FALSE)+
  autolayer(ts(fcvektorarima1, start = c(2019,10),frequency = 12), series = "ARIMA(1,0,0)(0,1,0)[12]", PI = FALSE)+
  autolayer(ts(fcvektorarima2, start = c(2019,10),frequency = 12), series = "ARIMA(0,1,3)", PI = FALSE)

#Plots for ML models
autoplot(tdata)+
  autolayer(ts(fcvektorknn, start = c(2019,10),frequency = 12), series = "KNN k=9", PI = FALSE) + 
  autolayer(ts(fcvektormlp, start = c(2019,10),frequency = 12), series = "Neural network-MLP", PI = FALSE)

#Comparing best from the groups
autoplot(tdata)+
  autolayer(rwd, series = "Drift method", PI = FALSE)+
  autolayer(hw, series = "Holt-Winter", PI = FALSE)+
  autolayer(ts(fcvektorarima1, start = c(2019,10),frequency = 12), series = "ARIMA(1,0,0)(0,1,0)[12]", PI = FALSE)+
  autolayer(ts(fcvektormlp, start = c(2019,10),frequency = 12), series = "Neural network-MLP", PI = FALSE)


############################################Appendix#######################################################

###########################################################################################################
#######################################6. Tranformation tehniques##########################################
###########################################################################################################

#1. Removing trend from series to get trend-stationary ts

#trend line function for later transformation use to remove trend from serie
#traintrend <-  predict(lm(ttrain~time(ttrain)))

#1.1. Method one
#Take first difference of the data to remove trend,but in this case we dont removed it
#dfdata <- diff(tdata)

#1.2. Method two
#Divide by trend to remove it, this works fine here
#dftrain <- ttrain / traintrend

#autoplot(dfdata) +
#  ggtitle("Trend stationary: Monthly changes in sales revenue")+
#  ylab("Revenue in thousands")


#2. Removing variance from series to get stationary ts

#Using Log transformation on trend stationary data
#plot(log(tdata))

#here we cannot remove it, unless we use daily sales and remove it there.
#For our experiment, it is not crucial


