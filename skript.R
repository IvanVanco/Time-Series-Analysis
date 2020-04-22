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

###########################################################################################################
##################1. Import data from MS SQL Server to data frame using ODBC connection - RODBC############
###########################################################################################################

connection <- odbcDriverConnect(connection = "Driver={SQL Server Native Client 11.0};server=sqlrep.algrosso.com;database=AlGrosso Analitycs;uid=ReadOnly;pwd=BBRO.124578;")

data <- sqlFetch(connection, "dbo.Test_ProdajaPoMesecima", colnames = FALSE, rownames = TRUE)

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
train <- data[1:33,]
test <- data[34:39,]

ttrain <- ts(train[,3], start = c(2017, 1), end = c(2019, 9), frequency = 12)
ttest <- ts(test[,3], start = c(2019, 10), end = c(2020, 3), frequency = 12)

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
meanfsummary <- c(sd(meanf$residuals, na.rm = TRUE), mean(abs(ttest-fcvektormeanf)), sqrt(mean((ttest-fcvektormeanf)^2)))
meanfsummary

#Real differences in values
as.vector(ttest - abs(fcvektormeanf))

autoplot(tdata)+
  autolayer(meanf, series = "Mean Method", PI = FALSE)

#################################################
#########5.2 Naive method - Random walk##########
#################################################
rwf <- rwf(ttrain, h=6) #or naive(ttrain, h=6)

fcvektorrw <- c(rep(rwf$mean[1], times = 6))

rwfsummary <- c(sd(rwf$residuals, na.rm = TRUE), mean(abs(ttest-fcvektorrw)), sqrt(mean((ttest-fcvektorrw)^2)))
rwfsummary

as.vector(ttest - abs(fcvektorrw))

autoplot(tdata)+
  autolayer(rwf, series = "Random walk", PI = FALSE)

##############################################################
######################5.3 Drift method########################
##############################################################
rwd <- rwf(ttrain, h=6, drift=TRUE)

fcvektorrwd <- c(rwd$mean[1],rwd$mean[2],rwd$mean[3],rwd$mean[4],rwd$mean[5],rwd$mean[6])

rwdsummary <- c(sd(rwd$residuals, na.rm = TRUE), mean(abs(ttest-fcvektorrwd)), sqrt(mean((ttest-fcvektorrwd)^2)))
rwdsummary

as.vector(ttest - abs(fcvektorrwd))

autoplot(tdata)+
  autolayer(rwd, series = "Drift method", PI = FALSE)

#################################################
###########5.4 Seasonal naive method#############
#################################################
snaive <- snaive(ttrain, h=6)

fcvektorsn <- c(snaive$mean[1],snaive$mean[2],snaive$mean[3],snaive$mean[4],snaive$mean[5],snaive$mean[6])

snaivesummary <- c(sd(snaive$residuals, na.rm = TRUE), mean(abs(ttest-fcvektorsn)), sqrt(mean((ttest-fcvektorsn)^2)))
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

setssummary <- c(sd(sets$residuals, na.rm = TRUE), mean(abs(ttest-fcvektorsets)), sqrt(mean((ttest-fcvektorsets)^2)))
setssummary

as.vector(ttest - abs(fcvektorsets))

autoplot(tdata)+
  autolayer(sets, series = "Simple ETS", PI = FALSE)

#############################
#5.2 Holt-Winters-HW#########
#############################
hw <- hw(ttrain, h = 6) #same as seasonal = "additive"

fcvektorhw <- c(hw$mean[1],hw$mean[2],hw$mean[3],hw$mean[4],hw$mean[5],hw$mean[6])

hwsummary <- c(sd(hw$residuals, na.rm = TRUE), mean(abs(ttest-fcvektorhw)), sqrt(mean((ttest-fcvektorhw)^2)))
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

fcvektoraets <- c(aets$mean[1],aets$mean[2],aets$mean[3],aets$mean[4],aets$mean[5],aets$mean[6])

aetssummary <- round(c(sd(aets$residuals, na.rm = TRUE), mean(abs(ttest-fcvektoraets)), sqrt(mean((ttest-fcvektoraets)^2))),4)
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

fcvektorma <- c(yma$pred[1],yma$pred[2],yma$pred[3],yma$pred[4],yma$pred[5],yma$pred[6])

masummary <- c(sqrt(arima2$sigma2), mean(abs(ttest-fcvektorma)), sqrt(mean((ttest-fcvektorma)^2)))
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

fcvektorar <- c(yar$pred[1],yar$pred[2],yar$pred[3],yar$pred[4],yar$pred[5],yar$pred[6])

arsummary <- c(sqrt(ar$sigma2), mean(abs(ttest-fcvektorar)), sqrt(mean((ttest-fcvektorar)^2)))
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

fcvektorarma <- c(yarma$pred[1],yarma$pred[2],yarma$pred[3],yarma$pred[4],yarma$pred[5],yarma$pred[6])

armasummary <- c(sqrt(arma$sigma2), mean(abs(ttest-fcvektorarma)), sqrt(mean((ttest-fcvektorarma)^2)))
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

fcvektorarima1 <- c(yarima1$pred[1],yarima1$pred[2],yarima1$pred[3],yarima1$pred[4],yarima1$pred[5],yarima1$pred[6])

arima1summary <- c(sqrt(arima1$sigma2), mean(abs(ttest-fcvektorarima1)), sqrt(mean((ttest-fcvektorarima1)^2)))
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

fcvektorarima2 <- c(yarima2$pred[1],yarima2$pred[2],yarima2$pred[3],yarima2$pred[4],yarima2$pred[5],yarima2$pred[6])

arima2summary <- c(sqrt(arima2$sigma2), mean(abs(ttest-fcvektorarima2)), sqrt(mean((ttest-fcvektorarima2)^2)))
arima2summary

as.vector(ttest - abs(fcvektorarima2))

autoplot(tdata)+
  autolayer(ts(fcvektorarima2, start = c(2019,10),frequency = 12), series = "ARIMA(0,1,3)", PI = FALSE)

###########################################
#6.4.3 ARIMA(0,1,1)(1,1,0)[12]#############
###########################################
arima3 <- auto.arima(tdata, d=1, D=1, 
                    stepwise = FALSE,
                    approximation = FALSE,
                    trace = TRUE)

arima3 <- arima(ttrain, order = c(0L, 1L, 1L),
               seasonal = list(order = c(1L, 1L, 0L), period = 12))

checkresiduals(arima3)

yarima3 <- predict(arima3, 6)

fcvektorarima3 <- c(yarima3$pred[1],yarima3$pred[2],yarima3$pred[3],yarima3$pred[4],yarima3$pred[5],yarima3$pred[6])

arima3summary <- c(sqrt(arima3$sigma2), mean(abs(ttest-fcvektorarima3)), sqrt(mean((ttest-fcvektorarima3)^2)))
arima3summary

as.vector(ttest - abs(fcvektorarima3))

autoplot(tdata)+
  autolayer(ts(fcvektorarima3, start = c(2019,10),frequency = 12), series = "ARIMA(0,1,1)(1,1,0)[12]", PI = FALSE)

#####################################################
#7. Choosing best fitting model for forecasting######
#####################################################
sumarno<- data.frame(rbind(meanfsummary, rwfsummary, rwdsummary, snaivesummary, setssummary, hwsummary, 
                          aetssummary, arsummary, masummary, armasummary, arima1summary, arima2summary, arima3summary),
           row.names = c("Mean", "Random walk-Naive", "Drift","S Naive", "Simple ETS", "HW", "Automated ETS", "AR(1)",
                         "MA(1)", "ARMA(1,1)", "ARIMA(1,0,0)(0,1,0)[12]", "ARIMA(0,1,3)", "ARIMA(0,1,1)(1,1,0)[12]"))
colnames(sumarno) <- c("SD Residuals", "MAE", "RMSE")
sumarno

autoplot(tdata)+
  autolayer(meanf, series = "Mean Method", PI = FALSE)+
  autolayer(rwf, series = "Random walk", PI = FALSE)+
  autolayer(rwd, series = "Drift method", PI = FALSE)+
  autolayer(snaive, series = "S Naive", PI = FALSE)+
  autolayer(sets, series = "Simple ETS", PI = FALSE)+
  autolayer(hw, series = "Holt-Winter", PI = FALSE)+
  autolayer(aets, series = "Automated ETS", PI = FALSE)+
  autolayer(ts(fcvektorma, start = c(2019,10),frequency = 12), series = "MA(1)", PI = FALSE)+
  autolayer(ts(fcvektorar, start = c(2019,10),frequency = 12), series = "AR(1)", PI = FALSE)+
  autolayer(ts(fcvektorarma, start = c(2019,10),frequency = 12), series = "ARMA(1,1)", PI = FALSE)+
  autolayer(ts(fcvektorarima1, start = c(2019,10),frequency = 12), series = "ARIMA(1,0,0)(0,1,0)[12]", PI = FALSE)+
  autolayer(ts(fcvektorarima2, start = c(2019,10),frequency = 12), series = "ARIMA(0,1,3)", PI = FALSE)+
  autolayer(ts(fcvektorarima3, start = c(2019,10),frequency = 12), series = "ARIMA(0,1,1)(1,1,0)[12]", PI = FALSE)

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


