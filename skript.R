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
#I used this different combination models, and model with k=9 is best fitted
knn_comparison <- list()
i <- 1
for(k in 2:9) {
  knn_fit <- knn_forecasting(ttrain, h = 6, lags = 1:12, k = c(k), cf = c("mean"))
  knn_pred <- as.numeric(knn_fit$prediction)
  knn_comparison[[i]] <- list(k = k, MAE = mean(abs(ttest-knn_pred)), RMSE = sqrt(mean((ttest-knn_pred)^2)))
  i <- i + 1
}
for(k in 2:8) {
  knn_fit <- knn_forecasting(ttrain, h = 6, lags = 1:12, k = c(k,k+1), cf = c("mean"))
  knn_pred <- as.numeric(knn_fit$prediction)
  knn_comparison[[i]] <- list(k = paste0("(",k,",",k+1,")"), 
                              MAE = mean(abs(ttest-knn_pred)), RMSE = sqrt(mean((ttest-knn_pred)^2)))
  i <- i + 1
}
for(k in 2:7) {
  knn_fit <- knn_forecasting(ttrain, h = 6, lags = 1:12, k = c(k,k+1,k+2), cf = c("mean"))
  knn_pred <- as.numeric(knn_fit$prediction)
  knn_comparison[[i]] <- list(k = paste0("(",k,",",k+1,",",k+2,")"), 
                              MAE = mean(abs(ttest-knn_pred)), RMSE = sqrt(mean((ttest-knn_pred)^2)))
  i <- i + 1
}

knn_comparison_df <- do.call(rbind.data.frame, knn_comparison)
View(knn_comparison_df[order(knn_comparison_df$RMSE),])

knn <- knn_forecasting(ttrain, h = 6, lags = 1:12, k = c(9), cf = c("mean"))  #best model
summary(knn)

fcvektorknn <- as.numeric(knn$prediction)

#Standard error for KNN is not calculated because there are no residuals or evaluated values for training data,
#and there are only predicted values for 6 months ahead. We will compare more important other two parametres
knnsummary <- c(res_sd = 0, 
                MAE = mean(abs(ttest-fcvektorknn)), 
                RSE = sqrt(mean((ttest-fcvektorknn)^2)))

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


