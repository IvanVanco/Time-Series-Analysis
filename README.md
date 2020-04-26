# **<p align="center">Time series analysis</p>**

## 1.  **Introduction**

A time series is a sequence of observations recorded in discrete time
points.\
They can be recorded in hourly (e.g. air temperature), daily (e.g. DJI
Average), monthly (e.g. sales) or yearly (e.g. GDP) time intervals.\
Real-world time series data commonly contains several components or
recognizable patterns.\
Those time series components can be divided into

1.  **Systematic** **components** - are those that allow modeling as
    they have some consistent or recurring pattern. Examples for
    systematic time series components are

    a.  **Trend** - A trend is a consistent increase or decrease in
        units of the underlying data. Consistent in this context means
        that there are consecutively increasing peaks during a so-called
        uptrend and consecutively decreasing lows during a so-called
        downtrend. Also, if there is no trend, this can be called
        **stationary**.

    b.  **Seasonality** - describes cyclical effects due to the time of
        the year. Daily data may show valleys on the weekend, weekly
        data may show peaks for the first week of every or some month(s)
        or monthly data may show lower levels for the winter or summer
        months respectively.

2.  **Non**-**systematic** components - any other observation in the
    data is categorized as unsystematic and is called error or
    remainder.

The general magnitude of the time series is also referred to with
"level".

**In our example, we will be looking wholesale confectionery products data. Our main focus is net revenue sales dataset from year 2017, until end of first quarter of 2020 (3 years and 3 months) by month interval.
We use one dimensional time series analysis techniques and machine learning algorithms, and compare them.
**

## 2.  **Loading libraries**

All necessary libraries are loaded and written all together at the
beginning of the project. This way we know what packages we used, and
all possible changes are made easier. **\
**In **fpp2** are all auto model and plotting functions.\
**RODBC** is used to create connection using ODBC driver, the most
popular and basic database connection driver.\
**Uroot** is used for later in chapter for changing model parameters .

<img src="https://github.com/IvanVanco/Time-Series-Analysis/blob/master/res/loadinglibrary.png">


## 3.  **Importing data**

We will use SQL Server to import data from view object named
**TEST\_ProdajaPoMesecima**.\
Firstly, we create connection object by giving them connection string,
contained of **driver's name**, **server's name**, **database**, **uid**
- username and **pwd** - password.\
Then, we call stored view from server, and get raw data inside data
variable.

<img src="https://github.com/IvanVanco/Time-Series-Analysis/blob/master/res/connection.png">

**Also, for MAC users**, I provided **data.csv** file. Set file path location to the file data.csv inside **path** variable (keep same structure using \\ marks).

Structured data is needed before procced into next step. For that we
will remove last month's incomplete data, present sales in thousands
values, and declare our data as time series data, by defining start
period and frequency of our data, which, for month, is 12.

<img src="https://github.com/IvanVanco/Time-Series-Analysis/blob/master/res/transformation1.png">

## 4.  **Exploratory analysis**

Our time series looks like this:

<img src="https://github.com/IvanVanco/Time-Series-Analysis/blob/master/res/explore.png" weight="350" height="350">

From graph, we can fill there is good positive trend. There are also
variations in our values, but in this analysis, we will not focus on
them.\
To inspect seasonality we will **ggseasonplot** function.

<img src="https://github.com/IvanVanco/Time-Series-Analysis/blob/master/res/exploreseason.png" weight="350" height="350">


We call see that in months of December, in mid-March we have strong
seasonality and weak in July-August.\
We need to be concern of this and use appropriate models, but I will
show all models, regards of these components.

## 5.  **Splitting data**

I choose proportion of 85:15 because of lack of data records. For each
data partition, we need to recreate time series copy.

<img src="https://github.com/IvanVanco/Time-Series-Analysis/blob/master/res/datapartition.png">


## 6.  **Model evaluation**

Some of the most popular evaluation metrics are:

1.  **Residual Standard Deviation** of training data **(SD res, or
    √sigma<sup>2</sup>)** -- lower values are good sign, but not enough

2.  **Mean Absolute Error (MAE)** of test data -- absolute prediction
    error, lower is better.

3.  **Root Mean Squared Error (RMSE)** of test data-- prediction
    square deviation error, lower is better.

Indicator number 2. and 3. are crucial for model grading.

Also, for visualization, we will be using these evaluation charts:

1.	**Graphical presentation of predicted vs real value** 

2.	**For ARIMA models, Correlogram, also known as ACF plot - ACF** is an auto-correlation function which gives us values of auto-correlation of any series with its lagged values. In simple terms, it describes how well the present value of the series is related with its past values.


## 7.  **Modelling -- simple techniques**

Simple forecasting techniques are used as benchmarks. They provide a
general understanding of historical data and to build intuition upon
which to add additional layers complexity and are good to test basic
nature about time series.\
Several such techniques are common in literature such as: **mean model,
naive forecast, random walk, drift method.**

### 7.1  **Mean model**

A mean model takes the mean of previous observations and uses that for
forecasting.\
**Formula**: **y<sub>t</sub> = mean(y<sub>t-x</sub>)** x- observation of model for whole
period\
**Model**: **meanf \<- meanf(ttrain, h=6)** h- period for prediction

| __SD RES__ | __MAE__ | __RMSE__ |
|-------------|------------|------------|
| 18,876.63   | 23,513.24  | 27,318.80  |
  
<img src="https://github.com/IvanVanco/Time-Series-Analysis/blob/master/res/meanmodel.png" weight="350" height="350">


### 7.2  **Naive method -- Random walk**

A random walk, on the other hand, would predict the next value, Ŷ(t),
that equals to the previous value plus a constant change.\
**Formula**: **y<sub>t</sub> = mean(y<sub>t-1</sub>)** \
**Model**: **rwf \<- rwf(ttrain, h=6)** or **naive(ttrain, h=6)** h-
period for prediction

| __SD RES__ | __MAE__ | __RMSE__ |
|-------------|------------|------------|
| 15,495.68   | 14,908.00  | 17,422.21  |

<img src="https://github.com/IvanVanco/Time-Series-Analysis/blob/master/res/randomwalk.png" weight="350" height="350">

### 7.3  **Drift method**

Equivalent to extrapolating a line drawn between first and last
observations. Forecast equal to last value plus average change. It is
first dynamic model.\
**Formula**: **y<sub>t</sub> = y<sub>t-1</sub> + mean(rest data)**\
**Model**: **rwd \<- rwf(ttrain, h=6, drift=TRUE)** h- period for
prediction

| __SD RES__ | __MAE__ | __RMSE__ |
|-------------|------------|------------|
| 15,495.68   | 14,648.55  | 16,021.92  |

<img src="https://github.com/IvanVanco/Time-Series-Analysis/blob/master/res/driftmodel.png" weight="350" height="350">

### 7.4  **Seasonal naïve method**

Forecasts are equal to last value from same season. Good for isolating
and testing seasonal effects on time series.\
**Formula**: **y<sub>t</sub> = y<sub>t-s</sub>**\
**Model**: **snaive \<- snaive(ttrain, h=6)** h- period for prediction

| __SD RES__ | __MAE__ | __RMSE__ |
|-------------|------------|------------|
| 16,453.78   | 17,185.17  | 18,487.01  |

<img src="https://github.com/IvanVanco/Time-Series-Analysis/blob/master/res/seasonnaive.png" weight="350" height="350">

## 8.  **Exponential smoothing models -- ETS**

Exponential smoothing is a popular forecasting method for short-term
predictions. Such forecasts of future values are based on past data
whereby the most recent observations are weighted more than less recent
observations. As part of this weighting, constants are
being ***smoothed***. Exponential smoothing introduces the idea of
building a forecasted value as the average figure from differently
weighted data points for the average calculation.

There are different exponential smoothing methods that differ from each
other in the components of the time series that are modeled.

### 8.1  **Simple ETS method**

Simple exponential smoothing assumes that the time series data has
only a level and some error (or remainder) but no trend or
seasonality. It uses only one smoothing constant. The smoothing
parameter α determines the distribution of weights of past
observations and with that how heavily a given time period is factored
into the forecasted value. If the smoothing parameter is close to 1,
recent observations carry more weight and if the smoothing parameter
is closer to 0, weights of older and recent observations are more
balanced.\
**Formula**: **F<sub>t</sub> = αy<sub>t-1</sub> + (1-α)F<sub>t-1</sub>** where\
F<sub>t</sub> , F<sub>t-1</sub> - forecast values for time t, t-1; y<sub>t-1</sub> actual value
for time t-1; α - smoothing constant\
**Model**: **sets \<- ses(ttrain, h=6)** h- period for prediction

| __SD RES__ | __MAE__ | __RMSE__ |
|-------------|------------|------------|
| 14,205.04   | 15,507.66  | 18,561.26  |
  
<img src="https://github.com/IvanVanco/Time-Series-Analysis/blob/master/res/SETS.png" weight="350" height="350">

### 8.2  **Holt-Winters exponential smoothing -- HW**

Holt-Winters exponential smoothing is a time series forecasting
approach that takes the overall level, trend and seasonality of the
underlying dataset into account for its forecast.\
\
Hence, the Holt-Winters model has three smoothing parameters
indicating the exponential decay from most recent to older
observations:\
α (alpha) for the level component,\
β (beta) for the trend component,\
γ (gamma) for the seasonality component.\
**Model**: **hw \<- hw(ttrain, h = 6, seasonal = "additive")** h-
period for prediction

| __SD RES__ | __MAE__ | __RMSE__ |
|-------------|------------|------------|
| 14,830.10   | 9,482.57   | 15,179.86  |

<img src="https://github.com/IvanVanco/Time-Series-Analysis/blob/master/res/hw.png" weight="350" height="350">


### 8.3  **Automated exponential smoothing**

There is also an automated exponential smoothing forecast. In this
case, it is automatically determined whether a time series component
is additive or multiplicative.\
The ets function takes several arguments including model. The model
argument takes three letters as input.\
The first letter denotes the error type (A, M or Z),\
the second letter denotes the trend type (N, A, M or Z),\
and the third letter denotes the seasonality type (N, A, M or Z).\
The letters stand for N = none, A = additive, M = multiplicative and Z
= automatically selected.\
**Model**: **aets \<- ets(ttrain)**\
**Note(thank’s to my professor)*** In this particular example, if we use aets function, it will be suggesting us **ETS (M,N,N)**, which is obviously not ideal model, considering the observed trend in data. In this case we must add model parameter in ets function like this (**ttrain, model = ("AAA")**)\

I will use these default values, only because my previous HW model is practically same as **ETS (A,A,A)**, but be concerned if you are using only this function.


| __SD RES__ | __MAE__ | __RMSE__ |
|-------------|------------|------------|
| 0,1913      | 14,908.23  | 17,422.64  |

<img src="https://github.com/IvanVanco/Time-Series-Analysis/blob/master/res/aets.png" weight="350" height="350">


## 9.  **ARIMA family models**

Auto Regressive Integrated Moving Average (ARIMA) is arguably the most
popular and widely used statistical forecasting technique. As the name
suggests, this family of techniques has 3 components:\
a) an "**autoregression**" component that models the relationship
between the series and it's lagged observations;\
b) a "**moving** **average**" model that models the forecast as a
function of lagged forecast errors; and\
c) an "**integrated**" component that makes the series stationary.

The model takes in the following parameter values:

p that defines the number of lags;

d that specifies the number of differences used; and

q that defines the size of moving average window

### 9.1  **Moving average model**

The moving-average model specifies that the output variable depends
linearly on the current and various past values of a stochastic
(imperfectly predictable) term.

The moving-average model is a special case and key component of the
more general ARMA and ARIMA (0, 0, q) models of time series, which
have a more complicated stochastic structure.\
**Model**: **MA(q): X<sub>t</sub> = µ + ϵ<sub>t</sub> + ϵ<sub>t-1</sub> θ<sub>1</sub>+...+ ϵ<sub>t-q</sub> θ<sub>q</sub>** -
where **μ** is the mean of the series, the ***θ*<sub>1</sub>,
\..., *θ<sub>q</sub>*** are the parameters of the model and
the ***ε<sub>t</sub>*, *ε<sub>t−1</sub>,\..., *ε<sub>t-q</sub>** are white noise error
terms. The value of ***q*** is called the order of the MA model. **I will demonstrate use of simplest MA (1) model, and later I will add more complexity.**

| __SD RES__ | __MAE__ | __RMSE__ |
|-------------|------------|------------|
| 12,806.54   | 20,977.35  | 26,129.44  |

**\
ACF function is inside dashed lines, there are no statistically significant autocorrelation, which is   positive sign**

<img src="https://github.com/IvanVanco/Time-Series-Analysis/blob/master/res/maadf.png">\
<img src="https://github.com/IvanVanco/Time-Series-Analysis/blob/master/res/maplot.png" weight="350" height="350">

### 9.2  **Autoregressive model**

The autoregressive model specifies that the output variable depends
linearly on its own previous values and on a stochastic element. Together with
the moving-average (MA) model, it is a special case and key component
of the more general autoregressive--moving-average (ARMA) and
autoregressive integrated moving average (ARIMA(p,0,0)) models of time
series. It is also a special case of the vector autoregressive model
(VAR), which consists of a system of more than one interlocking
stochastic difference equation in more than one evolving random
variable.\
**Model**: **AR(p): Y<sub>t</sub> = β<sub>0</sub>+β<sub>1</sub>y<sub>t-1</sub> +...+ β<sub>p</sub>y<sub>t-p</sub> +ϵ<sub>t</sub>** -
where **β<sub>0</sub>** is constant, the **β<sub>1</sub>**, \..., **β<sub>p</sub>**  are the
parameters of the model and the ***ε<sub>t</sub>*** is white noise. The value
of ***p*** is called the order of the AR model. **I will demonstrate use of simplest AR (1) model, and later I will add more complexity.**

| __SD RES__ | __MAE__ | __RMSE__ |
|-------------|------------|------------|
| 14,152.43   | 20,378.13  | 24,385.42  |

**\
ACF function is inside dashed lines, which is positive sign.**

<img src="https://github.com/IvanVanco/Time-Series-Analysis/blob/master/res/aradf.png">\
<img src="https://github.com/IvanVanco/Time-Series-Analysis/blob/master/res/arplot.png" weight="350" height="350">

### 9.3  **Autoregressive--moving-average model - ARMA**

The ARMA model consist of two parts. The AR part involves regressing
the variable on its own lagged (i.e., past) values. The MA part
involves modeling the error term as a linear combination of error
terms occurring contemporaneously and at various times in the past.\
**Model**: **ARMA(p, q): Y<sub>t</sub> = µ +β<sub>0</sub> +β<sub>1</sub>y<sub>t-1</sub> +...+ β<sub>p</sub>y<sub>t-p</sub>
+ϵ<sub>t</sub> + ϵ<sub>t-1</sub> θ<sub>1</sub>+...+ ϵ<sub>t-q</sub> θ<sub>q</sub>**\
**I will demonstrate use of simplest ARMA (1,1) model, which is composition of AR (1) and MA (1), to build, usually better, hybrid model. This is not final model. 
I will add more model complexity and also providing reasons for choosing particular parameters values.
**

| __SD RES__ | __MAE__ | __RMSE__ |
|-------------|------------|------------|
| 14,078.28   | 18,957.66  | 23,145.57  |

**\
ACF function is not fully inside dashed lines, which is sign there are
probably better models**

<img src="https://github.com/IvanVanco/Time-Series-Analysis/blob/master/res/armaadf.png">

<img src="https://github.com/IvanVanco/Time-Series-Analysis/blob/master/res/armaplot.png" weight="350" height="350">

### 9.4  **Autoregressive integrated moving average- ARIMA**

ARIMA models are applied in some cases where data show evidence of
non-stationarity, where an initial differencing step (corresponding to
the \"integrated\" part of the model) can be applied one or more times
to eliminate the non-stationarity.\
The I (for \"integrated\") indicates that the data values have been
replaced with the difference between their values and the previous
values (and this differencing process may have been performed more
than once). The purpose of each of these features is to make the model
fit the data as well as possible. There are two types of models:

1.  Non-seasonal ARIMA models are generally denoted ARIMA(p,d,q) where
    parameters p, d, and q are non-negative integers, p is the order
    (number of time lags) of the autoregressive model, d is the degree
    of differencing (the number of times the data have had past values
    subtracted), and q is the order of the moving-average model;

2.  Seasonal ARIMA models are usually denoted ARIMA(p,d,q)(P,D,Q)m,
    where m refers to the number of periods in each season, and the
    uppercase P,D,Q refer to the autoregressive, differencing, and
    moving average terms for the seasonal part of the ARIMA model.

    #### 9.4.1.  **ARIMA (1,0,0)(0,1,0)[12]**

I used auto.arima function from forecast package, which is good
function for exploring various types of fitted ARIMA models for
particular dataset. It will try all possible combinations of ARIMA
models by changing parameter values. We will start by first model,
that was suggested, seasonal ARIMA.

| __SD RES__ | __MAE__ | __RMSE__ |
|-------------|------------|------------|
| 13,981.54   | 11,087.77  | 15,449.20  |

**\
ACF function is inside dashed lines, which is positive sign.**

<img src="https://github.com/IvanVanco/Time-Series-Analysis/blob/master/res/arima1adf.png">

<img src="https://github.com/IvanVanco/Time-Series-Analysis/blob/master/res/arima1plot.png" weight="350" height="350">

   #### 9.4.2.  **ARIMA (0,1,3)**

Last model was suggested with few error messages. That errors pointed
that chosen seasonal unit root test encountered an error when testing
for the second difference (insufficient number of data after 1st
difference). By default, it used one seasonal difference.\
We will consider using a different seasonal unit root test (for
example Canova and Hansen (CH) test statistic)

| __SD RES__ | __MAE__ | __RMSE__ |
|-------------|------------|------------|
| 12,806.54   | 15,033.53  | 18,463.70  |

<img src="https://github.com/IvanVanco/Time-Series-Analysis/blob/master/res/arima2plot.png" weight="350" height="350">


## 10. **Conclusion - choosing best fitting model for forecasting**

<img src="https://github.com/IvanVanco/Time-Series-Analysis/blob/master/res/conclusion.png">

First four benchmark models, are good to test certain things about time
series, but are not used as final models. AR and MA models are good when
there are non-seasonal effects.

HW is clear winner here, and there are two main reasons(mainly against
ARIMA model):

1.  **Short time frame** -- 39 observation only in total;

2.  **Existence of structural fracture movements** - (cause is Corona
    virus). Structural fracture is a series of observations that is inconsistent with the previous time series. 
Corona virus has influenced on bringing several government decisions (interventions), that begun from middle of March 2020., such as:

    a.	**Some import products could not be imported into our country;**

    b.	**Sales value got decreased because sales field people work from home, and, thus, their efficiency decline down**


In more general occasions, ARIMA models are preferred. They are superior
models, and are widely used in practice.\
Also, there are many more complex model structures that are not part of
this research project, and for more in depth problem-oriented area (VAR,
GARCH, LSTM...). It would be pleasure for me to continue researching,
working and learning them .
