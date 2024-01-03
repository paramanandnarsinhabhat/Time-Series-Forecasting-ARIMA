#Importing Required Libraries
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt


from sklearn.metrics import mean_squared_error
from math import sqrt
from statistics import mean 

import warnings
warnings.filterwarnings("ignore")

train_data = pd.read_csv("/Users/paramanandbhat/Downloads/7.3_ARIMA_and_SARIMA_models/data/train_data.csv")
valid_data = pd.read_csv("/Users/paramanandbhat/Downloads/7.3_ARIMA_and_SARIMA_models/data/valid_data.csv")

print(train_data.shape)
print(train_data.head())

print(valid_data.shape)
print(valid_data.head())

# Required Preprocessing 
train_data.timestamp = pd.to_datetime(train_data['Date'],format='%Y-%m-%d')
train_data.index = train_data.timestamp

valid_data.timestamp = pd.to_datetime(valid_data['Date'],format='%Y-%m-%d')
valid_data.index = valid_data.timestamp


train_data.index = pd.DatetimeIndex(train_data.index, freq='D')
valid_data.index = pd.DatetimeIndex(valid_data.index, freq='D')


plt.figure(figsize=(12,8))

plt.plot(train_data.index, train_data['count'], label='train_data')
plt.plot(valid_data.index,valid_data['count'], label='valid')
plt.legend(loc='best')
plt.title("Train and Validation Data")
plt.show()

# dickey fuller, KPSS
from statsmodels.tsa.stattools import adfuller, kpss

def adf_test(timeseries):
    
    #Perform Dickey-Fuller test:
    print ('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput=pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])

    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print (dfoutput)

adf_test(train_data['count'])


def kpss_test(timeseries):
    print ('Results of KPSS Test:')
    kpsstest = kpss(timeseries, regression='c')
    kpss_output = pd.Series(kpsstest[0:3], index=['Test Statistic','p-value','Lags Used'])
    for key,value in kpsstest[3].items():
        kpss_output['Critical Value (%s)'%key] = value
    print (kpss_output)


kpss_test(train_data['count'])

# Making Series Stationary
train_data['count_diff'] = train_data['count'] - train_data['count'].shift(1)

plt.figure(figsize=(12,8))

plt.plot(train_data.index, train_data['count'], label='train_data')
plt.plot(train_data.index,train_data['count_diff'], label='stationary series')
plt.legend(loc='best')
plt.title("Stationary Series")
plt.show()

train_data['count_log'] = np.log(train_data['count'])
train_data['count_log_diff'] = train_data['count_log'] - train_data['count_log'].shift(1)

plt.figure(figsize=(12,8))

plt.plot(train_data.index,train_data['count_log_diff'], label='stationary series')
plt.legend(loc='best')
plt.title("Stationary Series")
plt.show()


# ACF and PACF plots
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
plot_acf(train_data['count_log_diff'].dropna(), lags=15)
plot_pacf(train_data['count_log_diff'].dropna(), lags=15)
plt.show()

from statsmodels.tsa.arima.model import ARIMA

# fit model
model = ARIMA(train_data['count_log'], order=(2,1,2))
model_fit = model.fit()

print(valid_data.shape)

# Generate forecasts
output = model_fit.forecast(steps=184)



# 'output' is already a numpy array, so we can assign it directly
valid_data['Arima'] = np.exp(output)


plt.figure(figsize=(12,8))

plt.plot(train_data.index, train_data['count'], label='train_data')
plt.plot(valid_data.index, valid_data['count'], label='valid')
plt.plot(valid_data.index, valid_data['Arima'], label='predicted')

plt.legend(loc='best')
plt.title("ARIMA model")
plt.show()


# calculating RMSE 
rmse = sqrt(mean_squared_error(valid_data['count'], valid_data['Arima']))
print('The RMSE value for ARIMA is', rmse)



