#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 12:48:09 2019

@author: nitinsinghal
"""

# Using Macrodata to predict Currency prices using Artificial Neural Networks (ANN) deep learning algorithm 
# For US, UK, EU, India. GBPUSD, EURUSD, USDINR
# Please refer the supporting document !!!!.docx for details

#Import libraries
import pandas as pd
# from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot
# Importing the Keras libraries for applying ANN
# import keras
from keras.models import Sequential
from keras.layers import Dense
# from keras import metrics

# Import the macro and ccy data for each country and ccy pair
# Make sure you change the file depending on currency pair you are evaluating

# Macro data - US, UK, EU, India.
usmacrodata = pd.read_csv('/Users/nitinsinghal/DropboxNew/Dropbox/MachineLearning/PublishedResearch/MacroCcyPrediction/Data/usmacrodata.csv')
eumacrodata = pd.read_csv('/Users/nitinsinghal/DropboxNew/Dropbox/MachineLearning/PublishedResearch/MacroCcyPrediction/Data/eurmacrodata.csv')
ukmacrodata = pd.read_csv('/Users/nitinsinghal/DropboxNew/Dropbox/MachineLearning/PublishedResearch/MacroCcyPrediction/Data/gbpmacrodata.csv')
inmacrodata = pd.read_csv('/Users/nitinsinghal/DropboxNew/Dropbox/MachineLearning/PublishedResearch/MacroCcyPrediction/Data/indmacrodata.csv')

# Ccy pair data - GBPUSD, EURUSD, USDINR
eurusddata = pd.read_csv('/Users/nitinsinghal/DropboxNew/Dropbox/MachineLearning/PublishedResearch/MacroCcyPrediction/Data/eurusd_Jan00Feb19.csv')
gbpusddata = pd.read_csv('/Users/nitinsinghal/DropboxNew/Dropbox/MachineLearning/PublishedResearch/MacroCcyPrediction/Data/gbpusd_Jan00Feb19.csv')
usdinrdata = pd.read_csv('/Users/nitinsinghal/DropboxNew/Dropbox/MachineLearning/PublishedResearch/MacroCcyPrediction/Data/usdinr_Jan00Feb19.csv')

# Take macro data from 2000
usmacrodata['date'] = pd.to_datetime(usmacrodata['date'])
usmacrodata = usmacrodata[(usmacrodata['date'] > '31/12/1999')]
eumacrodata['date'] = pd.to_datetime(eumacrodata['date'])
eumacrodata = eumacrodata[(eumacrodata['date'] > '31/12/1999')]
ukmacrodata['date'] = pd.to_datetime(ukmacrodata['date'])
ukmacrodata = ukmacrodata[(ukmacrodata['date'] > '31/12/1999')]
inmacrodata['date'] = pd.to_datetime(inmacrodata['date'])
inmacrodata = inmacrodata[(inmacrodata['date'] > '31/12/1999')]

# merge us eu, us uk, us inr macro data files
useumacrodata = pd.merge(usmacrodata, eumacrodata, how='left', on='date')
usukmacrodata = pd.merge(usmacrodata, ukmacrodata, how='left', on='date')
usinmacrodata = pd.merge(usmacrodata, inmacrodata, how='left', on='date')

#Change data format for matching with macro data 
eurusddata['Date'] = pd.to_datetime(eurusddata['date'])
eurusddata['Date'] = eurusddata['Date'].dt.strftime('%d/%m/%Y')
eurusddata = eurusddata.drop(['date'],axis=1)
gbpusddata['Date'] = pd.to_datetime(gbpusddata['date'])
gbpusddata['Date'] = gbpusddata['Date'].dt.strftime('%d/%m/%Y')
gbpusddata = gbpusddata.drop(['date'],axis=1)
usdinrdata['Date'] = pd.to_datetime(usdinrdata['date'])
usdinrdata['Date'] = usdinrdata['Date'].dt.strftime('%d/%m/%Y')
usdinrdata = usdinrdata.drop(['date'],axis=1)

# Function to Convert macrodata from monthly to daily
# Using resampling with forward fill of last known monthly value

def convert_monthly_to_daily(macrodata):
    macromonthlydata = macrodata.copy()
    print(type(macromonthlydata))
    macromonthlydata = macromonthlydata.drop_duplicates(subset=['date'], keep='first').copy()
    macromonthlydata = macromonthlydata.replace(to_replace=0, method='ffill')
    macromonthlydata = macromonthlydata.set_index(pd.DatetimeIndex(macromonthlydata['date']))
    macromonthlydata = macromonthlydata.drop(['date'],axis=1)
    
    # Resampling is missing in last 2 months so duplicate last month and add to last+1 month
    # DatetimeIndex uses Ymd format whereas shift treats data as Ydm
    # So shift last row to month end, then increment by 1 day to roll to next month
    # Now system recognises the correct data format for next month
    lastrow = macromonthlydata.tail(1)
    lastrowme = lastrow.shift(1, freq='M')
    lastrowmend = lastrowme.shift(1, freq='D')
    lastrowme1 = lastrow.shift(2, freq='M')
    lastrowde1nd = lastrowme1.shift(1, freq='D')
    
    # Drop the last row from macro data and then add copy as next 2 months
    macromonthlydata = macromonthlydata.drop(macromonthlydata.index[-1:])
    macromonthlydata = macromonthlydata.append(lastrowmend)
    macromonthlydata = macromonthlydata.append(lastrowde1nd)
    
    # Now resample daily using business day, forward filling with each months value
    # useumacrodailydata = macromonthlydata.resample('B').interpolate(method='linear')
    macrodailydata = macromonthlydata.resample('B').fillna(method='ffill')
    macrodailydata['DateKey'] = macrodailydata.index
    # Extra date for last+1 month which is out of real data date range is not required.
    macrodailydata = macrodailydata.drop(macrodailydata.index[-1:])
    macrodailydata = macrodailydata[1:]
    macrodailydata = macrodailydata[(macrodailydata['DateKey'] <= '2019-02-25')]

    return macrodailydata

# Apply the convert_monthly_to_daily function to each macro dataframe
useumacrodailydata = convert_monthly_to_daily(useumacrodata)
usukmacrodailydata = convert_monthly_to_daily(usukmacrodata)
usinmacrodailydata = convert_monthly_to_daily(usinmacrodata)

# Drop unwanted columns from merged macro data files
# useumacrodatafinal = useumacrodata.drop(['', ], axis=1)
# usukmacrodatafinal = usukmacrodata.drop(['', ], axis=1)
# usinmacrodatafinal = usinmacrodata.drop(['', ], axis=1)
# 'households_debt_to_gdp','stock_market_x', 'unemployed_persons', 'youth_unemployment_rate', 'record_month_y', 'record_year_y'

# Add SMA to ccy data
eurusddata['sma'] = eurusddata['eurusd'].rolling(window=30).mean()
eurusddata['DateKey'] = pd.to_datetime(eurusddata['Date'])
eurusddata = eurusddata.drop(['Date'],axis=1)
gbpusddata['sma'] = gbpusddata['gbpusd'].rolling(window=30).mean()
gbpusddata['DateKey'] = pd.to_datetime(gbpusddata['Date'])
gbpusddata = gbpusddata.drop(['Date'],axis=1)
usdinrdata['sma'] = usdinrdata['usdinr'].rolling(window=30).mean()
usdinrdata['DateKey'] = pd.to_datetime(usdinrdata['Date'])
usdinrdata = usdinrdata.drop(['Date'],axis=1)

# Merge ccy and macro data unsing actual date, not index key
mergedeurusdmacrodailydata = pd.merge(eurusddata, useumacrodailydata, how='left', on='DateKey')
mergedeurusdmacrodailydata = mergedeurusdmacrodailydata.fillna(0)
mergedgbpusdmacrodailydata = pd.merge(gbpusddata, usukmacrodailydata, how='left', on='DateKey')
mergedgbpusdmacrodailydata = mergedgbpusdmacrodailydata.fillna(0)
mergedusdinrmacrodailydata = pd.merge(usdinrdata, usinmacrodailydata, how='left', on='DateKey')
mergedusdinrmacrodailydata = mergedusdinrmacrodailydata.fillna(0)

######## Run Artificial Neural Networks Algo on EURUSD with US EU macro data ##########
# Create arrays of dependet and independent variables
# X = array of all independent variables (macro data)
# y is the dependent variable - ccy price stats
Xeurusd = mergedeurusdmacrodailydata.iloc[:, 3:79].values

yeurusddaily = mergedeurusdmacrodailydata.iloc[:,0].values
yeurusdsma = mergedeurusdmacrodailydata.iloc[:, 1].values

# EURUSD Daily -  ANN Algo run 
X_train_eurusddaily, X_test_eurusddaily, eurusddaily_train, eurusddaily_test = train_test_split(Xeurusd, yeurusddaily, test_size = 0.25)

# Feature Scaling the data to normalise it
# Only training data is standardized (fit) to get the parameters (mean, std dev) 
# The training data parameters are then used to scale (transform) test data
sc = StandardScaler()
X_train_eurusddaily = sc.fit_transform(X_train_eurusddaily)
X_test_eurusddaily = sc.transform(X_test_eurusddaily)

# Initialising the ANN
eurusdannregressor = Sequential()

# Adding the input layer and the first hidden layer
eurusdannregressor.add(Dense(units = 32, activation = 'relu', kernel_initializer = 'normal', input_dim = 76))

# Adding the second hidden layer
eurusdannregressor.add(Dense(units = 16, kernel_initializer = 'normal', activation = 'relu'))

# Adding the output layer
eurusdannregressor.add(Dense(units = 1, kernel_initializer = 'normal'))

# Compiling the ANN
# For continuous values, regression based ANN use sgd (Stochastic gradient descent optimizer)
# It Includes support for momentum, learning rate decay, and Nesterov momentum.
eurusdannregressor.compile(optimizer = 'sgd', loss = 'mean_squared_error', metrics = ['mse', 'mae', 'mape'])

# Fitting the ANN to the Training set
historyeurusd = eurusdannregressor.fit(X_train_eurusddaily, eurusddaily_train, batch_size = 10, epochs = 100)

# Making the predictions using ANN and evaluating the model
# Predicting the Test set results
eurusddaily_pred = eurusdannregressor.predict(X_test_eurusddaily)

# Quality of predictions - MSE,MAE, MAPE. 
# MSE measures square of difference between estimated and actual values
# plot metrics
pyplot.plot(historyeurusd.history['mean_squared_error'])
pyplot.plot(historyeurusd.history['mean_absolute_error'])
pyplot.plot(historyeurusd.history['mean_absolute_percentage_error'])
pyplot.show()


######## Run Artificial Neural Networks Algo on GBPUSD with US UK macro data ##########
# Create arrays of dependet and independent variables
# X = array of all independent variables (macro data)
# y is the dependent variable - ccy price stats
Xgbpusd = mergedgbpusdmacrodailydata.iloc[:, 3:80].values

ygbpusddaily = mergedgbpusdmacrodailydata.iloc[:,0].values
ygbpusdsma = mergedgbpusdmacrodailydata.iloc[:, 1].values

# gbpusd Daily -  ANN Algo run 
X_train_gbpusddaily, X_test_gbpusddaily, gbpusddaily_train, gbpusddaily_test = train_test_split(Xgbpusd, ygbpusddaily, test_size = 0.25)

# Feature Scaling the data to normalise it
# Only training data is standardized (fit) to get the parameters (mean, std dev) 
# The training data parameters are then used to scale (transform) test data
sc = StandardScaler()
X_train_gbpusddaily = sc.fit_transform(X_train_gbpusddaily)
X_test_gbpusddaily = sc.transform(X_test_gbpusddaily)

# Initialising the ANN
gbpusdannregressor = Sequential()

# Adding the input layer and the first hidden layer
gbpusdannregressor.add(Dense(units = 32, activation = 'relu', kernel_initializer = 'normal', input_dim = 77))

# Adding the second hidden layer
gbpusdannregressor.add(Dense(units = 16, kernel_initializer = 'normal', activation = 'relu'))

# Adding the output layer
gbpusdannregressor.add(Dense(units = 1, kernel_initializer = 'normal'))

# Compiling the ANN
# For continuous values, regression based ANN use sgd (Stochastic gradient descent optimizer)
# It Includes support for momentum, learning rate decay, and Nesterov momentum.


gbpusdannregressor.compile(optimizer = 'sgd', loss = 'mean_squared_error', metrics = ['mse', 'mae', 'mape'])

# Fitting the ANN to the Training set
historygbpusd = gbpusdannregressor.fit(X_train_gbpusddaily, gbpusddaily_train, batch_size = 10, epochs = 100)

# Making the predictions using ANN and evaluating the model

# Predicting the Test set results
gbpusddaily_pred = gbpusdannregressor.predict(X_test_gbpusddaily)

# Quality of predictions - MSE,MAE, MAPE. 
# MSE measures square of difference between estimated and actual values
# plot metrics
pyplot.plot(historygbpusd.history['mean_squared_error'])
pyplot.plot(historygbpusd.history['mean_absolute_error'])
pyplot.plot(historygbpusd.history['mean_absolute_percentage_error'])
pyplot.show()


######## Run Artificial Neural Networks Algo on USDINR with US IN macro data ##########
# Create arrays of dependet and independent variables
# X = array of all independent variables (macro data)
# y is the dependent variable - ccy price stats
Xusdinr = mergedusdinrmacrodailydata.iloc[:, 3:63].values

yusdinrdaily = mergedusdinrmacrodailydata.iloc[:,0].values
yusdinrsma = mergedusdinrmacrodailydata.iloc[:, 1].values

# usdinr Daily -  ANN Algo run 
X_train_usdinrdaily, X_test_usdinrdaily, usdinrdaily_train, usdinrdaily_test = train_test_split(Xusdinr, yusdinrdaily, test_size = 0.25)

# Feature Scaling the data to normalise it
# Only training data is standardized (fit) to get the parameters (mean, std dev) 
# The training data parameters are then used to scale (transform) test data
sc = StandardScaler()
X_train_usdinrdaily = sc.fit_transform(X_train_usdinrdaily)
X_test_usdinrdaily = sc.transform(X_test_usdinrdaily)

# Initialising the ANN
usdinrannregressor = Sequential()

# Adding the input layer and the first hidden layer
usdinrannregressor.add(Dense(units = 32, activation = 'relu', kernel_initializer = 'normal', input_dim = 60))

# Adding the second hidden layer
usdinrannregressor.add(Dense(units = 16, kernel_initializer = 'normal', activation = 'relu'))

# Adding the output layer
usdinrannregressor.add(Dense(units = 1, kernel_initializer = 'normal'))

# Compiling the ANN
# For continuous values, regression based ANN use sgd (Stochastic gradient descent optimizer)
# It Includes support for momentum, learning rate decay, and Nesterov momentum.


usdinrannregressor.compile(optimizer = 'sgd', loss = 'mean_squared_error', metrics = ['mse', 'mae', 'mape'])

# Fitting the ANN to the Training set
historyusdinr = usdinrannregressor.fit(X_train_usdinrdaily, usdinrdaily_train, batch_size = 10, epochs = 100)

# Making the predictions using ANN and evaluating the model

# Predicting the Test set results
usdinrdaily_pred = usdinrannregressor.predict(X_test_usdinrdaily)

# Quality of predictions - MSE,MAE, MAPE. 
# MSE measures square of difference between estimated and actual values
# plot metrics
pyplot.plot(historyusdinr.history['mean_squared_error'])
pyplot.plot(historyusdinr.history['mean_absolute_error'])
pyplot.plot(historyusdinr.history['mean_absolute_percentage_error'])
pyplot.show()
