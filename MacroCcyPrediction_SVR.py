#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 12:39:23 2019

@author: nitinsinghal
"""

# Using Macrodata to predict Currency prices using Support Vector Regression.
# For US, UK, EU, India. GBPUSD, EURUSD, USDINR
# Please refer the supporting document !!!!.docx for details

#Import libraries
import pandas as pd
# from datetime import datetime
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

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

######## Run Support Vector Regression Algo on EURUSD with US EU macro data ##########
# Create arrays of dependet and independent variables
# X = array of all independent variables (macro data)
# y is the dependent variable - ccy price stats
Xeurusd = mergedeurusdmacrodailydata.iloc[:, 3:79].values

yeurusddaily = mergedeurusdmacrodailydata.iloc[:,0].values
yeurusdsma = mergedeurusdmacrodailydata.iloc[:, 1].values

# EURUSD daily -  SVR Algo run 
X_train_eurusddaily, X_test_eurusddaily, eurusddaily_train, eurusddaily_test = train_test_split(Xeurusd, yeurusddaily, test_size = 0.25)

# Feature Scaling the data to normalise it
# Only training data is standardized (fit) to get the parameters (mean, std dev) 
# The training data parameters are then used to scale (transform) test data
sc = StandardScaler()
X_train_eurusddaily = sc.fit_transform(X_train_eurusddaily)
X_test_eurusddaily = sc.transform(X_test_eurusddaily)

# Fitting SVR algorithm to the Training set
# using SVR as predicted values are continues which requires regression not classification(SVM)
eurusdsvrlassifier = SVR(kernel = 'rbf', gamma='scale')
eurusdsvrlassifier.fit(X_train_eurusddaily, eurusddaily_train)

# Predicting the Test set results using the SVM algorithm
eurusddaily_pred = eurusdsvrlassifier.predict(X_test_eurusddaily)

# Quality of predictions - MSE,MAE, R2. MSE measures square of difference between estimated and actual values
# R2 measures how well future samples are likely to be predicted by the model, max 1
eurusdmse = mean_squared_error(eurusddaily_test , eurusddaily_pred)
eurusdmae = mean_absolute_error(eurusddaily_test , eurusddaily_pred)
eurusdr2 = r2_score(eurusddaily_test , eurusddaily_pred)

######## Run Support Vector Regression Algo on GBPUSD with US UK macro data ##########
# Create arrays of dependet and independent variables
# X = array of all independent variables (macro data)
# y is the dependent variable - ccy price stats
Xgbpusd = mergedgbpusdmacrodailydata.iloc[:, 3:80].values
ygbpusddaily = mergedgbpusdmacrodailydata.iloc[:,0].values
ygbpusdsma = mergedgbpusdmacrodailydata.iloc[:, 1].values

# GBPUSD daily -  SVR Algo run 
X_train_gbpusddaily, X_test_gbpusddaily, gbpusddaily_train, gbpusddaily_test = train_test_split(Xgbpusd, ygbpusddaily, test_size = 0.25)

# Feature Scaling the data to normalise it
# Only training data is standardized (fit) to get the parameters (mean, std dev) 
# The training data parameters are then used to scale (transform) test data
sc = StandardScaler()
X_train_gbpusddaily = sc.fit_transform(X_train_gbpusddaily)
X_test_gbpusddaily = sc.transform(X_test_gbpusddaily)

# Fitting SVR algorithm to the Training set
# using SVR as predicted values are continues which requires regression not classification(SVM)
gbpusdsvrlassifier = SVR(kernel = 'rbf', gamma='auto')
gbpusdsvrlassifier.fit(X_train_gbpusddaily, gbpusddaily_train)

# Predicting the Test set results using the SVM algorithm
gbpusddaily_pred = gbpusdsvrlassifier.predict(X_test_gbpusddaily)

# Quality of predictions - MSE,MAE, R2. MSE measures square of difference between estimated and actual values
# R2 measures how well future samples are likely to be predicted by the model, max 1
gbpusdmse = mean_squared_error(gbpusddaily_test , gbpusddaily_pred)
gbpusdmae = mean_absolute_error(gbpusddaily_test , gbpusddaily_pred)
gbpusdr2 = r2_score(gbpusddaily_test , gbpusddaily_pred)

######## Run Support Vector Regression Algo on USDINR with US IN macro data ##########
# Create arrays of dependet and independent variables
# X = array of all independent variables (macro data)
# y is the dependent variable - ccy price stats
Xusdinr = mergedusdinrmacrodailydata.iloc[:, 3:63].values
yusdinrdaily = mergedusdinrmacrodailydata.iloc[:,0].values
yusdinrsma = mergedusdinrmacrodailydata.iloc[:,1].values

# USDINR Daily  -  SVR Algo run 
X_train_usdinrdaily, X_test_usdinrdaily, usdinrdaily_train, usdinrdaily_test = train_test_split(Xusdinr, yusdinrdaily, test_size = 0.25)

# Feature Scaling the data to normalise it
# Only training data is standardized (fit) to get the parameters (mean, std dev) 
# The training data parameters are then used to scale (transform) test data
sc = StandardScaler()
X_train_usdinrdaily = sc.fit_transform(X_train_usdinrdaily)
X_test_usdinrdaily = sc.transform(X_test_usdinrdaily)

# Fitting SVR algorithm to the Training set
# using SVR as predicted values are continues which requires regression not classification(SVM)
usdinrsvrlassifier = SVR(kernel = 'rbf', gamma='scale')
usdinrsvrlassifier.fit(X_train_usdinrdaily, usdinrdaily_train)

# Predicting the Test set results using the SVM algorithm
usdinrdaily_pred = usdinrsvrlassifier.predict(X_test_usdinrdaily)

# Quality of predictions - MSE,MAE, R2. MSE measures square of difference between estimated and actual values
# R2 measures how well future samples are likely to be predicted by the model, max 1
usdinrmse = mean_squared_error(usdinrdaily_test , usdinrdaily_pred)
usdinrmae = mean_absolute_error(usdinrdaily_test , usdinrdaily_pred)
usdinrr2 = r2_score(usdinrdaily_test , usdinrdaily_pred)

