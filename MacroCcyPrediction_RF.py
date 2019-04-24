#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 16:53:51 2019

@author: nitinsinghal
"""

# Using Macrodata to predict Currency prices using Random Forest Regression
# For US, UK, EU, India. GBPUSD, EURUSD, USDINR
# Please refer the supporting document !!!!.docx for details

#Import libraries
import pandas as pd
#from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
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


######## Run Random Forest Algo on EURUSD with US EU macro data ##########
# Create arrays of dependent and independent variables
# X = array of all independent variables (macro data)
# y is the dependent variable - ccy price stats

# Running with all featuress
Xeurusd = mergedeurusdmacrodailydata.iloc[:, 3:79].values

yeurusddaily = mergedeurusdmacrodailydata.iloc[:,0].values
yeurusdsma = mergedeurusdmacrodailydata.iloc[:, 1].values

# EURUSD Daily SMA -  RF Algo run 
X_train_eurusdsma, X_test_eurusdsma, eurusdsma_train, eurusdsma_test = train_test_split(Xeurusd, yeurusdsma, test_size = 0.25)

# Feature Scaling the data to normalise it
# Only training data is standardized (fit) to get the parameters (mean, std dev) 
# The training data parameters are then used to scale (transform) test data
sc = StandardScaler()
X_train_eurusdsma = sc.fit_transform(X_train_eurusdsma)
X_test_eurusdsma = sc.transform(X_test_eurusdsma)

# Fitting Random Forest Regression to the dataset
# You can change the n_estimators and other parameters to get best fit 
eurusdsmarfreg = RandomForestRegressor(n_estimators=100, criterion='mse', min_samples_leaf=5, max_depth=10, 
                                min_samples_split=10, max_features=8, n_jobs=-1)
eurusdsmarfreg.fit(X_train_eurusdsma, eurusdsma_train)

# Predicting sma result with Random Forest Regression
eurusdsma_pred = eurusdsmarfreg.predict(X_test_eurusdsma)

# Check the importance of each feature
eurusdsmaimp = eurusdsmarfreg.feature_importances_
eurusdsmanfeat = eurusdsmarfreg.n_features_
# Quality of predictions - MSE,MAE, R2. MSE measures square of difference between estimated and actual values
# R2 measures how well future samples are likely to be predicted by the model, max 1
eurusdsmamse = mean_squared_error(eurusdsma_test , eurusdsma_pred)
eurusdsmamae = mean_absolute_error(eurusdsma_test , eurusdsma_pred)
eurusdsmar2 = r2_score(eurusdsma_test , eurusdsma_pred)

# EURUSD Daily -  RF Algo run 
X_train_eurusddaily, X_test_eurusddaily, eurusddaily_train, eurusddaily_test = train_test_split(Xeurusd, yeurusddaily, test_size = 0.25)

# Feature Scaling the data to normalise it
# Only training data is standardized (fit) to get the parameters (mean, std dev) 
# The training data parameters are then used to scale (transform) test data
sc = StandardScaler()
X_train_eurusddaily = sc.fit_transform(X_train_eurusddaily)
X_test_eurusddaily = sc.transform(X_test_eurusddaily)

# Fitting Random Forest Regression to the dataset
# You can change the n_estimators and other parameters to get best fit 
eurusddailyrfreg = RandomForestRegressor(n_estimators=100, criterion='mse', min_samples_leaf=5, max_depth=10,  
                                min_samples_split=10, max_features=8, n_jobs=-1)
eurusddailyrfreg.fit(X_train_eurusddaily, eurusddaily_train)

# Predicting sma result with Random Forest Regression
eurusddaily_pred = eurusddailyrfreg.predict(X_test_eurusddaily)

# Check the importance of each feature
eurusddailyimp = eurusddailyrfreg.feature_importances_
eurusddailynfeat = eurusddailyrfreg.n_features_
# Quality of predictions - MSE,MAE, R2. MSE measures square of difference between estimated and actual values
# R2 measures how well future samples are likely to be predicted by the model, max 1
eurusddailymse = mean_squared_error(eurusddaily_test , eurusddaily_pred)
eurusddailymae = mean_absolute_error(eurusddaily_test , eurusddaily_pred)
eurusddailyr2 = r2_score(eurusddaily_test , eurusddaily_pred)


######## Run Random Forest Algo on GBPUSD with US UK macro data ##########
# Create arrays of dependent and independent variables
# X = array of all independent variables (macro data)
# y is the dependent variable - ccy price stats
Xgbpusd = mergedgbpusdmacrodailydata.iloc[:, 3:80].values

ygbpusddaily = mergedgbpusdmacrodailydata.iloc[:,0].values
ygbpusdsma = mergedgbpusdmacrodailydata.iloc[:, 1].values

# GBPUSD Daily SMA -  RF Algo run 
X_train_gbpusdsma, X_test_gbpusdsma, gbpusdsma_train, gbpusdsma_test = train_test_split(Xgbpusd, ygbpusdsma, test_size = 0.25)

# Feature Scaling the data to normalise it
# Only training data is standardized (fit) to get the parameters (mean, std dev) 
# The training data parameters are then used to scale (transform) test data
sc = StandardScaler()
X_train_gbpusdsma = sc.fit_transform(X_train_gbpusdsma)
X_test_gbpusdsma = sc.transform(X_test_gbpusdsma)

# Fitting Random Forest Regression to the dataset
# You can change the n_estimators and other parameters to get best fit 
gbpusdrfreg = RandomForestRegressor(n_estimators=100, criterion='mse', min_samples_leaf=5, max_depth=10, 
                                min_samples_split=10, max_features=8, n_jobs=-1)
gbpusdrfreg.fit(X_train_gbpusdsma, gbpusdsma_train)

# Predicting sma result with Random Forest Regression
gbpusdsma_pred = gbpusdrfreg.predict(X_test_gbpusdsma)

# Check the importance of each feature
gbpusdimp = gbpusdrfreg.feature_importances_
gbpusdnfeat = gbpusdrfreg.n_features_
# Quality of predictions - MSE,MAE, R2. MSE measures square of difference between estimated and actual values
# R2 measures how well future samples are likely to be predicted by the model, max 1
gbpusdmse = mean_squared_error(gbpusdsma_test , gbpusdsma_pred)
gbpusdmae = mean_absolute_error(gbpusdsma_test , gbpusdsma_pred)
gbpusdr2 = r2_score(gbpusdsma_test , gbpusdsma_pred)

# GBPUSD Daily -  RF Algo run 
X_train_gbpusddaily, X_test_gbpusddaily, gbpusddaily_train, gbpusddaily_test = train_test_split(Xgbpusd, ygbpusddaily, test_size = 0.25)

# Feature Scaling the data to normalise it
# Only training data is standardized (fit) to get the parameters (mean, std dev) 
# The training data parameters are then used to scale (transform) test data
sc = StandardScaler()
X_train_gbpusddaily = sc.fit_transform(X_train_gbpusddaily)
X_test_gbpusddaily = sc.transform(X_test_gbpusddaily)

# Fitting Random Forest Regression to the dataset
# You can change the n_estimators and other parameters to get best fit 
gbpusddailyrfreg = RandomForestRegressor(n_estimators=100, criterion='mse', min_samples_leaf=5, max_depth=10,  
                                min_samples_split=10, max_features=8, n_jobs=-1)
gbpusddailyrfreg.fit(X_train_gbpusddaily, gbpusddaily_train)

# Predicting sma result with Random Forest Regression
gbpusddaily_pred = gbpusddailyrfreg.predict(X_test_gbpusddaily)

# Check the importance of each feature
gbpusddailyimp = gbpusddailyrfreg.feature_importances_
gbpusddailynfeat = gbpusddailyrfreg.n_features_
# Quality of predictions - MSE,MAE, R2. MSE measures square of difference between estimated and actual values
# R2 measures how well future samples are likely to be predicted by the model, max 1
gbpusddailymse = mean_squared_error(gbpusddaily_test , gbpusddaily_pred)
gbpusddailymae = mean_absolute_error(gbpusddaily_test , gbpusddaily_pred)
gbpusddailyr2 = r2_score(gbpusddaily_test , gbpusddaily_pred)


######## Run Random Forest Algo on USDINR with US IN macro data ##########
# Create arrays of dependent and independent variables
# X = array of all independent variables (macro data)
# y is the dependent variable - ccy price stats
Xusdinr = mergedusdinrmacrodailydata.iloc[:, 3:63].values

yusdinrdaily = mergedusdinrmacrodailydata.iloc[:,0].values
yusdinrsma = mergedusdinrmacrodailydata.iloc[:, 1].values

# USDINR Daily SMA -  RF Algo run 
X_train_usdinrsma, X_test_usdinrsma, usdinrsma_train, usdinrsma_test = train_test_split(Xusdinr, yusdinrsma, test_size = 0.25)

# Feature Scaling the data to normalise it
# Only training data is standardized (fit) to get the parameters (mean, std dev) 
# The training data parameters are then used to scale (transform) test data
sc = StandardScaler()
X_train_usdinrsma = sc.fit_transform(X_train_usdinrsma)
X_test_usdinrsma = sc.transform(X_test_usdinrsma)

# Fitting Random Forest Regression to the dataset
# You can change the n_estimators and other parameters to get best fit
usdinrrfreg = RandomForestRegressor(n_estimators=100, criterion='mse', min_samples_leaf=5, max_depth=10, 
                                min_samples_split=10, max_features=8, n_jobs=-1)
usdinrrfreg.fit(X_train_usdinrsma, usdinrsma_train)

# Predicting sma result with Random Forest Regression
usdinrsma_pred = usdinrrfreg.predict(X_test_usdinrsma)

# Check the importance of each feature
usdinrimp = usdinrrfreg.feature_importances_
usdinrnfeat = usdinrrfreg.n_features_
# Quality of predictions - MSE,MAE, R2. MSE measures square of difference between estimated and actual values
# R2 measures how well future samples are likely to be predicted by the model, max 1
usdinrmse = mean_squared_error(usdinrsma_test , usdinrsma_pred)
usdinrmae = mean_absolute_error(usdinrsma_test , usdinrsma_pred)
usdinrr2 = r2_score(usdinrsma_test , usdinrsma_pred)

# USDINR Daily -  RF Algo run 
X_train_usdinrdaily, X_test_usdinrdaily, usdinrdaily_train, usdinrdaily_test = train_test_split(Xusdinr, yusdinrdaily, test_size = 0.25)

# Feature Scaling the data to normalise it
# Only training data is standardized (fit) to get the parameters (mean, std dev) 
# The training data parameters are then used to scale (transform) test data
sc = StandardScaler()
X_train_usdinrdaily = sc.fit_transform(X_train_usdinrdaily)
X_test_usdinrdaily = sc.transform(X_test_usdinrdaily)

# Fitting Random Forest Regression to the dataset
# You can change the n_estimators and other parameters to get best fit 
usdinrdailyrfreg = RandomForestRegressor(n_estimators=100, criterion='mse', min_samples_leaf=5, max_depth=10,  
                                min_samples_split=10, max_features=8, n_jobs=-1)
usdinrdailyrfreg.fit(X_train_usdinrdaily, usdinrdaily_train)

# Predicting sma result with Random Forest Regression
usdinrdaily_pred = usdinrdailyrfreg.predict(X_test_usdinrdaily)

# Check the importance of each feature
usdinrdailyimp = usdinrdailyrfreg.feature_importances_
usdinrdailynfeat = usdinrdailyrfreg.n_features_
# Quality of predictions - MSE,MAE, R2. MSE measures square of difference between estimated and actual values
# R2 measures how well future samples are likely to be predicted by the model, max 1
usdinrdailymse = mean_squared_error(usdinrdaily_test , usdinrdaily_pred)
usdinrdailymae = mean_absolute_error(usdinrdaily_test , usdinrdaily_pred)
usdinrdailyr2 = r2_score(usdinrdaily_test , usdinrdaily_pred)

