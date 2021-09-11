# -*- coding: utf-8 -*-
"""
Created on Sat Feb  9 09:53:34 2019

@author: Kleyton
"""

import pandas as pd
import statsmodels.formula.api as smf
import numpy as np
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

# stock market
aord = pd.DataFrame.from_csv('C:/Users/Dinho/Desktop/Arquivos/4 Finance/3 Financial Analysis/2 StockData/Index/AORD.csv')
nikkei = pd.DataFrame.from_csv('C:/Users/Dinho/Desktop/Arquivos/4 Finance/3 Financial Analysis/2 StockData/Index/N225.csv')
hsi = pd.DataFrame.from_csv('C:/Users/Dinho/Desktop/Arquivos/4 Finance/3 Financial Analysis/2 StockData/Index/HSI.csv')
daxi = pd.DataFrame.from_csv('C:/Users/Dinho/Desktop/Arquivos/4 Finance/3 Financial Analysis/2 StockData/Index/DAX.csv')
cac40 = pd.DataFrame.from_csv('C:/Users/Dinho/Desktop/Arquivos/4 Finance/3 Financial Analysis/2 StockData/Index/CAC40.csv')
sp500 = pd.DataFrame.from_csv('C:/Users/Dinho/Desktop/Arquivos/4 Finance/3 Financial Analysis/2 StockData/Index/SP500.csv')
dji = pd.DataFrame.from_csv('C:/Users/Dinho/Desktop/Arquivos/4 Finance/3 Financial Analysis/2 StockData/Index/DJI.csv')
nasdaq = pd.DataFrame.from_csv('C:/Users/Dinho/Desktop/Arquivos/4 Finance/3 Financial Analysis/2 StockData/Index/NASDAQ.csv')
spy = pd.DataFrame.from_csv('C:/Users/Dinho/Desktop/Arquivos/4 Finance/3 Financial Analysis/2 StockData/Index/SPY.csv')

#Step 1 = Data Munging
# Due to the timezone issues, we extract and calculate appropriate stock market data for analysis
# Indicepanel is the DataFrame of our trading model
indicepanel = pd.DataFrame(index = spy.index)

indicepanel['spy'] = spy['Open'].shift(-1) - spy['Open']
indicepanel['spy_lag1']=indicepanel['spy'].shift(1)
indicepanel['sp500']=sp500["Open"]-sp500['Open'].shift(1)
indicepanel['nasdaq']=nasdaq['Open']-nasdaq['Open'].shift(1)
indicepanel['dji']=dji['Open']-dji['Open'].shift(1)

indicepanel['cac40']=cac40['Open']-cac40['Open'].shift(1)
indicepanel['daxi']=daxi['Open']-daxi['Open'].shift(1)

indicepanel['aord']=aord['Close']-aord['Open']
indicepanel['hsi']=hsi['Close']-hsi['Open']
indicepanel['nikkei']=nikkei['Close']-nikkei['Open']
indicepanel['Price']=spy['Open']

print(indicepanel.head())

# Lets check whether do we have NaN values in indicepanel
print(indicepanel.isnull().sum())

# save this indicepanel for part 4.5
path_save = 'C:/Users/Dinho/Desktop/Arquivos/4 Finance/3 Financial Analysis/2 StockData/Index/indicepanel.csv'
indicepanel.to_csv(path_save)

print(indicepanel.shape)

#split the data into (1)train set and (2)test set

Train = indicepanel.iloc[-2000:-1000, :]
Test = indicepanel.iloc[-1000:, :]
print(Train.shape, Test.shape)

# Generate scatter matrix among all stock markets (and the price of SPY) to observe the association

from pandas.tools.plotting import scatter_matrix
sm = scatter_matrix(Train, figsize=(10, 10))

# Find the indice with largest correlation
corr_array = Train.iloc[:, :-1].corr()['spy']
print(corr_array)

formula = 'spy~spy_lag1+sp500+nasdaq+dji+cac40+aord+daxi+nikkei+hsi'
lm = smf.ols(formula=formula, data=Train).fit()
lm.summary()

#Make Prediction
Train['PredictedY'] = lm.predict(Train)
Test['PredictedY'] = lm.predict(Test)
plt.scatter(Train['spy'], Train['PredictedY'])

# RMSE - Root Mean Squared Error, Adjusted R^2
def adjustedMetric(data, model, model_k, yname):
    data['yhat'] = model.predict(data)
    SST = ((data[yname] - data[yname].mean())**2).sum()
    SSR = ((data['yhat'] - data[yname].mean())**2).sum()
    SSE = ((data[yname] - data['yhat'])**2).sum()
    r2 = SSR/SST
    adjustR2 = 1 - (1-r2)*(data.shape[0] - 1)/(data.shape[0] -model_k -1)
    RMSE = (SSE/(data.shape[0] -model_k -1))**0.5
    return adjustR2, RMSE

def assessTable(test, train, model, model_k, yname):
    r2test, RMSEtest = adjustedMetric(test, model, model_k, yname)
    r2train, RMSEtrain = adjustedMetric(train, model, model_k, yname)
    assessment = pd.DataFrame(index=['R2', 'RMSE'], columns=['Train', 'Test'])
    assessment['Train'] = [r2train, RMSEtrain]
    assessment['Test'] = [r2test, RMSEtest]
    return assessment


# Get the assement table fo our model
assessTable(Test, Train, lm, 9, 'spy')




