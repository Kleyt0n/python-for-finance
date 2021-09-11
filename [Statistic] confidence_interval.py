# -*- coding: utf-8 -*-
"""
Created on Fri Feb  8 13:30:47 2019

@author: Kleyton Sales
"""

import pandas as pd
import numpy as np
from scipy.stats import norm 

stock = pd.DataFrame.from_csv('C:/Users/Dinho/Desktop/Arquivos/4 Finance/3 Financial Analysis/2 StockData/VALE.csv')
print(stock.head())

#Log Return for average stock return
stock['LogReturn'] = np.log(stock['Close'].shift(-1)) - np.log(stock['Close'])

#90% confidence interval for log return
sample_mean = stock['LogReturn'].mean()
sample_std = stock['LogReturn'].std(ddof=1)

#left and right quantile
z_left = norm.ppf(0.1)
z_right = norm.ppf(0.9)

#upper and lower bound
interval_left = sample_mean + z_left * sample_std
interval_right = sample_mean + z_right * sample_std
print('90% confidence interval is ',(interval_left,interval_right))
