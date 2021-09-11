# -*- coding: utf-8 -*-
"""
Created on Thu Feb  7 10:49:23 2019

@author: Dinho
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

stock = pd.DataFrame.from_csv('C:/Users/Dinho/Desktop/Arquivos/4 Finance/3 Financial Analysis/StockData/VALE.csv')
print(stock.head())

#Distribution of Log Return
#Let lay around with stock data by calculating the log daily return

stock['LogReturn'] = np.log(stock['Close']).shift(-1) - np.log(stock['Close'])
 
#Plot a histogram to show distribution of Log Return of stocks

from scipy.stats import norm
mu = stock['LogReturn'].mean()
sigma = stock['LogReturn'].std(ddof=1)

density = pd.DataFrame()
density['x'] = np.arange(stock['LogReturn'].min() - 0.01, stock['LogReturn'].max() + 0.01, 0.001)
density['pdf'] = norm.pdf(density['x'], mu, sigma)

stock['LogReturn'].hist(bins=60, figsize = (15,8))
plt.plot(density['x'], density['pdf'], color = 'red')
plt.show()

#Probability od the stock price will drop over a certain percentage in a day

# five percent
prob_return5 = norm.cdf(-0.05, mu, sigma)
print('The probability of droping over 5% in one day is ', prob_return5)

# two percent
prob_return2 = norm.cdf(-0.02,mu, sigma)
print('The probability of droping over 2% in one day is  ', prob_return2)

#Probability of the stock price will drop over a certain percentage in a year

#fourty percent
mu220 = 220 * mu
sigma220 = (220**0.5) * sigma
drop40 = norm.cdf(-0.4, mu220, sigma220)
print('The probability of droping over 40% in 220 days is ', drop40)

#twenty percent
drop20 = norm.cdf(-0.2, mu220, sigma220)
print('The probability of droping over 20% in 220 days is', drop20)

#Value at Risk (VaR)

VaR = norm.ppf(0.05, mu, sigma)
print('Singla day value at risk:', VaR)

#Quantile

#5% quantile
print('5% quantile is ', norm.ppf(0.05, mu, sigma))

#95% quantile
print('95% quantile is ', norm.ppf(0.95, mu, sigma))

#25% quantile
print('25% quantile is ', norm.ppf(0.25, mu, sigma))

#75% quantile 
print('75% quantile is ', norm.ppf(0.75, mu, sigma))