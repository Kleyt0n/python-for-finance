# -*- coding: utf-8 -*-
"""
Created on Fri Feb  8 13:52:47 2019

@author: Kleyton Sales
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm 

stock = pd.DataFrame.from_csv('C:/Users/Dinho/Desktop/Arquivos/4 Finance/3 Financial Analysis/2 StockData/VALE.csv')
stock['LogReturn'] = np.log(stock['Close'].shift(-1)) - np.log(stock['Close'])

#Log Return goes up and down during the period
stock['LogReturn'].plot(figsize = (20,8))
plt.axhline(0,color='red')
plt.show()

#Steps involved in test a claim by hypothesis testing
#Step 1 - Set hipothesis
#$H_0 : \mu = 0$
#$H_a : \mu \neq 0$

#Step 2 - Calculate test statistic
sample_mean = stock['LogReturn'].mean()
sample_std = stock['LogReturn'].std(ddof=1)
n = stock['LogReturn'].shape[0]

zhat = (sample_mean - 0)/(sample_std/n**0.5)
print('zhat is ', zhat)

#Step 3 - Set decision criteria
#Confidence level
alpha = 0.05

z_left = norm.ppf(alpha/2, 0, 1)
z_right = -z_left #z-distribution is symmetric
print(z_left,z_right)

#Step 4 - Make decision - shall we reject H0?
print('At significant level of {}, shall we reject: {}'.format(alpha, zhat > z_right or zhat < z_left))

#Alternative method: p-value
p = 1 - norm.cdf(zhat, 0, 1)
print(p)

print('At significant level of {}, shall we reject: {}'.format(alpha, p > alpha))