# -*- coding: utf-8 -*-
"""
Created on Thu Feb  7 12:50:30 2019

@author: Kleyton Sales
Code name: Building a simple trading strategy
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt    
from scipy.stats import norm


#import stock data, add two columns - MA10 and MA50
#dropna to remove any "NaN" data

stock = pd.DataFrame.from_csv('C:/Users/Dinho/Desktop/Arquivos/4 Finance/3 Financial Analysis/StockData/VALE.csv')
stock['MA10'] = stock['Close'].rolling(10).mean()
stock['MA50'] = stock['Close'].rolling(50).mean()
stock = stock.dropna()
print(stock.head())

# New column "Shares", if MA10>MA50, denote as 1 (long one share of stock), otherwise, denote as 0 (do nothing)

stock['Shares'] = [1 if stock.loc[i, 'MA10'] > stock.loc[i, 'MA50'] else 0 for i in stock.index]

#Add a new column "Profit" using List Comprehension, for any rows in fb, if Shares=1, the profit is calculated as the close price of 
#tomorrow - the close price of today. Otherwise the profit is 0.

#Plot a graph to show the Profit/Loss

stock['Close1'] = stock['Close'].shift(-1)
stock['Profit'] = [stock.loc[i, 'Close1'] - stock.loc[i, 'Close'] if stock.loc[i, 'Shares']==1 else 0 for i in stock.index]
stock['Profit'].plot()
plt.axhline(y=0, color='red')
plt.show()
#Use .cumsum() to calculate the accumulated wealth over the period

stock['wealth'] = stock['Profit'].cumsum()
print(stock.tail())

#plot the wealth to show the growth of profit over the period

stock['wealth'].plot()
plt.title('Total money you win is {}'.format(stock.loc[stock.index[-2], 'wealth']))
plt.show()