
# coding: utf-8

# In[6]:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from util import get_data, get_data2, plot_data, symbol_to_path
import datetime as dt
from matplotlib import gridspec
import RTLearner as rt
from marketsim import compute_portvals, portfolio_stat

# %matplotlib inline

import matplotlib
matplotlib.rcParams.update({'font.size': 22})


# calculate and return normalized dataframe
def normalize(df):
    #result = pd.DataFrame({'Normalized Price': df / df.ix[0]})
    return df / df.ix[0]

# calculate and return standardized dataframe
def standardize(df):
    return (df-df.mean(axis=0))/df.std(axis=0)

# calculate and return dataframe of simple moving average
def SMA(df, window=2):
    SMA = pd.DataFrame({'SMA':df.rolling(window).mean()})
    return SMA


# calculate and return dataframe of Bollinger Bands
def BB(df, window=20):
    n = 2
    average = df.rolling(window).mean()
    std = df.rolling(window).std()
    upper = average + n * std
    lower = average - n * std
    percentage = (df-lower)/(upper-lower)
    result = pd.DataFrame({'BB_upper': upper, 'BB_lower': lower, 'BB_%': percentage})
    return result


# calculate and return dataframe of Moving Average Convergence Divergence
def MACD(df, n_fast=12, n_slow=26, n_macd=9):
    ema_slow = df.ewm(ignore_na=False, span=n_slow, min_periods=1, adjust=True).mean()
    ema_fast = df.ewm(ignore_na=False, span=n_fast, min_periods=1, adjust=True).mean()
    MACD = (ema_fast-ema_slow).ewm(ignore_na=False, span=n_macd, min_periods=1, adjust=True).mean()
    result = pd.DataFrame({'MACD': MACD})
    return result


# calculate and return dataframe of Relative Strength Index
def RSI(df, window):
    delta = df.diff()
    up = delta.copy()
    up[up < 0] = 0
    down = delta.copy()
    down[down > 0] = 0
    rolling_up = up.rolling(window).mean()
    rolling_down = down.rolling(window).mean()
    RS = rolling_up / rolling_down.abs()
    RSI = 100.0 - (100.0 / (1.0 + RS))
    result = pd.DataFrame({'RSI': RSI})
    return result


# calculate and return dataframe of On Balane Volume
def OBV(df_price, df_volume):
    OBV = df_price.copy()
    OBV[:] = 0
    delta = df_price.diff()
    for i in range(1, df_price.shape[0]):
        if delta[i] > 0:
            OBV.ix[i] = OBV.ix[i-1] + df_volume.ix[i, 1]
        elif delta[i] < 0:
            OBV.ix[i] = OBV.ix[i-1] - df_volume.ix[i, 1]
        elif delta[i] == 0:
            OBV.ix[i] = OBV.ix[i-1] + 0

    result = pd.DataFrame({'OBV': OBV},dtype=int)
    return result

def author():
    return 'zzhao38'


# In[7]:

'''
generating order file for manual rule
'''

# in sample start and end dates
sd_train = dt.datetime(2008, 1, 1)
ed_train = dt.datetime(2009, 12, 31)

# out of sample start and end dates
sd_test = dt.datetime(2010, 1, 1)
ed_test = dt.datetime(2011, 12, 31)

# temporary start and end dates to get rid of NaN
sd_temp = dt.datetime(2007,11,1)
ed_temp = dt.datetime(2012,1,31)

dates_train = pd.date_range(sd_train, ed_train)
dates_test = pd.date_range(sd_test, ed_test)
dates_temp = pd.date_range(sd_temp,ed_temp)

symbols = ['AAPL']

# df_train = get_data2(symbols, dates_train)
# df_test = get_data2(symbols, dates_test)
df_temp = get_data2(symbols,dates_temp)

df_temp_price = df_temp.ix[:,0]
df_temp_volume = df_temp.ix[:,1]

df_temp_price = normalize(df_temp_price)

df_test_price_norm = normalize(df_temp_price.ix[dates_test].dropna())

sma = normalize(SMA(df_temp_price, 20).ix[dates_test,:].dropna())

price_to_sma = standardize(pd.DataFrame(df_test_price_norm)/sma.values)
bb = standardize(BB(df_temp_price, 20)).ix[dates_test,:].dropna()
macd = standardize(MACD(df_temp_price)).ix[dates_test,:].dropna()
rsi = standardize(RSI(df_temp_price, 10)).ix[dates_test,:].dropna()
obv = standardize(OBV(df_temp_price, df_temp_volume)).ix[dates_test,:].dropna()


# In[8]:

X_test = pd.concat([price_to_sma, bb['BB_%'], macd, rsi, obv],axis=1)
X_test.columns = ['Price/SMA', '%BB', 'MACD', 'RSI', 'OBV']

# define lower and upper bounds of buy and sell signal based on 5 criterias
BUY_criteria = [-1,-1,-2,-1.5,-1]
SELL_criteria = [1.5,1.5,1,-0.25,1.25]


# In[9]:

# construct a matrix of those buy and sell signals for each trading day
X_criteria = X_test.copy().as_matrix()
for i in range(5):
    X_criteria[X_criteria[:,i] <= BUY_criteria[i],i] = 1.0
    X_criteria[X_criteria[:,i] >= SELL_criteria[i],i] = -1.0
X_criteria[(X_criteria != 1.0) & (X_criteria != -1.0)] = 0.0


# In[11]:

# construct orderbook based on signal matrix
order_rule = pd.DataFrame()
order_rule = order_rule.append({'Date':X_test.index[0], 'Symbol':'AAPL', 'Order':'BUY', 'Shares':0},                                    ignore_index=True)
LONG_date = pd.DataFrame()
SHORT_date = pd.DataFrame()
for i in range(int((X_test.shape[0]-1)/21)):
    # if total number of buy signal - total number of sell signal > = 2, then buy
    if X_criteria[i].sum() >= 2.0:
        order_rule = order_rule.append({'Date':X_test.index[int(i*21)], 'Symbol':'AAPL', 'Order':'BUY', 'Shares':200},                                    ignore_index=True)
        order_rule = order_rule.append({'Date':X_test.index[int(i*21)+21], 'Symbol':'AAPL', 'Order':'SELL', 'Shares':200},                                    ignore_index=True)
        LONG_date = LONG_date.append({'Date':X_test.index[int(i*21)]}, ignore_index=True)
    # if total number of buy signal - total number of sell signal < = -1, then sell
    elif X_criteria[i].sum() <= -1.0:
        order_rule = order_rule.append({'Date':X_test.index[int(i*21)], 'Symbol':'AAPL', 'Order':'SELL', 'Shares':200},                                    ignore_index=True)
        order_rule = order_rule.append({'Date':X_test.index[int(i*21)+21], 'Symbol':'AAPL', 'Order':'BUY', 'Shares':200},                                    ignore_index=True)
        SHORT_date = SHORT_date.append({'Date':X_test.index[int(i*21)]}, ignore_index=True)
        
order_rule = order_rule.append({'Date':X_test.index[-1], 'Symbol':'AAPL', 'Order':'BUY', 'Shares':0},                                    ignore_index=True)
        
order_rule.index = order_rule['Date']
del order_rule['Date']

# save as .csv file
order_rule.to_csv('./orders/order_rule_outofsample.csv')


# In[ ]:



