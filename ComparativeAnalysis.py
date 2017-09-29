
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from util import get_data, get_data2, plot_data, symbol_to_path
import datetime as dt
from matplotlib import gridspec
import RTLearner as rt
from marketsim import compute_portvals, portfolio_stat

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


# In[2]:

# in sample start and end dates
sd_train = dt.datetime(2008, 1, 1)
ed_train = dt.datetime(2009, 12, 31)

# out of sample start and end dates
sd_test = dt.datetime(2010, 1, 1)
ed_test = dt.datetime(2011, 12, 31)

# temporary start and end dates to get rid of NaN
sd_temp = dt.datetime(2007,11,1)
ed_temp = dt.datetime(2011,12,31)

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

df_train_price_norm = normalize(df_temp_price.ix[dates_train].dropna())

sma = normalize(SMA(df_temp_price, 20).ix[dates_train,:].dropna())

price_to_sma = standardize(pd.DataFrame(df_train_price_norm)/sma.values)

bb = standardize(BB(df_temp_price, 20)).ix[dates_train,:].dropna()
macd = standardize(MACD(df_temp_price)).ix[dates_train,:].dropna()
rsi = standardize(RSI(df_temp_price, 10)).ix[dates_train,:].dropna()
obv = standardize(OBV(df_temp_price, df_temp_volume)).ix[dates_train,:].dropna()



# In[3]:

X_train = pd.concat([price_to_sma, bb['BB_%'], macd, rsi, obv],axis=1)
X_train.columns = ['Price/SMA', '%BB', 'MACD', 'RSI', 'OBV']

Y_train = df_temp_price.pct_change(21).shift(-21,axis=0)
Y_train = Y_train.ix[dates_train].dropna()


# In[4]:

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

X_test = pd.concat([price_to_sma, bb['BB_%'], macd, rsi, obv],axis=1)
X_test.columns = ['Price/SMA', '%BB', 'MACD', 'RSI', 'OBV']


# In[7]:

# define threshold for converting regression into classification
YSELL = -0.05
YBUY = 0.05

# convert Y from regression values into classification values
Y_train[Y_train<YSELL] = -1
Y_train[Y_train>YBUY] = 1
Y_train[(Y_train>=YSELL)&(Y_train<=YBUY)] = 0

# repeat the classification for repeat times
repeat = 100
Y_temp = np.zeros([X_test.shape[0],repeat])
for i in range(repeat):
    learner = rt.RTLearner(leaf_size=5, verbose = True)
    learner.addEvidence(np.array(X_train), np.array(Y_train))
    Y_temp[:,i] = learner.query(np.array(X_test))
Y_pred = Y_temp.mean(axis=1)
    
Y_pred = pd.DataFrame(Y_pred, index=X_test.index)

Y_pred[Y_pred<YSELL] = -1
Y_pred[Y_pred>YBUY] = 1
Y_pred[(Y_pred>=YSELL)&(Y_pred<=YBUY)] = 0

# for printing prediction accuracy
#m=pd.concat([Y_train,Y_pred], axis=1)
#m.ix[m.ix[:,0] == m.ix[:,1],'new']=1
#m.ix[m.ix[:,0] != m.ix[:,1],'new']=0
#m.ix[:,'new'].mean()

# convert Y_pred into orderbook
order_ML = pd.DataFrame()
# in case of no order for first date
order_ML = order_ML.append({'Date':Y_pred.index[0], 'Symbol':'AAPL', 'Order':'BUY', 'Shares':0},                                    ignore_index=True)

LONG_date = pd.DataFrame()
SHORT_date = pd.DataFrame()

for i in range(int((Y_pred.shape[0]-1)/21)):
    if Y_pred.ix[int(i*21),0] == 1.0:
        order_ML = order_ML.append({'Date':Y_pred.index[int(i*21)], 'Symbol':'AAPL', 'Order':'BUY', 'Shares':200},                                    ignore_index=True)
        order_ML = order_ML.append({'Date':Y_pred.index[int(i*21)+21], 'Symbol':'AAPL', 'Order':'SELL', 'Shares':200},                                    ignore_index=True)
        LONG_date = LONG_date.append({'Date':X_train.index[int(i*21)]}, ignore_index=True)
    elif Y_pred.ix[int(i*21),0] == -1.0:
        order_ML = order_ML.append({'Date':Y_pred.index[int(i*21)], 'Symbol':'AAPL', 'Order':'SELL', 'Shares':200},                                    ignore_index=True)
        order_ML = order_ML.append({'Date':Y_pred.index[int(i*21)+21], 'Symbol':'AAPL', 'Order':'BUY', 'Shares':200},                                    ignore_index=True)
        SHORT_date = SHORT_date.append({'Date':X_train.index[int(i*21)]}, ignore_index=True)

order_ML = order_ML.append({'Date':X_test.index[-1], 'Symbol':'AAPL', 'Order':'BUY', 'Shares':0},                                    ignore_index=True)

order_ML.index = order_ML['Date']
del order_ML['Date']

# save as .csv file
order_ML.to_csv('./orders/order_ML_outofsample.csv')



# In[13]:

# compute portfolio stats using orderbook
portfolio_benchmark = compute_portvals('./orders/order_benchmark_outofsample.csv', start_val=100000)
portfolio_rule = compute_portvals('./orders/order_rule_outofsample.csv', start_val=100000)
portfolio_ML = compute_portvals('./orders/order_ML_outofsample.csv', start_val=100000)



# In[15]:

plt.figure(1, figsize=(20,10))
plt.plot(normalize(portfolio_benchmark), color='black')
plt.plot(normalize(portfolio_rule), color='blue')
plt.plot(normalize(portfolio_ML),color='green')
#plt.ylim([0.75,1.8])
#plt.vlines(LONG_date,ymin=0.75,ymax=1.8, colors='green',linestyles='--')
#plt.vlines(SHORT_date,ymin=0.75,ymax=1.8, colors='red',linestyles='--')
plt.xlabel('Date')
plt.ylabel('Normalized Price')
plt.title('Comparison of strageties for test sample')
plt.legend(['Benchmark','Rule based', 'ML based'])
plt.show()


# In[16]:

# for test sample for best strategy

# calculate daily return
daily_return = df_test_price_norm.pct_change(1)
order_best = pd.DataFrame()

# If next day there is positive return => long today, close tomorrow 
# if next day there is negative return => short today, close tomorrow
for i in range(daily_return.shape[0]-1):
    if daily_return[i+1] > 0.0:
        order_best = order_best.append({'Date':daily_return.index[i], 'Symbol':'AAPL', 'Order':'BUY', 'Shares':200}, ignore_index=True)
        order_best = order_best.append({'Date':daily_return.index[i+1], 'Symbol':'AAPL', 'Order':'SELL', 'Shares':200}, ignore_index=True)
    elif daily_return[i+1] < 0.0:
        order_best = order_best.append({'Date':daily_return.index[i], 'Symbol':'AAPL', 'Order':'SELL', 'Shares':200},ignore_index=True)
        order_best = order_best.append({'Date':daily_return.index[i+1], 'Symbol':'AAPL', 'Order':'BUY', 'Shares':200},ignore_index=True)
        
order_best.index = order_best['Date']
del order_best['Date']

# save as .csv file
order_best.to_csv('./orders/order_best_outofsample.csv')


# In[23]:

portfolio_stat(compute_portvals('./orders/order_benchmark_outofsample.csv', start_val=100000))


# In[22]:

portfolio_stat(compute_portvals('./orders/order_best_outofsample.csv', start_val=100000))


# In[24]:

portfolio_stat(compute_portvals('./orders/order_rule_outofsample.csv', start_val=100000))


# In[25]:

portfolio_stat(compute_portvals('./orders/order_ML_outofsample.csv', start_val=100000))


# In[26]:

portfolio_stat(compute_portvals('./orders/order_benchmark.csv', start_val=100000))


# In[27]:

portfolio_stat(compute_portvals('./orders/order_rule.csv', start_val=100000))


# In[28]:

portfolio_stat(compute_portvals('./orders/order_ML.csv', start_val=100000))


# In[ ]:



