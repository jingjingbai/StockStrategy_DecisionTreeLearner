"""MC2-P1: Market simulator."""

import pandas as pd
import numpy as np
import datetime as dt
import os
from util import get_data, get_data2, plot_data
import matplotlib.pyplot as plt

def compute_portvals(orders_file = "./orders/order_benchmark.csv", start_val = 100000):

    # this is the function the autograder will call to test your code
    # TODO: Your code here

    # load CSV orderbook into df_orderbook
    df_orderbook = pd.read_csv(orders_file).sort_values(by='Date')

    # storage symbols into a list
    symbols = df_orderbook['Symbol'].unique().tolist()

    # obtain start and end dates info
    start = df_orderbook['Date'].iloc[0]
    end = df_orderbook['Date'].iloc[-1]

    # tranform start and end dates from strings to dt format
    start_date = dt.datetime.strptime(start, '%Y-%m-%d')
    end_date = dt.datetime.strptime(end, '%Y-%m-%d')

    # read in price info for given start and end date from order book
    df_prices = get_data(symbols, pd.date_range(start_date, end_date))
    df_SPY = df_prices['SPY'] # df with SPY price
    df_prices = df_prices.ix[:, 1:]  # remove SPY
    df_prices['Cash'] = 1 # add Cash as a column with value of 1

    # tranform BUY and SELL into + and -
    df_orderbook.loc[df_orderbook['Order']=='BUY', 'Order'] = 1
    df_orderbook.loc[df_orderbook['Order']=='SELL', 'Order'] = -1
    df_orderbook['Prices'] = df_orderbook['Order']*df_orderbook['Shares']

    # initialize df_trades for storaging trading events
    df_trades = df_prices.copy()
    df_trades[:] = 0

    # read order information into df_trades
    for i in df_orderbook.index:
        index_date = df_orderbook.loc[i, 'Date']
        if index_date in df_trades.index:
            df_trades.loc[index_date, df_orderbook.loc[i, 'Symbol']] += df_orderbook.loc[i, 'Prices']
        else:
            pass

    # add cash pool value
    df_trades['Cash'] = -(df_trades*df_prices).sum(axis=1)

    # set the transaction on the secret day to be zero
    secret_date = '2011-06-15'
    if secret_date in df_trades.index:
        df_trades.loc[secret_date] = 0
    else:
        pass

    # check for leverage and drop orders pass leverage limit
    # no leverage limitation for mc3p3
    '''
    for i in df_orderbook['Date'].unique():
        df_check = 0
        leverage = 0
        df_check = df_trades.loc[:i].cumsum().loc[i]
        leverage = (((df_check[symbols].abs())*(df_prices.loc[i,symbols])).sum())/((df_check*df_prices.loc[i]).sum()+start_val)
        # print [i,leverage]
        if leverage >= 1.5:
            df_trades.loc[i] = 0    
    '''

    # storage stock holding information in df_holdings
    df_holdings = df_trades.cumsum(axis=0)
    df_holdings['Cash'] = df_holdings['Cash'] + start_val

    portvals = pd.DataFrame()
    # calculate total portfolio value = stock value + cash value
    portvals['Portfolio'] = (df_holdings*df_prices).sum(axis=1)
    # portvals['SPY'] = df_SPY*start_val/df_SPY[0]
    
    return portvals



# define a function to calculate portfolio statistics
def portfolio_stat(port_val, sf = 252, rfr = 0.0):
    # Get daily portfolio value
    port_val = port_val/port_val.ix[0,:] # add code here to compute daily portfolio values

    # Get portfolio statistics (note: std_daily_ret = volatility)
    cr = port_val.ix[-1]/port_val.ix[0]-1 # calculate cumulative return
    daily_return = port_val.pct_change(int(252/sf)).dropna() # calculate daily return
    adr = daily_return.mean() # calculate average daily return
    sddr = daily_return.std() # calculate the standard deviation of daily return
    rfr_daily = (1+rfr)**(sf/252)-1 # calculate the daily rfr from given rfr within a certain sampling frequency
    daily_return_riskfree = daily_return - rfr_daily # calculate risk free daily return
    sr = (sf**0.5)*daily_return_riskfree.mean()/daily_return_riskfree.std() # calculate sharpe ratio
    
    return cr, adr, sddr, sr



def test_code(of):
    # this is a helper function you can use to test your code
    # note that during autograding his function will not be called.
    # Define input parameters

    sv = 100000

    # Process orders
    portvals = compute_portvals(orders_file = of, start_val = sv)
    
    if isinstance(portvals, pd.DataFrame):
        portvals = portvals[portvals.columns[0]] # just get the first column
    else:
        "warning, code did not return a DataFrame"
    
    return portvals    

def author():
    return 'zzhao38'


#if __name__ == "__main__":
#    test_code()
