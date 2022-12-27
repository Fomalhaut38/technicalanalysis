import pandas as pd
import numpy as np
import mplfinance as mpf
import pandas_ta as ta
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime
from datetime import timedelta
import matplotlib.pyplot as plt
import backtesting
from backtesting import Strategy, Backtest

pd.set_option('display.max_columns',None)

df = yf.download('BTC-USD', datetime.datetime(2017,1,1), datetime.date.today())
df.to_csv('database')
#pandas datareader is bugged but idk why. here use yfinance instead


bbands = ta.bbands(df['Close'], length = 20, std = 2.5)
bbands.to_csv('bollingbands')
df['SMA20'] = ta.sma(df['Close'], length = 20)
df['EMA40'] = ta.ema(df['Close'], length = 40)
df['SMA60'] = ta.sma(df['Close'], length = 60)
df['SMA200'] = ta.sma(df['Close'], length = 200)
df['RSI'] = ta.rsi(df['Close'], length = 12)
macd = ta.macd(df['Close'], fast = 12, slow = 26, signal = 9)

macd.to_csv('macd')


'''
add_plot = [mpf.make_addplot(bbands.iloc[:,0], type = 'line', color = 'tomato', width = 0.5),
            mpf.make_addplot(bbands.iloc[:,1], type = 'line', color = 'deepskyblue',width = 0.5),
            mpf.make_addplot(bbands.iloc[:,2], type = 'line', color = 'lime', width = 0.5),
            mpf.make_addplot(df['EMA40'], type = 'line',color = 'hotpink', width = 1),
            mpf.make_addplot(df['SMA200'], type = 'line', color = 'dodgerblue', width = 1),
            mpf.make_addplot(df['RSI'], type = 'line',color = 'r', panel = 2, width = 0.7, ylabel = 'RSI'),
            mpf.make_addplot(macd.iloc[:,1], type = 'bar', color = 'darkgreen', panel = 3, ylabel = 'MACD'),
            mpf.make_addplot(macd.iloc[:,2], type = 'line', color = 'olive', width = 1, panel = 3),
            mpf.make_addplot(macd.iloc[:,0], type = 'line', color = 'coral', width = 1, panel = 3)
            ]
mpf.plot(df, type = 'candle', style = 'yahoo', volume = True, title = 'Bitcoin/USE Trend', ylabel = 'Preis in USD',
         xlabel = 'Date', show_nontrading = False, addplot = add_plot)
'''


def addemasignal(df, backcandles):
    emasignal = [0]*len(df)
    for row in range(backcandles, len(df)):
        upt = 1
        dnt = 1
        for i in range(row - backcandles, row +1):
            if df['High'][i] >= df['EMA40'][i]:
                dnt = 0
            if df['Low'][i] <= df['EMA40'][i]:
                upt = 0
        if upt == 1 and dnt == 1:
            print(f'check trend loop')
            emasignal[row] = 3
        elif upt == 1:
            emasignal[row] = 2
        elif dnt ==1:
            emasignal[row] = 1
    print(f'{emasignal}\n{len(emasignal)}\n{len(df)}')
    df['EMA_Sig'] = emasignal

addemasignal(df,6)

def addorderlimit(df, percent):
    ordersignal = [0]*len(df)
    for i in range(1, len(df)):
        if df['EMA_Sig'][i] == 2 and df['Close'][i] <= bbands['BBM_20_2.5'][i]: #and df['RSI'][i] >= 50:
            ordersignal[i] = df['Close'][i] - df['Close'][i] * percent
        elif df['EMA_Sig'][i] == 1 and df['Close'][i] >= bbands['BBU_20_2.5'][i]: #and df['RSI'] <= 50:
            ordersignal[i] = df['Close'][i] + df['Close'][i] * percent
    df['Order_Sig'] = ordersignal
    print(ordersignal)

addorderlimit(df, 0.03)

def pointposbreak(x):
    if x['Order_Sig'] != 0:
        return x['Order_Sig']
    else:
        return np.nan
df['pointposbreak'] = df.apply(lambda row: pointposbreak(row), axis = 1)
print(df)
df.to_csv('df')


dfplot = df.copy()
fig = make_subplots(rows = 3, cols = 1, print_grid= True ,subplot_titles=('BTC Trend', 'RSI'))
fig.add_trace(go.Candlestick(x = dfplot.index, open = dfplot['Open'], high = dfplot['High'], low = dfplot['Low'], close = dfplot['Close']), row = 1, col = 1)
fig.add_trace(go.Scatter(x = dfplot.index, y = dfplot.EMA40, line = dict(color = 'orange', width = 1), name = 'EMA40'), row = 1, col = 1)
fig.add_trace(go.Scatter(x= dfplot.index, y = bbands['BBU_20_2.5'], line = dict(color = 'red', width = 1), name = 'bolling_up'), row = 1, col = 1)
fig.add_trace(go.Scatter(x = dfplot.index, y = bbands['BBL_20_2.5'], line = dict(color = 'blue', width = 1), name = 'bolling_low'), row = 1, col = 1)

fig.add_scatter(x = dfplot.index, y = dfplot['pointposbreak'], mode = 'markers', marker = dict(color = 'MediumPurple', size = 5), name = 'Signal', row = 1, col = 1)

fig.add_trace(go.Scatter(x = dfplot.index, y = dfplot.RSI, line = dict(color = 'red', width = 1 ), name = 'RSI'),
              row = 2, col = 1)
fig.add_trace(go.Bar(x = dfplot.index, y = macd.iloc[:,0], name = 'MACD', marker_color='darkgreen'),
              row = 3, col = 1)
fig.add_trace(go.Scatter(x = dfplot.index, y = macd.iloc[:,2], line = dict(color = 'olive', width = 1), name = 'MACD-s'),
              row = 3, col = 1)
fig.add_trace(go.Scatter(x = dfplot.index, y = macd.iloc[:,1], line = dict(color = 'coral', width = 1), name = 'MACD-h'),
              row = 3, col = 1)

fig.update_yaxes(range = [0,100], row = 2, col = 1)
fig.show()

def Signal():
    return dfplot.Order_Sig

class MyStrategy(Strategy):
    initsize = 0.5
    ordertime = []
    def init(self):
        super().init()
        self.signal = self.I(Signal)

    def next(self):
        super().init()
        for j in range(0,len(self.orders)):
            if self.data.index[-1] - self.ordertime[0] > timedelta(days = 5) :#days max to fulfill the trade?
                self.orders[0].cancel()
                self.ordertime.pop(0)

        if len(self.trades) >0:
            if self.data.index[-1] - self.trades[-1].entry_time >= timedelta(days = 10):
                self.trades[-1].close()
            if self.trades[-1].is_long and self.data.RSI[-1] >=50:
                self.trades[-1].close()
            elif self.trades[-1].is_short and self.data.RSI[-1] <=50:
                self.trades[-1].close()

        if self.signal != 0 and len(self.trades) ==0 and self.data.EMA_Sig ==2:
            #cancel previous orders
            for j in range(0, len(self.orders)):
                self.orders[0].cancel()
                self.ordertime.pop(0)

            #add new replacement order
            self.buy(sl = self.signal / 2, limit = self.signal, size = self.initsize)
            self.ordertime.append(self.data.index[-1])

        elif self.signal != 0 and len(self.trades) ==0 and self.data.EMA_Sig ==1:
            # cancel previous orders
            for j in range(0, len(self.orders)):
                self.orders[0].cancel()
                self.ordertime.pop(0)

            #add new replacement order
            self.sell(sl = self.signal * 2, limit = self.signal, size = self.initsize)
            self.ordertime.append(self.data.index[-1])

bt = Backtest(dfplot, MyStrategy, cash = 1000000, margin = 1/10, commission = .00)
stat = bt.run()
print(stat)






