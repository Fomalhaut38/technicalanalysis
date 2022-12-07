import pandas as pd
import mplfinance as mpf
import datetime
import pandas_ta as ta
import pandas_datareader.data as web
import matplotlib.pyplot as plt
pd.set_option('display.max_columns',None)


df = web.DataReader('BTC-USD', 'yahoo' ,datetime.datetime(2019,1,1), datetime.date.today())
bbands = ta.bbands(df['Close'], length = 20, std = 2.5)
df['SMA20'] = ta.sma(df['Close'], length = 20)
df['EMA40'] = ta.ema(df['Close'], length = 40)
df['SMA60'] = ta.sma(df['Close'], length = 60)
df['SMA200'] = ta.sma(df['Close'], length = 200)
df['RSI'] = ta.rsi(df['Close'], length = 12)
macd = ta.macd(df['Close'], fast = 12, slow = 26, signal = 9)
#dfSPY.dropna(inplace = True)
macd.to_csv('macd')
'''print(df)
print(bbands)
help(mpf.plot)'''

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




