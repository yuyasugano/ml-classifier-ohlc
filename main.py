#!/usr/bin/python
import csv
import time
import json
import talib
import requests
import numpy as np
import pandas as pd
from datetime import datetime, date, timedelta, timezone
from backtesting import Backtest
from backtesting import Strategy
from backtesting.lib import crossover
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.externals import joblib

headers = {'Content-Type': 'application/json'}
api_url_base = 'https://public.bitbank.cc'
pair = 'btc_jpy'
period = '1min'

today = datetime.today()
yesterday = today - timedelta(days=1)
today = "{0:%Y%m%d}".format(today)
yesterday = "{0:%Y%m%d}".format(yesterday)

pipe_knn = Pipeline([('scl', StandardScaler()), ('est', KNeighborsClassifier(n_neighbors=3))])
pipe_logistic = Pipeline([('scl', StandardScaler()), ('est', LogisticRegression(solver='lbfgs', multi_class='multinomial', random_state=39))])
pipe_rf = Pipeline([('scl', StandardScaler()), ('est', RandomForestClassifier(random_state=39))])
pipe_gb = Pipeline([('scl', StandardScaler()), ('est', GradientBoostingClassifier(random_state=39))])

def api_ohlcv(timestamp):
    api_url = '{0}/{1}/candlestick/{2}/{3}'.format(api_url_base, pair, period, timestamp)
    response = requests.get(api_url, headers=headers)

    if response.status_code == 200:
        ohlcv = json.loads(response.content.decode('utf-8'))['data']['candlestick'][0]['ohlcv']
        return ohlcv
    else:
        return None

def RF_Backtesting(df):
    """
    Return the label from calculating df(close, volume, diff1, diff2), at
    each step taking into account `n` previous values.
    """
    return pipe_rf.predict(df)

def GB_Backtesting(df):
    """
    Return the label from calculating df(close, volume, diff1, diff2), at
    each step taking into account `n` previous values.
    """
    return pipe_gb.predict(df)

class MachineLearning(Strategy):

    def init(self):
        # Precompute two moving averages
        close = pd.DataFrame({'Close': self.data.Close})
        open = pd.DataFrame({'Open': self.data.Open})
        high = pd.DataFrame({'High': self.data.High})
        low = pd.DataFrame({'Low': self.data.Low})
        volume = pd.DataFrame({'Volume': self.data.Volume})
        self.df = close.join(open).join(high).join(low).join(volume)
        self.df['diff1'] = self.df['Close'] - self.df['Open']
        self.df['diff2'] = self.df['High'] - self.df['Low']
        self.label = self.I(GB_Backtesting, self.df.drop(['Open', 'High', 'Low'], axis=1))
        # self.label = self.I(RF_Backtesting, self.df.drop(['Open', 'High', 'Low'], axis=1))
    
    def next(self):
        if self.label == 1:
            self.buy()
        elif self.label == -1:
            self.sell()

def main():
    ohlcv = api_ohlcv('20190901')
    open, high, low, close, volume, timestamp = [],[],[],[],[],[]

    for i in ohlcv:
        open.append(int(i[0]))
        high.append(int(i[1]))
        low.append(int(i[2]))
        close.append(int(i[3]))
        volume.append(float(i[4]))
        time_str = str(i[5])
        timestamp.append(datetime.fromtimestamp(int(time_str[:10])).strftime('%Y/%m/%d %H:%M:%M'))

    date_time_index = pd.to_datetime(timestamp) # convert to DateTimeIndex type
    df = pd.DataFrame({'open': open, 'high': high, 'low': low, 'close': close, 'volume': volume}, index=date_time_index)
    # adjustment for JST if required
    # df.index += pd.offsets.Hour(9)
    print(df.shape)
    print(df.columns)

    # pct_change
    f = lambda x: 1 if x>0.0001 else -1 if x<-0.0001 else 0 if -0.0001<=x<=0.0001 else np.nan
    y = df.rename(columns={'close': 'y'}).loc[:, 'y'].pct_change(1).shift(-1).fillna(0)
    X = df.copy()
    y_ = pd.DataFrame(y.map(f), columns=['y'])
    df_ = pd.concat([df, y_], axis=1)
    y_std = (y - y.mean())/y.std() # Standalization
    X['diff1'] = X.close - X.open
    X['diff2'] = X.high - X.low
    X_ = X.drop(['open', 'high', 'low'], axis=1)
    X_.join(y_).head(10)

    # check the shape
    print('----------------------------------------------------------------------------------------')
    print('X shape: (%i,%i)' % X.shape)
    print('y shape: (%i,%i)' % y_.shape)
    print('----------------------------------------------------------------------------------------')
    print(y_.groupby('y').size())
    print('y=1 up, y=0 stay, y=-1 down')
    print('----------------------------------------------------------------------------------------')

    X_train, X_test, y_train, y_test = train_test_split(X_, y_, test_size=0.33, random_state=42)
    print('X_train shape: {}'.format(X_train.shape))
    print('X_test shape: {}'.format(X_test.shape))
    print('y_train shape: {}'.format(y_train.shape))
    print('y_test shape: {}'.format(y_test.shape))

    pipe_names = ['KNN','Logistic','RandomForest','GradientBoosting']
    pipe_lines = [pipe_knn, pipe_logistic, pipe_rf, pipe_gb]

    for (i, pipe) in enumerate(pipe_lines):
        pipe.fit(X_train, y_train.values.ravel())
        print('%s: %.3f' % (pipe_names[i] + ' Train Accuracy', accuracy_score(y_train.values.ravel(), pipe.predict(X_train))))
        print('%s: %.3f' % (pipe_names[i] + ' Test Accuracy', accuracy_score(y_test.values.ravel(), pipe.predict(X_test))))
        print('%s: %.3f' % (pipe_names[i] + ' Train F1 Score', f1_score(y_train.values.ravel(), pipe.predict(X_train), average='weighted')))
        print('%s: %.3f' % (pipe_names[i] + ' Test F1 Score', f1_score(y_test.values.ravel(), pipe.predict(X_test), average='weighted')))

    df_ = df.copy()
    df_.columns = ['Close','Open','High','Low','Volume']
    print('{0}\n{1}'.format(yesterday, df_.head(5)))
    bt = Backtest(df_, MachineLearning, cash=1, commission=.002)
    print('Backtesting result:\n', bt.run())

if __name__ == '__main__':
    main()

