import pandas as pd
import numpy as np

dfraw = pd.read_csv('data/data_div.csv')

df = dfraw[['isin', 'periods', 'gpr2m', 'cbgpr2m', 'opr2m', 'cbopr2m','b2m', 'e2m', 'c2m', 'divyield', 'size', 'universe_returns', 'backtest returns']].copy()

df['universe_returns'] = df['universe_returns']/100
df['backtest returns'] = df['backtest returns']/100
df.rename(columns={'periods':'date', 'universe_returns':'pct_change'}, inplace=True)
df['date'] = pd.to_datetime(df['date'])

df.loc[df.groupby('date')['size'].nlargest(1000).index.levels[1]].drop(['size'],axis=1).to_pickle('data_us_value_div.pkl')
