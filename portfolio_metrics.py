import pandas as pd
import numpy as np
from datetime import date

def portfolio_metrics(returns_series, strategy_name = 'Strategy_name', strategy_type = 'Strategy_type', returns_freq = 'monthly', show_drawdown = True, five_days_cal = True):
    """
    Calculates and returns a summary of portfolio metrics from monthly / daily returns.
    The metrics:
    - A.Ret: Average annual return
    - A.ann_std: Average annual Standard Deviation
    - 3MDD (3M drawdown): Decline from portfolio's peak value to the lowest value over a period of 3 months
    - 12MDD (12M drawdown): Decline from portfolio's peak value to the lowest value over a period of 12 months
    - Sharpe: The ratio between risk and retrun
    
    Parameters
    ----------
    returns_series : Pandas.Series
        Series of monthly retruns. 
        *The index of the series should be DatetimeIndex*
    strategy_name: String
        The name of the strategy e.g. Book2Market, Momentum, LSTM etc.
    strategy_type: String
        The type the strategy e.g. 'Top 20', 'Bottom 20', 'Benchmark' etc.
    returns_freq: String
        The frequency of the retruns series - 'daily'/'monthly' 
    Returns
    -------
    Pandas.DataFrame
        One row DataFrame with portfolio metrics.
    
    """
    if returns_freq == 'daily':
        if five_days_cal:
            freq = 252
        else:
            freq = 365
    elif returns_freq == 'monthly':
        freq = 12
    
    strategy_name = strategy_name if isinstance(strategy_name, str) else str(strategy_name)
    strategy_type = strategy_type if isinstance(strategy_type, str) else str(strategy_type)

    strategy_name = strategy_name.replace('_',' ')
    strategy_type = strategy_type.replace('_',' ')

    returns = returns_series.copy()
    returns.sort_index(ascending=True, inplace =True)
    
    report = pd.DataFrame()
    report[strategy_name] = [strategy_type]
    report.set_index(strategy_name, inplace = True)
    periods = [3,5,2010,15]  
    for period in periods:
        if period == 2010:
            rets = returns[returns.index >= f'2009-12-31']
            report[f'{period} A.Ret'] = [ann_ret(rets, freq = freq)]
            report[f'{period} A.Std'] = [ann_std(rets, freq = freq)]
            if show_drawdown:
                #report[f'{period} 3MDD'] = [drawdown(rets,3)]
                report[f'{period} 12MDD'] = [drawdown(rets,12)]
            report[f'{period} Sharpe'] = [ann_ret(rets, freq = freq) / ann_std(rets, freq = freq)]
        else:
            report[f'{period}Y A.Ret'] = [ann_ret(returns[-period * freq:], freq = freq)]
            report[f'{period}Y A.std'] = [ann_std(returns[-period * freq:], freq = freq)]
            if show_drawdown:
                #report[f'{period}Y 3MDD'] = [drawdown(returns[-period * freq:],int(freq / 4))]
                report[f'{period}Y 12MDD'] = [drawdown(returns[-period * freq:],freq)]
            report[f'{period}Y Sharpe'] = [ann_ret(returns[-period * freq:], freq = freq) / ann_std(returns[-period * freq:], freq = freq)]
    report = report.round(2)
    return report

def yearly_returns(returns_series, strategy_name = 'strategy_name', strategy_type = 'strategy_type'):
    """
    Calculates and returns a summary of portfolio retruns by year from monthly / daily returns.
    
    Parameters
    ----------
    returns_series : Pandas.Series
        Series of monthly / daily retruns. 
        *The index of the series should be DatetimeIndex*
    strategy_name: String
        The name of the strategy e.g. Book2Market, Momentum, LSTM etc.
    strategy_type: String
        The type the strategy e.g. 'Top 20', 'Bottom 20', 'Benchmark' etc.

    Returns
    -------
    Pandas.DataFrame
        One row DataFrame with portfolio yearly_returns.
    """
    returns = returns_series.copy()
    returns.sort_index(ascending = True, inplace = True)
    returns.index.name = strategy_name
    returns.name = strategy_type
    yearly_returns = returns.groupby(returns.index.year).apply(total_ret)
    yearly_returns.sort_index(ascending = False, inplace = True)
    yearly_returns = yearly_returns.to_frame().T
    yearly_returns.columns = ['YTD' if col == date.today().year else col for col in yearly_returns.columns]
    return yearly_returns.round(1)

def total_ret(x):
    """
    Calculates Total cumalative Return
    """
    return (x.add(1).prod() - 1) * 100

def ann_ret(x, freq):
    """
    Calculates Average Annual Return
    """
    return np.round((x.add(1).prod() ** (freq / len(x)) - 1) * 100,1)
def ann_std(x, freq):
    """
    Calculates Average annual Standard Deviation
    """
    return np.round(np.std(x) * np.sqrt(freq) * 100,1)
def drawdown(x,n):
    """
    Calculates Max Drawdown in n periods
    """
    return np.round(x.rolling(n,n).apply(lambda x: np.prod(1 + x) - 1, raw = False).min() * 100,1)


