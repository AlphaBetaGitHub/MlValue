from sparta.ab.dal import fetch_market_data, fetch_financial_data, fetch_industry_type
from sparta.tomer.alpha_go.consts import COUNTRY_CODE, BOOK_VALUE_CODE, REVENUE_CODE,\
                                         SHARES_CODE, MODE, FEAT_NORM, NUM_FEAT, \
                                         FILLNA_ZERO_FIELDS, FILLNA_MEDIAN
import numpy as np
import datetime
from sklearn import preprocessing
import pandas as pd
import pickle as pkl
from sparta.tomer.alpha_go.feature_engineering_utils import FeatureConstructor, SklearnWrapper
from pathlib import Path
import pdb
def get_universe_data(LOCAL_PATH, UNIVERSE_FILE_NAME):
    # type: (str, str) -> pd.DataFrame
    """
    Get the universe stocks by month
    :param LOCAL_PATH: path to the universe file
    :param US_UNIVERSE_FILE_NAME: name of universe file
    :return: df with universe stocks
    """
    uni = pd.read_csv(Path(LOCAL_PATH)/UNIVERSE_FILE_NAME, usecols=['date', 'name', 'isin', 'country', 'sector'])
    uni = uni[['date','name','isin','country','sector']]
    '''raw_data = [i.strip('\n').split('\t') for i in open(LOCAL_PATH + UNIVERSE_FILE_NAME)]
    uni = pd.DataFrame(raw_data, columns=['date', 'name', 'ticker', 'isin', 'country', 'sector'])
    uni = uni.iloc[1:]
    uni.drop('ticker', axis=1, inplace=True)
    '''
    uni['date'] = pd.to_datetime(uni['date'])

    # for testing
    # us_uni = pd.read_excel(LOCAL_PATH + 'uni_test_file.xlsx')  # TODO: Remove this
    # dedup rows
    uni = uni.drop_duplicates(subset=['date', 'isin'])

    # make sure dates are since 1999
    uni = uni[uni.date > datetime.datetime(1999, 1, 1)]
    return uni


def fetch_data(us_uni, fetch_new_data):
    # type: (pd.DataFrame, bool) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame)
    """
    Get market, financial and industry data from api or from pickle file
    :param us_uni: df of universe stocks
    :param fetch_new_data: fetch new data or from pickle file
    :return: 3 dfs
    """
    isin_list = list(us_uni['isin'].unique())
    min_date = (min(us_uni.date) - datetime.timedelta(days=365)).replace(day=1)  # get a warm up period
    max_date = max(us_uni.date)
    if fetch_new_data:
        market_data = fetch_market_data(
            start_date=min_date,
            end_date=max_date,
            symbols=isin_list,
            symbols_type='isin',
            columns=('price_close_adj', 'volume'),
            excountry=COUNTRY_CODE
        )
        market_data.to_pickle(r'market_data.pkl')
    else:
        market_data = pd.read_pickle(r'market_data.pkl')
    if fetch_new_data:
        financial_data = fetch_financial_data(
            start_date=min_date,
            end_date=max_date,
            symbols=isin_list,
            symbols_type='isin',
            restatement=2,
            period_type=4,  # LTM
            columns=(BOOK_VALUE_CODE, REVENUE_CODE, SHARES_CODE)
        )
        financial_data.to_pickle(r'financial_data.pkl')
    else:
        financial_data = pd.read_pickle(r'financial_data.pkl')

    if fetch_new_data:
        industry_type_data = fetch_industry_type(
            symbols=isin_list,
            symbols_type='isin'
        )
        industry_type_data.to_pickle(r'industry_type_data.pkl')
    else:
        industry_type_data = pd.read_pickle(r'industry_type_data.pkl')

    return market_data, financial_data, industry_type_data


def prepare_and_merge_data(market_data, financial_data, industry_type_data, us_uni):
    # type: (pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame) -> pd.DataFrame
    """
    process the data and merge it into one df.
    Processing the data involves creating 2 features, clean na's, calc monthly returns.
    :param market_data: market data
    :param financial_data: financial data
    :param industry_type_data: industry data
    :param us_uni: stocks universe
    :return: processed df
    """
    market_data['date'] = pd.to_datetime(market_data.date, format='%Y-%m-%d')
    market_data['month'] = 100 * market_data['date'].dt.year + market_data['date'].dt.month  # get a int with the
    # adjust for None tickers (delisted companies)
    #ticker_dict = dict(zip(us_uni['isin'], us_uni['ticker']))
    #market_data['ticker'].replace(None, ticker_dict.get('isin'), inplace=True)

    # year and month attach
    market_data = market_data.drop_duplicates(subset=['isin', 'date'], keep='first').sort_values(
        by=['isin', 'date']).reset_index(drop=True)  # dedup
    #market_data['price_close_adj'] = market_data.groupby('tradingitemid')['price_close_adj'].fillna(method='ffill',limit=30)

    # add max daily return feature
    market_data['pct_daily_change'] = market_data.groupby('tradingitemid')['price_close_adj'].pct_change()#(fill_method='ffill')
    market_data['maxret'] = market_data.groupby(['month', 'tradingitemid']).pct_daily_change.transform('max')

    # add return volatility feature
    market_data['retvol'] = market_data.groupby(['month', 'tradingitemid']).pct_daily_change.transform('std')

    # drop unnecessary column
    market_data.drop('pct_daily_change', axis=1, inplace=True)

    # Move to monthly return
    market_data.set_index('date', inplace=True)
    market_data = market_data.groupby('tradingitemid').resample('M').agg({
        'price_close_adj': 'last', 'volume': 'sum', #'ticker': 'last',
        'companyid': 'last', 'isin': 'last', 'maxret': 'last', 'retvol': 'last', 'month': 'last'})
    market_data['pct_change'] = market_data.groupby('tradingitemid')['price_close_adj'].pct_change().shift(-1)#(fill_method='ffill').shift(-1)
    market_data.reset_index(inplace=True)

    # clean financial data
    financial_data = financial_data[['filingdate', 'isin', 'dataitemid', 'dataitemvalue']]
    financial_data['filingdate'] = pd.to_datetime(financial_data.filingdate, format='%Y-%m-%d')

    financial_data.sort_values(by=['isin', 'filingdate'], inplace=True)
    financial_data.drop_duplicates(subset=['isin', 'filingdate', 'dataitemid'], keep='first', inplace=True)
    financial_data.reset_index(drop=True, inplace=True)

    financial_data.rename(columns={'filingdate': 'date'}, inplace=True)
    # sort and merge financial and market data
    market_data.sort_values(by=['date', 'isin'], inplace=True)
    financial_data.sort_values(by=['date', 'isin'], inplace=True)

    # divide data to items
    book_value_data, equity_data, revenue_data = financial_data[financial_data.dataitemid == BOOK_VALUE_CODE], \
                                                 financial_data[financial_data.dataitemid == SHARES_CODE], \
                                                 financial_data[financial_data.dataitemid == REVENUE_CODE]

    all_data = market_data.copy(deep=True)

    all_data = pd.merge_asof(all_data, book_value_data, on='date', by='isin').drop('dataitemid', axis=1).rename(
        columns={'dataitemvalue': 'book_value'})
    all_data = pd.merge_asof(all_data, equity_data, on='date', by='isin').drop('dataitemid', axis=1).rename(
        columns={'dataitemvalue': 'shares'})
    all_data = pd.merge_asof(all_data, revenue_data, on='date', by='isin').drop('dataitemid', axis=1).rename(
        columns={'dataitemvalue': 'revenue'})

    all_data.sort_values(['isin', 'date'], inplace=True)

    # merge industry_type_data
    industry_type_data = industry_type_data[['companyid', 'isin', 'simpleindustryid']]
    all_data = all_data.merge(industry_type_data, on=['companyid', 'isin'])

    # filter out to only relevant months
    all_data = filter_relevant_months(all_data, us_uni)

    return all_data


def construct_features(all_data):
    # type: (pd.DataFrame) -> pd.DataFrame
    """
    Create the features for the model
    :param all_data: combined processed df
    :return: df with features
    """
    feature_constructor = FeatureConstructor(all_data)

    return feature_constructor.construct_all_features()


def filter_relevant_months(all_data, us_uni):
    # type: (pd.DataFrame, pd.DataFrame) -> pd.DataFrame
    """
    Filter the processed df with the relevant months from the universe
    :param all_data: processed df
    :param us_uni: stocks universe
    :return: filtered df
    """
    us_uni['month'] = 100 * us_uni['date'].dt.year + us_uni['date'].dt.month
    universe_months = us_uni[['isin', 'month']]

    return universe_months.merge(all_data, on=['isin', 'month'], how='left')


def handle_missing_values(enriched_data):
    # type: (pd.DataFrame) -> pd.DataFrame
    """
    fill and remove missing values and remove unnecessary columns
    :param enriched_data: df with features data
    :return: clean df
    """

    # make sure dates are since 2000
    enriched_data = enriched_data[enriched_data.date > datetime.datetime(2000, 1, 1)]

    # Check for nan values
    nan_pct = (enriched_data.isnull().sum(axis=0) / len(enriched_data))
    print(nan_pct[nan_pct > 0.0])

    # deal with extreme values
    enriched_data.replace([np.inf, -np.inf], np.nan, inplace=True)
    # fill nan values with the cross section media
    factors = enriched_data.columns[2:2+NUM_FEAT]
    for f in FILLNA_ZERO_FIELDS:
        enriched_data[f] = enriched_data[f].fillna(0)
    if FILLNA_MEDIAN:
        for fidx,f in enumerate(factors):
            enriched_data[f] = enriched_data[f].fillna(enriched_data.groupby('date')[f].transform('median'))

    # lose irrelevant columns in order to spot the missing na
    #enriched_data = enriched_data.sort_values(by=['tradingitemid', 'month']).reset_index(drop=True)
    #enriched_data.drop(
    #    ['price_close_adj', 'volume', 'book_value', 'shares', 'revenue', 'simpleindustryid', #'ticker',
    #     'companyid', 'month', 'tradingitemid'], axis=1,
    #    inplace=True)

    # Check for nan values
    nan_pct = (enriched_data.isnull().sum(axis=0) / len(enriched_data))
    print(f'Dropping rows with missing values: {nan_pct[nan_pct > 0.0]}')

    # drop any rows with missing values
    enriched_data.dropna(inplace=True)

    return enriched_data


def split_and_scale_data(enriched_data, year):
    # type: (pd.DataFrame) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame)
    """
    get enriched data, split it and scale it
    :param enriched_data: features df
    :return: splitted and scaled df
    """
    #scaler = preprocessing.StandardScaler()
    #enriched_data = enriched_data.groupby('date').apply(SklearnWrapper(scaler))
    train = enriched_data[(enriched_data.date < datetime.datetime(year, 1, 1)) &
                          (enriched_data.date >= datetime.datetime(year-5, 1, 1))]
    mean = train['pct_change'].mean()
    std = train['pct_change'].std()
    bins = list(np.arange(0,1.01,0.1))

    #train = train[(train['pct_change'] > mean + 0.25*std) | (train['pct_change'] < mean - 0.25*std)]
    test = enriched_data[(enriched_data.date >= datetime.datetime(year, 1, 1)) & 
                         (enriched_data.date < datetime.datetime(year+1, 1, 1))]
    if FEAT_NORM == 'RANK':
        factors = train.columns[2:2+NUM_FEAT]
        for fidx,f in enumerate(factors):
            #train[f+'_bins'] = pd.DataFrame(train.groupby('date')[f].apply(lambda x: pd.qcut(x.rank(method='first'),bins,labels=False,duplicates='drop')))
            #test[f+'_bins'] = pd.DataFrame(test.groupby('date')[f].apply(lambda x: pd.qcut(x.rank(method='first'),bins,labels=False,duplicates='drop')))
            train[f+'_rank'] = pd.DataFrame(train.groupby('date')[f].apply(lambda x: (x.rank()/x.rank().max()-0.5)*2))
            test[f+'_rank'] = pd.DataFrame(test.groupby('date')[f].apply(lambda x: (x.rank()/x.rank().max()-0.5)*2))
        #train['pct_change'] = pd.DataFrame(train.groupby('date')['pct_change'].apply(lambda x: (x.rank()/x.rank().max()-0.5)*2))
        #test['pct_change'] = pd.DataFrame(test.groupby('date')['pct_change'].apply(lambda x: (x.rank()/x.rank().max()-0.5)*2))
        mean = train['pct_change'].mean()
        std = train['pct_change'].std()
        bins = list(np.arange(0,1.01,0.1))
        #train = train[(train['pct_change'] > mean + 0.25*std) | (train['pct_change'] < mean - 0.25*std)]
        train = train[['isin', 'date'] + [f+'_rank' for f in factors] + ['pct_change', 'backtest returns']]
        test = test[['isin', 'date'] + [f+'_rank' for f in factors] + ['pct_change', 'backtest returns']]
    train.set_index(['isin', 'date'], drop=True, inplace=True)
    test.set_index(['isin', 'date'], drop=True, inplace=True)
    train_x, train_y, train_ret = train.drop(['pct_change',  'backtest returns'], axis=1), train[['pct_change']], train[['backtest returns']]
    test_x, test_y, test_ret = test.drop(['pct_change', 'backtest returns'], axis=1), test[['pct_change']], test[['backtest returns']]
    '''train_x =  (train.groupby('isin').shift().iloc[:,:-1]-train.iloc[:,:-1]).dropna(axis=0).drop(['pct_change'],axis=1)
    train_y =  (train.groupby('isin').shift().iloc[:,:-1]-train.iloc[:,:-1]).dropna(axis=0)[['pct_change']]
    train_ret = train.loc[~(train.groupby('isin').shift().iloc[:,:-1]-train.iloc[:,:-1]).isna().any(axis=1)][['backtest returns']] 
    test_x =  (test.groupby('isin').shift().iloc[:,:-1]-test.iloc[:,:-1]).dropna(axis=0).drop(['pct_change'],axis=1)
    test_y =  (test.groupby('isin').shift().iloc[:,:-1]-test.iloc[:,:-1]).dropna(axis=0)[['pct_change']]
    test_ret = test.loc[~(test.groupby('isin').shift().iloc[:,:-1]-test.iloc[:,:-1]).isna().any(axis=1)][['backtest returns']]''' 
    if MODE == 'CLS':
        test_y =  pd.DataFrame(test_y.groupby('date')['pct_change'].apply(lambda x: pd.qcut(x.rank(method='first'),bins,labels=False,duplicates='drop')))
        train_y =  pd.DataFrame(train_y.groupby('date')['pct_change'].apply(lambda x: pd.qcut(x.rank(method='first'),bins,labels=False,duplicates='drop')))
        '''train_y[(train_y > 5)] = 20
        train_y[(train_y < 4)] = 0
        train_y[(train_y > 3) & (train_y < 6)] = 10
        train_y = train_y / 10
        test_y[(test_y > 5)] = 20
        test_y[(test_y < 4)] = 0
        test_y[(test_y > 3) & (test_y < 6)] = 10
        test_y = test_y / 10'''
    if FEAT_NORM == 'RANK':
        train_scaled_x = train_x
        test_scaled_x = test_x
    else:
        scaler = preprocessing.StandardScaler()
        train_scaled_x = train_x.groupby('date').apply(SklearnWrapper(scaler))
        scaler = preprocessing.StandardScaler()
        test_scaled_x = test_x.groupby('date').apply(SklearnWrapper(scaler))
    #feat_scaler = preprocessing.RobustScaler()
    #train_scaled_x = feat_scaler.fit_transform(train_x.values)
    #train_scaled_x = pd.DataFrame(train_scaled_x, columns=train_x.columns, index=train_x.index)
    #test_scaled_x = feat_scaler.transform(test_x.values)
    #test_scaled_x = pd.DataFrame(test_scaled_x, columns=test_x.columns, index=test_x.index)
    #with open(f'data/results/feat_scaler-{year}.pkl','wb') as f:
    #    pkl.dump(feat_scaler, f)

    if MODE == 'CLS':
        train_scaled_y = train_y.copy()
        test_scaled_y = test_y.copy()
    else:
        #targ_scaler = preprocessing.MinMaxScaler()
        #train_scaled_y = targ_scaler.fit_transform(train_y.values)
        train_scaled_y = train_y#pd.DataFrame(train_scaled_y, columns=train_y.columns, index=train_y.index)
        #test_scaled_y = targ_scaler.transform(test_y.values)
        test_scaled_y = test_y#pd.DataFrame(test_scaled_y, columns=test_y.columns, index=test_y.index)
        #with open(f'data/results/targ_scaler-{year}.pkl','wb') as f:
        #    pkl.dump(targ_scaler, f)

    train_data = train_scaled_x.merge(train_y, left_index=True, right_index=True)
    return train_scaled_x, train_scaled_y, train_ret, test_scaled_x, test_scaled_y, test_ret, train_data
