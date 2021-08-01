import sys
sys.path.append('../../../')
import os
import random
import numpy as np
import tensorflow as tf
#from tfdeterminism import patch
#patch()
SEED = 42
os.environ['PYTHONHASHSEED']=str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
#tf.set_random_seed(SEED)
from sparta.tomer.alpha_go.consts import LOCAL_PATH, UNIVERSE_FILE_NAME,\
                                         PORTFOLIO_SIZE, DATA_STYLE, DATA_SRC,\
                                         DATA_SEP, DATA_FIELDS, RENAME_FIELDS,\
                                         DATE_FORMAT

import fire

from sparta.tomer.alpha_go.models_utils import run_random_forest_model, run_xgboost_model, \
     run_nn_model

from sparta.tomer.alpha_go.preprocessing_utils import fetch_data, get_universe_data, prepare_and_merge_data, \
    construct_features, \
    handle_missing_values, split_and_scale_data
from sparta.tomer.alpha_go.report_builder import ReportBuilder

import pdb
import pandas as pd

def get_data():
    if DATA_STYLE == 'ALPHAGO_PAPER':
        uni = get_universe_data(LOCAL_PATH, UNIVERSE_FILE_NAME)
        # get market and financial data
        market_data, financial_data, industry_type_data = fetch_data(uni, fetch_new_data)
    
        # prepare and merge market and financial data
        if fetch_new_data:
            all_data = prepare_and_merge_data(market_data, financial_data, industry_type_data, uni)
            all_data.to_pickle(DATA_SRC)
        else:
            all_data = pd.read_pickle(DATA_SRC)
        # construct ratios
        df = construct_features(all_data)
    elif DATA_STYLE == 'UNPROCESSED_CSV':
        df = pd.read_csv(DATA_SRC, sep=DATA_SEP)[DATA_FIELDS]
        df.rename(columns=RENAME_FIELDS, inplace=True)
        df['pct_change'] = df['pct_change']/100
        df['backtest returns'] = df['backtest returns']/100
        df['date'] = pd.to_datetime(df['date'],format=DATE_FORMAT)
    elif DATA_STYLE == 'PROCESSED_PICKLE':    
        df = pd.read_pickle(DATA_SRC)
    return df
def main(fetch_new_data=False, find_best_param=False, workers=6):
    # type: (bool, bool, int) -> None
    """
    implementation of the Alpha Go Everywhere paper
    :param fetch_new_data: whether or not fetch new data from API
    :param find_best_param: whether or not find the best models params (via grid search)
    :param workers: number of workers
    :return: evaluation reports for each model
    """
    df = get_data()
    enriched_data = handle_missing_values(df)
    reports = {}
    pf = {}
    for year in range(2010, 2022):
        # divide df to train, validation and test and normalize them
        X_train, y_train, y_train_unscaled, X_test, y_test, y_test_unscaled, train_data = split_and_scale_data(enriched_data, year)
        print(f'{year} train: {X_train.shape}, test: {X_test.shape}')
        '''ytrn = y_train_unscaled.rename(columns={'pct_change':'train_return'})
        ytes = y_test_unscaled.rename(columns={'pct_change':'test_return'})
        y = pd.concat([ytrn, ytes])
        X = pd.concat([X_train, X_test])
        X_train.plot.kde().get_figure().savefig(f'{year}-trn-returns.png')
        X_test.plot.kde().get_figure().savefig(f'{year}-tes-returns.png')
        #y.plot.hist().get_figure().savefig(f'{year}-returns.png')
        '''
         
        # random forest algorithm
        rf = 'random_forest'
        #rf_y_pred = pd.read_pickle(f'data/results_std/{rf}/{year}_{rf}_predictions.pkl')
        rf_y_pred = run_random_forest_model(year, X_train, y_train, X_test, 
                                            y_test, y_test_unscaled,
                                            train_data, find_best_param, workers)
        pf[f'{year}_{rf}'] = reports.setdefault(rf,ReportBuilder()).run(year, rf_y_pred, PORTFOLIO_SIZE, rf)
        
        # xgboost algorithm
        print(f'{year} train: {X_train.shape}, test: {X_test.shape}')    
        xg = 'xgboost'
        #xg_y_pred = pd.read_pickle(f'data/results_std/{xg}/{year}_{xg}_predictions.pkl')
        xg_y_pred = run_xgboost_model(year, X_train, y_train, X_test, y_test,
                                      y_test_unscaled, train_data, 
                                      find_best_param, workers)
        pf[f'{year}_{xg}'] = reports.setdefault(xg,ReportBuilder()).run(year, xg_y_pred, PORTFOLIO_SIZE, xg)
        # 1nn algorithm
        
         
        for nn in ('NN1','NN2','NN3','NN4','NN5'):
            print(f'{year} train: {X_train.shape}, test: {X_test.shape}')
            #nn_y_pred = pd.read_pickle(f'data/results_std/{nn}/{year}_{nn}_predictions.pkl')
            nn_y_pred = run_nn_model(year, X_train, y_train, X_test, y_test,
                                     y_test_unscaled, train_data, nn, 
                                     find_best_param, workers)
            pf[f'{year}_{nn}'] = reports.setdefault(nn,ReportBuilder()).run(year, nn_y_pred, PORTFOLIO_SIZE, nn)
        
if __name__ == '__main__':
    fire.Fire(main)
