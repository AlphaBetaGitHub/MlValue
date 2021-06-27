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
from consts import PORTFOLIO_SIZE

import fire

from models_utils import run_random_forest_model, run_xgboost_model, run_nn_model

from preprocessing_utils import handle_missing_values, split_and_scale_data

from report_builder import ReportBuilder

import pdb
import pandas as pd
def main(find_best_param=False, workers=6):
    data = pd.read_pickle('data_us_value.pkl')
    # handle missing values
    data = handle_missing_values(data)
    reports = {}
    pf = {}
    for year in range(2017, 2022):
        # divide df to train, validation and test and normalize them
        X_train, y_train, y_train_unscaled, X_test, y_test, y_test_unscaled, train_data = split_and_scale_data(enriched_data, year)
        print(f'{year} train: {X_train.shape}, test: {X_test.shape}')
         
        # random forest algorithm
        rf = 'random_forest'
        rf_y_pred = run_random_forest_model(year, X_train, y_train, X_test, 
                                            y_test, y_test_unscaled,
                                            train_data, find_best_param,
                                            workers)
        pf[f'{year}_{rf}'] = reports.setdefault(rf,ReportBuilder()).run(year, rf_y_pred, PORTFOLIO_SIZE, rf)
        
        # xgboost algorithm
        print(f'{year} train: {X_train.shape}, test: {X_test.shape}')    
        xg = 'xgboost'
        xg_y_pred = run_xgboost_model(year, X_train, y_train, X_test, y_test,
                                      y_test_unscaled, train_data, 
                                      find_best_param, workers)
        pf[f'{year}_{xg}'] = reports.setdefault(xg,ReportBuilder()).run(year, xg_y_pred, PORTFOLIO_SIZE, xg)
        # 1nn algorithm
         
         
        for nn in ('NN1','NN2','NN3','NN4','NN5'): 
            print(f'{year} train: {X_train.shape}, test: {X_test.shape}')
            nn_y_pred = run_nn_model(year, X_train, y_train, X_test, y_test,
                                     y_test_unscaled, train_data, nn, 
                                     find_best_param, workers)
            pf[f'{year}_{nn}'] = reports.setdefault(nn,ReportBuilder()).run(year, nn_y_pred, PORTFOLIO_SIZE, nn)

if __name__ == '__main__':
    fire.Fire(main)
