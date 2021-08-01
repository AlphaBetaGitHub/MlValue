import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import cross_val_score
from xgboost import XGBRegressor

from sparta.tomer.alpha_go.consts import VALIDATION_SIZE
from sparta.tomer.alpha_go.model_constructor import ModelConstructor
from sparta.tomer.alpha_go.nn_constructor import Net
import pdb

def cv_comparison(models, X, y, cv):
    # Define a function that compares the CV performance of a set of predetermined models

    cv_accuracies = pd.DataFrame()
    maes = []
    r2s = []

    # Loop through the models, run a CV, add the average scores to the DataFrame and the scores of
    # all CVs to the list
    for model in models:
        mae = -np.round(cross_val_score(model, X, y, scoring='neg_mean_absolute_error', cv=cv), 4)
        maes.append(mae)
        mae_avg = round(mae.mean(), 4)
        r2 = np.round(cross_val_score(model, X, y, scoring='r2', cv=cv), 4)
        r2s.append(r2)
        r2_avg = round(r2.mean(), 4)
        cv_accuracies[str(model)] = [mae_avg, r2_avg]
    cv_accuracies.index = ['Mean Absolute Error', 'Mean Squared Error', 'R^2', 'Accuracy']
    return cv_accuracies, maes, r2s


def run_random_forest_model(year, X_train, y_train, 
                            X_test, y_test, y_test_unscaled, train_data, 
                            find_best_param, workers):
    # type: (pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, bool, int) -> pd.DataFrame
    """
    Initiate random forest model and provide its predictions
    :param X_train: train set
    :param y_train: train labels
    :param X_test: test set
    :param y_test: test labels
    :param train_data: train set with labels
    :param find_best_param: if true will initiate grid search for optimal parameters
    :param workers: number of workers
    :return: predictions
    """

    model_constructor = ModelConstructor(year, X_train, y_train, X_test, y_test, train_data,
                                         RandomForestRegressor(n_jobs=workers, random_state=42),
                                         VALIDATION_SIZE, workers)
    y_pred = model_constructor.deploy_rf_model(find_best_param)
    model_constructor.evaluate(y_pred)

    predictions = X_test.copy(deep=True)

    predictions['y_pred'] = y_pred
    predictions['y_test'] = y_test_unscaled
    predictions.to_pickle(f'data/results/random_forest/{year}_random_forest_predictions.pkl')
    #train_data.to_pickle(r'random_forest_train_data.pkl')

    return predictions


def run_xgboost_model(year, X_train, y_train, X_test, y_test, y_test_unscaled,
                      train_data, find_best_param, workers):
    # type: (pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, bool, int) -> pd.DataFrame
    """
    Initiate xgboost model and provide its predictions
    :param X_train: train set
    :param y_train: train labels
    :param X_test: test set
    :param y_test: test labels
    :param train_data: train set with labels
    :param find_best_param: if true will initiate grid search for optimal parameters
    :param workers: number of workers
    :return: predictions
    """

    model_constructor = ModelConstructor(year, X_train, y_train, X_test, y_test, train_data,
                                         XGBRegressor(n_jobs=workers, random_state=42),
                                         VALIDATION_SIZE, workers)
    y_pred = model_constructor.deploy_xgboost_model(find_best_param)
    model_constructor.evaluate(y_pred)

    predictions = X_test.copy(deep=True)

    predictions['y_pred'] = y_pred
    predictions['y_test'] = y_test_unscaled
    predictions.to_pickle(f'data/results/xgboost/{year}_xgboost_predictions.pkl')

    return predictions

'''
def run_5_nn_model(year, X_train, y_train, X_test, y_test, train_data, find_best_param, workers):
    # type: (pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, bool, int) -> pd.DataFrame
    """
    Initiate neural network with 5 layers model and provide its predictions
    :param X_train: train set
    :param y_train: train labels
    :param X_test: test set
    :param y_test: test labels
    :param train_data: train set with labels
    :param find_best_param: if true will initiate grid search for optimal parameters
    :param workers: number of workers
    :return: predictions
    """

    nn_model = Net(year, X_train, y_train, X_test, y_test, train_data, '5_layer', find_best_param, VALIDATION_SIZE, workers)
    nn_model.fit_model()
    y_pred = nn_model.predict
    nn_model.loss_evaluate()
    nn_model.evaluate(y_pred)

    predictions = X_test.copy(deep=True)
    predictions['y_pred'] = y_pred
    predictions['y_test'] = y_test
    predictions.to_pickle(f'data/results/5_layer/{year}_5_nn_output_predictions.pkl')

    return predictions


def run_3_nn_model(year, X_train, y_train, X_test, y_test, train_data, find_best_param, workers):
    # type: (pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, bool, int) -> pd.DataFrame
    """
    Initiate neural network with 3 layers model and provide its predictions
    :param X_train: train set
    :param y_train: train labels
    :param X_test: test set
    :param y_test: test labels
    :param train_data: train set with labels
    :param find_best_param: if true will initiate grid search for optimal parameters
    :param workers: number of workers
    :return: predictions
    """

    nn_model = Net(year, X_train, y_train, X_test, y_test, train_data, '3_layer', find_best_param, VALIDATION_SIZE, workers)
    nn_model.fit_model()
    y_pred = nn_model.predict
    nn_model.loss_evaluate()
    nn_model.evaluate(y_pred)

    predictions = X_test.copy(deep=True)
    predictions['y_pred'] = y_pred
    predictions['y_test'] = y_test
    predictions.to_pickle(f'data/results/3_layer/{year}_3_nn_output_predictions.pkl')

    return predictions


def run_1_nn_model(year, X_train, y_train, X_test, y_test, train_data, find_best_param, workers):
    # type: (pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, bool, int) -> pd.DataFrame
    """
    Initiate neural network with 1 layers model and provide its predictions
    :param X_train: train set
    :param y_train: train labels
    :param X_test: test set
    :param y_test: test labels
    :param train_data: train set with labels
    :param find_best_param: if true will initiate grid search for optimal parameters
    :param workers: number of workers
    :return: predictions
    """

    nn_model = Net(year, X_train, y_train, X_test, y_test, train_data, '1_layer', find_best_param, VALIDATION_SIZE, workers)
    nn_model.fit_model()
    y_pred = nn_model.predict
    nn_model.loss_evaluate()
    nn_model.evaluate(y_pred)

    predictions = X_test.copy(deep=True)
    predictions['y_pred'] = y_pred
    predictions['y_test'] = y_test
    predictions.to_pickle(f'data/results/1_layer/{year}_1_nn_output_predictions.pkl')

    return predictions
'''

def run_nn_model(year, X_train, y_train, X_test, y_test, y_test_unscaled,
                 train_data, nn, find_best_param, workers):
    # type: (pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, bool, int) -> pd.DataFrame
    """
    Initiate neural network with 1 layers model and provide its predictions
    :param X_train: train set
    :param y_train: train labels
    :param X_test: test set
    :param y_test: test labels
    :param train_data: train set with labels
    :param find_best_param: if true will initiate grid search for optimal parameters
    :param workers: number of workers
    :return: predictions
    """
    nn_model = Net(year, X_train, y_train, X_test, y_test, train_data, nn, find_best_param, VALIDATION_SIZE, workers)
    nn_model.fit_model()
    y_pred = nn_model.predict
    nn_model.loss_evaluate()
    #nn_model.evaluate(y_pred)

    predictions = X_test.copy(deep=True)
    predictions['y_pred'] = y_pred
    predictions['y_test'] = y_test_unscaled
    predictions.to_pickle(f'data/results/{nn}/{year}_{nn}_predictions.pkl')

    return predictions
