import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from xgboost import XGBRegressor
import pandas as pd
import pdb

class ModelConstructor(object):

    def __init__(self, year, X_train, y_train, X_test, y_test, train_data, model, validation_size, workers):
        self.year = year
        self._train_data = train_data.copy(deep=True)
        self.X_train = X_train
        self.y_train = np.array(y_train).reshape(-1, )
        self.X_test = X_test
        self.y_test = np.array(y_test).reshape(-1, )
        self.model = model
        self.predefined_split_sample_size = 12
        self.feature_names = self._train_data.columns.to_list()[:-1]
        self.best_params = None
        self.validation_size = validation_size
        self.workers = workers
        self.features_display = X_test.columns.to_list()

    @staticmethod
    def rf_params():
        rf_n_estimators = [int(x) for x in np.linspace(200, 1000, 5)]
        rf_n_estimators.append(1500)
        rf_n_estimators.append(2000)
        rf_n_estimators = [int(x) for x in np.linspace(200, 1000, 3)]

        # Maximum number of levels in tree
        rf_max_depth = [int(x) for x in np.linspace(5, 55, 11)]

        # Add the default as a possible value
        rf_max_depth.append(None)
        # rf_max_depth = [5, 10]

        # Number of features to consider at every split
        rf_max_features = ['auto', 'sqrt', 'log2']
        # rf_max_features = ['log2']

        # Criterion to split on
        rf_criterion = ['mse', 'mae']
        # rf_criterion = ['mae']

        # Minimum number of samples required to split a node
        rf_min_samples_split = [int(x) for x in np.linspace(2, 10, 9)]
        # rf_min_samples_split = [2, 5, 7]

        # Minimum decrease in impurity required for split to happen
        rf_min_impurity_decrease = [0.0, 0.05, 0.1]
        # rf_min_impurity_decrease = [0.05, 0.1]

        # Method of selecting samples for training each tree
        rf_bootstrap = [True, False]
        # rf_bootstrap = [True]

        return {'n_estimators': rf_n_estimators,
                'max_depth': rf_max_depth,
                'max_features': rf_max_features,
                'criterion': rf_criterion,
                'min_samples_split': rf_min_samples_split,
                'min_impurity_decrease': rf_min_impurity_decrease,
                'bootstrap': rf_bootstrap,
                'random_state': [42]
                }

    @staticmethod
    def get_rf_best_params():

        return {'n_estimators': 300,
                'max_depth': 6,
                'max_features': 5,
                'criterion': 'mae',
                'bootstrap': True,
                'random_state': 42
                }

    @staticmethod
    def xgboost_params():

        xgb_n_estimators = [int(x) for x in np.linspace(200, 2000, 20)]
        # xgb_n_estimators = [200, 2000]

        # Maximum number of levels in tree
        xgb_max_depth = [int(x) for x in np.linspace(2, 20, 10)]
        # xgb_max_depth = [2, 20, 10]

        # Minimum number of instaces needed in each node
        xgb_min_child_weight = [int(x) for x in np.linspace(1, 10, 10)]
        # xgb_min_child_weight = [1, 10, 10]

        # Learning rate
        xgb_eta = [x for x in np.linspace(0.1, 0.6, 6)]
        # xgb_eta = [0.1, 0.6]

        # Minimum loss reduction required to make further partition
        xgb_gamma = [int(x) for x in np.linspace(0, 0.5, 6)]
        # xgb_gamma = [0.1, 0.5]

        return {'n_estimators': xgb_n_estimators,
                'max_depth': xgb_max_depth,
                'min_child_weight': xgb_min_child_weight,
                'learning_rate': xgb_eta,
                'gamma': xgb_gamma,
                'random_state': [42]
                }

    @staticmethod
    def get_xgboost_best_params():
        return {'n_estimators': 1000,
                'max_depth': 2,
                'learning_rate': 0.1,
                'random_state': 42
                }

    # def grid_and_train_model(self, grid_params):
    #     self.train_data.reset_index(inplace=True)
    #     self.train_data.sort_values('date', inplace=True)
    #     self.date_index = pd.to_datetime(self.train_data['date'].unique()).sort_values()
    #
    #     min_training_periods = 12 * 2
    #     self.training_dates = self.date_index[min_training_periods::6]
    #
    #     for i in range(min_training_periods, len(self.date_index)):
    #         self.test_date = self.date_index[i]
    #         self.train_dates = self.date_index[i - min_training_periods:i]
    #
    #         self.valid_dates = self.train_dates[-self.predefined_split_sample_size:]
    #         self.train_dates = self.train_dates[:-self.predefined_split_sample_size]
    #         self.X_train = np.array(
    #             self.train_data.loc[self.train_data['date'].isin(self.train_dates), self.feature_names])
    #         self.y_train = np.array(
    #             self.train_data.loc[self.train_data['date'].isin(self.train_dates), 'pct_change'])
    #         self.X_valid = np.array(
    #             self.train_data.loc[self.train_data['date'].isin(self.valid_dates), self.feature_names])
    #         self.y_valid = np.array(
    #             self.train_data.loc[self.train_data['date'].isin(self.valid_dates), 'pct_change'])
    #         self.test_fold = list(np.ones(len(self.y_train) - len(self.y_valid))) + list(np.zeros(len(self.y_valid)))
    #
    #         self.X_test = np.array(self.train_data.loc[self.train_data['date'] == self.test_date, self.feature_names])
    #         self.y_test = np.array(self.train_data.loc[self.train_data['date'] == self.test_date, 'pct_change'])
    #
    #         cv = [[c for c in PredefinedSplit(self.test_fold).split()][0]]
    #
    #         grid_search = GridSearchCV(estimator=self.model, param_grid=grid_params, refit=False,
    #                                    scoring='neg_mean_squared_error', cv=cv, n_jobs=-1, verbose=3).fit(self.X_train,
    #                                                                                           self.y_train)
    #
    #         self.model = grid_search.best_estimator_
    #         print(f'Model: {self.model} best params are: {grid_search.best_estimator_}')
    #
    #         self.model.fit(self.X_train, self.y_train)
    #
    #         pickle.dump(self.model, open(Path(LOCAL_PATH) / f'model_{self.test_date: %Y-%M-%d}.pkl', 'wb'))

    # def grid_search(self, grid_params, validation_size):
    #
    #     self.X_train.append(self.y_train)
    #     val_dates = train_dates[-validation_size:]
    #
    #     n_test_obs = processed_data['date'].isin(train_dates).sum()
    #     n_valid_obs = processed_data['date'].isin(valid_dates).sum()
    #
    #     test_fold_encoding = list(np.concatenate([np.ones(n_test_obs - n_valid_obs), np.zeros(n_valid_obs)]))
    #
    #     cv = [[c for c in PredefinedSplit(test_fold=test_fold_encoding).split()][0]]
    #
    #     grid_search = GridSearchCV(estimator=self.model, param_grid=grid_params, refit=False,
    #                                scoring='neg_mean_squared_error', cv=cv, n_jobs=1).fit(self.X_train,
    #                                                                                       self.y_train)
    #
    #     print(f'Model: {self.model} best params are: {grid_search.best_estimator_}')
    #     return grid_search.best_estimator_

    def _grid_and_train_model(self, grid_params):

        self._train_data.reset_index(inplace=True)
        self._train_data.sort_values('date', inplace=True)
        self.train_dates = pd.to_datetime(self._train_data['date'].unique()).sort_values()

        val_dates = self.train_dates[-self.validation_size:]

        n_test_obs = self._train_data['date'].isin(self.train_dates).sum()
        n_valid_obs = self._train_data['date'].isin(val_dates).sum()

        test_fold_encoding = list(np.concatenate([np.ones(n_test_obs - n_valid_obs), np.zeros(n_valid_obs)]))

        cv = [[c for c in PredefinedSplit(test_fold=test_fold_encoding).split()][0]]

        grid_search = GridSearchCV(estimator=self.model, param_grid=grid_params, refit=True,
                                   scoring='neg_mean_squared_error', cv=cv, n_jobs=1).fit(self.X_train,
                                                                                          self.y_train)
        print(f'Model: {self.model} best params are: {grid_search.best_params_}')

        self.best_params = grid_search.best_params_

    def deploy_rf_model(self, find_best_param):
        print('-' * 25, ' Initialize RandomForest Regressor ', '-' * 25)

        if find_best_param:
            rf_params = self.rf_params()
            self._grid_and_train_model(rf_params)
        else:
            self.best_params = self.get_rf_best_params()
        #self.model = RandomForestRegressor(n_estimators=300,max_depth=6,random_state = 42,n_jobs=8)

        # assemble best param with new model
        self.model = RandomForestRegressor(n_estimators=self.best_params['n_estimators'],
                                           max_depth=self.best_params['max_depth'],
                                           random_state=self.best_params['random_state'],
                                           #max_features=self.best_params['max_features'],
                                           criterion='mse',#self.best_params['criterion'],
                                           #bootstrap=self.best_params['bootstrap'],
                                           n_jobs=self.workers,
                                           )
        self.model.fit(self.X_train, self.y_train)

        print('Random forest model feature importance:')
        fidx = self.model.feature_importances_.argsort()
        for name, importance in zip(np.array(self.feature_names)[fidx], self.model.feature_importances_[fidx]): 
            print(name, "=", importance)

        return self.model.predict(self.X_test)

    def deploy_xgboost_model(self, find_best_param):
        print('-' * 25, ' Initialize xgboost Regressor ', '-' * 25)
        if find_best_param:
            xgboost_params = self.xgboost_params()
            self._grid_and_train_model(xgboost_params)
        else:
            self.best_params = self.get_xgboost_best_params()

        self.model = XGBRegressor(
            n_estimators=self.best_params['n_estimators'],
            max_depth=self.best_params['max_depth'],
            random_state=self.best_params['random_state'],
            learning_rate=self.best_params['learning_rate'],
            n_jobs=self.workers,
        )
        self.model.fit(self.X_train, self.y_train)

        fidx = self.model.feature_importances_.argsort()
        print('Xgboost model feature importance:')
        for name, importance in zip(np.array(self.feature_names)[fidx], self.model.feature_importances_[fidx]):
            print(name, "=", importance)

        # assemble best param with new model
        return self.model.predict(self.X_test)

    def evaluate(self, y_pred):
        """Evaluate ML models"""
        results = {}
        results['mae'] = mean_absolute_error(self.y_test, y_pred)
        results['r2_score'] = r2_score(self.y_test, y_pred)
        results['rmse'] = np.mean((y_pred - self.y_test) ** 2) ** .5
        for metric in ['mae', 'r2_score', 'rmse']:
            print(
                f'{metric.capitalize()} Test: {results[metric]}')
