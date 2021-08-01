from tensorflow.python.keras import regularizers
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from tensorflow.python.keras.losses import mean_absolute_error
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, BatchNormalization, Dropout
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.python.keras.optimizer_v2.gradient_descent import SGD
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.wrappers.scikit_learn import KerasRegressor
from sklearn import preprocessing
from sparta.tomer.alpha_go.consts import EPOCHS_NUMBER, LR, BATCH_SIZE, LOCAL_PATH, PATIENCE, L1, MODE, NUM_FEAT

from datetime import datetime
import pdb
import os
import random
class Net(tf.keras.Model):

    def __init__(self, year, X_train, y_train, X_test, y_test, train_data, model, find_best_param, validation_size, workers):
        super().__init__()
        self.num_feat = NUM_FEAT
        self.year = year
        self.model_type = model
        self.model_file_name = LOCAL_PATH + f'/results/{model}/{year}_best_model_.h5'
        self.split_train_data(X_train, y_train, find_best_param)
        if MODE == 'CLS':
            self.enc = preprocessing.OneHotEncoder(handle_unknown='ignore')
            self.y_train = self.enc.fit_transform(self.y_train.reshape(-1,1)).toarray()
            self.y_val = self.enc.fit_transform(self.y_val.reshape(-1,1)).toarray()
            self.y_test = self.enc.transform(y_test.values.reshape(-1,1)).toarray()
            self.finact = 'softmax'
            self.loss = 'categorical_crossentropy'
            self.finout = 3
        else:
            self.finact = 'linear'
            self.loss = 'mse'
            self.finout = 1
            self.y_test = np.array(y_test).reshape(-1, )

        self._train_data = train_data.copy(deep=True)
        self.X_test = X_test
        #'glorot_normal' = tf.keras.initializers.GlorotNormal(seed=42)
        self.find_best_param = find_best_param
        '''self.best_params = {'NN5': {'dropout': 0.25, 'lr': 1e-2}, 'NN4':{'dropout':0.0, 'lr': 0.1},
                            'NN3': {'dropout': 0.0, 'lr': 1e-4}, 'NN2':{'dropout':0.25, 'lr': 1e-3},
                            'NN1': {'dropout': 0.0, 'lr': 1e-4}}'''
        '''self.best_params = {'NN5': {'dropout': 0.0, 'lr': LR}, 'NN4':{'dropout':0.0, 'lr': LR},
                            'NN3': {'dropout': 0.0, 'lr': LR}, 'NN2':{'dropout':0.0, 'lr': LR},
                            'NN1': {'dropout': 0.0, 'lr': LR}}'''
        '''self.best_params = {'NN5': {'dropout': 0.25, 'lr': LR}, 'NN4':{'dropout':0.0, 'lr': LR},
                            'NN3': {'dropout': 0.0, 'lr': LR}, 'NN2':{'dropout':0.25, 'lr': LR},
                            'NN1': {'dropout': 0.0, 'lr': LR}}'''
        self.best_params = {'NN5': {'dropout': 0.5, 'lr': LR*9/10000}, 'NN4':{'dropout':0.5, 'lr': LR*9/10000},
                            'NN3': {'dropout': 0.5, 'lr': LR*3/100}, 'NN2':{'dropout':0.25, 'lr': LR*3/100},
                            'NN1': {'dropout': 0.0, 'lr': 9*LR/10}}
        self.model = getattr(self,model)(**self.best_params[model])
        self.validation_size = validation_size
        self.workers = workers

    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'year': self.year,
            'model_file_name': self.model_file_name,
            '_train_data': self._train_data,
            'X_train': self.X_train,
            'X_val': self.X_val,
            'y_train': self.y_train,
            'y_val': self.y_val,
            'X_test': self.X_test,
            'y_test': self.y_test,
            'initializer': 'glorot_normal',
            'model': self.model,
            'find_best_param': self.find_best_param,
            'validation_size': self.validation_size,
            'workers': self.workers
        })
        return config

    # noinspection PyAttributeOutsideInit
    def split_train_data(self, X_train, y_train, find_best_param):

        if not find_best_param:
            self.X_train = X_train[X_train.index.get_level_values(1) <= datetime(self.year-1, 6, 1)]
            self.X_val = X_train[X_train.index.get_level_values(1) > datetime(self.year-1, 6, 1)]
            self.y_train = np.array(y_train[y_train.index.get_level_values(1) <= datetime(self.year-1, 6, 1)]).reshape(-1, )
            self.y_val = np.array(y_train[y_train.index.get_level_values(1) > datetime(self.year-1, 6, 1)]).reshape(-1, )
        else:
            self.X_train = X_train#[X_train.index.get_level_values(1) < datetime(2008, 1, 1)]
            #self.X_val = X_train[X_train.index.get_level_values(1) > datetime(2008, 1, 1)]
            self.y_train = np.array(y_train).reshape(-1,)#[y_train.index.get_level_values(1) < datetime(2008, 1, 1)]).reshape(-1, )
            #self.y_val = np.array(y_train[y_train.index.get_level_values(1) > datetime(2008, 1, 1)]).reshape(-1, )

    def model_resolver(self, model):
        if model == 'NN5':
            print('-' * 25, ' Initialize 5 layer neural network ', '-' * 25)
            return self.NN5(**self.best_params[model])
        elif model == 'NN4':
            print('-' * 25, ' Initialize 4 layer neural network ', '-' * 25)
            return self.NN4(**self.best_params[model])
        elif model == 'NN3':
            print('-' * 25, ' Initialize 3 layer neural network ', '-' * 25)
            return self.NN3()
        elif model == 'NN2':
            print('-' * 25, ' Initialize 2 layer neural network ', '-' * 25)
            return self.NN2()
        elif model == 'NN1':
            print('-' * 25, ' Initialize 1 layer neural network ', '-' * 25)
            return self.NN1()
        else:
            raise Exception('Model not supported')

    def NN5(self, dropout=0.0, l1=L1, lr=1e-2):
        model = Sequential()
        model.add(
            Dense(32, input_dim=self.num_feat, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=l1),
                  kernel_initializer='glorot_normal'))
        model.add(Dropout(dropout))
        model.add(BatchNormalization())
        model.add(Dense(16, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=l1),
                  kernel_initializer='glorot_normal'))
        model.add(Dropout(dropout))
        model.add(BatchNormalization())
        model.add(Dense(8, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=l1),
                  kernel_initializer='glorot_normal'))
        model.add(Dropout(dropout))
        model.add(BatchNormalization())
        model.add(Dense(4, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=l1),
                  kernel_initializer='glorot_normal'))
        model.add(BatchNormalization())
        model.add(Dense(2, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=l1),
                  kernel_initializer='glorot_normal'))
        model.add(BatchNormalization())
        model.add(Dense(self.finout, activation=self.finact, kernel_regularizer=regularizers.l1_l2(l1=l1),
                  kernel_initializer='glorot_normal'))
        model.compile(optimizer=SGD(learning_rate=lr), loss=self.loss)

        return model

    def NN4(self, dropout=0.0, l1=L1, lr=1e-2):

        model = Sequential()
        model.add(
            Dense(32, input_dim=self.num_feat, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=l1),
                  kernel_initializer='glorot_normal'))
        model.add(Dropout(dropout))
        model.add(BatchNormalization())
        model.add(Dense(16, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=l1),
                  kernel_initializer='glorot_normal'))
        model.add(Dropout(dropout))
        model.add(BatchNormalization())
        model.add(Dense(8, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=l1),
                  kernel_initializer='glorot_normal'))
        model.add(Dropout(dropout))
        model.add(BatchNormalization())
        model.add(Dense(4, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=l1),
                  kernel_initializer='glorot_normal'))
        model.add(BatchNormalization())
        model.add(Dense(self.finout, activation=self.finact, kernel_regularizer=regularizers.l1_l2(l1=l1),
                  kernel_initializer='glorot_normal'))
        model.compile(optimizer=SGD(learning_rate=lr), loss=self.loss)

        return model


    def NN3(self, dropout=0.0, l1=L1, lr=1e-2):
        model = Sequential()
        model.add(
            Dense(32, input_dim=self.num_feat, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=l1),
                  kernel_initializer='glorot_normal'))
        model.add(Dropout(dropout))
        model.add(BatchNormalization())
        model.add(Dense(16, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=l1),
                  kernel_initializer='glorot_normal'))
        model.add(Dropout(dropout))
        model.add(BatchNormalization())
        model.add(Dense(8, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=l1),
                  kernel_initializer='glorot_normal'))
        model.add(BatchNormalization())
        model.add(Dense(self.finout, activation=self.finact, kernel_regularizer=regularizers.l1_l2(l1=l1),
                  kernel_initializer='glorot_normal'))
        model.compile(optimizer=SGD(learning_rate=lr), loss=self.loss)

        return model



    def NN2(self, dropout=0.0, l1=L1, lr=1e-2):

        model = Sequential()
        model.add(Dense(32, input_dim=self.num_feat, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=l1),
                  kernel_initializer='glorot_normal'))
        model.add(Dropout(dropout))
        model.add(BatchNormalization())
        model.add(Dense(16, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=l1),
                  kernel_initializer='glorot_normal'))
        model.add(BatchNormalization())
        model.add(Dense(self.finout, activation=self.finact, kernel_regularizer=regularizers.l1_l2(l1=l1),
                  kernel_initializer='glorot_normal'))
        model.compile(optimizer=SGD(learning_rate=lr), loss=self.loss)

        return model

    def NN1(self, dropout=0.0, l1=L1, lr=1e-2):
        model = Sequential()
        model.add(Dense(32, input_dim=self.num_feat, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=l1),
                  kernel_initializer='glorot_normal'))
        model.add(BatchNormalization())
        model.add(Dense(self.finout, activation=self.finact, kernel_regularizer=regularizers.l1_l2(l1=l1),
                  kernel_initializer='glorot_normal'))
        model.compile(optimizer=SGD(learning_rate=lr), loss=self.loss)

        return model

    def fit_model(self):

        if self.find_best_param:
            self._grid_and_fit()
            '''self.model.fit(self.X_train, self.y_train, epochs=self.best_params['epochs'],
                           batch_size=self.best_params['batch_size'], workers=self.workers,
                           callbacks=[
                               ModelCheckpoint(self.model_file_name, monitor='loss', mode='min', save_best_only=True),
                               EarlyStopping(monitor='loss', mode='min', verbose=1, patience=PATIENCE)])
            print(self.model.summary())'''

        else:
            SEED = 42
            os.environ['PYTHONHASHSEED']=str(SEED)
            random.seed(SEED)
            np.random.seed(SEED)
            tf.random.set_seed(SEED)
            
            trainmin, trainmax = self.X_train.index.get_level_values(1).min(), self.X_train.index.get_level_values(1).max()
            testmin, testmax = self.X_test.index.get_level_values(1).min(), self.X_test.index.get_level_values(1).max()
            valmin, valmax = self.X_val.index.get_level_values(1).min(), self.X_val.index.get_level_values(1).max()
            self.model.fit(self.X_train, self.y_train, verbose=0, epochs=EPOCHS_NUMBER, batch_size=BATCH_SIZE,
                           workers=self.workers,
                           callbacks=[
                               ModelCheckpoint(self.model_file_name, monitor='val_loss', mode='min',
                                               save_best_only=True),
                               EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=PATIENCE)],
                           validation_data=(self.X_val, self.y_val))
            print(self.model.summary())
            print(f"train: {trainmin} {trainmax}") 
            print(f"val: {valmin} {valmax}") 
            print(f"test: {testmin} {testmax}") 

    @property
    def predict(self):
        #r1 = self.model.predict(self.X_test[:200000])
        #r2 = self.model.predict(self.X_test[200000:])
        #r = np.concatenate([r1,r2]).reshape(-1)
        if MODE == 'CLS':
            r = np.argmax(self.model.predict(self.X_test),axis=1) + np.max(self.model.predict(self.X_test),axis=1)
        else:
            r = self.model.predict(self.X_test).reshape(-1)
        return r
    def loss_evaluate(self):
        loss = self.model.evaluate(self.X_test, self.y_test)
        print(f"loss: {loss}")

    def evaluate(self, y_pred):
        """Evaluate NN models"""
        results = {'mae': mean_absolute_error(self.y_test, y_pred), 'r2_score': r2_score(self.y_test, y_pred),
                   'rmse': np.mean((y_pred - self.y_test) ** 2) ** .5}

        for metric in ['mae', 'r2_score', 'rmse']:
            print(
                f'{metric.capitalize()} Test: {results[metric]}')

    def _grid_and_fit(self):

        # grid params
        lr = [1e-1, 1e-2, 1e-3, 1e-4]
        dropout = [0.0, 0.25, 0.5]
        #batch_size = [10, 20, 30]
        #epochs = [50, 100, 200]

        # dictionary of the grid search parameters
        #param_grid = dict(batch_size=batch_size, epochs=epochs,
        #                  shuffle=[False])
        param_grid = dict(lr=lr, dropout=dropout)

        self._train_data.reset_index(inplace=True)
        self._train_data.sort_values('date', inplace=True)
        self.train_dates = pd.to_datetime(self._train_data['date'].unique()).sort_values()

        val_dates = self.train_dates[-self.validation_size:]

        n_test_obs = self._train_data['date'].isin(self.train_dates).sum()
        n_valid_obs = self._train_data['date'].isin(val_dates).sum()

        test_fold_encoding = list(np.concatenate([np.ones(n_test_obs - n_valid_obs), np.zeros(n_valid_obs)]))

        cv = [[c for c in PredefinedSplit(test_fold=test_fold_encoding).split()][0]]

        # Build and fit the GridSearchCV
        nnbuilder = getattr(self, self.model_type)
        pdb.set_trace()
        kmodel = KerasRegressor(build_fn=nnbuilder, verbose=1)
        grid_search = GridSearchCV(estimator=kmodel, param_grid=param_grid,
                                   cv=cv, verbose=10, scoring='neg_mean_squared_error', refit=False)
        grid_search.fit(self._train_data.drop('pct_change', axis=1).iloc[:,2:].values, np.array(self._train_data['pct_change']).reshape(-1,))

        print(f'Model: {self.model_type} best params are: {grid_search.best_params_}')
        pdb.set_trace()
        self.best_params = grid_search.best_params_
