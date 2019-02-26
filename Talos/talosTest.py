#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 12:50:22 2018

@author: sameepshah
"""

import talos as ta
import pandas as pd

#%matplotlib inline

x, y = ta.datasets.iris()

from talos.model.early_stopper import early_stopper
from talos.model.normalizers import lr_normalizer

from keras.models import Sequential
from keras.layers import Dropout, Dense

def iris_model(x_train, y_train, x_val, y_val, params):
    
    model = Sequential()                            
    model.add(Dense(params['first_neuron'],
                    input_dim=x_train.shape[1],
                    activation='relu'))
    
    model.add(Dropout(params['dropout']))
    model.add(Dense(y_train.shape[1],
                    activation=params['last_activation']))

    model.compile(optimizer=params['optimizer'](lr=lr_normalizer(params['lr'], params['optimizer'])),
                  loss=params['loss'],
                  metrics=['acc'])

    out = model.fit(x_train, y_train,
                    batch_size=params['batch_size'],
                    epochs=params['epochs'],
                    verbose=0,
                    validation_data=[x_val, y_val],
                    callbacks=early_stopper(params['epochs'], mode='strict'))
    
    return out, model


from keras.optimizers import Adam, Nadam
from keras.activations import softmax
from keras.losses import categorical_crossentropy, logcosh

p = {'lr': (0.1, 10, 10),
     'first_neuron':[4, 8, 16, 32, 64, 128],
     'batch_size': [2, 3, 4],
     'epochs': [200],
     'dropout': (0, 0.40, 10),
     'optimizer': [Adam, Nadam],
     'loss': [categorical_crossentropy, logcosh],
     'last_activation': [softmax],
     'weight_regulizer':[None]}


h = ta.Scan(x, y, params=p,
            model=iris_model,
            dataset_name='iris',
            experiment_no='1',
            grid_downsample=.01)


# accessing the results data frame
h.data.head()

# accessing epoch entropy values for each round
h.peak_epochs_df

# access the summary details
h.details

# accessing the saved models
h.saved_models

# accessing the saved weights for models
h.saved_weights

# use Scan object as input
r = ta.Reporting(h)

# use filename as input
r = ta.Reporting('iris_1.csv')


# access the dataframe with the results
r.data.head(-3)