from __future__ import division
import pandas as pd
from pandas.plotting import radviz
from pandas.plotting import parallel_coordinates

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from keras import layers
from keras import models
import numpy as np

(data, targets) = load_iris(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(data, targets, test_size=.33)

mean = X_train.mean(axis=0)
X_train -= mean
std = X_train.std(axis=0)
X_train /= std

X_test -= mean
X_test /= std

def build_model():
    model = models.Sequential()
    model.add(layers.Dense(4, activation='relu', input_shape=(4,)))
    model.add(layers.Dense(4, activation='relu',))
    model.add(layers.Dense(3, activation='softmax'))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model

k = 4
num_val_samples = len(X_train) // k
num_epochs = 100
all_mae_histories = []

for i in range(k):
    print 'Processing fold #', k
    val_data = X_train[i * num_val_samples: (i + 1) * num_val_samples]
    val_targets = y_train[i * num_val_samples: (i + 1) * num_val_samples]
    
    partial_train_data = np.concatenate([X_train[:i * num_val_samples],
                                         X_train[(i + 1) * num_val_samples:]],
                                        axis=0)
    partial_train_targets = np.concatenate([y_train[:i * num_val_samples],
                                            y_train[(i + 1) * num_val_samples:]],
                                           axis=0)
    model = build_model()
    
    history = model.fit(partial_train_data, partial_train_targets,
                       validation_data=(val_data, val_targets),
                       epochs=num_epochs, batch_size=1, verbose=0)
    mae_history = history.history['val_mean_absolute_error']
    all_mae_histories.append(mae_history)