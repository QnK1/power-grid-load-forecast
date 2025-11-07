from utils.load_data import load_test_data, load_training_data, DataLoadingParams, decode_ml_outputs
import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow import keras
from pathlib import Path
import matplotlib.pyplot as plt
import os
import math
from models.modular_system.modular_system import Modular_system
from tensorflow.keras.losses import MeanAbsolutePercentageError

FEATURE_COLUMNS = [
    'load_timestamp_-1', 'load_timestamp_-2', 'load_timestamp_-3',
    'load_previous_day_timestamp_-2', 'load_previous_day_timestamp_-1',
    'load_previous_day_timestamp_0', 'load_previous_day_timestamp_1',
    'load_previous_day_timestamp_2',
    'prev_3_temperature_timestamps_mean',
    'prev_day_temperature_5_timestamps_mean',
    'day_of_week_sin', 'day_of_week_cos',
    'day_of_year_sin', 'day_of_year_cos',
    'timestamp_day'    
]

params = DataLoadingParams()
params.freq = '1h'
params.prev_load_values = 3
params.prev_day_load_values= (-2, 2)
params.prev_load_as_mean = False
params.prev_day_load_as_mean = False
params.prev_temp_values = 3
params.prev_temp_as_mean = True
params.prev_day_temp_values = (-2, 2)
params.prev_day_temp_as_mean = True
params.interpolate_empty_values = True
params.include_timeindex = True

test_data, raw_test_data = load_test_data(params)
train_data, raw_train_data = load_training_data(params)
test_data = test_data.sort_index()
raw_test_data = raw_test_data.sort_index()
train_data = train_data.sort_index()
raw_train_data = raw_train_data.sort_index()

test_load = decode_ml_outputs(np.array(test_data['load']).reshape(-1, 1), raw_test_data).flatten()
train_load = decode_ml_outputs(np.array(train_data['load']).reshape(-1, 1), raw_train_data).flatten()

# x = [i for i in range(24*7)]
# for i in range(7):
#     plt.plot(x, train_load[i*24*365:24*7+i*24*365], label='{train_load.index[i*24*365].strftime("%Y/%m/%d)}')
#     plt.show()


# print(test_data[['timestamp_day']].head(24))

# for i in range(24):
#     print(np.sin(2 * np.pi * i / 23) == test_data[['hour_of_day_sin']].iloc[i].item(),
#           np.cos(2 * np.pi * i / 23) == test_data[['hour_of_day_cos']].iloc[i].item())


# Metoda tf.config.list_physical_devices zwraca listę dostępnych fizycznych urządzeń
X_train = test_data[FEATURE_COLUMNS] 
y_train = test_data['load']

X_test = train_data[FEATURE_COLUMNS]
y_test = train_data['load']


model = Modular_system([25], 24)
model.compile(optimizer='adam', loss=MeanAbsolutePercentageError(), metrics=['mae', 'mse', 'mape'])
hour_tf = tf.convert_to_tensor(X_train.index.hour.values, dtype=tf.int32)
model.fit({'features': X_train, 'model_index': hour_tf}, y_train, epochs=20)


