from utils.load_data import load_test_data, load_training_data, DataLoadingParams, decode_ml_outputs
from models.modular_system.modular_system import train_models
import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow import keras
from pathlib import Path
import matplotlib.pyplot as plt
import os



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

test_data, raw_test_data = load_test_data(params)
train_data, raw_train_data = load_training_data(params)
test_data = test_data.sort_index()
raw_test_data = raw_test_data.sort_index()
train_data = train_data.sort_index()
raw_train_data = raw_train_data.sort_index()

test_load = decode_ml_outputs(np.array(test_data['load']).reshape(-1, 1), raw_test_data).flatten()
train_load = decode_ml_outputs(np.array(train_data['load']).reshape(-1, 1), raw_train_data).flatten()

x = [i for i in range(24*7)]
for i in range(7):
    plt.plot(x, train_load[i*24*365:24*7+i*24*365], label='{train_load.index[i*24*365].strftime("%Y/%m/%d)}')
    plt.show()