from utils.load_data import load_test_data, load_training_data, DataLoadingParams, decode_ml_outputs
from models.modular_system.modular_system import Modular_system
import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow import keras
from pathlib import Path
import matplotlib.pyplot as plt
import os
from tensorflow.keras.losses import MeanAbsolutePercentageError
from tensorflow.keras.models import load_model


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

PREV_LOADS = ['load_timestamp_-1', 'load_timestamp_-2', 'load_timestamp_-3']

@tf.function(reduce_retracing=True)
def predict_step(model, x_input):
    return model(x_input, training=False)

def freq_from_number(num_models):
    if num_models == 12:
        return '2h'
    elif num_models == 24:
        return '1h'
    elif num_models == 48:
        return '30min'
    elif num_models == 96:
        return '15min'
    else:
        raise ValueError("unknown num_models")
    
def time_index_from_freq(timestamp, freq):
    delta = pd.to_timedelta(freq)
    step_seconds = delta.total_seconds()
    midnight = timestamp.normalize()
    seconds_since_midnight = (timestamp - midnight).total_seconds()
    index = (seconds_since_midnight // step_seconds).astype(int)
    return index

def train_and_evaluate_model(hidden_layer: list[int], epoch: int, num_models: int = 24, train = True, forecast_range = 20, verbose=0, plot=False):
    """
    Function for creating and training models with desired hidden layers, epochs and frequency. For example:
    train_and_evaluate_models(hidden_layers = [[15, 20, 25]], epochs = [40, 50], freq = "1h", train = True, forecast_range = 20)

    hidden_layers: list of hidden layers sizes
                for example hidden_layers = [[24, 12], [20]] -> two models for each epoch number
                first with two hidden layers with [24, 12] nodes and second with one hidden layer  with [20] nodes
    epochs: list of epochs numbers for training for each model
                for example epochs = [10, 20] means to train each network 10 and 20 epochs
    freq: frequency of data ONLY "15min", "30min", "1h" or "2h"
    train: if True, models will be trained
    forecast_range: number of samples to forecast and calculate mape
    
    """
    freq = freq_from_number(num_models)
    params = DataLoadingParams()
    params.freq = freq
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
    train_data, raw_train_data = load_training_data(params)

    project_folder = Path(__file__).parent.parent.parent
    model_folder = project_folder / "models" / "modular_system" /  "models"
    hidden_str = "-".join(map(str, hidden_layer))

    if train:
        print('Beginning model training')
        model = Modular_system(hidden_layers=hidden_layer, num_models=num_models)
        model.compile(optimizer='adam', loss=MeanAbsolutePercentageError(), metrics=['mae', 'mse', 'mape'])
        indexes = time_index_from_freq(train_data.index, freq)
        indexes_tf = tf.convert_to_tensor(indexes, dtype=tf.int32)
        model.fit({'features': train_data[FEATURE_COLUMNS], 'model_index': indexes_tf},
                  train_data['load'], epochs=epoch, verbose=verbose, batch_size=32)  
        model.save(model_folder / f'modular_system_{hidden_str}_{epoch}_{freq}.keras')
        print('Models trained!')
    else:
        model = load_model(model_folder / f'modular_system_{hidden_str}_{epoch}_{freq}.keras')
   
    test_data, raw_data = load_test_data(params)
    test_data = test_data.sort_index()

 
    grouped = test_data.groupby(test_data.index.time, sort=True)
    n_groups = len(grouped)


    
    X_data_test = test_data[FEATURE_COLUMNS]
    y_data_test = test_data['load']

    real_values = {i: [] for i in range(num_models)}
    pred_values = {i: [] for i in range(num_models)}
    reals = []
    preds = []

    

    prev1_load = X_data_test[0:len(X_data_test)-forecast_range]['load_timestamp_-1'].copy()
    prev2_load = X_data_test[0:len(X_data_test)-forecast_range]['load_timestamp_-2'].copy()
    prev3_load = X_data_test[0:len(X_data_test)-forecast_range]['load_timestamp_-3'].copy()
    print('Evaluation:')
    for i in range(forecast_range):
        percent = (i+1)/forecast_range*100
        print(f'\r{percent:6.2f}%', end='')
        stop = len(X_data_test)-forecast_range+i
        X_data_test.loc[X_data_test.index[i:stop], 'load_timestamp_-1'] = prev1_load
        X_data_test.loc[X_data_test.index[i:stop], 'load_timestamp_-2'] = prev2_load
        X_data_test.loc[X_data_test.index[i:stop], 'load_timestamp_-3'] = prev3_load
        X_test = X_data_test.iloc[i:stop]
        indexes = time_index_from_freq(X_test.index, freq)
        indexes_tf = tf.convert_to_tensor(indexes, dtype=tf.int32)
        y_pred = model.predict({'features': X_test, 'model_index': indexes_tf})
        prev3_load, prev2_load, prev1_load = prev2_load, prev1_load, y_pred
        preds.append(np.copy(y_pred))
        reals.append(np.copy(X_test.to_numpy()))
        model_indexes = time_index_from_freq(X_data_test.index, freq)
        for idx, row in enumerate(y_pred):
            model_index = model_indexes[idx]
            pred_values[model_index].append(row[0])
            real_values[model_index].append(y_data_test.iloc[idx+i])
            
        path = Path(__file__).parent / "results"
        path.mkdir(parents=True, exist_ok=True)
        with open(path / f"eval_results_{hidden_str}_{epoch}_{freq}.txt", 'w') as f:
            f.write(f"starting_hour, mape_{forecast_range}_*_{freq}\n")
            for i in range(num_models):
                y_real = decode_ml_outputs(np.array(real_values[i]).reshape(-1, 1), raw_train_data).flatten()
                y_pred = decode_ml_outputs(np.array(pred_values[i]).reshape(-1, 1), raw_train_data).flatten()
                mape = np.mean(np.abs((y_real - y_pred) / y_real) * 100)
                f.write(f"{i}, {mape}\n")


        







    # if plot:
    #     path = Path(__file__).parent / "plots"
    #     os.makedirs(path, exist_ok=True)
    #     y_real = decode_ml_outputs(np.array(reals).reshape(-1, 1), raw_train_data).flatten()
    #     y_pred = decode_ml_outputs(np.array(preds).reshape(-1, 1), raw_train_data).flatten()
    #     x = list(range(forecast_range))
    #     for day in range(7):
    #         for plot_index in range(len(models)):
    #             index = day*forecast_range*24 + plot_index*forecast_range
    #             plt.plot(x, y_real[index:index+forecast_range], label="real")
    #             plt.plot(x, y_pred[index:index+forecast_range], label="pred")
    #             plt.xticks(x, dates[index:index+forecast_range], rotation=80)
    #             name = f'plot_day-{day}_start-model-{plot_index}'
    #             plt.legend()
    #             plt.tight_layout()
    #             plt.savefig(f'{path}/{name}.png')
    #             plt.clf()

if __name__ == "__main__":
    train_and_evaluate_model(hidden_layer=[25], epoch=20, num_models=24,
                             train=True, forecast_range=20, verbose=1, plot=False)
