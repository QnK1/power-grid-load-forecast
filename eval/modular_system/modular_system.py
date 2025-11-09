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
from analysis.model_analysis import ModelPlotCreator


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

def train_and_evaluate_model(hidden_layer: list[int], epochs: list[int], num_models: int = 24, train = True,
                             forecast_range = 24, verbose=0, plot=False, plot_range=24):
    """
    Function for creating and training models with desired hidden layers, epochs and frequency. For example:
    train_and_evaluate_models(hidden_layers = [[15, 20, 25]], epochs = [40, 50], freq = "1h", train = True, forecast_range = 20)

    :param hidden_layers: list of hidden layers sizes
                for example hidden_layers = [24, 12]
    :param epochs: list of epochs numbers for training for each model
                for example epochs = [10, 20] means to train each network 10 and 20 epochs
    :param num_models: number of models to train
    :param train: if True, models will be trained
    :param forecast_range: number of samples to forecast and calculate mape
    :param verbose: 0, 1 or 2. parametr passed to model.fit(verbose=verbose)
    :param plot: bool, True if you want to plot results, False if you don't 
    :param plot_range: int, number of plots
    
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
    train_data = train_data.sort_index()
    train_data = train_data.iloc[3:]          # deleting first 3 rows where load of prev hours is the same
    train_data = train_data.sample(frac=1)
    

    project_folder = Path(__file__).parent.parent.parent
    model_folder = project_folder / "models" / "modular_system" /  "models"
    hidden_str = "-".join(map(str, hidden_layer))

    if train:
        print('Beginning model training')
        epoch_done = 0
        model = Modular_system(hidden_layers=hidden_layer, num_models=num_models)
        model.compile(optimizer='adam', loss=MeanAbsolutePercentageError(), metrics=['mae', 'mse', 'mape'])
        indexes = time_index_from_freq(train_data.index, freq)
        indexes_tf = tf.convert_to_tensor(indexes, dtype=tf.int32)
        for epoch in sorted(epochs):
            model.fit({'features': train_data[FEATURE_COLUMNS], 'model_index': indexes_tf},
                    train_data['load'], epochs=(epoch-epoch_done), verbose=verbose, batch_size=32)  
            model.save(model_folder / f'modular_system_{hidden_str}_{epoch}_{freq}.keras')
            epoch_done = epoch
        print('Models trained!')
   
    test_data, raw_data = load_test_data(params)

    test_data = test_data.sort_index()
    test_data = test_data.iloc[3:]          # deleting first 3 rows where load of prev hours is the same
 
    grouped = test_data.groupby(test_data.index.time, sort=True)
    n_groups = len(grouped)    
    

    
    print('Evaluation:')
    for epoch in epochs:
        print(f'Eval of modular_system_{hidden_str}_{epoch}')
        X_load = test_data['load'].copy()
        X_data_test = test_data[FEATURE_COLUMNS].copy()
        y_data_test = test_data['load'].copy()
        prev1_load = X_data_test[0:len(X_data_test)-forecast_range]['load_timestamp_-1'].values
        prev2_load = X_data_test[0:len(X_data_test)-forecast_range]['load_timestamp_-2'].values
        prev3_load = X_data_test[0:len(X_data_test)-forecast_range]['load_timestamp_-3'].values
        real_values = {i: [] for i in range(num_models)}
        pred_values = {i: [] for i in range(num_models)}
        reals = []
        preds = []
        model = keras.models.load_model(model_folder / f'modular_system_{hidden_str}_{epoch}_{freq}.keras')
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
            y_pred = model.predict({'features': X_test, 'model_index': indexes_tf}, verbose=0)
            prev3_load, prev2_load, prev1_load = prev2_load, prev1_load, y_pred.flatten()
            
            model_indexes = time_index_from_freq(X_data_test.index, freq)
            for idx, row in enumerate(y_pred):
                model_index = model_indexes[idx]
                pred_values[model_index].append(row[0])
                real_values[model_index].append(y_data_test.iloc[idx+i])
                preds.append(row[0])
                reals.append(y_data_test.iloc[idx+i])
        print()
     
        path = Path(__file__).parent / "results"
        path.mkdir(parents=True, exist_ok=True)
        with open(path / f"eval_results_{hidden_str}_{epoch}_{freq}.txt", 'w') as f:
            f.write(f"starting_hour, mape_{forecast_range}_*_{freq}\n")
            for i in range(num_models):
                y_real = decode_ml_outputs(np.array(real_values[i]).reshape(-1, 1), raw_train_data).flatten()
                y_pred = decode_ml_outputs(np.array(pred_values[i]).reshape(-1, 1), raw_train_data).flatten()
                mape = np.mean(np.abs(y_real - y_pred) / (np.abs(y_real) + 1e-6) * 100)
                f.write(f"{i}, {mape}\n")
        if plot:
            plot_creator = ModelPlotCreator()
            reals = np.array(reals).reshape(forecast_range, -1).T.flatten()
            preds = np.array(preds).reshape(forecast_range, -1).T.flatten()
            reals = decode_ml_outputs(reals.reshape(-1, 1), raw_train_data).flatten()
            preds = decode_ml_outputs(preds.reshape(-1, 1), raw_train_data).flatten()
            for i in range(plot_range):
                plot_creator.plot_predictions(
                    y_real=reals[i*forecast_range : (i+1)*forecast_range], 
                    y_pred=preds[i*forecast_range : (i+1)*forecast_range], 
                    model_name=f'modular_system_{hidden_str}_{epoch}_{freq}_{i}',
                    folder=f'modular_system/model_{hidden_str}_{epoch}',
                    freq=freq,
                    save_plot=True,
                    show_plot=False
                )

if __name__ == "__main__":

    # hiddens = [[5]]
    # epochs = [5,10,15,20]
    hiddens = [[20,15,10,5], [20,15,10,5,1], [25,20,15,10,5], [25,20,15,10,5,1],
               [30,25,20,15,10,5], [30,25,20,15,10,5,1], [35,30,25,20,15,10,5], [35,30,25,20,15,10,5,1],
               [20,5], [25,5], [30,5], [20,5,1], [25,5,1], [30,5,1]]
    epochs = [10,20,30,40,50,60,70,80,90,100]
    for hid in hiddens:
        train_and_evaluate_model(hidden_layer=hid, epochs=epochs, num_models=24, train=True,
                                 forecast_range=24, verbose=1, plot=True, plot_range=72)
