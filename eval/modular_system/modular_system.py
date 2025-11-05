from utils.load_data import load_test_data, DataLoadingParams, decode_ml_outputs
from models.modular_system.modular_system import train_models
import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow import keras
from pathlib import Path

FEATURE_COLUMNS = [
    'load_timestamp_-1', 'load_timestamp_-2', 'load_timestamp_-3',
    'load_previous_day_timestamp_-2', 'load_previous_day_timestamp_-1',
    'load_previous_day_timestamp_0', 'load_previous_day_timestamp_1',
    'load_previous_day_timestamp_2',
    'prev_3_temperature_timestamps_mean',
    'prev_day_temperature_5_timestamps_mean',
    'day_of_week_sin', 'day_of_week_cos',
    'day_of_year_sin', 'day_of_year_cos'    
]

PREV_LOADS = ['load_timestamp_-1', 'load_timestamp_-2', 'load_timestamp_-3']

@tf.function(reduce_retracing=True)
def predict_step(model, x_input):
    return model(x_input, training=False)

def time_to_index(time, freq):
    time_str = time.strftime("%H:%M")
    h, m = map(int, time_str.split(':'))
    if freq == '2h':
        return h/2
    elif freq == '1h':
        return h
    elif freq == '30min':
        return h * 2 + (m // 30)
    elif freq == '15min':
        return h * 4 + (m // 15)
    else:
        raise ValueError("unknown frequency")

def train_and_evaluate_models(hidden_layers: list[list[int]], epochs: list[int], freq: str = "1h", train = True, forecast_range = 20, verbose=0):
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

    if train:
        print('Beginning model training')
        train_models(hidden_layers, epochs, freq, cur_verbose=verbose)  
        print('Models trained!')
        
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
    test_data, raw_data = load_test_data(params)
    project_folder = Path(__file__).parent.parent.parent
    model_folder = project_folder / "models" / "modular_system" /  "models"
 
    grouped = test_data.groupby(test_data.index.time, sort=True)
    n_groups = len(grouped)

    for hidden_layer in hidden_layers:
        hidden_str = "-".join(map(str, hidden_layer))
        for epoch_goal in epochs:
            models = {}
            for i in range(n_groups):
                
                file_name = f"model_{i}_{hidden_str}_{epoch_goal}_{freq}_0.keras"
                model = model_folder / file_name
                models[i] = keras.models.load_model(model)

            X_test = test_data[FEATURE_COLUMNS]
            y_test = test_data['load']

            real_values = {i: [] for i in range(len(models))}
            pred_values = {i: [] for i in range(len(models))}
            for i_total in range(len(test_data) - forecast_range):
                if i_total % 1000 == 0 and i_total > 0:
                    print(f'Processed {i_total} of {len(test_data) - forecast_range} samples')
                X_current = X_test.iloc[i_total:i_total + forecast_range].copy().reset_index(drop=True)
                load_prev1, load_prev2, load_prev3 = X_current[PREV_LOADS].iloc[0]
                for i_pred in range(forecast_range):
                    X_current.at[i_pred, 'load_timestamp_-1'] = load_prev1
                    X_current.at[i_pred, 'load_timestamp_-2'] = load_prev2
                    X_current.at[i_pred, 'load_timestamp_-3'] = load_prev3
                    model_index = time_to_index(X_test.index[i_total + i_pred], freq)
                    model = models[model_index]
                    x_input = tf.convert_to_tensor(X_current.iloc[i_pred:i_pred+1].to_numpy(), dtype=tf.float32)
                    y_pred = float(predict_step(model, x_input)[0,0])
                    pred_values[model_index].append(y_pred)
                    real_values[model_index].append(y_test.iloc[i_total + i_pred])
                    load_prev3, load_prev2, load_prev1 = load_prev2, load_prev1, y_pred
            print('All samples processed')
            path = Path(__file__).parent / "results"
            path.mkdir(parents=True, exist_ok=True)
            with open(path / f"eval_results_{hidden_str}_{epoch_goal}_{freq}.txt", 'w') as f:
                f.write(f"starting_hour, mape_{forecast_range}_*_{freq}\n")
                for i in range(len(models)):
                    y_real = decode_ml_outputs(np.array(real_values[i]).reshape(-1, 1), raw_data).flatten()
                    y_pred = decode_ml_outputs(np.array(pred_values[i]).reshape(-1, 1), raw_data).flatten()
                    mape = np.mean(np.abs((y_real - y_pred) / y_real) * 100)
                    f.write(f"{i}, {mape}\n")


if __name__ == "__main__":
    train_and_evaluate_models([[9]], [100], "1h", train=True, forecast_range=20, verbose=1)

