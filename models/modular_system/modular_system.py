from utils.load_data import load_training_data, DataLoadingParams
from models.rule_aided.base import get_special_days_dict
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.losses import MeanAbsolutePercentageError
import pandas as pd
from pathlib import Path

# This list defines the 14 features the model expects, matching the docstring.
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

def get_model(hidden_layers: list[int]):
    """Function for creating model with desired hidden layers."""
    model = keras.Sequential()
    model.add(layers.Input(shape=(14,)))
    for layer in hidden_layers:
        model.add(layers.Dense(layer, activation='sigmoid'))
    model.add(layers.Dense(1))
    model.compile(optimizer='adam', loss=MeanAbsolutePercentageError(), metrics=['mae', 'mape'])
    return model

def train_models(hidden_layers: list[list[int]], epochs: list[int], freq: str = "1h", id: int = 0, drop_special_days: bool = False):
    """Function for creating and training models with desired hidden layers, epochs and frequency.

    :param hidden_layers: list of hidden layers sizes
                for example hidden_layers = [[24, 12], [20]] -> two models for each epoch number
                first with two hidden layers with [24, 12] nodes and second with one hidden layer  with [20] nodes
    :param epochs: list of epochs numbers for training for each model
                for example epochs = [10, 20] means to train each network 10 and 20 epochs
    :param freq: frequency of data ONLY "15min", "30min", "1h" or "2h"
    :param drop_special_days: whether to drop special day row, for the rule-aided system

    each model input has 14 features:
    1,2,3: Power load at the three previous hours L(i - 1), L(i - 2), L(i - 3).
    4,5,6,7,8: Power load on the previous day at the same hour and for neighbouring ones: L(i - 22), L(i - 23), L(i - 24), L(i - 25), L(i - 26).
    9: Mean temperature at last three hours: (T(i - 1) + T(i - 2) + T(i - 3))/3.
    10: Mean temperature on the previous day at the same hour and for neighbouring hours: (T(i - 22) + T(i - 23) + T(i - 24) + T(i - 25) + T(i - 26))/5
    11,12: The number of the day in week as sin and cos.
    13,14: The number of the day in year as sin and cos.
    
    This function saves each model to subfloder "models"
    each model is saved as 'model_{iteration}_{hidden_layers}_{epoch}_{frequency}.h5'
                for example model_0_8-12-4_20_1h.h5
    """

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

    data, _ = load_training_data(params)
    
    # needed by the rule-aided model
    if drop_special_days:
        special_days_dict = get_special_days_dict()
        special_dates = [pd.to_datetime(k) for k in special_days_dict.keys()]
        
        data = data.reset_index()
        data = data[~data['date'].isin(special_dates)]
        data = data.set_index('date')
    
    current_folder = Path(__file__).parent
    model_folder = current_folder / "models"
    model_folder.mkdir(exist_ok=True)

    grouped = data.groupby(data.index.time, sort=True)
    for i, (time, group) in enumerate(grouped):
        X_train = group[FEATURE_COLUMNS]
        y_train= group['load']
        # we might add data randomisation here
        for hidden_layer in hidden_layers:
            epoch_done = 0
            model = get_model(hidden_layers=hidden_layer)
            for epoch_goal in sorted(epochs):
                model.fit(X_train, y_train, epochs=epoch_goal-epoch_done, verbose=0)
                hidden_str = "-".join(map(str, hidden_layer))
                file_name = f"model_{i}_{hidden_str}_{epoch_goal}_{freq}_{id}.keras"
                model_path = model_folder / file_name
                model.save(model_path)
                epoch_done = epoch_goal
