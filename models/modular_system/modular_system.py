from utils.load_data import load_training_data, DataLoadingParams
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.losses import MeanAbsolutePercentageError
import pandas as pd
from pathlib import Path




def get_model(hidden_layers: list[int]):
    """Function for creating model with desired hidden layers.

    :param hidden_layers: list of hidden layers sizes
                for example hidden_layers = [24, 12] for two hidden layers with 24 and 12 nodes.
    :return: model
    """
    model = keras.Sequential()
    model.add(layers.Dense(hidden_layers[0], activation='sigmoid', input_shape=(14,)))
    for layer in hidden_layers[1:]:
        model.add(layers.Dense(layer, activation='sigmoid'))
    model.add(layers.Dense(1))
    model.compile(
        optimizer='adam',
        loss=MeanAbsolutePercentageError(),  # MAPE
        metrics=['mae', 'mape']
    )
    return model

def create_and_train_models(hidden_layers: list[list[int]], epochs: list[int], freq: str = "1h"):
    """Function for creating and training models with desired hidden layers, epochs and frequency.

    :param hidden_layers: list of hidden layers sizes
                for example hidden_layers = [[24, 12], [20]] -> two models for each epoch number
                first with two hidden layers with [24, 12] nodes and second with one hidden layer  with [20] nodes
    :param epochs: list of epochs numbers for training for each model
    :param freq: frequency of data

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
    params.prev_day_temp_as_mean = True
    params.freq = freq
    data, _ = load_training_data(params)
    current_folder = Path(__file__).parent
    model_folder = current_folder / "models"
    model_folder.mkdir(exist_ok=True)

    for i, (_, group_df) in enumerate(data.groupby(['hour_of_day_sin', 'hour_of_day_cos'])):
        X_train = pd.DataFrame()
        X_train[['load_timestamp_-1', 'load_timestamp_-2', 'load_timestamp_-3']] = group_df[['load_timestamp_-1', 'load_timestamp_-2', 'load_timestamp_-3']]
        # ADD THERE LOADING FROM PREVIOUS DAY
        X_train[['prev_3_temperature_timestamps_mean']] = group_df[['prev_3_temperature_timestamps_mean']]
        X_train[['prev_day_temperature_5_timestamps_mean']] = group_df[['prev_day_temperature_5_timestamps_mean']]
        X_train[['day_of_week_sin', 'day_of_week_cos']] = group_df[['day_of_week_sin', 'day_of_week_cos']]
        X_train[['day_of_year_sin', 'day_of_year_cos']] = group_df[['day_of_year_sin', 'day_of_year_cos']]
        Y_train = pd.DataFrame()
        Y_train['load'] = group_df['load']
        for hidden_layer in hidden_layers:
            for epoch in epochs:
                model = get_model(hidden_layers=hidden_layer)
                # model.fit(X_train, Y_train, epochs=epoch)
                hidden_str = "-".join(map(str, hidden_layer))
                model_path = model_folder / "model_" + str(i) + "_" + hidden_str + "_" + str(epoch) + "_" + freq + ".h5"
                model.save(model_path)





