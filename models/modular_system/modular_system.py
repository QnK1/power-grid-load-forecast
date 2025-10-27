from utils.load_data import load_training_data, DataLoadingParams
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
    model.add(layers.Dense(hidden_layers[0], activation='sigmoid', input_shape=(14,)))
    for layer in hidden_layers[1:]:
        model.add(layers.Dense(layer, activation='sigmoid'))
    model.add(layers.Dense(1))
    model.compile(optimizer='adam', loss=MeanAbsolutePercentageError(), metrics=['mae', 'mape'])
    return model

def create_and_train_models(hidden_layers: list[list[int]], epochs: list[int], freq: str = "1h"):
    """Function for creating and training models..."""
    params = DataLoadingParams()
    # --- Tell the data loader to create the 5 separate 'previous day' features ---
    params.prev_day_load_as_mean = False 
    params.prev_day_temp_as_mean = True
    params.freq = freq
    
    data, _ = load_training_data(params)
    current_folder = Path(__file__).parent
    model_folder = current_folder / "models"
    model_folder.mkdir(exist_ok=True)

    for i, (_, group_df) in enumerate(data.groupby(['hour_of_day_sin', 'hour_of_day_cos'])):
        # Check if all required 14 features are present in the loaded data
        if not all(feature in group_df.columns for feature in FEATURE_COLUMNS):
            print(f"Skipping hour {i} due to missing feature columns.")
            continue

        # Select the correct 14 features
        X_train = group_df[FEATURE_COLUMNS]
        Y_train = group_df[['load']]
        
        for hidden_layer in hidden_layers:
            for epoch in epochs:
                model = get_model(hidden_layers=hidden_layer)
                model.fit(X_train, Y_train, epochs=epoch, verbose=0)
                hidden_str = "-".join(map(str, hidden_layer))
                # --- fixed the path joining bug ---
                file_name = f"model_{i}_{hidden_str}_{epoch}_{freq}.h5"
                model_path = model_folder / file_name
                model.save(model_path)