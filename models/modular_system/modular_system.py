from utils.load_data import load_training_data, DataLoadingParams
from models.rule_aided.base import get_special_days_dict
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.losses import MeanAbsolutePercentageError
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.callbacks import EarlyStopping
from keras.saving import register_keras_serializable
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
    'day_of_year_sin', 'day_of_year_cos',
    'timestamp_day'
]

@register_keras_serializable()
class Modular_system(keras.Model):
    """
    Class for creating Modular system with desired hidden layers.
    :param hidden_layers: list of hidden layers sizes
    :param num_models: number of models in modular system
                        choose from 12 (freq=2h), 24 (freq=1h), 48 (freq=30min), 96 (freq=15min)

    to fit or predict Modular system model needs dict with keys 'features' -> df of features
                                        and 'model_index' -> 0, 1, ..., num_models-1
    model.fit({'features': X_train, 'model_index': hour_of_day}, y_train, epochs=20)

    """
    def __init__(self,  hidden_layers: list[int], num_models: int = 24, **kwargs):
        super(Modular_system, self).__init__(**kwargs)
        self.num_models = num_models
        self.hidden_layers = hidden_layers
        self.models = []
        for _ in range(self.num_models):
            model = get_model(hidden_layers)
            self.models.append(model)
    def get_config(self):
        config = super().get_config()
        config.update({
            "hidden_layers": self.hidden_layers,
            "num_models": self.num_models,
        })
        return config
    @tf.function
    def call(self, inputs, training=False):
        X_features = inputs['features']
        H_index = inputs['model_index']
        H_index = tf.cast(H_index, tf.int32)
        all_model_outputs = [model(X_features) for model in self.models]
        stacked_outputs = tf.stack(all_model_outputs, axis=0) 
        stacked_outputs = tf.transpose(stacked_outputs, perm=[1, 0, 2])
        batch_indices = tf.range(tf.shape(X_features)[0])
        indices_to_gather = tf.stack([batch_indices, H_index], axis=1)
        final_output = tf.gather_nd(stacked_outputs, indices_to_gather)
        return final_output
        

def get_model(hidden_layers: list[int]):
    """Function for creating model with desired hidden layers."""
    model = keras.Sequential()
    model.add(layers.Input(shape=(15,)))
    for layer in hidden_layers:
        model.add(layers.Dense(layer, activation='relu'))
        # model.add(layers.Dropout(0.2))
    model.add(layers.Dense(1))
    model.compile(optimizer='adam', loss=MeanAbsolutePercentageError(), metrics=['mae', 'mse', 'mape'])
    return model
